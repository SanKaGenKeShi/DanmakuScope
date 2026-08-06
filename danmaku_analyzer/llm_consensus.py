"""双路推理共识度量 - JSD 计算、共识水平判定与输出合并策略"""

from typing import List, Dict, Optional

import numpy as np

from .llm_models import ConsensusLevel, LLMOutput
from .utils.logger import get_logger

logger = get_logger(__name__)


def calculate_jsd(outputs: List[Dict]) -> float:
    """多维共识度量：各维 JSD 按 ln(n) 归一化到 [0,1] 后取均值，消除类别数差异导致的主导偏差"""
    if len(outputs) < 2:
        return 0.0
    
    try:
        categorical_dims = [
            ("emotion", "label", ["positive", "neutral", "negative"]),
            ("interaction_type", "label", ["check_in", "identity_claim", "mocking", "info_request", "expression", "other"]),
            ("orthography", "status", ["standard", "community_variant", "non_standard_typo"]),
        ]
        
        jsd_scores = []
        for dim_key, label_field, labels in categorical_dims:
            distributions = [
                categorical_distribution(
                    output.get(dim_key, {}).get(label_field),
                    output.get(dim_key, {}).get("confidence", 0.5),
                    labels,
                )
                for output in outputs
            ]
            jsd_scores.append(_normalized_jsd(distributions))
        
        # 合作原则无 confidence 字段，按 violated 二元确定性分布参与度量
        cp_distributions = [
            np.array([1.0, 0.0]) if output.get("cooperative_principle", {}).get("violated", False)
            else np.array([0.0, 1.0])
            for output in outputs
        ]
        jsd_scores.append(_normalized_jsd(cp_distributions))
        
        return float(np.mean(jsd_scores))
        
    except Exception as e:
        # 无法计算等同于不确定，按最大散度处理（LOW 共识、权重 0.2），遵循保守策略
        logger.warning(f"JSD 计算失败，按最大散度处理: {e}")
        return 1.0


def categorical_distribution(label: Optional[str], confidence: float, labels: List[str]) -> np.ndarray:
    """标签+置信度 → 概率分布；未知标签按均匀分布（不偏向任何一侧）"""
    dist = np.ones(len(labels)) / len(labels)
    if label in labels:
        idx = labels.index(label)
        dist[idx] = confidence
        remaining = (1 - confidence) / (len(labels) - 1)
        for i in range(len(labels)):
            if i != idx:
                dist[i] = remaining
    return dist


def _jsd_of(distributions: List[np.ndarray]) -> float:
    # epsilon 平滑避免零概率除零
    eps = 1e-10
    distributions = [d + eps for d in distributions]
    distributions = [d / d.sum() for d in distributions]
    
    avg_dist = np.mean(distributions, axis=0)
    jsd = 0.0
    for dist in distributions:
        kl = np.sum(dist * np.log(dist / avg_dist))
        jsd += kl
    jsd /= len(distributions)
    
    return float(jsd)


def _normalized_jsd(distributions: List[np.ndarray]) -> float:
    """JSD 除以其理论上限 ln(m)（m 为分布数，支撑集互斥时取到），映射到 [0,1] 使各维可比"""
    max_jsd = np.log(len(distributions))
    if max_jsd <= 0:
        return 0.0
    return float(_jsd_of(distributions) / max_jsd)


def determine_consensus_level(jsd_score: float, threshold_low: float, threshold_medium: float) -> ConsensusLevel:
    if jsd_score < threshold_low:
        return ConsensusLevel.HIGH
    elif jsd_score < threshold_medium:
        return ConsensusLevel.MEDIUM
    else:
        return ConsensusLevel.LOW


def merge_outputs(outputs: List[Dict], consensus_level: ConsensusLevel) -> LLMOutput:
    """多路输出合并：高共识取首路，否则按各维度 confidence 总和择优"""
    if not outputs:
        return LLMOutput.default()
    
    if consensus_level == ConsensusLevel.HIGH:
        return LLMOutput.from_dict(outputs[0])
    
    # 非高共识时按各维度 confidence 总和择优，避免仅凭情感单维自信度选路
    def total_confidence(output: Dict) -> float:
        return sum(
            output.get(dim, {}).get(field, 0.5)
            for dim, field in (("emotion", "confidence"), ("interaction_type", "confidence"), ("orthography", "confidence"))
        )
    
    best_output = max(outputs, key=total_confidence)
    return LLMOutput.from_dict(best_output)


def calculate_weight_multiplier(consensus_level: ConsensusLevel, low_consensus_weight: float) -> float:
    if consensus_level == ConsensusLevel.LOW:
        return low_consensus_weight
    else:
        return 1.0
