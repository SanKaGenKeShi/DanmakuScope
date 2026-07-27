"""
聚合器模块 - 嵌套聚合（分区/热区）
分层键：tname（官方分区） + hot_zone（热区）
tags 仅作为元数据附加，不参与分组
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from collections import defaultdict

from .hard_metrics import HardMetricsResult
from .llm_client import DualPathResult
from .utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class AggregatedData:
    """聚合数据"""
    # 分组键
    tname: str  # 官方分区
    zone_type: str  # hot_zone 或 cold_zone
    
    # 元数据
    tags: List[str] = field(default_factory=list)  # 标签（仅元数据）
    video_count: int = 0  # 视频数
    segment_count: int = 0  # 段数
    danmaku_count: int = 0  # 弹幕数
    
    # 硬统计聚合
    avg_word_length: float = 0.0
    content_word_density: float = 0.0
    punctuation_emoji_rate: float = 0.0
    pos_distribution: Dict[str, float] = field(default_factory=dict)
    syllable_distribution: Dict[str, float] = field(default_factory=dict)
    orthography_hard_metrics: Dict[str, float] = field(default_factory=dict)
    
    # 软标签聚合
    emotion_distribution: Dict[str, float] = field(default_factory=dict)
    sentence_function_distribution: Dict[str, float] = field(default_factory=dict)
    interaction_type_distribution: Dict[str, float] = field(default_factory=dict)
    orthography_status_distribution: Dict[str, float] = field(default_factory=dict)
    
    # 共识统计
    high_consensus_rate: float = 0.0
    medium_consensus_rate: float = 0.0
    low_consensus_rate: float = 0.0
    avg_weight_multiplier: float = 1.0
    
    # 统计验证（Wilson 置信区间）
    consensus_ci: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "tname": self.tname,
            "zone_type": self.zone_type,
            "tags": self.tags,
            "video_count": self.video_count,
            "segment_count": self.segment_count,
            "danmaku_count": self.danmaku_count,
            "hard_metrics": {
                "avg_word_length": round(self.avg_word_length, 4),
                "content_word_density": round(self.content_word_density, 4),
                "punctuation_emoji_rate": round(self.punctuation_emoji_rate, 4),
                "pos_distribution": self.pos_distribution,
                "syllable_distribution": self.syllable_distribution,
                "orthography_hard_metrics": self.orthography_hard_metrics,
            },
            "soft_labels": {
                "emotion_distribution": self.emotion_distribution,
                "sentence_function_distribution": self.sentence_function_distribution,
                "interaction_type_distribution": self.interaction_type_distribution,
                "orthography_status_distribution": self.orthography_status_distribution,
            },
            "consensus_stats": {
                "high_consensus_rate": round(self.high_consensus_rate, 4),
                "medium_consensus_rate": round(self.medium_consensus_rate, 4),
                "low_consensus_rate": round(self.low_consensus_rate, 4),
                "avg_weight_multiplier": round(self.avg_weight_multiplier, 4),
                "consensus_ci": self.consensus_ci,
            },
        }


@dataclass
class DanmakuRecord:
    """弹幕记录（用于聚合）"""
    tname: str
    zone_type: str
    tags: List[str]
    hard_metrics: HardMetricsResult
    llm_result: DualPathResult
    segment_id: int = 0  # 所属段索引，用于去重段级硬统计


class Aggregator:

    def aggregate(self, records: List[DanmakuRecord]) -> List[AggregatedData]:
        if not records:
            return []
        
        logger.info(f"开始聚合，共 {len(records)} 条记录")
        
        # 按 (tname, zone_type) 分组
        groups: Dict[tuple, List[DanmakuRecord]] = defaultdict(list)
        
        for record in records:
            key = (record.tname, record.zone_type)
            groups[key].append(record)
        
        # 聚合每个组
        aggregated_list = []
        for (tname, zone_type), group_records in groups.items():
            aggregated = self._aggregate_group(tname, zone_type, group_records)
            aggregated_list.append(aggregated)
        
        logger.info(f"聚合完成，共 {len(aggregated_list)} 个组")
        return aggregated_list
    
    def _aggregate_group(
        self, 
        tname: str, 
        zone_type: str, 
        records: List[DanmakuRecord]
    ) -> AggregatedData:
        # 收集所有标签（去重）
        all_tags = set()
        for record in records:
            all_tags.update(record.tags)
        
        # 初始化聚合数据
        # 注意：同一 segment 的多条记录共享同一 hard_metrics，需按 segment_id 去重
        unique_segments = {}
        for r in records:
            if r.segment_id not in unique_segments:
                unique_segments[r.segment_id] = r.hard_metrics
        
        aggregated = AggregatedData(
            tname=tname,
            zone_type=zone_type,
            tags=list(all_tags),
            video_count=1,  # 当前流水线每次运行处理单个视频
            segment_count=len(unique_segments),
            danmaku_count=sum(hm.total_danmaku_count for hm in unique_segments.values()),
        )
        
        # 聚合硬统计
        self._aggregate_hard_metrics(aggregated, records)
        
        # 聚合软标签
        self._aggregate_soft_labels(aggregated, records)
        
        # 聚合共识统计
        self._aggregate_consensus_stats(aggregated, records)
        
        return aggregated
    
    def _aggregate_hard_metrics(
        self, 
        aggregated: AggregatedData, 
        records: List[DanmakuRecord]
    ):
        """聚合硬统计（按 segment_id 去重，避免同段重复计数）"""
        if not records:
            return
        
        # 去重：每个 segment 只计一次硬统计
        unique_segments = {}
        for r in records:
            if r.segment_id not in unique_segments:
                unique_segments[r.segment_id] = r.hard_metrics
        unique_metrics = list(unique_segments.values())
        
        # 加权平均（按弹幕数加权）
        total_weight = sum(hm.total_danmaku_count for hm in unique_metrics)
        
        if total_weight > 0:
            aggregated.avg_word_length = sum(
                hm.avg_word_length * hm.total_danmaku_count 
                for hm in unique_metrics
            ) / total_weight
            
            aggregated.content_word_density = sum(
                hm.content_word_density * hm.total_danmaku_count 
                for hm in unique_metrics
            ) / total_weight
            
            aggregated.punctuation_emoji_rate = sum(
                hm.punctuation_emoji_rate * hm.total_danmaku_count 
                for hm in unique_metrics
            ) / total_weight
        
        # 聚合词性分布
        pos_counter = defaultdict(float)
        for hm in unique_metrics:
            for pos, ratio in hm.pos_distribution.items():
                pos_counter[pos] += ratio * hm.total_danmaku_count
        if total_weight > 0:
            aggregated.pos_distribution = {
                pos: count / total_weight 
                for pos, count in pos_counter.items()
            }
        
        # 聚合音节分布
        syllable_counter = defaultdict(float)
        for hm in unique_metrics:
            for syllable_type, ratio in hm.syllable_distribution.items():
                syllable_counter[syllable_type] += ratio * hm.total_danmaku_count
        if total_weight > 0:
            aggregated.syllable_distribution = {
                syllable_type: count / total_weight 
                for syllable_type, count in syllable_counter.items()
            }
        
        # 聚合正字法硬指标
        ortho_counter = defaultdict(float)
        for hm in unique_metrics:
            for metric, value in hm.orthography_hard_metrics.items():
                ortho_counter[metric] += value * hm.total_danmaku_count
        if total_weight > 0:
            aggregated.orthography_hard_metrics = {
                metric: count / total_weight 
                for metric, count in ortho_counter.items()
            }
    
    def _aggregate_soft_labels(
        self, 
        aggregated: AggregatedData, 
        records: List[DanmakuRecord]
    ):
        """聚合软标签"""
        if not records:
            return
        
        # 情感分布
        emotion_counter = defaultdict(int)
        # 句类分布
        sentence_function_counter = defaultdict(int)
        # 互动类型分布
        interaction_type_counter = defaultdict(int)
        # 正字法状态分布
        orthography_status_counter = defaultdict(int)
        
        # 软标签每条记录代表一次 LLM 分析，权重为 weight_multiplier（零丢弃铁律：低共识=0.2）
        total_weight = sum(r.llm_result.weight_multiplier for r in records)
        
        for record in records:
            llm_output = record.llm_result.output
            weight = record.llm_result.weight_multiplier
            
            # 情感
            emotion_counter[llm_output.emotion.label] += weight
            
            # 句类
            sentence_function_counter[llm_output.sentence_function.label] += weight
            
            # 互动类型
            interaction_type_counter[llm_output.interaction_type.label] += weight
            
            # 正字法状态
            orthography_status_counter[llm_output.orthography.status] += weight
        
        # 计算分布
        if total_weight > 0:
            aggregated.emotion_distribution = {
                label: count / total_weight 
                for label, count in emotion_counter.items()
            }
            aggregated.sentence_function_distribution = {
                label: count / total_weight 
                for label, count in sentence_function_counter.items()
            }
            aggregated.interaction_type_distribution = {
                label: count / total_weight 
                for label, count in interaction_type_counter.items()
            }
            aggregated.orthography_status_distribution = {
                label: count / total_weight 
                for label, count in orthography_status_counter.items()
            }
    
    def _aggregate_consensus_stats(
        self, 
        aggregated: AggregatedData, 
        records: List[DanmakuRecord]
    ):
        """聚合共识统计（每条记录等权计数）"""
        if not records:
            return
        
        from .llm_client import ConsensusLevel
        
        high_count = 0
        medium_count = 0
        low_count = 0
        total = len(records)
        
        for record in records:
            if record.llm_result.consensus_level == ConsensusLevel.HIGH:
                high_count += 1
            elif record.llm_result.consensus_level == ConsensusLevel.MEDIUM:
                medium_count += 1
            else:
                low_count += 1
        
        if total > 0:
            aggregated.high_consensus_rate = high_count / total
            aggregated.medium_consensus_rate = medium_count / total
            aggregated.low_consensus_rate = low_count / total
            aggregated.avg_weight_multiplier = sum(
                r.llm_result.weight_multiplier for r in records
            ) / total
