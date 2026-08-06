"""
统计验证模块 - 描述性统计 + 置信区间 + 可选显著性检验
包含 Wilson 置信区间、描述性统计、Mann-Whitney U 检验（可选）；
语料库级比较检验（Kruskal-Wallis/Dunn 等）由 corpus_visualizer 生成的 R 脚本执行，不在 Python 侧重复实现
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from scipy import stats

from .config import get_settings
from .utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ConfidenceInterval:
    """置信区间"""
    lower: float
    upper: float
    point_estimate: float
    confidence_level: float
    method: str  # "wilson" 或 "normal"
    
    def to_dict(self) -> dict:
        return {
            "lower": round(self.lower, 4),
            "upper": round(self.upper, 4),
            "point_estimate": round(self.point_estimate, 4),
            "confidence_level": self.confidence_level,
            "method": self.method,
        }


@dataclass
class DescriptiveStats:
    """描述性统计"""
    mean: float
    median: float
    std: float
    min: float
    max: float
    count: int
    q25: float
    q75: float
    
    def to_dict(self) -> dict:
        return {
            "mean": round(self.mean, 4),
            "median": round(self.median, 4),
            "std": round(self.std, 4),
            "min": round(self.min, 4),
            "max": round(self.max, 4),
            "count": self.count,
            "q25": round(self.q25, 4),
            "q75": round(self.q75, 4),
        }


@dataclass
class SignificanceTestResult:
    """显著性检验结果"""
    test_name: str
    statistic: float
    p_value: float
    is_significant: bool
    alpha: float
    note: str  # 标记为"探索性（Exploratory）"
    
    def to_dict(self) -> dict:
        return {
            "test_name": self.test_name,
            "statistic": round(self.statistic, 4),
            "p_value": round(self.p_value, 4),
            "is_significant": self.is_significant,
            "alpha": self.alpha,
            "note": self.note,
        }


class StatisticalValidator:
    
    def __init__(self):
        self.settings = get_settings()
        self.confidence_level = self.settings.CONFIDENCE_LEVEL
        self.enable_significance_testing = self.settings.ENABLE_SIGNIFICANCE_TESTING
        self.significance_alpha = self.settings.SIGNIFICANCE_ALPHA
    
    def wilson_confidence_interval(
        self, 
        successes: int, 
        total: int,
        confidence_level: Optional[float] = None
    ) -> ConfidenceInterval:
        if confidence_level is None:
            confidence_level = self.confidence_level
        
        if total == 0:
            return ConfidenceInterval(
                lower=0.0,
                upper=0.0,
                point_estimate=0.0,
                confidence_level=confidence_level,
                method="wilson",
            )
        
        alpha = 1 - confidence_level
        z = stats.norm.ppf(1 - alpha / 2)
        
        p_hat = successes / total
        denominator = 1 + z**2 / total
        center = (p_hat + z**2 / (2 * total)) / denominator
        margin = z * np.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * total)) / total) / denominator
        
        lower = max(0.0, center - margin)
        upper = min(1.0, center + margin)
        
        return ConfidenceInterval(
            lower=lower,
            upper=upper,
            point_estimate=p_hat,
            confidence_level=confidence_level,
            method="wilson",
        )
    
    def descriptive_statistics(self, values: List[float]) -> DescriptiveStats:
        if not values:
            return DescriptiveStats(
                mean=0.0,
                median=0.0,
                std=0.0,
                min=0.0,
                max=0.0,
                count=0,
                q25=0.0,
                q75=0.0,
            )
        
        arr = np.array(values)
        
        return DescriptiveStats(
            mean=float(np.mean(arr)),
            median=float(np.median(arr)),
            std=float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
            min=float(np.min(arr)),
            max=float(np.max(arr)),
            count=len(arr),
            q25=float(np.percentile(arr, 25)),
            q75=float(np.percentile(arr, 75)),
        )
    
    def mann_whitney_u_test(
        self, 
        sample1: List[float], 
        sample2: List[float],
        alpha: Optional[float] = None
    ) -> Optional[SignificanceTestResult]:
        """Mann-Whitney U（仅当 ENABLE_SIGNIFICANCE_TESTING=True 时执行）"""
        if not self.enable_significance_testing:
            logger.info("显著性检验未启用，跳过 Mann-Whitney U 检验")
            return None
        
        if alpha is None:
            alpha = self.significance_alpha
        
        if len(sample1) < 3 or len(sample2) < 3:
            logger.warning("样本量太小，无法执行 Mann-Whitney U 检验")
            return None
        
        try:
            statistic, p_value = stats.mannwhitneyu(sample1, sample2, alternative='two-sided')
            
            result = SignificanceTestResult(
                test_name="Mann-Whitney U",
                statistic=float(statistic),
                p_value=float(p_value),
                is_significant=p_value < alpha,
                alpha=alpha,
                note="探索性（Exploratory） - 不用于定论",
            )
            
            logger.info(f"Mann-Whitney U 检验完成：statistic={statistic:.4f}, p={p_value:.4f}")
            return result
            
        except Exception as e:
            logger.error(f"Mann-Whitney U 检验失败: {e}")
            return None
    
    def compare_groups(
        self, 
        group1_values: List[float], 
        group2_values: List[float],
        group1_name: str = "Group1",
        group2_name: str = "Group2"
    ) -> Dict:
        result = {
            "group1": {
                "name": group1_name,
                "descriptive": self.descriptive_statistics(group1_values).to_dict(),
            },
            "group2": {
                "name": group2_name,
                "descriptive": self.descriptive_statistics(group2_values).to_dict(),
            },
            "significance_test": None,
        }
        
        if self.enable_significance_testing:
            test_result = self.mann_whitney_u_test(group1_values, group2_values)
            if test_result:
                result["significance_test"] = test_result.to_dict()
        
        return result
    
    def validate_sample_size(
        self, 
        sample_size: int, 
        min_size: int = 30
    ) -> Tuple[bool, str]:
        if sample_size >= min_size:
            return True, f"样本量充足 ({sample_size} >= {min_size})"
        else:
            return False, f"样本量不足 ({sample_size} < {min_size})，跳过置信区间，标记为 insufficient_sample"


