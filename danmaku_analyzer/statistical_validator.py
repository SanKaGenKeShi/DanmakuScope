"""
统计验证模块 - 描述性统计 + 置信区间 + 可选显著性检验
包含 Wilson 置信区间、描述性统计、Mann-Whitney U 检验（可选），
以及语料库级比较检验：Kruskal-Wallis / Dunn 事后检验 / BH-FDR / Cramér's V / Cliff's delta（手写零运行时 scipy 依赖）
"""

import math
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
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


@dataclass
class KruskalWallisResult:
    """Kruskal-Wallis H 检验结果（含结校正）"""
    test_name: str
    statistic: float
    df: int
    p_value: float
    is_significant: bool
    alpha: float
    n_per_group: Dict[str, int] = field(default_factory=dict)
    note: str = ""

    def to_dict(self) -> dict:
        return {
            "test_name": self.test_name,
            "statistic": round(self.statistic, 4),
            "df": self.df,
            "p_value": round(self.p_value, 6),
            "is_significant": self.is_significant,
            "alpha": self.alpha,
            "n_per_group": self.n_per_group,
            "note": self.note,
        }


@dataclass
class DunnPairResult:
    """Dunn 事后检验单对比较结果"""
    group_a: str
    group_b: str
    z_statistic: float
    p_value: float
    p_adjusted: float  # BH-FDR 校正后
    cliffs_delta: float
    effect_magnitude: str
    is_significant: bool

    def to_dict(self) -> dict:
        return {
            "group_a": self.group_a,
            "group_b": self.group_b,
            "z_statistic": round(self.z_statistic, 4),
            "p_value": round(self.p_value, 6),
            "p_adjusted": round(self.p_adjusted, 6),
            "cliffs_delta": round(self.cliffs_delta, 4),
            "effect_magnitude": self.effect_magnitude,
            "is_significant": self.is_significant,
        }


# ========== 手写分布函数（避免运行时 scipy 依赖） ==========

def _normal_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _gamma_p_series(a: float, x: float) -> float:
    """正则化下不完全 gamma P(a,x) 级数展开（x < a+1 时收敛快）"""
    ap = a
    summ = 1.0 / a
    delta = summ
    for _ in range(500):
        ap += 1.0
        delta *= x / ap
        summ += delta
        if abs(delta) < abs(summ) * 1e-13:
            break
    return summ * math.exp(-x + a * math.log(x) - math.lgamma(a))


def _gamma_q_cf(a: float, x: float) -> float:
    """正则化上不完全 gamma Q(a,x) 连分数展开（x >= a+1 时收敛快）"""
    tiny = 1e-300
    b = x + 1.0 - a
    c = 1.0 / tiny
    d = 1.0 / b
    h = d
    for i in range(1, 500):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + an / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-13:
            break
    return math.exp(-x + a * math.log(x) - math.lgamma(a)) * h


def _chi2_sf(x: float, df: int) -> float:
    """卡方分布生存函数 P(X > x) = Q(df/2, x/2)"""
    if df <= 0:
        raise ValueError(f"自由度必须为正: {df}")
    if x <= 0:
        return 1.0
    a, xx = df / 2.0, x / 2.0
    if xx < a + 1.0:
        return 1.0 - _gamma_p_series(a, xx)
    return _gamma_q_cf(a, xx)


def _rankdata(values: List[float]) -> List[float]:
    """平均秩（同值取秩均值）"""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    return ranks


def _tie_sum(sorted_values: List[float]) -> float:
    """结校正项 Σ(t³ - t)，t 为每个同值组的重复数"""
    total = 0.0
    i = 0
    n = len(sorted_values)
    while i < n:
        j = i
        while j + 1 < n and sorted_values[j + 1] == sorted_values[i]:
            j += 1
        t = j - i + 1
        if t > 1:
            total += t ** 3 - t
        i = j + 1
    return total


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

    # ========== 语料库级比较检验（统计单位 = 视频级观测） ==========

    def kruskal_wallis(
        self,
        groups: Dict[str, List[float]],
        alpha: Optional[float] = None,
    ) -> Optional[KruskalWallisResult]:
        """多组秩和检验（含结校正）；groups = {组名: 视频级观测列表}"""
        if alpha is None:
            alpha = self.significance_alpha

        groups = {name: list(values) for name, values in groups.items() if values}
        if len(groups) < 2:
            logger.warning(f"Kruskal-Wallis 需要至少 2 个非空组，当前 {len(groups)} 组")
            return None
        min_n = self.settings.CORPUS_MIN_VIDEOS_PER_PARTITION
        for name, values in groups.items():
            if len(values) < min_n:
                logger.warning(f"组 {name} 视频数 {len(values)} < {min_n}，Kruskal-Wallis 结果仅供参考")
        if any(len(values) < 2 for values in groups.values()):
            logger.warning("存在样本量 < 2 的组，无法执行 Kruskal-Wallis 检验")
            return None

        names = sorted(groups.keys())
        all_values: List[float] = []
        for name in names:
            all_values.extend(groups[name])
        n_total = len(all_values)
        ranks = _rankdata(all_values)
        ties = _tie_sum(sorted(all_values))

        h_stat = 0.0
        offset = 0
        for name in names:
            n_i = len(groups[name])
            r_i = sum(ranks[offset:offset + n_i])
            h_stat += r_i ** 2 / n_i
            offset += n_i
        h_stat = 12.0 / (n_total * (n_total + 1)) * h_stat - 3.0 * (n_total + 1)
        if ties > 0:
            h_stat /= 1.0 - ties / (n_total ** 3 - n_total)

        df = len(names) - 1
        p_value = _chi2_sf(h_stat, df)
        return KruskalWallisResult(
            test_name="Kruskal-Wallis H",
            statistic=h_stat,
            df=df,
            p_value=p_value,
            is_significant=p_value < alpha,
            alpha=alpha,
            n_per_group={name: len(groups[name]) for name in names},
            note="探索性（Exploratory） - 统计单位为视频",
        )

    def dunn_posthoc(
        self,
        groups: Dict[str, List[float]],
        alpha: Optional[float] = None,
    ) -> List[DunnPairResult]:
        """Dunn 事后两两比较，p 值经 BH-FDR 校正，附 Cliff's delta 效应量"""
        if alpha is None:
            alpha = self.significance_alpha

        groups = {name: list(values) for name, values in groups.items() if values}
        names = sorted(groups.keys())
        if len(names) < 2:
            logger.warning("Dunn 事后检验需要至少 2 个非空组")
            return []

        all_values: List[float] = []
        for name in names:
            all_values.extend(groups[name])
        n_total = len(all_values)
        ranks = _rankdata(all_values)
        ties = _tie_sum(sorted(all_values))

        mean_ranks: Dict[str, float] = {}
        offset = 0
        for name in names:
            n_i = len(groups[name])
            mean_ranks[name] = sum(ranks[offset:offset + n_i]) / n_i
            offset += n_i

        # 结校正后的合并秩方差（Dunn 1964）
        sigma2 = (n_total * (n_total + 1) - ties / (n_total - 1)) / 12.0 if n_total > 1 else 0.0

        pairs = []
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = names[i], names[j]
                se = math.sqrt(sigma2 * (1.0 / len(groups[a]) + 1.0 / len(groups[b])))
                z = (mean_ranks[a] - mean_ranks[b]) / se if se > 0 else 0.0
                p = 2.0 * (1.0 - _normal_cdf(abs(z)))
                delta, magnitude = self.cliffs_delta(groups[a], groups[b])
                pairs.append({"group_a": a, "group_b": b, "z": z, "p": p,
                              "cliffs_delta": delta, "effect_magnitude": magnitude})

        raw_p = [pair["p"] for pair in pairs]
        adjusted = self.benjamini_hochberg(raw_p)
        results = []
        for pair, p_adj in zip(pairs, adjusted):
            results.append(DunnPairResult(
                group_a=pair["group_a"],
                group_b=pair["group_b"],
                z_statistic=pair["z"],
                p_value=pair["p"],
                p_adjusted=p_adj,
                cliffs_delta=pair["cliffs_delta"],
                effect_magnitude=pair["effect_magnitude"],
                is_significant=p_adj < alpha,
            ))
        return results

    @staticmethod
    def benjamini_hochberg(p_values: List[float]) -> List[float]:
        """BH 阶梯向上 FDR 校正，返回与输入同序的调整后 p 值"""
        m = len(p_values)
        if m == 0:
            return []
        order = sorted(range(m), key=lambda i: p_values[i], reverse=True)
        adjusted = [0.0] * m
        running_min = 1.0
        for rank_from_top, idx in enumerate(order, start=1):
            rank = m - rank_from_top + 1  # 升序秩
            running_min = min(running_min, p_values[idx] * m / rank)
            adjusted[idx] = min(1.0, running_min)
        return adjusted

    def cramers_v(self, contingency: List[List[float]]) -> Tuple[float, float, int]:
        """列联表卡方 + Cramér's V 效应量，返回 (V, p_value, df)"""
        rows = [row for row in contingency if any(c > 0 for c in row)]
        if len(rows) < 2:
            raise ValueError("列联表至少需要 2 行非零观测")
        n_cols = len(rows[0])
        if any(len(row) != n_cols for row in rows):
            raise ValueError("列联表各行列数不一致")
        if n_cols < 2:
            raise ValueError("列联表至少需要 2 列")

        row_totals = [sum(row) for row in rows]
        col_totals = [sum(row[c] for row in rows) for c in range(n_cols)]
        n_total = sum(row_totals)
        if n_total <= 0:
            raise ValueError("列联表总观测数为 0")

        chi2 = 0.0
        for r, row in enumerate(rows):
            for c, observed in enumerate(row):
                expected = row_totals[r] * col_totals[c] / n_total
                if expected > 0:
                    chi2 += (observed - expected) ** 2 / expected
        df = (len(rows) - 1) * (n_cols - 1)
        p_value = _chi2_sf(chi2, df)
        v = math.sqrt(chi2 / (n_total * (min(len(rows), n_cols) - 1)))
        return v, p_value, df

    @staticmethod
    def cliffs_delta(x: List[float], y: List[float]) -> Tuple[float, str]:
        """非参数效应量；阈值参照 Romano et al. (2006)"""
        if not x or not y:
            raise ValueError("Cliff's delta 需要两个非空样本")
        more = sum(1 for xi in x for yi in y if xi > yi)
        less = sum(1 for xi in x for yi in y if xi < yi)
        delta = (more - less) / (len(x) * len(y))
        abs_d = abs(delta)
        if abs_d < 0.147:
            magnitude = "negligible"
        elif abs_d < 0.33:
            magnitude = "small"
        elif abs_d < 0.474:
            magnitude = "medium"
        else:
            magnitude = "large"
        return delta, magnitude


