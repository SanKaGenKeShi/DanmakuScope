"""
统计验证模块 - 描述性统计 + 置信区间 + 可选显著性检验 + 语料库级推断统计
Wilson 置信区间、描述性统计、Mann-Whitney U 检验（单视频段间，可选）；
语料库级跨分区推断（Kruskal-Wallis + 逐对 Mann-Whitney U + Cliff's delta，基于 scipy）
由 corpus_compare 编排，输出未校正 p 值的 statistical_tests.csv
"""

import itertools
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from .config import get_settings
from .corpus_builder import SCALAR_FIELDS
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


STATISTICAL_TESTS_COLUMNS = [
    "metric", "test_type", "group1", "group2", "n1", "n2",
    "statistic", "p_value", "effect_size", "effect_magnitude", "note",
]

UNCORRECTED_P_NOTE = "未校正 p 值（未实施多重比较校正）"


def cohen_kappa(labels_a: List, labels_b: List) -> Optional[float]:
    """两组类别标注的 Cohen's Kappa；长度不一致/空序列/期望一致率退化时返回 None"""
    if not labels_a or len(labels_a) != len(labels_b):
        return None
    n = len(labels_a)
    counter_a, counter_b = Counter(labels_a), Counter(labels_b)
    po = sum(a == b for a, b in zip(labels_a, labels_b)) / n
    pe = sum((counter_a[c] / n) * (counter_b[c] / n) for c in set(counter_a) | set(counter_b))
    if pe >= 1.0:
        # 两路均恒定同一类别时无随机一致性基线，完全一致视为 1.0，否则不可定义
        return 1.0 if po == 1.0 else None
    return (po - pe) / (1 - pe)


@dataclass
class ComparisonResult:
    """语料库级比较检验结果：行集合 + 落盘能力"""
    rows: List[Dict] = field(default_factory=list)
    enabled: bool = True

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows, columns=STATISTICAL_TESTS_COLUMNS)

    def to_csv(self, path: str) -> str:
        self.to_dataframe().to_csv(path, index=False, encoding='utf-8-sig')
        return path


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

    def kruskal_wallis_test(self, groups: Dict[str, List[float]], metric: str) -> Optional[Dict]:
        """Kruskal-Wallis H 检验（scipy）+ ε² 效应量，返回未校正 p 值的 CSV 行；可检验分组 < 2 返回 None"""
        names = sorted(groups)
        if len(names) < 2:
            return None
        samples = [groups[name] for name in names]
        n_total = sum(len(s) for s in samples)
        k = len(samples)
        if len({v for s in samples for v in s}) <= 1:
            statistic, p_value, effect_size = 0.0, 1.0, 0.0
            note = UNCORRECTED_P_NOTE + "；全部观测值相同，检验退化"
        else:
            statistic, p_value = stats.kruskal(*samples)
            # ε² = (H - k + 1) / (n - k)，负值截断 0；每组观测 >= 2 时分母恒正，否则退化为 0
            effect_size = round(max(0.0, (float(statistic) - k + 1) / (n_total - k)), 4) if n_total > k else 0.0
            note = UNCORRECTED_P_NOTE
        return {
            "metric": metric,
            "test_type": "Kruskal-Wallis",
            "group1": "",
            "group2": "",
            "n1": n_total,
            "n2": "",
            "statistic": round(float(statistic), 4),
            "p_value": round(float(p_value), 6),
            "effect_size": effect_size,
            "effect_magnitude": "",
            "note": note,
        }

    def cliff_delta(self, sample1: List[float], sample2: List[float]) -> float:
        """Cliff's delta 效应量（纯算术实现）：(x>y 对数 - x<y 对数) / (n1*n2)"""
        more = less = 0
        for x in sample1:
            for y in sample2:
                if x > y:
                    more += 1
                elif x < y:
                    less += 1
        return (more - less) / (len(sample1) * len(sample2))

    @staticmethod
    def _cliff_delta_magnitude(delta: float) -> str:
        """效应量分级阈值参照 Romano et al. (2006)"""
        d = abs(delta)
        if d < 0.147:
            return "negligible"
        if d < 0.33:
            return "small"
        if d < 0.474:
            return "medium"
        return "large"

    def pairwise_mann_whitney(self, groups: Dict[str, List[float]], metric: str) -> List[Dict]:
        """逐对 Mann-Whitney U 两两比较（scipy），输出未校正 p 值并附 Cliff's delta"""
        rows = []
        for name1, name2 in itertools.combinations(sorted(groups), 2):
            a, b = groups[name1], groups[name2]
            if len(set(a) | set(b)) <= 1:
                statistic, p_value = len(a) * len(b) / 2, 1.0
                note = UNCORRECTED_P_NOTE + "；全部观测值相同，检验退化"
            else:
                statistic, p_value = stats.mannwhitneyu(a, b, alternative='two-sided')
                note = UNCORRECTED_P_NOTE
            delta = self.cliff_delta(a, b)
            rows.append({
                "metric": metric,
                "test_type": "Mann-Whitney U",
                "group1": name1,
                "group2": name2,
                "n1": len(a),
                "n2": len(b),
                "statistic": round(float(statistic), 4),
                "p_value": round(float(p_value), 6),
                "effect_size": round(delta, 4),
                "effect_magnitude": self._cliff_delta_magnitude(delta),
                "note": note,
            })
        return rows

    def zone_paired_compare(self, csv_path: str) -> List[Dict]:
        """冷热区视频内配对 Wilcoxon 符号秩检验（单分区合并模式的情境变异轴）：
        每视频提供热/冷区一对观测，配对数 < CORPUS_MIN_VIDEOS_PER_PARTITION 或无双区观测返回空"""
        df = pd.read_csv(csv_path, encoding='utf-8-sig')
        if "zone_type" not in df.columns or "bvid" not in df.columns:
            return []
        zone = df["zone_type"].astype(str)
        # 防御重复观测（同 bvid 同区多行）：保留首条，避免配对错位与样本量虚高
        hot = df[zone == "hot_zone"].set_index("bvid")
        hot = hot[~hot.index.duplicated(keep="first")]
        cold = df[zone == "cold_zone"].set_index("bvid")
        cold = cold[~cold.index.duplicated(keep="first")]
        paired_bvids = sorted(set(hot.index) & set(cold.index))
        min_pairs = get_settings().CORPUS_MIN_VIDEOS_PER_PARTITION
        if len(paired_bvids) < min_pairs:
            if paired_bvids:
                logger.warning(f"冷热区配对数不足（{len(paired_bvids)} < {min_pairs}），跳过配对检验")
            return []

        rows: List[Dict] = []
        for metric in SCALAR_FIELDS:
            if metric not in hot.columns or metric not in cold.columns:
                continue
            a = pd.to_numeric(hot.loc[paired_bvids, metric], errors="coerce")
            b = pd.to_numeric(cold.loc[paired_bvids, metric], errors="coerce")
            mask = a.notna() & b.notna()
            a, b = a[mask], b[mask]
            if len(a) < min_pairs:
                continue
            if (a.values == b.values).all():
                statistic, p_value = len(a) * len(a) / 2, 1.0
                note = UNCORRECTED_P_NOTE + "；全部配对差为零，检验退化"
            else:
                try:
                    statistic, p_value = stats.wilcoxon(a.values, b.values)
                    note = UNCORRECTED_P_NOTE
                except ValueError:
                    statistic, p_value = len(a) * len(a) / 2, 1.0
                    note = UNCORRECTED_P_NOTE + "；非零差异不足，检验退化"
            delta = self.cliff_delta(a.tolist(), b.tolist())
            rows.append({
                "metric": metric,
                "test_type": "Wilcoxon 符号秩（配对）",
                "group1": "hot_zone",
                "group2": "cold_zone",
                "n1": len(a),
                "n2": len(b),
                "statistic": round(float(statistic), 4),
                "p_value": round(float(p_value), 6),
                "effect_size": round(delta, 4),
                "effect_magnitude": self._cliff_delta_magnitude(delta),
                "note": note,
            })
        if rows:
            logger.info(f"冷热区配对检验完成: {len(paired_bvids)} 对视频观测，{len(rows)} 行结果（未校正 p 值）")
        return rows

    def corpus_compare(self, csv_path: str, groupby_col: Optional[str] = None) -> ComparisonResult:
        """语料库级推断编排：消费视频级观测表（勿用组级汇总表）。
        分组键自动分流：多分区按 tname 比对；单分区且开启时间分桶时按 time_period 历时比对；
        单分区另补冷热区视频内配对检验（情境变异轴）；组内视频数 < CORPUS_MIN_VIDEOS_PER_PARTITION
        标 insufficient_sample 跳过；无任何可执行检验时输出注记行；仅受 ENABLE_CORPUS_STATISTICS 控制"""
        settings = get_settings()
        if not settings.ENABLE_CORPUS_STATISTICS:
            logger.info("语料库级推断统计未启用（ENABLE_CORPUS_STATISTICS=False），跳过 corpus_compare")
            return ComparisonResult(rows=[], enabled=False)

        df = pd.read_csv(csv_path, encoding='utf-8-sig')
        if "tname" not in df.columns:
            raise ValueError(f"观测表缺少分组列 tname: {csv_path}")

        if groupby_col is None:
            tnames = df["tname"].dropna().astype(str).str.strip()
            tnames = tnames[tnames != ""]
            single_partition = tnames.nunique() <= 1
            temporal_usable = (
                "time_period" in df.columns
                and df["time_period"].notna().any()
                and (df["time_period"].astype(str).str.strip() != "").any()
                and df.loc[df["time_period"].notna(), "time_period"].astype(str).nunique() >= 2
            )
            groupby_col = "time_period" if (single_partition and temporal_usable) else "tname"

        if groupby_col not in df.columns:
            raise ValueError(f"观测表缺少分组列 {groupby_col}: {csv_path}")
        df = df[df[groupby_col].notna() & (df[groupby_col].astype(str).str.strip() != "")]

        min_videos = settings.CORPUS_MIN_VIDEOS_PER_PARTITION
        counts = df.groupby(groupby_col).size()
        valid = sorted(counts[counts >= min_videos].index.astype(str).tolist())

        rows: List[Dict] = []
        axis_label = "时段" if groupby_col == "time_period" else "分区"
        for raw_name, count in sorted(counts.items(), key=lambda kv: str(kv[0])):
            # 时段列 CSV 往返后可能为数值 dtype，统一转字符串避免键查错位
            name = str(raw_name)
            n = int(count)
            sufficient = n >= min_videos
            rows.append({
                "metric": "",
                "test_type": "sample_status",
                "group1": name,
                "group2": "",
                "n1": n,
                "n2": "",
                "statistic": "",
                "p_value": "",
                "effect_size": "",
                "effect_magnitude": "",
                "note": "sample_sufficient" if sufficient else f"insufficient_sample（视频数 {n} < {min_videos}，不参与检验）",
            })

        if len(valid) >= 2:
            # 时段轴在 note 追加检验轴标注，供导出端措辞区分（分区间/时段间）
            axis_note_suffix = "" if groupby_col == "tname" else "；检验轴：时段"
            sub = df[df[groupby_col].astype(str).isin(valid)]
            for metric in SCALAR_FIELDS:
                if metric not in sub.columns:
                    continue
                groups: Dict[str, List[float]] = {}
                for name in valid:
                    values = pd.to_numeric(sub.loc[sub[groupby_col].astype(str) == name, metric], errors='coerce').dropna().tolist()
                    if len(values) >= 2:
                        groups[name] = values
                if len(groups) < 2:
                    continue
                kw_row = self.kruskal_wallis_test(groups, metric)
                if kw_row:
                    kw_row["note"] += axis_note_suffix
                    rows.append(kw_row)
                mwu_rows = self.pairwise_mann_whitney(groups, metric)
                for mwu_row in mwu_rows:
                    mwu_row["note"] += axis_note_suffix
                rows.extend(mwu_rows)
            logger.info(f"语料库级推断检验完成: 按{axis_label}分组 {len(valid)} 个有效组，{len(rows)} 行结果（未校正 p 值）")
        else:
            logger.warning(f"有效{axis_label}不足（{len(valid)} < 2，门槛：每组视频数 >= {min_videos}），未执行组间检验")

        # 单分区合并模式：补冷热区视频内配对检验（多分区场景避免跨分区混杂不执行）
        if df["tname"].astype(str).str.strip().nunique() <= 1:
            paired_rows = self.zone_paired_compare(csv_path)
            rows.extend(paired_rows)
            if len(valid) < 2 and not paired_rows:
                rows.append({
                    "metric": "", "test_type": "note", "group1": "", "group2": "",
                    "n1": "", "n2": "", "statistic": "", "p_value": "",
                    "effect_size": "", "effect_magnitude": "",
                    "note": "单一分区且未启用时间分桶/冷热区双区保留，无可用比较轴，未执行推断检验",
                })

        return ComparisonResult(rows=rows)


