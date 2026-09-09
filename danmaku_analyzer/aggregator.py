"""
聚合器模块 - 嵌套聚合（分区/热区）
分层键：tname（官方分区） + hot_zone（热区）
tags 仅作为元数据附加，不参与分组
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from collections import defaultdict

from .hard_metrics import HardMetricsResult
from .llm_models import ConsensusLevel, DualPathResult
from .utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class AggregatedData:
    tname: str
    zone_type: str  # hot_zone 或 cold_zone
    
    tags: List[str] = field(default_factory=list)  # 仅元数据，不参与分组
    video_count: int = 0
    segment_count: int = 0
    danmaku_count: int = 0
    total_word_count: int = 0
    total_char_count: int = 0

    avg_word_length: float = 0.0
    content_word_density: float = 0.0
    punctuation_emoji_rate: float = 0.0
    pos_distribution: Dict[str, float] = field(default_factory=dict)
    syllable_distribution: Dict[str, float] = field(default_factory=dict)
    orthography_hard_metrics: Dict[str, float] = field(default_factory=dict)
    
    emotion_distribution: Dict[str, float] = field(default_factory=dict)
    sentence_function_distribution: Dict[str, float] = field(default_factory=dict)
    interaction_type_distribution: Dict[str, float] = field(default_factory=dict)
    orthography_status_distribution: Dict[str, float] = field(default_factory=dict)
    cooperative_principle_violation_rate: Optional[float] = None
    label_weight_sums: Dict[str, float] = field(default_factory=dict)
    valid_label_counts: Dict[str, int] = field(default_factory=dict)
    failed_record_count: int = 0
    degraded_record_count: int = 0
    
    high_consensus_rate: float = 0.0
    medium_consensus_rate: float = 0.0
    low_consensus_rate: float = 0.0
    avg_weight_multiplier: float = 1.0
    llm_record_count: int = 0  # 参与共识统计的 LLM 记录数（共识率与 CI 的基数）
    
    consensus_ci: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> dict:
        return {
            "tname": self.tname,
            "zone_type": self.zone_type,
            "tags": self.tags,
            "video_count": self.video_count,
            "segment_count": self.segment_count,
            "total_word_count": self.total_word_count,
            "total_char_count": self.total_char_count,
            "label_weight_sums": self.label_weight_sums,
            "valid_label_counts": self.valid_label_counts,
            "failed_record_count": self.failed_record_count,
            "degraded_record_count": self.degraded_record_count,
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
                "cooperative_principle_violation_rate": round(self.cooperative_principle_violation_rate, 4) if self.cooperative_principle_violation_rate is not None else None,
            },
            "consensus_stats": {
                "high_consensus_rate": round(self.high_consensus_rate, 4),
                "medium_consensus_rate": round(self.medium_consensus_rate, 4),
                "low_consensus_rate": round(self.low_consensus_rate, 4),
                "avg_weight_multiplier": round(self.avg_weight_multiplier, 4),
                "llm_record_count": self.llm_record_count,
                "consensus_ci": self.consensus_ci,
            },
        }

    def to_flat_dict(self) -> dict:
        """扁平序列化（LLM 报告输入用，字段与历史手工拆解保持一致）"""
        return {
            "tname": self.tname,
            "zone_type": self.zone_type,
            "danmaku_count": self.danmaku_count,
            "video_count": self.video_count,
            "segment_count": self.segment_count,
            "total_word_count": self.total_word_count,
            "total_char_count": self.total_char_count,
            "label_weight_sums": self.label_weight_sums,
            "valid_label_counts": self.valid_label_counts,
            "failed_record_count": self.failed_record_count,
            "degraded_record_count": self.degraded_record_count,
            "emotion_distribution": self.emotion_distribution,
            "sentence_function_distribution": self.sentence_function_distribution,
            "interaction_type_distribution": self.interaction_type_distribution,
            "orthography_status_distribution": self.orthography_status_distribution,
            "cooperative_principle_violation_rate": self.cooperative_principle_violation_rate,
            "high_consensus_rate": self.high_consensus_rate,
            "medium_consensus_rate": self.medium_consensus_rate,
            "low_consensus_rate": self.low_consensus_rate,
            "avg_weight_multiplier": self.avg_weight_multiplier,
            "llm_record_count": self.llm_record_count,
            "avg_word_length": self.avg_word_length,
            "content_word_density": self.content_word_density,
            "punctuation_emoji_rate": self.punctuation_emoji_rate,
            "pos_distribution": self.pos_distribution,
            "syllable_distribution": self.syllable_distribution,
            "orthography_hard_metrics": self.orthography_hard_metrics,
        }


@dataclass
class DanmakuRecord:
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
        
        groups: Dict[tuple, List[DanmakuRecord]] = defaultdict(list)
        
        for record in records:
            key = (record.tname, record.zone_type)
            groups[key].append(record)
        
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
        all_tags = set()
        for record in records:
            all_tags.update(record.tags)
        
        unique_metrics = self._unique_segment_metrics(records)
        
        aggregated = AggregatedData(
            tname=tname,
            zone_type=zone_type,
            tags=list(all_tags),
            video_count=1,  # 当前流水线每次运行处理单个视频
            segment_count=len(unique_metrics),
            danmaku_count=sum(hm.total_danmaku_count for hm in unique_metrics),
        )
        
        self._aggregate_hard_metrics(aggregated, unique_metrics)
        self._aggregate_soft_labels(aggregated, records)
        self._aggregate_consensus_stats(aggregated, records)
        
        return aggregated
    
    def _unique_segment_metrics(self, records: List[DanmakuRecord]) -> List[HardMetricsResult]:
        """按 segment_id 去重段级硬统计，避免同段多条记录重复计数"""
        unique_segments = {}
        for r in records:
            if r.segment_id not in unique_segments:
                unique_segments[r.segment_id] = r.hard_metrics
        return list(unique_segments.values())
    
    def _aggregate_hard_metrics(
        self, 
        aggregated: AggregatedData, 
        unique_metrics: List[HardMetricsResult]
    ):
        """聚合硬统计（输入已按 segment_id 去重）"""
        if not unique_metrics:
            return
        
        total_danmaku = sum(hm.total_danmaku_count for hm in unique_metrics)
        total_words = sum(hm.total_word_count for hm in unique_metrics)
        total_chars = sum(hm.total_char_count for hm in unique_metrics)
        aggregated.total_word_count = total_words
        aggregated.total_char_count = total_chars

        if total_words:
            aggregated.avg_word_length = total_chars / total_words
            aggregated.content_word_density = sum(
                hm.content_word_density * hm.total_word_count for hm in unique_metrics
            ) / total_words
        if total_danmaku:
            aggregated.punctuation_emoji_rate = sum(
                hm.punctuation_emoji_rate * hm.total_danmaku_count for hm in unique_metrics
            ) / total_danmaku

        for attribute, denominator, total in (
            ("pos_distribution", "total_word_count", total_words),
            ("syllable_distribution", "total_word_count", total_words),
            ("orthography_hard_metrics", "total_char_count", total_chars),
        ):
            counts = defaultdict(float)
            for metrics in unique_metrics:
                for label, ratio in getattr(metrics, attribute).items():
                    counts[label] += ratio * getattr(metrics, denominator)
            setattr(aggregated, attribute, {
                label: count / total if total else 0.0 for label, count in counts.items()
            })
    
    def _aggregate_soft_labels(
        self, 
        aggregated: AggregatedData, 
        records: List[DanmakuRecord]
    ):
        if not records:
            return
        
        dimensions = (
            ("emotion", "label", "emotion_distribution"),
            ("sentence_function", "label", "sentence_function_distribution"),
            ("interaction_type", "label", "interaction_type_distribution"),
            ("orthography", "status", "orthography_status_distribution"),
            ("cooperative_principle", "violated", None),
        )
        for dimension, label_field, attribute in dimensions:
            counts = defaultdict(float)
            weight_sum = 0.0
            valid_count = 0
            for record in records:
                value = getattr(record.llm_result.output, dimension)
                if value is None:
                    continue
                weight = record.llm_result.weight_multiplier
                counts[getattr(value, label_field)] += weight
                weight_sum += weight
                valid_count += 1
            aggregated.label_weight_sums[dimension] = weight_sum
            aggregated.valid_label_counts[dimension] = valid_count
            if attribute is None:
                aggregated.cooperative_principle_violation_rate = (
                    counts[True] / weight_sum if weight_sum else None
                )
            else:
                setattr(aggregated, attribute, {
                    label: count / weight_sum for label, count in counts.items()
                } if weight_sum else {})
    
    def _aggregate_consensus_stats(
        self, 
        aggregated: AggregatedData, 
        records: List[DanmakuRecord]
    ):
        """聚合共识统计（每条记录等权计数）"""
        if not records:
            return
        
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
            aggregated.llm_record_count = total
            aggregated.failed_record_count = sum(r.llm_result.analysis_status == "failed" for r in records)
            aggregated.degraded_record_count = sum(r.llm_result.analysis_status == "degraded" for r in records)
            aggregated.high_consensus_rate = high_count / total
            aggregated.medium_consensus_rate = medium_count / total
            aggregated.low_consensus_rate = low_count / total
            aggregated.avg_weight_multiplier = sum(
                r.llm_result.weight_multiplier for r in records
            ) / total
