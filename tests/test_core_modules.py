"""
核心模块单元测试 - HardMetricsAnalyzer / Aggregator / TimelineSegmenter
不依赖网络，纯本地逻辑验证
"""

import pytest
import json
import numpy as np
from dataclasses import dataclass

from danmaku_analyzer.hard_metrics import HardMetricsAnalyzer, HardMetricsResult
from danmaku_analyzer.aggregator import Aggregator, AggregatedData, DanmakuRecord
from danmaku_analyzer.reporter import Reporter
from danmaku_analyzer.timeline_segmenter import TimelineSegmenter, TimeSegment
from danmaku_analyzer.crawler import DanmakuItem
from danmaku_analyzer.llm_models import (
    DualPathResult, LLMOutput, ConsensusLevel,
    EmotionOutput, CooperativePrincipleOutput,
    InteractionTypeOutput, SentenceFunctionOutput, OrthographyOutput,
)


# ========== 辅助工厂函数 ==========

def make_danmaku_item(content: str, time_sec: float, uid_hash: str = "abc123") -> DanmakuItem:
    """创建测试用 DanmakuItem"""
    return DanmakuItem(
        uid_hash=uid_hash,
        content=content,
        time_sec=time_sec,
        identity_type="real_user",
    )


def make_llm_output(
    emotion_label: str = "positive",
    sentence_function_label: str = "exclamation",
    interaction_type_label: str = "expression",
    orthography_status: str = "standard",
) -> LLMOutput:
    """创建测试用 LLMOutput"""
    return LLMOutput(
        emotion=EmotionOutput(label=emotion_label, confidence=0.9),
        cooperative_principle=CooperativePrincipleOutput(violated=False),
        interaction_type=InteractionTypeOutput(label=interaction_type_label, confidence=0.8),
        sentence_function=SentenceFunctionOutput(label=sentence_function_label, confidence=0.85),
        orthography=OrthographyOutput(status=orthography_status, confidence=0.95),
    )


def make_dual_path_result(
    output: LLMOutput = None,
    consensus_level: ConsensusLevel = ConsensusLevel.HIGH,
    weight_multiplier: float = 1.0,
) -> DualPathResult:
    """创建测试用 DualPathResult"""
    if output is None:
        output = make_llm_output()
    return DualPathResult(
        output=output,
        consensus_level=consensus_level,
        jsd_score=0.05,
        weight_multiplier=weight_multiplier,
        raw_outputs=[],
        prompt_version="v2.2.0",
    )


def make_hard_metrics_result(
    total_danmaku_count: int = 10,
    avg_word_length: float = 2.0,
    content_word_density: float = 0.5,
    punctuation_emoji_rate: float = 0.3,
) -> HardMetricsResult:
    """创建测试用 HardMetricsResult"""
    return HardMetricsResult(
        pos_distribution={"n": 0.3, "v": 0.3, "a": 0.1, "d": 0.1, "r": 0.2},
        syllable_distribution={"单音节": 0.4, "双音节": 0.4, "三+音节": 0.2},
        avg_word_length=avg_word_length,
        content_word_density=content_word_density,
        punctuation_emoji_rate=punctuation_emoji_rate,
        orthography_hard_metrics={
            "uppercase_abbr_per_1000": 1.0,
            "number_symbol_per_1000": 0.5,
            "emoticon_per_1000": 0.2,
        },
        total_danmaku_count=total_danmaku_count,
        total_word_count=total_danmaku_count * 5,
        total_char_count=total_danmaku_count * 10,
    )


def make_danmaku_record(
    tname: str = "游戏",
    zone_type: str = "hot_zone",
    tags: list = None,
    segment_id: int = 0,
    hard_metrics: HardMetricsResult = None,
    llm_result: DualPathResult = None,
) -> DanmakuRecord:
    """创建测试用 DanmakuRecord"""
    return DanmakuRecord(
        tname=tname,
        zone_type=zone_type,
        tags=tags or ["TAG1"],
        hard_metrics=hard_metrics or make_hard_metrics_result(),
        llm_result=llm_result or make_dual_path_result(),
        segment_id=segment_id,
    )


# ========== HardMetricsAnalyzer 测试 ==========

class TestHardMetricsAnalyzer:
    """硬统计分析器测试"""

    def setup_method(self):
        """初始化分析器（默认 ENABLE_LLM_TOKENIZER=False，纯 jieba）"""
        self.analyzer = HardMetricsAnalyzer()

    def test_empty_list(self):
        """空弹幕列表返回空结果"""
        result = self.analyzer.analyze([])
        assert result.total_danmaku_count == 0
        assert result.total_word_count == 0
        assert result.total_char_count == 0
        assert result.avg_word_length == 0.0
        assert result.content_word_density == 0.0
        assert result.punctuation_emoji_rate == 0.0
        assert result.pos_distribution == {}
        assert result.syllable_distribution == {}

    def test_single_danmaku(self):
        """单条弹幕基本统计"""
        result = self.analyzer.analyze(["你好世界"])
        assert result.total_danmaku_count == 1
        assert result.total_word_count > 0
        assert result.total_char_count > 0
        assert result.avg_word_length > 0
        # 无标点/emoji
        assert result.punctuation_emoji_rate == 0.0

    def test_punctuation_emoji_rate(self):
        """标点/表情携带率计算"""
        danmaku_list = [
            "太好看了！",       # 有标点
            "哈哈哈",          # 无标点
            "真的吗？",        # 有标点
            "666",            # 无标点
        ]
        result = self.analyzer.analyze(danmaku_list)
        # 4 条中 2 条有标点 → 0.5
        assert result.punctuation_emoji_rate == pytest.approx(0.5)

    def test_pos_distribution_sums_to_one(self):
        """词性分布占比之和约为 1"""
        danmaku_list = ["这个游戏真的太好玩了", "我觉得剧情很精彩", "哈哈哈笑死"]
        result = self.analyzer.analyze(danmaku_list)
        total_ratio = sum(result.pos_distribution.values())
        assert total_ratio == pytest.approx(1.0, abs=0.01)

    def test_syllable_distribution_sums_to_one(self):
        """音节分布占比之和约为 1"""
        danmaku_list = ["这个游戏真的太好玩了", "我觉得剧情很精彩"]
        result = self.analyzer.analyze(danmaku_list)
        total_ratio = sum(result.syllable_distribution.values())
        assert total_ratio == pytest.approx(1.0, abs=0.01)

    def test_content_word_density_range(self):
        """实词密度在 [0, 1] 范围内"""
        danmaku_list = ["科学和技术的力量", "美丽的自然风光", "他快速地跑过去"]
        result = self.analyzer.analyze(danmaku_list)
        assert 0.0 <= result.content_word_density <= 1.0

    def test_orthography_hard_metrics_keys(self):
        """正字法硬指标包含三个键"""
        result = self.analyzer.analyze(["AWSL", "23333", "普通弹幕"])
        assert "uppercase_abbr_per_1000" in result.orthography_hard_metrics
        assert "number_symbol_per_1000" in result.orthography_hard_metrics
        assert "emoticon_per_1000" in result.orthography_hard_metrics

    def test_uppercase_abbr_detection(self):
        """大写缩写检测"""
        # 含大写缩写的弹幕
        result_with = self.analyzer.analyze(["AWSL 太棒了", "YYDS 永远的神"])
        # 纯中文弹幕
        result_without = self.analyzer.analyze(["太棒了", "永远的神"])
        assert result_with.orthography_hard_metrics["uppercase_abbr_per_1000"] > 0
        assert result_without.orthography_hard_metrics["uppercase_abbr_per_1000"] == 0.0

    def test_number_symbol_detection(self):
        """数字表意串检测"""
        result = self.analyzer.analyze(["23333 笑死", "666666"])
        assert result.orthography_hard_metrics["number_symbol_per_1000"] > 0


# ========== Aggregator 测试 ==========

class TestAggregator:
    """聚合器测试"""

    def setup_method(self):
        """初始化聚合器"""
        self.aggregator = Aggregator()

    def test_empty_records(self):
        """空记录列表返回空结果"""
        result = self.aggregator.aggregate([])
        assert result == []

    def test_single_record(self):
        """单条记录聚合"""
        record = make_danmaku_record(tname="游戏", zone_type="hot_zone")
        result = self.aggregator.aggregate([record])
        assert len(result) == 1
        assert result[0].tname == "游戏"
        assert result[0].zone_type == "hot_zone"
        assert result[0].segment_count == 1

    def test_grouping_by_tname_and_zone(self):
        """按 (tname, zone_type) 分组"""
        records = [
            make_danmaku_record(tname="游戏", zone_type="hot_zone", segment_id=0),
            make_danmaku_record(tname="游戏", zone_type="hot_zone", segment_id=1),
            make_danmaku_record(tname="音乐", zone_type="cold_zone", segment_id=2),
        ]
        result = self.aggregator.aggregate(records)
        assert len(result) == 2
        # 验证分组正确
        group_keys = {(r.tname, r.zone_type) for r in result}
        assert ("游戏", "hot_zone") in group_keys
        assert ("音乐", "cold_zone") in group_keys

    def test_segment_dedup_in_hard_metrics(self):
        """同 segment_id 的记录硬统计只计一次"""
        hm = make_hard_metrics_result(total_danmaku_count=10)
        records = [
            make_danmaku_record(segment_id=0, hard_metrics=hm),
            make_danmaku_record(segment_id=0, hard_metrics=hm),  # 同段重复
        ]
        result = self.aggregator.aggregate(records)
        # segment_count 应为 1（去重后）
        assert result[0].segment_count == 1
        # danmaku_count 应为 10（只计一次）
        assert result[0].danmaku_count == 10

    def test_soft_label_distribution(self):
        """软标签分布聚合"""
        # 两条记录：一条 positive，一条 negative
        output_pos = make_llm_output(emotion_label="positive")
        output_neg = make_llm_output(emotion_label="negative")
        records = [
            make_danmaku_record(
                segment_id=0,
                llm_result=make_dual_path_result(output=output_pos, weight_multiplier=1.0),
            ),
            make_danmaku_record(
                segment_id=1,
                llm_result=make_dual_path_result(output=output_neg, weight_multiplier=1.0),
            ),
        ]
        result = self.aggregator.aggregate(records)
        emotion_dist = result[0].emotion_distribution
        assert emotion_dist["positive"] == pytest.approx(0.5)
        assert emotion_dist["negative"] == pytest.approx(0.5)

    def test_weight_multiplier_affects_distribution(self):
        """权重乘数影响软标签分布"""
        output_pos = make_llm_output(emotion_label="positive")
        output_neg = make_llm_output(emotion_label="negative")
        records = [
            make_danmaku_record(
                segment_id=0,
                llm_result=make_dual_path_result(output=output_pos, weight_multiplier=1.0),
            ),
            make_danmaku_record(
                segment_id=1,
                llm_result=make_dual_path_result(
                    output=output_neg,
                    consensus_level=ConsensusLevel.LOW,
                    weight_multiplier=0.2,
                ),
            ),
        ]
        result = self.aggregator.aggregate(records)
        emotion_dist = result[0].emotion_distribution
        # positive: 1.0 / 1.2 ≈ 0.833, negative: 0.2 / 1.2 ≈ 0.167
        assert emotion_dist["positive"] == pytest.approx(1.0 / 1.2, abs=0.01)
        assert emotion_dist["negative"] == pytest.approx(0.2 / 1.2, abs=0.01)

    def test_consensus_stats(self):
        """共识统计聚合"""
        records = [
            make_danmaku_record(
                segment_id=0,
                llm_result=make_dual_path_result(consensus_level=ConsensusLevel.HIGH),
            ),
            make_danmaku_record(
                segment_id=1,
                llm_result=make_dual_path_result(consensus_level=ConsensusLevel.MEDIUM),
            ),
            make_danmaku_record(
                segment_id=2,
                llm_result=make_dual_path_result(
                    consensus_level=ConsensusLevel.LOW, weight_multiplier=0.2
                ),
            ),
        ]
        result = self.aggregator.aggregate(records)
        agg = result[0]
        assert agg.high_consensus_rate == pytest.approx(1 / 3)
        assert agg.medium_consensus_rate == pytest.approx(1 / 3)
        assert agg.low_consensus_rate == pytest.approx(1 / 3)

    def test_tags_dedup(self):
        """标签去重"""
        records = [
            make_danmaku_record(tags=["TAG_A", "TAG_B"], segment_id=0),
            make_danmaku_record(tags=["TAG_B", "TAG_C"], segment_id=1),
        ]
        result = self.aggregator.aggregate(records)
        assert set(result[0].tags) == {"TAG_A", "TAG_B", "TAG_C"}

    def test_to_dict(self):
        """AggregatedData.to_dict 结构验证"""
        record = make_danmaku_record()
        result = self.aggregator.aggregate([record])
        d = result[0].to_dict()
        assert "tname" in d
        assert "hard_metrics" in d
        assert "soft_labels" in d
        assert "consensus_stats" in d


# ========== TimelineSegmenter 测试 ==========

class TestTimelineSegmenterFixed:
    """时序切分器 - 固定模式测试"""

    def setup_method(self):
        """以 fixed 模式初始化（真实构造后覆写字段，避免跳过构造函数）"""
        self.segmenter = TimelineSegmenter()
        self.segmenter.min_segment_samples = 5
        self.segmenter.segmentation_mode = "fixed"

    def test_empty_list(self):
        """空弹幕列表"""
        result = self.segmenter.segment([])
        assert result == []

    def test_fixed_equal_split(self):
        """固定模式等分"""
        # 12 条弹幕，每段 5 条 → 3 段（5 + 5 + 2）
        danmaku_list = [make_danmaku_item(f"弹幕{i}", float(i)) for i in range(12)]
        result = self.segmenter.segment(danmaku_list)
        assert len(result) == 3
        assert result[0].danmaku_count == 5
        assert result[1].danmaku_count == 5
        assert result[2].danmaku_count == 2

    def test_fixed_exact_multiple(self):
        """弹幕数恰好为 step 整数倍"""
        danmaku_list = [make_danmaku_item(f"弹幕{i}", float(i)) for i in range(10)]
        result = self.segmenter.segment(danmaku_list)
        assert len(result) == 2
        assert all(s.danmaku_count == 5 for s in result)

    def test_fixed_single_segment(self):
        """弹幕数不足一段"""
        danmaku_list = [make_danmaku_item(f"弹幕{i}", float(i)) for i in range(3)]
        result = self.segmenter.segment(danmaku_list)
        assert len(result) == 1
        assert result[0].danmaku_count == 3

    def test_zone_labeling_single_segment(self):
        """单段默认标记为 cold_zone"""
        danmaku_list = [make_danmaku_item(f"弹幕{i}", float(i)) for i in range(3)]
        result = self.segmenter.segment(danmaku_list)
        assert result[0].zone_type == "cold_zone"

    def test_indices_continuity(self):
        """弹幕索引连续性"""
        danmaku_list = [make_danmaku_item(f"弹幕{i}", float(i)) for i in range(12)]
        result = self.segmenter.segment(danmaku_list)
        all_indices = []
        for seg in result:
            all_indices.extend(seg.danmaku_indices)
        assert all_indices == list(range(12))


class TestTimelineSegmenterDynamic:
    """时序切分器 - 动态模式测试"""

    def setup_method(self):
        """以 dynamic 模式初始化（真实构造后覆写字段，避免跳过构造函数）"""
        self.segmenter = TimelineSegmenter()
        self.segmenter.min_segment_samples = 5
        self.segmenter.segmentation_mode = "dynamic"

    def test_dynamic_produces_segments(self):
        """动态模式能产生切分结果"""
        # 模拟密度突变：前 20 条密集（0-2秒），后 20 条稀疏（10-30秒）
        dense = [make_danmaku_item(f"密集{i}", i * 0.1) for i in range(20)]
        sparse = [make_danmaku_item(f"稀疏{i}", 10.0 + i * 1.0) for i in range(20)]
        danmaku_list = dense + sparse
        result = self.segmenter.segment(danmaku_list)
        assert len(result) >= 1
        # 所有弹幕都被分配
        total_count = sum(s.danmaku_count for s in result)
        assert total_count == 40

    def test_merge_small_segments(self):
        """过小段合并逻辑"""
        # 构造 segments 手动测试 _merge_small_segments
        seg_small = TimeSegment(
            start_time=0.0, end_time=1.0,
            danmaku_indices=[0, 1],  # 2 条 < min_segment_samples=5
            density=2.0, zone_type="",
        )
        seg_normal = TimeSegment(
            start_time=1.0, end_time=10.0,
            danmaku_indices=[2, 3, 4, 5, 6, 7, 8, 9],
            density=0.89, zone_type="",
        )
        merged = self.segmenter._merge_small_segments([seg_normal, seg_small])
        # 小段应被合并到前一段
        assert len(merged) == 1
        assert merged[0].danmaku_count == 10

    def test_label_zones_hot_cold(self):
        """hot/cold zone 标记"""
        # 构造密度差异明显的段
        segments = [
            TimeSegment(start_time=0, end_time=10, danmaku_indices=list(range(10)),
                        density=1.0, zone_type=""),
            TimeSegment(start_time=10, end_time=20, danmaku_indices=list(range(10, 20)),
                        density=1.0, zone_type=""),
            TimeSegment(start_time=20, end_time=21, danmaku_indices=list(range(20, 50)),
                        density=30.0, zone_type=""),  # 极高密度
        ]
        result = self.segmenter._label_zones(segments)
        # 高密度段应标为 hot_zone
        assert result[2].zone_type == "hot_zone"
        # 低密度段应标为 cold_zone
        assert result[0].zone_type == "cold_zone"
        assert result[1].zone_type == "cold_zone"

    def test_density_signal_computation(self):
        """密度信号计算"""
        timestamps = np.array([0.5, 0.8, 1.2, 1.5, 1.9, 5.0])
        signal = self.segmenter._compute_density_signal(timestamps, window_size=1.0)
        # min_time=0.5, 窗口: [0.5,1.5)→3条, [1.5,2.5)→2条, ...
        assert len(signal) >= 5
        # 第一个窗口 [0.5, 1.5) 有 3 条 (0.5, 0.8, 1.2)
        assert signal[0] == 3
        # 第二个窗口 [1.5, 2.5) 有 2 条 (1.5, 1.9)
        assert signal[1] == 2

    def test_all_segments_have_zone_type(self):
        """所有段都有 zone_type 标记"""
        dense = [make_danmaku_item(f"弹幕{i}", i * 0.05) for i in range(30)]
        sparse = [make_danmaku_item(f"弹幕{i}", 5.0 + i * 2.0) for i in range(30)]
        result = self.segmenter.segment(dense + sparse)
        for seg in result:
            assert seg.zone_type in ("hot_zone", "cold_zone")


# ========== 集成冒烟测试 ==========

class TestIntegrationSmoke:
    """跨模块冒烟测试（仍不依赖网络）"""

    def test_hard_metrics_to_aggregator(self):
        """HardMetrics 结果能顺利传入 Aggregator"""
        analyzer = HardMetricsAnalyzer()
        hm_result = analyzer.analyze(["哈哈哈", "太好看了！", "666"])

        record = DanmakuRecord(
            tname="动画",
            zone_type="cold_zone",
            tags=["测试"],
            hard_metrics=hm_result,
            llm_result=make_dual_path_result(),
            segment_id=0,
        )
        aggregator = Aggregator()
        agg_result = aggregator.aggregate([record])
        assert len(agg_result) == 1
        assert agg_result[0].danmaku_count == 3


# ========== Reporter metadata 透传测试 ==========

class TestReporterMetadata:
    """metadata.json 字段透传测试（为语料库回读提供视频身份字段）"""

    def test_extra_fields_written_to_metadata_json(self, tmp_path):
        """pipeline 传入的 pubdate/view_count/danmaku_count/pipeline_version 原样写入"""
        reporter = Reporter(output_dir=str(tmp_path))
        aggregated = Aggregator().aggregate([make_danmaku_record()])
        extra = {
            "bvid": "BV1test000001",
            "title": "测试视频",
            "tname": "游戏",
            "tags": ["TAG1"],
            "pubdate": "2025-08-05T12:00:00",
            "view_count": 12345,
            "danmaku_count": 40,
            "pipeline_version": "0.3.0-beta",
        }
        filepath = reporter._generate_metadata(aggregated, extra)
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for key, value in extra.items():
            assert data[key] == value, f"字段 {key} 未正确透传"

    def test_internal_fields_still_present(self, tmp_path):
        """内部生成字段（prompt_version/generated_at/汇总统计）不受透传影响"""
        reporter = Reporter(output_dir=str(tmp_path))
        aggregated = Aggregator().aggregate([make_danmaku_record()])
        filepath = reporter._generate_metadata(aggregated, {"bvid": "BV1x"})
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        assert "prompt_version" in data
        assert "generated_at" in data
        assert data["total_videos"] == 1
        assert data["partitions"] == ["游戏"]
