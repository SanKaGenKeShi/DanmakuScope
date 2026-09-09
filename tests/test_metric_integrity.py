import io
from types import SimpleNamespace

import pandas as pd
import pytest

from danmaku_analyzer.aggregator import AggregatedData, Aggregator, DanmakuRecord
from danmaku_analyzer.hard_metrics import HardMetricsAnalyzer
from danmaku_analyzer.llm_models import ConsensusLevel, DualPathResult, LLMOutput
from danmaku_analyzer.reporter import Reporter


def test_hard_statistics_are_invariant_under_segmentation():
    analyzer = HardMetricsAnalyzer.__new__(HardMetricsAnalyzer)
    texts = ["猫AB!!"] * 30 + ["以及" * 9] * 30
    tokens = [[("猫", "n"), ("AB", "eng")]] * 30 + [[("以及", "c")] * 9] * 30
    whole = analyzer._compute_stats(texts, tokens)
    segments = [analyzer._compute_stats(texts[:30], tokens[:30]),
                analyzer._compute_stats(texts[30:], tokens[30:])]
    aggregate = AggregatedData(tname="游戏", zone_type="hot_zone")
    Aggregator()._aggregate_hard_metrics(aggregate, segments)
    for attribute in ("avg_word_length", "content_word_density", "punctuation_emoji_rate",
                      "pos_distribution", "syllable_distribution", "orthography_hard_metrics"):
        assert getattr(aggregate, attribute) == pytest.approx(getattr(whole, attribute))
    assert aggregate.total_word_count == whole.total_word_count
    assert aggregate.total_char_count == whole.total_char_count


def test_failure_records_keep_hard_statistics_and_valid_label_denominators():
    analyzer = HardMetricsAnalyzer.__new__(HardMetricsAnalyzer)
    metrics = analyzer._compute_stats(["猫"], [[("猫", "n")]])
    valid = DualPathResult(LLMOutput.default(), ConsensusLevel.HIGH, 0, 1, [], "test")
    missing = DualPathResult(LLMOutput.missing(), ConsensusLevel.LOW, 1, 0.2, [], "test", analysis_status="failed")
    records = [DanmakuRecord("游戏", "hot_zone", [], metrics, result, index)
               for index, result in enumerate([valid, missing])]
    aggregate = Aggregator().aggregate(records)[0]
    assert aggregate.danmaku_count == 2
    assert aggregate.llm_record_count == 2
    assert aggregate.failed_record_count == 1
    assert aggregate.low_consensus_rate == 0.5
    assert aggregate.emotion_distribution == {"neutral": 1.0}
    assert aggregate.label_weight_sums["emotion"] == 1.0
    assert aggregate.valid_label_counts["emotion"] == 1


def test_report_exports_denominators_and_keeps_unknown_labels_blank(tmp_path):
    aggregate = AggregatedData(tname="游戏", zone_type="hot_zone", danmaku_count=30,
                               total_word_count=50, total_char_count=100, llm_record_count=10,
                               failed_record_count=10)
    Aggregator()._aggregate_soft_labels(aggregate, [SimpleNamespace(llm_result=SimpleNamespace(
        output=LLMOutput.missing(), weight_multiplier=0.2,
    ))])
    reports = Reporter(str(tmp_path)).generate_reports([aggregate])
    lexical = pd.read_csv(reports["lexical_by_partition"])
    assert lexical.iloc[0]["total_word_count"] == 50
    assert lexical.iloc[0]["total_char_count"] == 100
    emotion = pd.read_csv(reports["emotion"])
    assert emotion.iloc[0]["label_weight_sum"] == 0
    assert emotion.iloc[0]["valid_label_count"] == 0
    assert pd.isna(emotion.iloc[0]["neutral"])
    consensus = pd.read_csv(reports["consensus_stats"])
    assert consensus.iloc[0]["llm_record_count"] == 10
    assert consensus.iloc[0]["failed_record_count"] == 10


def test_kappa_export_does_not_turn_missing_into_false_or_zero(tmp_path):
    reporter = Reporter(str(tmp_path))
    path = reporter._generate_kappa_ready([{
        "llm_output": LLMOutput.missing().to_dict(), "analysis_status": "failed",
        "requested_paths": 2, "successful_paths": 0, "sentence_function_source": "unavailable",
    }])
    with open(path, encoding="utf-8-sig") as stream:
        row = pd.read_csv(io.StringIO(stream.read())).iloc[0]
    assert row["analysis_status"] == "failed"
    assert row["successful_paths"] == 0
    assert pd.isna(row["cooperative_principle_violated"])
    assert pd.isna(row["emotion_confidence"])


def test_empty_analysis_reports_have_readable_headers(tmp_path):
    reports = Reporter(str(tmp_path)).generate_reports([])
    for key in ["lexical_by_partition", "orthography", "sentence_function", "emotion", "interaction_type", "consensus_stats"]:
        table = pd.read_csv(reports[key])
        assert table.empty
        assert "zone_type" in table.columns
