"""隔离的 CSV/ZIP 往返回归：来源表、真实分母、缺失值与独立视频观测。"""

import json
import zipfile
from unittest.mock import patch

import pandas as pd
import pytest

import danmaku_analyzer.config as config_module
from danmaku_analyzer.corpus_builder import (
    NON_DIST_COLUMNS,
    CorpusBuilder,
    validate_zip_archive,
)
from danmaku_analyzer.statistical_validator import StatisticalValidator


@pytest.fixture(autouse=True)
def isolated_settings(monkeypatch, tmp_path):
    settings = config_module.Settings.model_construct(DATA_ROOT=str(tmp_path), CORPUS_ZONE_POLICY="weighted")
    monkeypatch.setattr(config_module, "_settings", settings)
    return settings


def write_zip(tmp_path, bvid, tables, tname="游戏", pubdate="2025-01-01T00:00:00"):
    path = tmp_path / f"{bvid}.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("metadata.json", json.dumps({
            "bvid": bvid, "tname": tname, "pubdate": pubdate, "prompt_version": "v2.3.0",
        }))
        for name, rows in tables.items():
            archive.writestr(name, pd.DataFrame(rows).to_csv(index=False).encode("utf-8-sig"))
    return str(path)


def row(zone="hot_zone", count=100, **values):
    return {"tname": "游戏", "zone_type": zone, "danmaku_count": count, **values}


def full_tables():
    return {
        "table_lexical_by_partition.csv": [
            row(total_word_count=10, total_char_count=20, avg_word_length=2.0,
                content_word_density=0.2, punctuation_emoji_rate=0.4, pos_n=1.0, syllable_one=1.0),
            row("cold_zone", 20, total_word_count=90, total_char_count=360, avg_word_length=4.0,
                content_word_density=0.8, punctuation_emoji_rate=0.1, pos_v=1.0, syllable_two=1.0),
        ],
        "table_emotion.csv": [
            row(label_weight_sum=2.0, valid_label_count=2, positive=1.0, cooperative_principle_violation_rate=0.5),
            row("cold_zone", 20, label_weight_sum=8.0, valid_label_count=8, negative=1.0, cooperative_principle_violation_rate=0.0),
        ],
        "table_sentence_function.csv": [
            row(label_weight_sum=6.0, valid_label_count=6, interrogative=1.0),
            row("cold_zone", 20, label_weight_sum=4.0, valid_label_count=4, declarative=1.0),
        ],
        "table_interaction_type.csv": [
            row(label_weight_sum=2.0, valid_label_count=2, technical_exchange=1.0),
            row("cold_zone", 20, label_weight_sum=8.0, valid_label_count=8, emotional_resonance=1.0),
        ],
        "table_orthography.csv": [
            row(total_char_count=20, label_weight_sum=2.0, valid_label_count=2, hard_typo=0.5, soft_standard=1.0),
            row("cold_zone", 20, total_char_count=360, label_weight_sum=8.0, valid_label_count=8,
                hard_typo=0.1, soft_community_variant=1.0),
        ],
        "table_consensus_stats.csv": [
            row(llm_record_count=2, failed_record_count=0, degraded_record_count=0,
                high_consensus_rate=1.0, medium_consensus_rate=0.0, low_consensus_rate=0.0, avg_weight_multiplier=1.0),
            row("cold_zone", 20, llm_record_count=8, failed_record_count=0, degraded_record_count=0,
                high_consensus_rate=0.0, medium_consensus_rate=0.0, low_consensus_rate=1.0, avg_weight_multiplier=0.2),
        ],
    }


class TestCorpusDenominatorIntegrity:

    def test_sparse_csv_roundtrip_uses_each_metric_denominator(self, tmp_path):
        builder = CorpusBuilder()
        path = write_zip(tmp_path, "BVweighted", full_tables())
        metadata, tables = builder.read_zip(path)
        assert pd.isna(tables["table_emotion.csv"].iloc[0]["negative"])
        summary = builder.summarize_video(metadata, tables)[0]
        assert summary.danmaku_count == 120
        assert summary.scalars["avg_word_length"] == pytest.approx(3.8)
        assert summary.scalars["content_word_density"] == pytest.approx(0.74)
        assert summary.scalars["punctuation_emoji_rate"] == pytest.approx(0.35)
        assert summary.scalars["high_consensus_rate"] == pytest.approx(0.2)
        assert summary.scalars["avg_weight_multiplier"] == pytest.approx(0.36)
        assert summary.scalars["cooperative_principle_violation_rate"] == pytest.approx(0.1)
        expected = {"positive": 0.2, "negative": 0.8, "pos_n": 0.1, "pos_v": 0.9,
                    "syllable_one": 0.1, "syllable_two": 0.9, "interrogative": 0.6,
                    "declarative": 0.4, "technical_exchange": 0.2, "emotional_resonance": 0.8,
                    "soft_standard": 0.2, "soft_community_variant": 0.8,
                    "hard_typo": (0.5 * 20 + 0.1 * 360) / 380}
        assert summary.distributions == pytest.approx(expected)
        for pair in [("positive", "negative"), ("pos_n", "pos_v"), ("syllable_one", "syllable_two"),
                     ("interrogative", "declarative"), ("technical_exchange", "emotional_resonance"),
                     ("soft_standard", "soft_community_variant")]:
            assert sum(summary.distributions[key] for key in pair) == pytest.approx(1.0)
        assert not NON_DIST_COLUMNS.intersection(summary.distributions)
        assert summary.distribution_sources["positive"] == "table_emotion.csv"
        assert summary.denominators["table_emotion.csv:label_weight_sum"] == 10
        assert not summary.legacy_denominators
        assert not summary.warnings

    def test_cross_video_distributions_keep_denominators_and_table_availability(self, tmp_path):
        builder = CorpusBuilder()
        paths = [write_zip(tmp_path, "BVfirst", full_tables())]
        paths.append(write_zip(tmp_path, "BVsecond", {
            "table_lexical_by_partition.csv": [row(count=1000, total_word_count=1000, total_char_count=1000,
                avg_word_length=1.0, content_word_density=0.9, punctuation_emoji_rate=0.0, pos_n=1.0)],
            "table_emotion.csv": [row(count=1000, label_weight_sum=90, valid_label_count=90, positive=1.0,
                cooperative_principle_violation_rate=0.3)],
            "table_orthography.csv": [row(count=1000, total_char_count=1000, label_weight_sum=90,
                valid_label_count=90, hard_typo=0.9, soft_standard=1.0)],
        }))
        paths.append(write_zip(tmp_path, "BVmissing", {
            "table_sentence_function.csv": [row(count=10000, label_weight_sum=1, valid_label_count=1, declarative=1.0)],
        }))
        result = builder.build_from_zips(paths, str(tmp_path / "out"))
        group = pd.read_csv(result.csv_path).iloc[0]
        assert group["video_count"] == 3
        assert group["positive_mean"] == pytest.approx(0.92)
        assert group["negative_mean"] == pytest.approx(0.08)
        assert group["pos_n_mean"] == pytest.approx((10 + 1000) / 1100)
        assert group["pos_v_mean"] == pytest.approx(90 / 1100)
        assert group["soft_standard_mean"] == pytest.approx(0.92)
        assert group["hard_typo_mean"] == pytest.approx((0.5 * 20 + 0.1 * 360 + 0.9 * 1000) / 1380)
        assert group["content_word_density_mean"] == pytest.approx((0.74 + 0.9) / 2)
        assert group["cooperative_principle_violation_rate_mean"] == pytest.approx(0.2)
        videos = pd.read_csv(result.videos_csv_path).set_index("bvid")
        assert videos.loc["BVsecond", "negative"] == 0.0
        assert pd.isna(videos.loc["BVmissing", "positive"])
        assert pd.isna(videos.loc["BVmissing", "content_word_density"])
        assert any("缺少表 table_lexical_by_partition.csv" in warning for warning in result.warnings)

    def test_missing_scalar_table_stays_missing_in_video_and_group(self, tmp_path):
        tables = {"table_emotion.csv": [row(label_weight_sum=2, valid_label_count=2, positive=1.0)]}
        builder = CorpusBuilder()
        summary = builder.summarize_video(*builder.read_zip(write_zip(tmp_path, "BVmissing", tables)))[0]
        assert pd.isna(summary.scalars["content_word_density"])
        assert pd.isna(summary.scalars["cooperative_principle_violation_rate"])
        group = builder._aggregate_group("游戏", "", "", [summary])
        assert pd.isna(group["content_word_density_mean"])
        assert pd.isna(group["content_word_density_std"])
        assert any("缺少必要指标 cooperative_principle_violation_rate" in warning for warning in summary.warnings)

    @pytest.mark.parametrize("invalid_denominator", [0, float("nan"), -1])
    def test_invalid_new_denominator_never_falls_back_or_invents_zero(self, tmp_path, invalid_denominator):
        tables = {"table_emotion.csv": [row(label_weight_sum=invalid_denominator,
            valid_label_count=0, positive=0.0, negative=float("nan"), cooperative_principle_violation_rate=0.0)]}
        builder = CorpusBuilder()
        summary = builder.summarize_video(*builder.read_zip(write_zip(tmp_path, "BVzero", tables)))[0]
        assert pd.isna(summary.distributions["positive"])
        assert pd.isna(summary.distributions["negative"])
        assert pd.isna(summary.scalars["cooperative_principle_violation_rate"])
        assert not summary.legacy_denominators
        group = builder._aggregate_group("游戏", "", "", [summary])
        assert pd.isna(group["positive_mean"])

    def test_partially_missing_scalar_uses_only_observed_denominators(self, tmp_path):
        tables = full_tables()
        tables["table_lexical_by_partition.csv"][0]["content_word_density"] = float("nan")
        builder = CorpusBuilder()
        summary = builder.summarize_video(*builder.read_zip(write_zip(tmp_path, "BVpartial", tables)))[0]
        assert summary.scalars["content_word_density"] == pytest.approx(0.8)
        assert summary.denominators["metric:content_word_density"] == 90
        assert any("content_word_density 缺失" in warning for warning in summary.warnings)

    def test_zone_missing_entire_table_does_not_enter_distribution_denominator(self, tmp_path):
        tables = full_tables()
        tables["table_emotion.csv"] = tables["table_emotion.csv"][:1]
        builder = CorpusBuilder()
        summary = builder.summarize_video(*builder.read_zip(write_zip(tmp_path, "BVpartial", tables)))[0]
        assert summary.distributions["positive"] == 1.0
        assert summary.denominators["table_emotion.csv:label_weight_sum"] == 2
        assert any("cold_zone 缺少表 table_emotion.csv" in warning for warning in summary.warnings)

    def test_legacy_sparse_report_warns_and_uses_count_estimate(self, tmp_path):
        tables = full_tables()
        diagnostics = NON_DIST_COLUMNS - {"tname", "zone_type", "danmaku_count"}
        legacy = {name: [{key: value for key, value in item.items() if key not in diagnostics} for item in rows]
                  for name, rows in tables.items()}
        builder = CorpusBuilder()
        with patch("danmaku_analyzer.corpus_builder.logger") as logger:
            result = builder.build_from_zips([write_zip(tmp_path, "BVlegacy", legacy)], str(tmp_path / "out"))
        group = pd.read_csv(result.csv_path).iloc[0]
        assert group["positive_mean"] == pytest.approx(100 / 120)
        assert group["positive_mean"] + group["negative_mean"] == pytest.approx(1.0)
        assert any("兼容估计" in warning and "不能精确重聚合" in warning for warning in result.warnings)
        assert any("缺少分母 label_weight_sum" in call.args[0] for call in logger.warning.call_args_list)

    def test_mixed_denominators_are_explicitly_reported(self, tmp_path):
        tables = full_tables()
        old = {"table_emotion.csv": [row(positive=1.0, cooperative_principle_violation_rate=0.5)]}
        paths = [write_zip(tmp_path, "BVnew", tables), write_zip(tmp_path, "BVold", old)]
        result = CorpusBuilder().build_from_zips(paths, str(tmp_path / "out"))
        assert any("混合新旧分母" in warning for warning in result.warnings)
        assert any("兼容估计" in warning for warning in result.warnings)

    def test_legacy_all_nan_labels_stay_unknown(self, tmp_path):
        tables = {"table_emotion.csv": [row(positive=float("nan"), negative=float("nan"),
            cooperative_principle_violation_rate=float("nan"))]}
        builder = CorpusBuilder()
        summary = builder.summarize_video(*builder.read_zip(write_zip(tmp_path, "BVunknown", tables)))[0]
        assert all(pd.isna(value) for value in summary.distributions.values())
        assert pd.isna(builder._aggregate_group("游戏", "", "", [summary])["positive_mean"])

    def test_duplicate_zip_does_not_increase_video_count(self, tmp_path):
        path = write_zip(tmp_path, "BVrepeat", full_tables())
        result = CorpusBuilder().build_from_zips([path, path], str(tmp_path / "out"))
        assert pd.read_csv(result.csv_path).iloc[0]["video_count"] == 1
        assert len(pd.read_csv(result.videos_csv_path)) == 1
        assert any("重复视频观测" in warning for warning in result.warnings)

    def test_diff_distinguishes_missing_from_zero(self, tmp_path):
        paths = []
        for name, density in [("missing", float("nan")), ("zero", 0.0)]:
            path = tmp_path / f"{name}.csv"
            pd.DataFrame([{"bvid": "BVdiff", "content_word_density": density}]).to_csv(path, index=False)
            paths.append(str(path))
        report = CorpusBuilder().diff(*paths)
        assert len(report.changed) == 1
        assert pd.isna(report.changed[0]["fields"]["content_word_density"]["old"])
        assert report.changed[0]["fields"]["content_word_density"]["new"] == 0.0
        assert not CorpusBuilder().diff(paths[0], paths[0]).changed


class TestCorpusZoneIntegration:

    def test_missing_table_reduces_only_its_metrics_effective_video_n(self, tmp_path):
        paths = []
        for partition in ("游戏", "音乐"):
            for index in range(3):
                tables = full_tables()
                if partition == "游戏" and index == 0:
                    del tables["table_lexical_by_partition.csv"]
                paths.append(write_zip(tmp_path, f"BV{partition}{index}", tables, tname=partition))
        result = CorpusBuilder().build_from_zips(paths, str(tmp_path / "out"))
        frame = StatisticalValidator().corpus_compare(result.videos_csv_path).to_dataframe()
        tests = frame[frame["test_type"].isin(["Mann-Whitney U", "Kruskal-Wallis"])]
        assert "content_word_density" not in set(tests["metric"])
        assert "high_consensus_rate" in set(tests["metric"])
        status = frame[(frame["test_type"] == "sample_status") & (frame["metric"] == "content_word_density")].iloc[0]
        assert status["group1"] == "游戏"
        assert status["n1"] == 2

    @pytest.mark.parametrize("video_count", [2, 3])
    def test_all_policy_zip_roundtrip_counts_unique_videos_per_zone(self, tmp_path, isolated_settings, video_count):
        isolated_settings.CORPUS_ZONE_POLICY = "all"
        paths = [write_zip(tmp_path, f"BV{partition}{index}", full_tables(), tname=partition)
                 for partition in ("游戏", "音乐") for index in range(video_count)]
        result = CorpusBuilder().build_from_zips(paths + paths[:1], str(tmp_path / "out"))
        frame = StatisticalValidator().corpus_compare(result.videos_csv_path).to_dataframe()
        statuses = frame[(frame["test_type"] == "sample_status") & (frame["metric"] == "")]
        assert len(statuses) == 4
        assert set(statuses["n1"]) == {video_count}
        tests = frame[frame["test_type"].isin(["Mann-Whitney U", "Kruskal-Wallis"])]
        if video_count < 3:
            assert tests.empty
        else:
            assert not tests.empty
            pairs = tests[tests["test_type"] == "Mann-Whitney U"]
            assert set(pairs["n1"]) == {3}
            assert set(pairs["n2"]) == {3}
            assert pairs["note"].str.contains("冷热区分层").all()


class TestZipIntegrity:

    def test_empty_archive_is_not_valid(self, tmp_path):
        path = tmp_path / "empty.zip"
        with zipfile.ZipFile(path, "w"):
            pass
        assert not validate_zip_archive(str(path), 0)

    def test_every_member_is_read_not_just_first(self, tmp_path):
        path = tmp_path / "members.zip"
        with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as archive:
            archive.writestr("first.txt", "first-payload")
            archive.writestr("second.txt", "second-payload")
        assert validate_zip_archive(str(path), 2)
        path.write_bytes(path.read_bytes().replace(b"second-payload", b"broken-payload"))
        assert not validate_zip_archive(str(path), 2)
