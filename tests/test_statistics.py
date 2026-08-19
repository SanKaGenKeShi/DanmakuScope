"""
语料库级推断统计单元测试 - KW / 逐对 MWU / Cliff's delta / corpus_compare
以 scipy 为参照交叉验证，并覆盖样本量门槛与开关语义
"""

import pandas as pd
import pytest
from scipy import stats

from danmaku_analyzer.config import get_settings
from danmaku_analyzer.corpus_builder import SCALAR_FIELDS
from danmaku_analyzer.statistical_validator import (
    STATISTICAL_TESTS_COLUMNS,
    StatisticalValidator,
    cohen_kappa,
)


@pytest.fixture
def validator():
    return StatisticalValidator()


def write_videos_csv(tmp_path, rows) -> str:
    path = str(tmp_path / "corpus_videos.csv")
    pd.DataFrame(rows).to_csv(path, index=False, encoding='utf-8-sig')
    return path


def make_video(bvid: str, tname: str, density: float, consensus: float = 0.5) -> dict:
    row = {
        "bvid": bvid, "tname": tname, "pubdate": "2025-01-01T00:00:00",
        "prompt_version": "v2.2.1", "zone_type": "", "danmaku_count": 100,
    }
    for name in SCALAR_FIELDS:
        row[name] = consensus
    row["content_word_density"] = density
    return row


class TestCliffDelta:

    def test_identical_samples_zero(self, validator):
        assert validator.cliff_delta([1.0, 2.0], [1.0, 2.0]) == pytest.approx(0.0)

    def test_full_separation(self, validator):
        assert validator.cliff_delta([1.0, 2.0], [3.0, 4.0]) == pytest.approx(-1.0)
        assert validator.cliff_delta([3.0, 4.0], [1.0, 2.0]) == pytest.approx(1.0)

    def test_symmetry(self, validator):
        a, b = [1.0, 3.0, 5.0], [2.0, 4.0, 6.0]
        assert validator.cliff_delta(a, b) == pytest.approx(-validator.cliff_delta(b, a))

    def test_manual_value(self, validator):
        # a=[1,2] b=[2,3]：(1,2)<,(1,3)<,(2,2)=,(2,3)< → (0-3)/4
        assert validator.cliff_delta([1.0, 2.0], [2.0, 3.0]) == pytest.approx(-0.75)

    def test_magnitude_boundaries(self):
        assert StatisticalValidator._cliff_delta_magnitude(0.1) == "negligible"
        assert StatisticalValidator._cliff_delta_magnitude(-0.2) == "small"
        assert StatisticalValidator._cliff_delta_magnitude(0.4) == "medium"
        assert StatisticalValidator._cliff_delta_magnitude(-0.5) == "large"


class TestKruskalWallis:

    def test_matches_scipy(self, validator):
        groups = {"A": [1.0, 2.0, 3.0, 4.0], "B": [5.0, 6.0, 7.0, 8.0], "C": [3.0, 4.0, 5.0, 6.0]}
        row = validator.kruskal_wallis_test(groups, "content_word_density")
        expected_h, expected_p = stats.kruskal(groups["A"], groups["B"], groups["C"])
        assert row["test_type"] == "Kruskal-Wallis"
        assert row["statistic"] == pytest.approx(expected_h, abs=1e-4)
        assert row["p_value"] == pytest.approx(expected_p, abs=1e-6)
        assert row["n1"] == 12
        assert "未校正" in row["note"]

    def test_single_group_returns_none(self, validator):
        assert validator.kruskal_wallis_test({"A": [1.0, 2.0]}, "m") is None

    def test_identical_values_degrades(self, validator):
        row = validator.kruskal_wallis_test({"A": [1.0, 1.0], "B": [1.0, 1.0]}, "m")
        assert row["p_value"] == pytest.approx(1.0)
        assert "退化" in row["note"]


class TestPairwiseMannWhitney:

    def test_three_groups_yield_three_pairs(self, validator):
        groups = {"A": [1.0, 2.0, 3.0], "B": [4.0, 5.0, 6.0], "C": [2.0, 3.0, 4.0]}
        rows = validator.pairwise_mann_whitney(groups, "content_word_density")
        assert len(rows) == 3
        pairs = {(r["group1"], r["group2"]) for r in rows}
        assert pairs == {("A", "B"), ("A", "C"), ("B", "C")}

    def test_matches_scipy(self, validator):
        groups = {"A": [1.0, 2.0, 3.0, 4.0], "B": [5.0, 6.0, 7.0, 8.0]}
        row = validator.pairwise_mann_whitney(groups, "content_word_density")[0]
        expected_u, expected_p = stats.mannwhitneyu(groups["A"], groups["B"], alternative='two-sided')
        assert row["group1"] == "A" and row["group2"] == "B"
        assert row["statistic"] == pytest.approx(expected_u, abs=1e-4)
        assert row["p_value"] == pytest.approx(expected_p, abs=1e-6)
        assert row["effect_size"] == pytest.approx(-1.0, abs=1e-4)
        assert row["effect_magnitude"] == "large"
        assert "未校正" in row["note"]

    def test_identical_values_degrades(self, validator):
        row = validator.pairwise_mann_whitney({"A": [2.0, 2.0], "B": [2.0, 2.0]}, "m")[0]
        assert row["p_value"] == pytest.approx(1.0)
        assert row["effect_size"] == pytest.approx(0.0)


class TestCorpusCompare:

    def _three_partition_csv(self, tmp_path):
        rows = [make_video(f"BV1g{i}", "游戏", 0.3 + i * 0.05) for i in range(3)]
        rows += [make_video(f"BV1m{i}", "音乐", 0.6 + i * 0.05) for i in range(3)]
        rows += [make_video("BV1k0", "知识", 0.9)]
        return write_videos_csv(tmp_path, rows)

    def test_valid_partitions_produce_kw_and_pairwise(self, validator, tmp_path):
        csv_path = self._three_partition_csv(tmp_path)
        result = validator.corpus_compare(csv_path)
        df = result.to_dataframe()
        assert list(df.columns) == STATISTICAL_TESTS_COLUMNS

        kw_rows = df[df["test_type"] == "Kruskal-Wallis"]
        assert len(kw_rows) == len(SCALAR_FIELDS)
        for _, row in kw_rows.iterrows():
            assert "未校正" in row["note"]

        mwu_rows = df[df["test_type"] == "Mann-Whitney U"]
        # 有效分区 {游戏, 音乐} → C(2,2)=1 对 × 每指标
        assert len(mwu_rows) == len(SCALAR_FIELDS)
        assert set(mwu_rows["group1"]) <= {"游戏", "音乐"}
        assert "知识" not in set(mwu_rows["group2"])

    def test_insufficient_partition_marked_and_excluded(self, validator, tmp_path):
        csv_path = self._three_partition_csv(tmp_path)
        result = validator.corpus_compare(csv_path)
        df = result.to_dataframe()
        status = df[df["test_type"] == "sample_status"].set_index("group1")
        assert "insufficient_sample" in status.loc["知识", "note"]
        assert status.loc["游戏", "note"] == "sample_sufficient"
        assert status.loc["知识", "n1"] == 1

    def test_cross_validate_kw_with_scipy(self, validator, tmp_path):
        csv_path = self._three_partition_csv(tmp_path)
        result = validator.corpus_compare(csv_path)
        df = result.to_dataframe()
        kw = df[(df["test_type"] == "Kruskal-Wallis") & (df["metric"] == "content_word_density")].iloc[0]
        expected_h, expected_p = stats.kruskal(
            [0.3, 0.35, 0.4], [0.6, 0.65, 0.7],
        )
        assert kw["statistic"] == pytest.approx(expected_h, abs=1e-4)
        assert kw["p_value"] == pytest.approx(expected_p, abs=1e-6)

    def test_single_valid_group_no_between_test_but_status_rows(self, validator, tmp_path):
        rows = [make_video(f"BV1g{i}", "游戏", 0.3 + i * 0.05) for i in range(3)]
        rows += [make_video("BV1m0", "音乐", 0.6)]
        csv_path = write_videos_csv(tmp_path, rows)
        result = validator.corpus_compare(csv_path)
        df = result.to_dataframe()
        # 有效组仅 1 个不执行组间检验，但样本状态行照常输出；多分区场景不补配对/注记行
        assert (df["test_type"] == "sample_status").all()
        status = df.set_index("group1")["note"]
        assert status["游戏"] == "sample_sufficient"
        assert "insufficient_sample" in status["音乐"]
        assert list(df.columns) == list(STATISTICAL_TESTS_COLUMNS)

    def test_disabled_switch_returns_disabled_empty(self, validator, tmp_path, monkeypatch):
        monkeypatch.setattr(get_settings(), "ENABLE_CORPUS_STATISTICS", False)
        csv_path = self._three_partition_csv(tmp_path)
        result = validator.corpus_compare(csv_path)
        assert result.enabled is False
        assert result.rows == []

    def test_missing_groupby_column_raises(self, validator, tmp_path):
        path = str(tmp_path / "no_group.csv")
        pd.DataFrame([{"bvid": "BV1x", "content_word_density": 0.5}]).to_csv(path, index=False)
        with pytest.raises(ValueError):
            validator.corpus_compare(path)

    def test_to_csv_writes_header_only_when_empty(self, validator, tmp_path):
        out = str(tmp_path / "statistical_tests.csv")
        from danmaku_analyzer.statistical_validator import ComparisonResult
        ComparisonResult(rows=[]).to_csv(out)
        df = pd.read_csv(out, encoding='utf-8-sig')
        assert list(df.columns) == STATISTICAL_TESTS_COLUMNS
        assert len(df) == 0


class TestPluralModes:
    """复数分析：单分区历时比较 / 冷热区视频内配对 / 无比较轴注记"""

    def _make_row(self, bvid, tname, density, time_period="", zone_type=""):
        row = make_video(bvid, tname, density)
        row["time_period"] = time_period
        row["zone_type"] = zone_type
        return row

    def test_single_partition_temporal_groups_by_period(self, validator, tmp_path):
        rows = [self._make_row(f"BV1a{i}", "游戏", 0.3 + i * 0.05, time_period="2023") for i in range(3)]
        rows += [self._make_row(f"BV1b{i}", "游戏", 0.6 + i * 0.05, time_period="2024") for i in range(3)]
        csv_path = write_videos_csv(tmp_path, rows)
        df = validator.corpus_compare(csv_path).to_dataframe()
        # 单分区×多时段 → 自动按时段分组执行组间检验
        assert set(df[df["test_type"] == "sample_status"]["group1"]) == {"2023", "2024"}
        assert len(df[df["test_type"] == "Kruskal-Wallis"]) == len(SCALAR_FIELDS)
        mwu = df[df["test_type"] == "Mann-Whitney U"]
        assert {(r["group1"], r["group2"]) for r in mwu.to_dict("records")} == {("2023", "2024")}

    def test_single_partition_no_axis_emits_note(self, validator, tmp_path):
        rows = [self._make_row(f"BV1a{i}", "游戏", 0.3 + i * 0.05) for i in range(3)]
        csv_path = write_videos_csv(tmp_path, rows)
        df = validator.corpus_compare(csv_path).to_dataframe()
        note_rows = df[df["test_type"] == "note"]
        assert len(note_rows) == 1
        assert "无可用比较轴" in note_rows.iloc[0]["note"]

    def test_zone_paired_wilcoxon_matches_scipy(self, validator, tmp_path):
        rows = []
        for i in range(4):
            rows.append(self._make_row(f"BV1z{i}", "游戏", 0.3 + i * 0.02, zone_type="hot_zone"))
            rows.append(self._make_row(f"BV1z{i}", "游戏", 0.6 + i * 0.02, zone_type="cold_zone"))
        csv_path = write_videos_csv(tmp_path, rows)
        paired = validator.zone_paired_compare(csv_path)
        assert len(paired) == len(SCALAR_FIELDS)
        for row in paired:
            assert row["test_type"] == "Wilcoxon 符号秩（配对）"
            assert row["group1"] == "hot_zone" and row["group2"] == "cold_zone"
            assert row["n1"] == 4
        density_row = next(r for r in paired if r["metric"] == "content_word_density")
        hot = [0.3 + i * 0.02 for i in range(4)]
        cold = [0.6 + i * 0.02 for i in range(4)]
        expected_w, expected_p = stats.wilcoxon(hot, cold)
        assert density_row["statistic"] == pytest.approx(expected_w, abs=1e-4)
        assert density_row["p_value"] == pytest.approx(expected_p, abs=1e-6)
        assert density_row["effect_size"] == pytest.approx(-1.0, abs=1e-4)

    def test_zone_paired_requires_min_pairs(self, validator, tmp_path):
        rows = []
        for i in range(2):
            rows.append(self._make_row(f"BV1z{i}", "游戏", 0.3, zone_type="hot_zone"))
            rows.append(self._make_row(f"BV1z{i}", "游戏", 0.6, zone_type="cold_zone"))
        csv_path = write_videos_csv(tmp_path, rows)
        assert validator.zone_paired_compare(csv_path) == []

    def test_corpus_compare_includes_paired_for_single_partition(self, validator, tmp_path):
        rows = []
        for i in range(3):
            rows.append(self._make_row(f"BV1z{i}", "游戏", 0.3 + i * 0.02, zone_type="hot_zone"))
            rows.append(self._make_row(f"BV1z{i}", "游戏", 0.6 + i * 0.02, zone_type="cold_zone"))
        csv_path = write_videos_csv(tmp_path, rows)
        df = validator.corpus_compare(csv_path).to_dataframe()
        assert (df["test_type"] == "Wilcoxon 符号秩（配对）").any()
        assert not (df["test_type"] == "note").any()

    def test_zone_paired_dedups_duplicate_bvid(self, validator, tmp_path):
        # 同 bvid 同区重复观测（不同输入解析到同一视频的残留场景）不得抬升配对数或错位
        rows = []
        for i in range(3):
            rows.append(self._make_row(f"BV1z{i}", "游戏", 0.3, zone_type="hot_zone"))
            rows.append(self._make_row(f"BV1z{i}", "游戏", 0.6, zone_type="cold_zone"))
        rows.append(self._make_row("BV1z0", "游戏", 0.9, zone_type="hot_zone"))
        rows.append(self._make_row("BV1z0", "游戏", 0.1, zone_type="cold_zone"))
        csv_path = write_videos_csv(tmp_path, rows)
        paired = validator.zone_paired_compare(csv_path)
        assert paired and all(r["n1"] == 3 for r in paired)


class TestCohenKappa:

    def test_perfect_agreement(self):
        assert cohen_kappa(["a", "b", "a"], ["a", "b", "a"]) == pytest.approx(1.0)

    def test_hand_computed_value(self):
        # po=0.75, pe=0.5 → kappa=0.5
        labels_a = ["pos", "pos", "neg", "neg"]
        labels_b = ["pos", "neg", "neg", "neg"]
        assert cohen_kappa(labels_a, labels_b) == pytest.approx(0.5)

    def test_chance_agreement_near_zero(self):
        # 边际分布对称且按期望比例一致时 kappa 趋近 0
        labels_a = ["a", "a", "b", "b"]
        labels_b = ["a", "b", "a", "b"]
        assert cohen_kappa(labels_a, labels_b) == pytest.approx(0.0)

    def test_empty_or_mismatched_returns_none(self):
        assert cohen_kappa([], []) is None
        assert cohen_kappa(["a"], ["a", "b"]) is None

    def test_degenerate_both_constant_same(self):
        assert cohen_kappa(["a", "a"], ["a", "a"]) == pytest.approx(1.0)

    def test_supports_boolean_labels(self):
        assert cohen_kappa([True, False], [True, False]) == pytest.approx(1.0)
