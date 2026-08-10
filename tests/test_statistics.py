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

    def test_single_valid_group_empty_result(self, validator, tmp_path):
        rows = [make_video(f"BV1g{i}", "游戏", 0.3 + i * 0.05) for i in range(3)]
        rows += [make_video("BV1m0", "音乐", 0.6)]
        csv_path = write_videos_csv(tmp_path, rows)
        result = validator.corpus_compare(csv_path)
        assert result.rows == []
        assert result.to_dataframe().empty
        assert len(result.to_dataframe().columns) == len(STATISTICAL_TESTS_COLUMNS)

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
