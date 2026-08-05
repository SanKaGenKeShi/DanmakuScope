"""
语料库级比较统计单元测试 - Kruskal-Wallis / Dunn / BH-FDR / Cramér's V / Cliff's delta
手写实现以 scipy 为参照交叉验证（scipy 为既有依赖，仅测试时使用）
"""

import math

import pytest
from scipy import stats

from danmaku_analyzer.statistical_validator import (
    StatisticalValidator,
    _chi2_sf,
    _normal_cdf,
    _rankdata,
)


@pytest.fixture
def validator():
    return StatisticalValidator()


# ========== 手写分布函数 ==========

class TestHandwrittenDistributions:

    @pytest.mark.parametrize("x,df", [(1.0, 1), (3.84, 1), (5.99, 2), (7.81, 3), (0.5, 4), (20.0, 5)])
    def test_chi2_sf_matches_scipy(self, x, df):
        assert _chi2_sf(x, df) == pytest.approx(stats.chi2.sf(x, df), abs=1e-8)

    @pytest.mark.parametrize("z", [-3.0, -1.96, 0.0, 0.5, 1.96, 3.5])
    def test_normal_cdf_matches_scipy(self, z):
        assert _normal_cdf(z) == pytest.approx(stats.norm.cdf(z), abs=1e-10)

    def test_rankdata_with_ties(self):
        ranks = _rankdata([3.0, 1.0, 2.0, 2.0, 5.0])
        assert ranks == [4.0, 1.0, 2.5, 2.5, 5.0]


# ========== Kruskal-Wallis ==========

class TestKruskalWallis:

    def test_matches_scipy_with_ties(self, validator):
        groups = {
            "游戏": [0.31, 0.42, 0.38, 0.45, 0.40],
            "音乐": [0.55, 0.61, 0.58, 0.63, 0.52],
            "科技": [0.40, 0.48, 0.51, 0.44, 0.47],
        }
        result = validator.kruskal_wallis(groups)
        expected = stats.kruskal(groups["游戏"], groups["音乐"], groups["科技"])
        assert result.statistic == pytest.approx(expected.statistic, rel=1e-6)
        assert result.p_value == pytest.approx(expected.pvalue, rel=1e-6)
        assert result.df == 2
        assert result.n_per_group == {"游戏": 5, "音乐": 5, "科技": 5}

    def test_two_groups_matches_scipy(self, validator):
        groups = {"A": [1.0, 2.0, 3.0, 4.0], "B": [5.0, 6.0, 7.0, 8.0]}
        result = validator.kruskal_wallis(groups)
        expected = stats.kruskal(groups["A"], groups["B"])
        assert result.statistic == pytest.approx(expected.statistic, rel=1e-6)
        assert result.p_value == pytest.approx(expected.pvalue, rel=1e-6)
        assert result.is_significant

    def test_single_group_returns_none(self, validator):
        assert validator.kruskal_wallis({"A": [1.0, 2.0, 3.0]}) is None

    def test_group_too_small_returns_none(self, validator):
        assert validator.kruskal_wallis({"A": [1.0], "B": [2.0, 3.0, 4.0]}) is None

    def test_empty_groups_filtered(self, validator):
        result = validator.kruskal_wallis({"A": [], "B": [1.0, 2.0], "C": [3.0, 4.0]})
        assert result is not None
        assert result.n_per_group == {"B": 2, "C": 2}


# ========== Dunn 事后检验 ==========

class TestDunnPosthoc:

    def test_three_groups_detects_true_difference(self, validator):
        # n=12/组时相邻组 z≈2.8，BH 校正后仍显著
        groups = {
            "低": [0.10, 0.12, 0.11, 0.13, 0.09, 0.12, 0.10, 0.11, 0.12, 0.13, 0.10, 0.09],
            "中": [0.30, 0.32, 0.28, 0.31, 0.29, 0.33, 0.30, 0.31, 0.29, 0.32, 0.30, 0.28],
            "高": [0.55, 0.58, 0.60, 0.57, 0.56, 0.59, 0.55, 0.57, 0.58, 0.60, 0.56, 0.59],
        }
        results = validator.dunn_posthoc(groups)
        assert len(results) == 3
        # 三组完全分离，所有两两比较均应显著
        assert all(r.is_significant for r in results)
        # 组序与 Cliff's delta 符号一致："低" vs "高" 应为负
        low_high = [r for r in results if r.group_a == "低" and r.group_b == "高"][0]
        assert low_high.cliffs_delta < 0
        assert low_high.effect_magnitude == "large"

    def test_no_difference_not_significant(self, validator):
        groups = {"A": [0.48, 0.5, 0.52, 0.5], "B": [0.49, 0.51, 0.5, 0.5]}
        results = validator.dunn_posthoc(groups)
        assert len(results) == 1
        assert not results[0].is_significant
        assert results[0].effect_magnitude == "negligible"

    def test_z_matches_manual_formula(self, validator):
        groups = {"A": [1.0, 2.0, 3.0], "B": [4.0, 5.0, 6.0]}
        results = validator.dunn_posthoc(groups)
        # 无结：sigma2 = N(N+1)/12 = 3.5；z = (2-5)/sqrt(3.5*(2/3))
        expected_z = (2.0 - 5.0) / math.sqrt(3.5 * (1 / 3 + 1 / 3))
        assert results[0].z_statistic == pytest.approx(expected_z, rel=1e-9)

    def test_single_group_returns_empty(self, validator):
        assert validator.dunn_posthoc({"A": [1.0, 2.0]}) == []


# ========== BH-FDR ==========

class TestBenjaminiHochberg:

    def test_known_adjustment(self):
        adjusted = StatisticalValidator.benjamini_hochberg([0.01, 0.02, 0.03, 0.5])
        # 升序秩 1..4：p*m/rank = [0.04, 0.04, 0.04, 0.5]，阶梯取累计最小
        assert adjusted == pytest.approx([0.04, 0.04, 0.04, 0.5])

    def test_preserves_input_order(self):
        adjusted = StatisticalValidator.benjamini_hochberg([0.5, 0.01, 0.03])
        # 升序为 [0.01(秩1), 0.03(秩2), 0.5(秩3)] → [0.03, 0.045, 0.5]，按原序还原
        assert adjusted == pytest.approx([0.5, 0.03, 0.045])

    def test_capped_at_one(self):
        adjusted = StatisticalValidator.benjamini_hochberg([0.9, 0.95])
        # 与 statsmodels multipletests(method='fdr_bh') 一致：阶梯最小值不回增
        assert adjusted == pytest.approx([0.95, 0.95])

    def test_empty(self):
        assert StatisticalValidator.benjamini_hochberg([]) == []


# ========== Cramér's V ==========

class TestCramersV:

    def test_perfect_association_v_is_one(self, validator):
        v, p, df = validator.cramers_v([[10, 0], [0, 10]])
        assert v == pytest.approx(1.0)
        assert df == 1
        assert p < 0.001

    def test_independence_v_near_zero(self, validator):
        v, p, df = validator.cramers_v([[25, 25], [25, 25]])
        assert v == pytest.approx(0.0)
        assert p == pytest.approx(1.0)

    def test_matches_scipy_chi2(self, validator):
        table = [[12, 8, 5], [7, 15, 10], [3, 6, 20]]
        v, p, df = validator.cramers_v(table)
        expected = stats.chi2_contingency(table, correction=False)
        assert df == expected.dof
        assert p == pytest.approx(expected.pvalue, rel=1e-6)
        n_total = sum(sum(row) for row in table)
        expected_v = math.sqrt(expected.statistic / (n_total * (min(3, 3) - 1)))
        assert v == pytest.approx(expected_v, rel=1e-9)

    def test_zero_row_filtered(self, validator):
        v, p, df = validator.cramers_v([[0, 0], [10, 5], [3, 8]])
        assert df == 1

    def test_invalid_table_raises(self, validator):
        with pytest.raises(ValueError):
            validator.cramers_v([[10, 5]])
        with pytest.raises(ValueError):
            validator.cramers_v([[1, 2], [3]])


# ========== Cliff's delta ==========

class TestCliffsDelta:

    def test_complete_separation(self, validator):
        delta, magnitude = StatisticalValidator.cliffs_delta([1, 2, 3], [4, 5, 6])
        assert delta == -1.0
        assert magnitude == "large"

    def test_identical_distributions(self, validator):
        delta, magnitude = StatisticalValidator.cliffs_delta([1, 2, 3], [1, 2, 3])
        assert delta == 0.0
        assert magnitude == "negligible"

    def test_partial_overlap_magnitude_boundaries(self):
        # 16 对比较：more=10, less=4 → delta=0.375 → medium
        delta, magnitude = StatisticalValidator.cliffs_delta([3, 3, 4, 4], [1, 2, 3, 5])
        assert delta == pytest.approx(0.375)
        assert magnitude == "medium"

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            StatisticalValidator.cliffs_delta([], [1.0])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
