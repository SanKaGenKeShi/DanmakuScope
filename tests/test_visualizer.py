"""
R 可视化脚本模板单元测试 - 模板渲染完整性 + statistical_tests.csv 列名契约 + 文件写出
"""

import os

import pandas as pd
import pytest

from danmaku_analyzer.corpus_builder import SCALAR_FIELDS
from danmaku_analyzer.corpus_visualizer import (
    STATS_CSV_FILENAME,
    CorpusVisualizer,
    R_SCRIPT_FILENAME,
)
from danmaku_analyzer.statistical_validator import STATISTICAL_TESTS_COLUMNS


@pytest.fixture
def visualizer():
    return CorpusVisualizer()


class TestRenderRScript:

    def test_placeholders_resolved(self, visualizer):
        script = visualizer.render_r_script()
        assert "{scalars}" not in script
        assert "{partitions}" not in script
        assert "{csv_filename}" not in script
        assert "{stats_filename}" not in script
        assert "{{" not in script

    def test_default_filenames_embedded(self, visualizer):
        script = visualizer.render_r_script()
        assert '"corpus_videos.csv"' in script
        assert f'"{STATS_CSV_FILENAME}"' in script

    def test_custom_csv_filename_embedded(self, visualizer):
        script = visualizer.render_r_script(csv_filename="my_obs.csv")
        assert '"my_obs.csv"' in script

    def test_all_scalar_fields_embedded(self, visualizer):
        script = visualizer.render_r_script()
        for name in SCALAR_FIELDS:
            assert f'"{name}"' in script

    def test_partitions_injected(self, visualizer):
        script = visualizer.render_r_script(partitions=["音乐", "游戏"])
        assert 'partitions <- c("音乐", "游戏")' in script

    def test_empty_partitions_render_empty_vector(self, visualizer):
        script = visualizer.render_r_script()
        assert "partitions <- c()" in script

    def test_reads_precomputed_statistics_only_no_recompute(self, visualizer):
        script = visualizer.render_r_script()
        assert f'read.csv(stats_path' in script
        assert 'test_type == "Kruskal-Wallis"' in script
        # 检验由 Python 侧预计算，R 脚本不得重复计算
        assert "kruskal.test(" not in script
        assert "pairwise.wilcox.test" not in script

    def test_no_multiple_comparison_correction(self, visualizer):
        script = visualizer.render_r_script()
        assert "p.adjust" not in script
        assert '"BH"' not in script
        assert "p_adjusted" not in script
        assert "未校正" in script

    def test_statistical_tests_csv_column_contract(self, visualizer):
        """R 模板硬编码消费 statistical_tests.csv 列名，Python 侧 schema 变更必须双侧同步"""
        script = visualizer.render_r_script()
        for column in ("metric", "test_type", "p_value"):
            assert column in STATISTICAL_TESTS_COLUMNS
            assert column in script

    def test_key_r_constructs_present(self, visualizer):
        script = visualizer.render_r_script()
        assert "geom_boxplot" in script
        assert "geom_jitter" in script
        assert "facet_wrap" in script
        assert "ggsave" in script
        assert 'fileEncoding = "UTF-8-BOM"' in script

    def test_r_braces_balanced(self, visualizer):
        script = visualizer.render_r_script()
        assert script.count("{") == script.count("}")


class TestWriteRScript:

    def test_writes_file_and_returns_path(self, visualizer, tmp_path):
        out_dir = str(tmp_path / "reports")
        path = visualizer.write_r_script(out_dir)
        assert path == os.path.join(out_dir, R_SCRIPT_FILENAME)
        assert os.path.exists(path)

    def test_creates_missing_directory(self, visualizer, tmp_path):
        out_dir = str(tmp_path / "deep" / "nested" / "dir")
        path = visualizer.write_r_script(out_dir)
        assert os.path.exists(path)

    def test_partitions_read_from_videos_csv(self, visualizer, tmp_path):
        out_dir = str(tmp_path)
        pd.DataFrame([
            {"bvid": "BV1a", "tname": "游戏", "content_word_density": 0.5},
            {"bvid": "BV1b", "tname": "音乐", "content_word_density": 0.6},
            {"bvid": "BV1c", "tname": "游戏", "content_word_density": 0.7},
        ]).to_csv(os.path.join(out_dir, "corpus_videos.csv"), index=False, encoding='utf-8-sig')

        path = visualizer.write_r_script(out_dir)
        with open(path, encoding="utf-8") as f:
            content = f.read()
        # 分区注入按 Unicode 排序去重：游戏 < 音乐
        assert 'partitions <- c("游戏", "音乐")' in content

    def test_missing_videos_csv_yields_no_partitions(self, visualizer, tmp_path):
        path = visualizer.write_r_script(str(tmp_path))
        with open(path, encoding="utf-8") as f:
            content = f.read()
        assert "partitions <- c()" in content

    def test_utf8_lf_line_endings(self, visualizer, tmp_path):
        path = visualizer.write_r_script(str(tmp_path))
        with open(path, "rb") as f:
            raw = f.read()
        assert b"\r\n" not in raw
