"""
R 可视化脚本模板单元测试 - 模板渲染完整性 + 文件写出
"""

import os

import pytest

from danmaku_analyzer.corpus_builder import SCALAR_FIELDS
from danmaku_analyzer.corpus_visualizer import CorpusVisualizer, R_SCRIPT_FILENAME


@pytest.fixture
def visualizer():
    return CorpusVisualizer()


class TestRenderRScript:

    def test_placeholders_resolved(self, visualizer):
        script = visualizer.render_r_script()
        assert "{scalars}" not in script
        assert "{csv_filename}" not in script
        assert "{{" not in script

    def test_default_csv_filename_embedded(self, visualizer):
        script = visualizer.render_r_script()
        assert '"corpus_videos.csv"' in script

    def test_custom_csv_filename_embedded(self, visualizer):
        script = visualizer.render_r_script(csv_filename="my_obs.csv")
        assert '"my_obs.csv"' in script

    def test_all_scalar_fields_embedded(self, visualizer):
        script = visualizer.render_r_script()
        for name in SCALAR_FIELDS:
            assert f'"{name}"' in script

    def test_key_r_constructs_present(self, visualizer):
        script = visualizer.render_r_script()
        assert "kruskal.test" in script
        assert "pairwise.wilcox.test" in script
        assert 'p.adjust.method = "BH"' in script
        assert "geom_boxplot" in script
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

    def test_written_content_matches_render(self, visualizer, tmp_path):
        path = visualizer.write_r_script(str(tmp_path))
        with open(path, encoding="utf-8") as f:
            content = f.read()
        assert content == visualizer.render_r_script()

    def test_utf8_lf_line_endings(self, visualizer, tmp_path):
        path = visualizer.write_r_script(str(tmp_path))
        with open(path, "rb") as f:
            raw = f.read()
        assert b"\r\n" not in raw
