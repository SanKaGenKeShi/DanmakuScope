"""
产出标准化单元测试 - Exporter 多格式导出 / MethodologyGenerator / 双路 Kappa 质控指标
全部离线：构造假 CSV/ZIP 与内存记录验证
"""

import json
import os
import zipfile
from io import StringIO

import pandas as pd
import pytest

from danmaku_analyzer.exporter import Exporter
from danmaku_analyzer.methodology import METHODOLOGY_FILENAME, MethodologyGenerator
from danmaku_analyzer.llm_models import DualPathResult, ConsensusLevel, LLMOutput
from danmaku_analyzer.aggregator import DanmakuRecord
from danmaku_analyzer.pipeline import _build_quality_metrics
from danmaku_analyzer.reporter import Reporter


@pytest.fixture
def exporter():
    return Exporter()


def make_summary_rows():
    return [
        {"tname": "游戏", "video_count": 2, "total_danmaku": 200,
         "content_word_density_mean": 0.5, "content_word_density_std": 0.1,
         "avg_word_length_mean": 2.1, "avg_word_length_std": 0.2},
        {"tname": "音乐", "video_count": 3, "total_danmaku": 450,
         "content_word_density_mean": 0.6, "content_word_density_std": 0.05,
         "avg_word_length_mean": 1.9, "avg_word_length_std": 0.3},
    ]


def write_csv(path, rows):
    pd.DataFrame(rows).to_csv(path, index=False, encoding='utf-8-sig')
    return str(path)


class TestLatexExport:

    def test_booktabs_structure_and_escaping(self, exporter):
        df = pd.DataFrame([{"tname": "游戏_区", "rate": 0.25, "note": "a&b%c"}])
        latex = exporter.to_latex_table(df, caption="测试表", label="tab:test")
        assert "\\begin{tabular}" in latex
        assert "\\toprule" in latex and "\\bottomrule" in latex
        assert "\\caption{测试表}" in latex
        assert "\\label{tab:test}" in latex
        assert r"游戏\_区" in latex
        assert r"a\&b\%c" in latex
        assert "0.25" in latex

    def test_from_csv_latex(self, exporter, tmp_path):
        csv_path = write_csv(tmp_path / "summary.csv", make_summary_rows())
        content = exporter.from_csv(csv_path, "latex")
        assert content.count("\\begin{table}") == 1
        # 下划线在 LaTeX 中需转义
        assert "content\\_word\\_density\\_mean" in content

    def test_export_writes_tex_file(self, exporter, tmp_path):
        csv_path = write_csv(tmp_path / "summary.csv", make_summary_rows())
        out = exporter.export(csv_path, "latex", str(tmp_path / "out"))
        assert out.endswith("summary_latex.tex")
        assert os.path.exists(out)


class TestApaExport:

    def test_mean_std_pairs_reported(self, exporter):
        text = exporter.to_apa_text(pd.DataFrame(make_summary_rows()))
        assert "分区 游戏" in text
        assert "k = 2 个视频" in text
        assert "N = 200 条弹幕" in text
        assert "content_word_density M = 0.500, SD = 0.100" in text

    def test_stats_to_apa_kw_and_pairwise(self, exporter):
        stats_df = pd.DataFrame([
            {"metric": "content_word_density", "test_type": "Kruskal-Wallis",
             "group1": "", "group2": "", "n1": 5, "n2": "", "statistic": 6.25,
             "p_value": 0.012, "effect_size": "", "effect_magnitude": "", "note": "未校正"},
            {"metric": "content_word_density", "test_type": "Mann-Whitney U",
             "group1": "游戏", "group2": "音乐", "n1": 2, "n2": 3, "statistic": 0.0,
             "p_value": 0.008, "effect_size": -1.0, "effect_magnitude": "large", "note": "未校正"},
        ])
        text = exporter.stats_to_apa(stats_df)
        assert "Kruskal-Wallis H = 6.250" in text
        assert "p = .012" in text
        assert "游戏 vs 音乐" in text
        assert "Cliff's δ = -1.000（large）" in text
        assert "未校正" in text

    def test_stats_to_apa_insufficient_sample_row(self, exporter):
        stats_df = pd.DataFrame([
            {"metric": "", "test_type": "sample_status", "group1": "知识", "group2": "",
             "n1": 1, "n2": "", "statistic": "", "p_value": float("nan"),
             "effect_size": "", "effect_magnitude": "", "note": "insufficient_sample"},
        ])
        text = exporter.stats_to_apa(stats_df)
        assert "知识" in text
        assert "未纳入推断检验" in text

    def test_p_below_threshold_reported_as_lt(self, exporter):
        assert exporter._apa_p(0.0001) == "< .001"

    def test_export_writes_txt_file(self, exporter, tmp_path):
        csv_path = write_csv(tmp_path / "summary.csv", make_summary_rows())
        out = exporter.export(csv_path, "apa", str(tmp_path / "out"))
        assert out.endswith("summary_apa.txt")
        assert "M = 0.500" in open(out, encoding="utf-8").read()


class TestZipExport:

    def _make_corpus_zip(self, tmp_path, with_stats=True):
        zip_path = str(tmp_path / "[corpus]_2videos.zip")
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            buffer = StringIO()
            pd.DataFrame(make_summary_rows()).to_csv(buffer, index=False, encoding='utf-8-sig')
            zipf.writestr("corpus_summary.csv", buffer.getvalue().encode('utf-8-sig'))
            if with_stats:
                buffer = StringIO()
                pd.DataFrame([
                    {"metric": "content_word_density", "test_type": "Kruskal-Wallis",
                     "group1": "", "group2": "", "n1": 5, "n2": "", "statistic": 6.25,
                     "p_value": 0.012, "effect_size": "", "effect_magnitude": "", "note": "未校正"},
                ]).to_csv(buffer, index=False, encoding='utf-8-sig')
                zipf.writestr("statistical_tests.csv", buffer.getvalue().encode('utf-8-sig'))
        return zip_path

    def test_corpus_zip_latex(self, exporter, tmp_path):
        content = exporter.from_zip(self._make_corpus_zip(tmp_path), "latex")
        assert "\\begin{tabular}" in content

    def test_corpus_zip_apa_includes_statistics(self, exporter, tmp_path):
        content = exporter.from_zip(self._make_corpus_zip(tmp_path), "apa")
        assert "content_word_density M = 0.500" in content
        assert "Kruskal-Wallis" in content

    def test_single_video_zip_latex(self, exporter, tmp_path):
        zip_path = str(tmp_path / "[BV1x]test.zip")
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            zipf.writestr("metadata.json", json.dumps({"bvid": "BV1x"}))
            buffer = StringIO()
            pd.DataFrame([{"tname": "游戏", "high_consensus_rate": 0.8}]).to_csv(
                buffer, index=False, encoding='utf-8-sig')
            zipf.writestr("table_consensus_stats.csv", buffer.getvalue().encode('utf-8-sig'))
        content = exporter.from_zip(zip_path, "latex")
        assert content.count("\\begin{table}") == 1

    def test_zip_without_tables_raises(self, exporter, tmp_path):
        zip_path = str(tmp_path / "empty.zip")
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            zipf.writestr("metadata.json", "{}")
        with pytest.raises(ValueError):
            exporter.from_zip(zip_path, "latex")


class TestExportErrors:

    def test_unknown_format_raises(self, exporter, tmp_path):
        csv_path = write_csv(tmp_path / "x.csv", [{"a": 1}])
        with pytest.raises(ValueError):
            exporter.export(csv_path, "docx")

    def test_unsupported_extension_raises(self, exporter, tmp_path):
        path = tmp_path / "notes.txt"
        path.write_text("x", encoding="utf-8")
        with pytest.raises(ValueError):
            exporter.export(str(path), "latex")


class TestMethodology:

    @pytest.fixture
    def metadata(self):
        return {
            "bvid": "BV1test", "title": "测试视频", "tname": "游戏",
            "tags": ["弹幕", "测试"], "pubdate": "2025-01-01T00:00:00",
            "view_count": 10000, "danmaku_count": 1500,
            "danmaku_source": "protobuf", "pipeline_version": "0.3.4-beta",
            "batch_segment_analysis": False,
        }

    def test_render_contains_core_parameters(self, metadata, tmp_path):
        text = MethodologyGenerator(str(tmp_path)).render(metadata)
        for expected in ("BV1test", "测试视频", "游戏", "Wilson", "Kruskal-Wallis",
                         "0.3.4-beta", "零丢弃", "宁可误判为社区变体"):
            assert expected in text

    def test_xml_source_marks_truncation_caveat(self, metadata, tmp_path):
        metadata["danmaku_source"] = "xml"
        text = MethodologyGenerator(str(tmp_path)).render(metadata)
        assert "截断数据" in text

    def test_batch_mode_conditional_block(self, metadata, tmp_path):
        gen = MethodologyGenerator(str(tmp_path))
        assert "段内批量推理" not in gen.render(metadata)
        metadata["batch_segment_analysis"] = True
        assert "段内批量推理" in gen.render(metadata)

    def test_sampling_override(self, metadata, tmp_path):
        text = MethodologyGenerator(str(tmp_path)).render(
            metadata, sampling={"freq_based": True, "top_n": 15})
        assert "频次排序" in text
        assert "15" in text

    def test_write_creates_file(self, metadata, tmp_path):
        path = MethodologyGenerator(str(tmp_path)).write(metadata)
        assert path == os.path.join(str(tmp_path), METHODOLOGY_FILENAME)
        assert os.path.exists(path)


class TestReporterDelegation:

    def test_generate_methodology_writes_into_output_dir(self, tmp_path):
        reporter = Reporter(output_dir=str(tmp_path))
        path = reporter.generate_methodology({"bvid": "BV1d", "title": "t", "tname": "游戏"})
        assert os.path.dirname(path) == str(tmp_path)
        assert os.path.basename(path) == METHODOLOGY_FILENAME

    def test_export_formatted_delegates_to_exporter(self, tmp_path):
        csv_path = write_csv(tmp_path / "summary.csv", make_summary_rows())
        reporter = Reporter(output_dir=str(tmp_path / "out"))
        path = reporter.export_formatted(csv_path, "latex")
        assert path.endswith(".tex")


def make_record(raw_outputs):
    return DanmakuRecord(
        tname="游戏", zone_type="hot_zone", tags=[], hard_metrics=None,
        llm_result=DualPathResult(
            output=LLMOutput.default(), consensus_level=ConsensusLevel.HIGH,
            jsd_score=0.0, weight_multiplier=1.0,
            raw_outputs=raw_outputs, prompt_version="v-test",
        ),
        segment_id=0,
    )


def raw_output(emotion="positive", violated=False):
    return {
        "emotion": {"label": emotion, "confidence": 0.9},
        "cooperative_principle": {"violated": violated, "maxim": None},
        "interaction_type": {"label": "expression", "confidence": 0.9},
        "sentence_function": {"label": "assertion", "confidence": 0.9},
        "orthography": {"status": "standard", "confidence": 0.9},
    }


class TestQualityMetrics:

    def test_dual_path_kappa_computed(self):
        records = [
            make_record([raw_output(emotion="positive"), raw_output(emotion="positive")]),
            make_record([raw_output(emotion="positive"), raw_output(emotion="negative")]),
            make_record([raw_output(emotion="negative"), raw_output(emotion="negative")]),
            make_record([raw_output(emotion="negative"), raw_output(emotion="positive")]),
        ]
        metrics = _build_quality_metrics(records)
        assert metrics["dual_path_samples"] == 4
        # 四样本两两对半一致：po=0.5, pe=0.5 → kappa=0
        assert metrics["cohen_kappa"]["emotion.label"] == pytest.approx(0.0)
        # 其余维度全一致 → 1.0
        assert metrics["cohen_kappa"]["cooperative_principle.violated"] == pytest.approx(1.0)
        # 句类为单路任务不参与
        assert not any(k.startswith("sentence_function") for k in metrics["cohen_kappa"])

    def test_single_path_returns_empty(self):
        records = [make_record([raw_output()])]
        assert _build_quality_metrics(records) == {}

    def test_none_raw_outputs_skipped(self):
        records = [make_record([None, None]), make_record([raw_output(), raw_output()])]
        metrics = _build_quality_metrics(records)
        assert metrics["dual_path_samples"] == 1
