"""HTML 可视化报告单测：渲染内容/离线自包含/markdown 转义/降级容错/语料库级报告（全部离线）"""

import os

import pandas as pd

from danmaku_analyzer.aggregator import AggregatedData
from danmaku_analyzer.html_report import HtmlReportGenerator, _markdown_to_html
from danmaku_analyzer.reporter import Reporter


def make_aggregated():
    return [AggregatedData(
        tname="游戏", zone_type="hot_zone", danmaku_count=100, segment_count=5,
        emotion_distribution={"positive": 0.6, "neutral": 0.3, "negative": 0.1},
        sentence_function_distribution={"assertion": 0.7, "exclamation": 0.3},
        interaction_type_distribution={"check_in": 0.4, "other": 0.6},
        orthography_status_distribution={"standard": 0.9, "community_variant": 0.1},
        high_consensus_rate=0.8, medium_consensus_rate=0.15, low_consensus_rate=0.05,
        consensus_ci={"lower": 0.7, "upper": 0.88, "status": "ok"},
    )]


def make_metadata():
    return {
        "bvid": "BV1test", "title": "测试视频", "tname": "游戏",
        "tags": ["弹幕"], "pubdate": "2025-01-01T00:00:00",
        "view_count": 10000, "danmaku_count": 100, "danmaku_source": "protobuf",
        "pipeline_version": "0.3.7-beta", "prompt_version": "v2.3.0",
        "generated_at": "2026-08-19T12:00:00",
    }


class TestHtmlRender:

    def test_render_contains_core_sections(self):
        html_text = HtmlReportGenerator(".").render(make_aggregated(), make_metadata())
        for expected in ("测试视频", "BV1test", "游戏", "描写层概览", "语用层分布",
                         "双路共识统计", "正面", "陈述", "打卡报到", "规范书写"):
            assert expected in html_text

    def test_self_contained_no_external_resources(self):
        html_text = HtmlReportGenerator(".").render(make_aggregated(), make_metadata())
        assert "<style>" in html_text
        assert "<svg" in html_text
        # 无外链资源（SVG xmlns 为命名空间标识符非网络请求），无脚本
        stripped = html_text.replace('xmlns="http://www.w3.org/2000/svg"', "")
        assert "http://" not in stripped and "https://" not in stripped
        assert "<script" not in html_text.lower()

    def test_llm_report_embedded(self):
        md = "# 摘要\n\n**重点**发现：情感偏正面。\n\n- 观察一\n- 观察二"
        html_text = HtmlReportGenerator(".").render(make_aggregated(), make_metadata(), md)
        assert "社会语言学分析报告" in html_text
        assert "<strong>重点</strong>" in html_text
        assert "<li>观察一</li>" in html_text

    def test_markdown_injection_escaped(self):
        html_text = _markdown_to_html("<script>alert(1)</script>")
        assert "<script>" not in html_text
        assert "&lt;script&gt;" in html_text

    def test_insufficient_sample_ci_rendered(self):
        item = make_aggregated()[0]
        item.consensus_ci = {"status": "insufficient_sample"}
        html_text = HtmlReportGenerator(".").render([item], make_metadata())
        assert "样本不足" in html_text

    def test_xml_source_caveat(self):
        metadata = make_metadata()
        metadata["danmaku_source"] = "xml"
        html_text = HtmlReportGenerator(".").render(make_aggregated(), metadata)
        assert "XML 兜底" in html_text

    def test_write_creates_file(self, tmp_path):
        path = HtmlReportGenerator(str(tmp_path)).write(make_aggregated(), make_metadata())
        assert os.path.exists(path) and os.path.basename(path) == "report.html"


class TestReporterHtmlDelegation:

    def test_generate_html_report_with_zh_twin(self, tmp_path):
        reporter = Reporter(output_dir=str(tmp_path))
        path = reporter.generate_html_report(make_aggregated(), make_metadata())
        assert os.path.basename(path) == "report.html"
        assert os.path.exists(os.path.join(str(tmp_path), "分析报告.html"))
        assert "report_zh" in reporter.zh_reports

    def test_metadata_enriched_with_prompt_version(self, tmp_path):
        reporter = Reporter(output_dir=str(tmp_path))
        reporter.generate_html_report(make_aggregated(), {"bvid": "BV1x"})
        content = open(os.path.join(str(tmp_path), "report.html"), encoding="utf-8").read()
        assert "Prompt v" in content


class TestCorpusHtml:

    def test_render_corpus_merge_mode_with_tests(self):
        summary = pd.DataFrame([{"tname": "游戏", "video_count": 3, "content_word_density_mean": 0.5}])
        tests_df = pd.DataFrame([{
            "metric": "content_word_density", "test_type": "Wilcoxon 符号秩（配对）",
            "group1": "hot_zone", "group2": "cold_zone", "n1": 3, "n2": 3,
            "statistic": 0.0, "p_value": 0.05, "effect_size": -1.0,
            "effect_magnitude": "large", "note": "未校正",
        }])
        html_text = HtmlReportGenerator(".").render_corpus(
            summary, tests_df, {"tnames": ["游戏"], "video_count": 3, "total_danmaku": 300})
        assert "合并分析" in html_text
        assert "Wilcoxon 符号秩（配对）" in html_text
        assert "<table" in html_text

    def test_render_corpus_compare_mode_without_tests(self):
        summary = pd.DataFrame([{"tname": "游戏"}, {"tname": "音乐"}])
        html_text = HtmlReportGenerator(".").render_corpus(
            summary, None, {"tnames": ["游戏", "音乐"], "video_count": 6, "total_danmaku": 600})
        assert "比对分析" in html_text
        assert "未执行推断检验" in html_text

    def test_write_corpus_creates_file(self, tmp_path):
        path = HtmlReportGenerator(str(tmp_path)).write_corpus(
            pd.DataFrame([{"tname": "游戏"}]), None,
            {"tnames": ["游戏"], "video_count": 1, "total_danmaku": 10})
        assert os.path.basename(path) == "corpus_report.html"
