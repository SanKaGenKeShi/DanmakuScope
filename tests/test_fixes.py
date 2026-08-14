"""
v0.2.1-beta 修复回归测试：
extra_body 显式传递 / LLM 分词 POS 归一化 / 报告生成重试 /
共识 CI 落表 / 包级懒加载
"""

import asyncio
import inspect
import json
import subprocess
import sys
import types

import pandas as pd
import pytest

import danmaku_analyzer
from danmaku_analyzer.hard_metrics import HardMetricsAnalyzer, _normalize_pos
from danmaku_analyzer.llm_client import LLMClient
from danmaku_analyzer.report_generator import AnalysisReportGenerator
from danmaku_analyzer.reporter import Reporter
from danmaku_analyzer.aggregator import Aggregator
from danmaku_analyzer import report_generator as report_gen_module

from test_core_modules import make_danmaku_record


# ========== 假 LLM 后端（实现 LLMBackend 协议的 complete 接口） ==========

class _FakeBackend:
    def __init__(self, payloads):
        self.payloads = list(payloads)
        self.call_count = 0
        self.captured_kwargs = None

    async def complete(self, **kwargs):
        self.captured_kwargs = kwargs
        self.call_count += 1
        payload = self.payloads.pop(0)
        if isinstance(payload, Exception):
            raise payload
        return payload


_FakeAsyncClient = _FakeBackend  # 旧名别名，注入点无需改动


# ========== 问题2：LLM 分词 POS 归一化 ==========

class TestPosNormalization:

    def test_english_aliases_mapped(self):
        assert _normalize_pos("noun") == "n"
        assert _normalize_pos("Verb") == "v"
        assert _normalize_pos("ADJ") == "a"
        assert _normalize_pos("pronoun") == "r"
        assert _normalize_pos("proper_noun") == "nz"

    def test_jieba_style_passthrough(self):
        assert _normalize_pos("n") == "n"
        assert _normalize_pos("nz") == "nz"
        assert _normalize_pos("vg") == "vg"

    def test_unknown_tag_kept(self):
        assert _normalize_pos("mystery") == "mystery"

    def test_llm_tokenize_normalizes_and_sends_extra_body(self):
        analyzer = HardMetricsAnalyzer()
        analyzer.llm_client = _FakeAsyncClient([json.dumps([["你好", "noun"], ["跑", "verb"]])])
        analyzer.llm_model = "fake-model"
        analyzer.enable_thinking = False
        analyzer.llm_semaphore = asyncio.Semaphore(4)
        result = asyncio.run(analyzer._llm_tokenize("你好跑"))
        assert result == [("你好", "n"), ("跑", "v")]
        kwargs = analyzer.llm_client.captured_kwargs
        assert kwargs["extra_body"] == {
            "enable_thinking": False,
            "chat_template_kwargs": {"enable_thinking": False},
        }


# ========== 问题1：extra_body 始终显式传递 ==========

class TestExtraBodyAlwaysSent:

    @pytest.mark.parametrize("flag", [True, False])
    def test_llm_client_call_llm(self, flag):
        client = LLMClient()
        client.enable_thinking = flag
        fake = _FakeAsyncClient(['{"emotion": {"label": "positive", "confidence": 0.9}}'])
        result = asyncio.run(client._call_llm(fake, "m", "sys", "user", 0.1))
        assert result["emotion"]["label"] == "positive"
        assert fake.captured_kwargs["extra_body"] == {
            "enable_thinking": flag,
            "chat_template_kwargs": {"enable_thinking": flag},
        }

    @pytest.mark.parametrize("flag", [True, False])
    def test_report_generator_call_llm(self, flag):
        gen = AnalysisReportGenerator()
        gen.enable_thinking = flag
        fake = _FakeAsyncClient(["报告正文"])
        gen.client = fake
        content = asyncio.run(gen._call_llm("sys", "user"))
        assert content == "报告正文"
        assert fake.captured_kwargs["extra_body"] == {
            "enable_thinking": flag,
            "chat_template_kwargs": {"enable_thinking": flag},
        }


# ========== 请求超时接配置 ==========

class TestLLMTimeoutConfig:

    def test_llm_client_timeouts_follow_config(self, monkeypatch):
        from danmaku_analyzer.llm_config import get_llm_settings
        cfg = get_llm_settings()
        monkeypatch.setattr(cfg, "COMPLEX_LLM_TIMEOUT", 77.0)
        monkeypatch.setattr(cfg, "SIMPLE_LLM_TIMEOUT", 55.0)
        client = LLMClient()
        assert float(client.complex_client.timeout) == pytest.approx(77.0)
        assert float(client.simple_client.timeout) == pytest.approx(55.0)

    def test_report_generator_timeout_follows_config(self, monkeypatch):
        from danmaku_analyzer.llm_config import get_llm_settings
        cfg = get_llm_settings()
        monkeypatch.setattr(cfg, "ANALYSIS_REPORT_LLM_TIMEOUT", 99.0)
        gen = AnalysisReportGenerator()
        assert float(gen.client.timeout) == pytest.approx(99.0)


# ========== LLM 配置写回 .env（单一数据源） ==========

class TestWriteLlmEnv:

    def test_inplace_replace_preserves_comments_and_others(self, tmp_path):
        from danmaku_analyzer.prefs import write_llm_env
        env = tmp_path / ".env"
        env.write_text(
            "# 注释保留\nCOMPLEX_LLM_MODEL=old-model\nOTHER_KEY=keep\n",
            encoding="utf-8",
        )
        write_llm_env({"COMPLEX_LLM_MODEL": "new-model"}, str(env))
        lines = env.read_text(encoding="utf-8").splitlines()
        assert lines == ["# 注释保留", "COMPLEX_LLM_MODEL=new-model", "OTHER_KEY=keep"]

    def test_missing_keys_appended(self, tmp_path):
        from danmaku_analyzer.prefs import write_llm_env
        env = tmp_path / ".env"
        env.write_text("COMPLEX_LLM_MODEL=m\n", encoding="utf-8")
        write_llm_env({"COMPLEX_LLM_TIMEOUT": 120.0, "SIMPLE_LLM_TIMEOUT": 90.0}, str(env))
        text = env.read_text(encoding="utf-8")
        assert "COMPLEX_LLM_TIMEOUT=120.0" in text
        assert "SIMPLE_LLM_TIMEOUT=90.0" in text


# ========== 问题3+4：死参数移除 + tenacity 重试 ==========

class TestReportGeneratorRetry:

    @pytest.fixture(autouse=True)
    def _fast_retry(self):
        # 指数退避等待时间压到 0，避免拖慢测试；用例结束后还原，避免污染同 session 其他测试
        from tenacity import wait_none
        retry_state = AnalysisReportGenerator._call_llm.retry
        original_wait = retry_state.wait
        retry_state.wait = wait_none()
        yield
        retry_state.wait = original_wait

    def test_generate_signature_has_no_prompt_version(self):
        assert "prompt_version" not in inspect.signature(AnalysisReportGenerator.generate).parameters

    def test_transient_failure_then_success(self):
        gen = AnalysisReportGenerator()
        gen.client = _FakeAsyncClient([RuntimeError("瞬时故障"), RuntimeError("瞬时故障"), "报告正文"])
        content = asyncio.run(gen.generate([{"tname": "游戏"}], {"tname": "游戏"}))
        assert content == "报告正文"
        assert gen.client.call_count == 3

    def test_empty_content_triggers_retry(self):
        gen = AnalysisReportGenerator()
        gen.client = _FakeAsyncClient(["", "报告正文"])
        content = asyncio.run(gen.generate([], {"tname": "游戏"}))
        assert content == "报告正文"
        assert gen.client.call_count == 2

    def test_all_retries_exhausted_returns_none(self):
        gen = AnalysisReportGenerator()
        gen.client = _FakeAsyncClient([RuntimeError("x")] * 3)
        assert asyncio.run(gen.generate([], {"tname": "游戏"})) is None
        assert gen.client.call_count == 3


# ========== 打开系统终端的平台分支 ==========

class TestOpenSystemTerminal:

    def _launch(self, platform, monkeypatch):
        from danmaku_analyzer.tui.app import DanmakuTUI
        argvs = []
        monkeypatch.setattr(subprocess, "Popen", lambda argv, **kwargs: argvs.append(argv))
        monkeypatch.setattr(sys, "platform", platform)
        stub = types.SimpleNamespace(
            notify=lambda *a, **k: None,
            _cli_script_path=DanmakuTUI._cli_script_path,
        )
        DanmakuTUI._open_system_terminal(stub)
        return argvs

    def test_macos_uses_osascript_do_script(self, monkeypatch):
        """`open -a Terminal` 会把参数当文件打开，执行命令须经 AppleScript do script"""
        argvs = self._launch("darwin", monkeypatch)
        assert argvs[0][:2] == ["osascript", "-e"]
        script = argvs[0][2]
        assert script.startswith('tell application "Terminal" to do script "')
        assert "danmaku-analyzer" in script and script.rstrip('"').endswith("login")

    def test_linux_uses_x_terminal_emulator(self, monkeypatch):
        argvs = self._launch("linux", monkeypatch)
        assert argvs == [["x-terminal-emulator", "-e", "danmaku-analyzer login"]]

    def test_windows_uses_cmd_start(self, monkeypatch):
        argvs = self._launch("win32", monkeypatch)
        assert argvs[0][:4] == ["cmd", "/c", "start", "cmd"]

    def test_cli_script_path_prefers_sibling_of_executable(self, monkeypatch, tmp_path):
        from danmaku_analyzer.tui.app import DanmakuTUI
        exe = tmp_path / "python"
        exe.write_text("")
        script = tmp_path / "danmaku-analyzer"
        script.write_text("")
        monkeypatch.setattr(sys, "executable", str(exe))
        monkeypatch.setattr(sys, "platform", "linux")
        assert DanmakuTUI._cli_script_path() == str(script)

    def test_cli_script_path_falls_back_to_command_name(self, monkeypatch, tmp_path):
        from danmaku_analyzer.tui.app import DanmakuTUI
        exe = tmp_path / "python"
        exe.write_text("")
        monkeypatch.setattr(sys, "executable", str(exe))
        monkeypatch.setattr(sys, "platform", "linux")
        assert DanmakuTUI._cli_script_path() == "danmaku-analyzer"


# ========== 问题5：consensus_ci 写入共识统计表 ==========

class TestConsensusTableCI:

    def _build_agg(self, ci):
        agg = Aggregator().aggregate([make_danmaku_record()])[0]
        agg.consensus_ci = ci
        return agg

    def _read_table(self, tmp_path, ci):
        reporter = Reporter(output_dir=str(tmp_path))
        path = reporter._generate_consensus_table([self._build_agg(ci)])
        return pd.read_csv(path, encoding="utf-8-sig")

    def test_ci_values_written(self, tmp_path):
        ci = {"lower": 0.5, "upper": 0.7, "point_estimate": 0.6,
              "confidence_level": 0.95, "method": "wilson"}
        df = self._read_table(tmp_path, ci)
        assert df.iloc[0]["high_consensus_ci_lower"] == pytest.approx(0.5)
        assert df.iloc[0]["high_consensus_ci_upper"] == pytest.approx(0.7)
        assert df.iloc[0]["high_consensus_ci_status"] == "ok"

    def test_insufficient_sample_status(self, tmp_path):
        ci = {"status": "insufficient_sample", "sample_size": 5, "min_required": 30}
        df = self._read_table(tmp_path, ci)
        assert df.iloc[0]["high_consensus_ci_status"] == "insufficient_sample"
        assert pd.isna(df.iloc[0]["high_consensus_ci_lower"])

    def test_missing_ci_defaults_ok(self, tmp_path):
        df = self._read_table(tmp_path, None)
        assert df.iloc[0]["high_consensus_ci_status"] == "ok"


# ========== 问题7：包级懒加载 ==========

class TestLazyPackageExports:

    def test_version_directly_available(self):
        assert danmaku_analyzer.__version__

    def test_lazy_attr_matches_submodule(self):
        from danmaku_analyzer.corpus_visualizer import CorpusVisualizer
        assert danmaku_analyzer.CorpusVisualizer is CorpusVisualizer

    def test_unknown_attr_raises(self):
        with pytest.raises(AttributeError):
            _ = danmaku_analyzer.NotExistSymbol

    def test_dir_contains_exports(self):
        names = dir(danmaku_analyzer)
        assert "CorpusStore" in names and "get_settings" in names


# ========== 问题1：cli 轻量命令不拉入 pipeline 重型依赖 ==========

class TestCliLazyPipeline:

    def test_cli_import_does_not_load_pipeline(self):
        # 子进程隔离验证：导入 cli 后 pipeline（及其拖入的 jieba/pandas/openai）未被加载
        code = (
            "import sys; from danmaku_analyzer import cli; "
            "assert 'danmaku_analyzer.pipeline' not in sys.modules; "
            "assert 'jieba' not in sys.modules; "
            "print('OK')"
        )
        result = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True
        )
        assert result.returncode == 0, result.stderr
        assert "OK" in result.stdout

    def test_cli_module_has_no_top_level_pipeline_symbols(self):
        from danmaku_analyzer import cli as cli_module
        assert not hasattr(cli_module, "analyze_video")
        assert not hasattr(cli_module, "AnalysisResult")


# ========== 问题3：fetch_all 透传 cid ==========

class TestFetchAllCidPassthrough:

    def test_fetch_all_passes_meta_cid(self):
        from danmaku_analyzer.crawler import BilibiliCrawler, VideoMeta
        from datetime import datetime

        crawler = BilibiliCrawler()
        captured = {}

        async def fake_metadata(bvid):
            return VideoMeta(
                bvid=bvid, title="t", tname="游戏",
                pubdate=datetime(2025, 1, 1), cid=12345,
            )

        async def fake_danmaku(bvid, cid=None):
            captured["cid"] = cid
            return [], "protobuf"

        crawler.fetch_video_metadata = fake_metadata
        crawler.fetch_danmaku = fake_danmaku
        asyncio.run(crawler.fetch_all("BV1xx"))
        assert captured["cid"] == 12345

    def test_fetch_all_cid_zero_falls_back_to_none(self):
        from danmaku_analyzer.crawler import BilibiliCrawler, VideoMeta
        from datetime import datetime

        crawler = BilibiliCrawler()
        captured = {}

        async def fake_metadata(bvid):
            return VideoMeta(
                bvid=bvid, title="t", tname="游戏",
                pubdate=datetime(2025, 1, 1),
            )

        async def fake_danmaku(bvid, cid=None):
            captured["cid"] = cid
            return [], "protobuf"

        crawler.fetch_video_metadata = fake_metadata
        crawler.fetch_danmaku = fake_danmaku
        asyncio.run(crawler.fetch_all("BV1xx"))
        assert captured["cid"] is None


# ========== 问题4：聚合组截断告警 ==========

class TestReportTruncationWarning:

    def _capture_warnings(self, monkeypatch):
        warnings = []
        monkeypatch.setattr(report_gen_module.logger, "warning",
                            lambda msg, *a, **k: warnings.append(msg))
        return warnings

    def test_more_than_3_groups_warns(self, monkeypatch):
        warnings = self._capture_warnings(monkeypatch)
        gen = AnalysisReportGenerator()
        groups = [{"tname": "游戏", "zone_type": "hot_zone"} for _ in range(5)]
        gen._build_user_prompt(groups, {"tname": "游戏"})
        assert len(warnings) == 1
        assert "前 3 组" in warnings[0] and "2 组" in warnings[0]

    def test_three_or_fewer_groups_no_warning(self, monkeypatch):
        warnings = self._capture_warnings(monkeypatch)
        gen = AnalysisReportGenerator()
        gen._build_user_prompt([{"tname": "游戏"}] * 3, {"tname": "游戏"})
        assert warnings == []


# ========== 问题5：interaction_type 模板补 confidence ==========

class TestPromptTemplateCompleteness:

    def test_interaction_type_template_has_confidence(self):
        from danmaku_analyzer.prompt_builder import PromptBuilder
        system_prompt = PromptBuilder().build_system_prompt("游戏", [])
        for line in system_prompt.splitlines():
            if '"interaction_type"' in line:
                assert '"confidence"' in line
                break
        else:
            pytest.fail("模板中未找到 interaction_type 行")

    def test_prompt_version_bumped(self):
        from danmaku_analyzer.llm_config import get_llm_settings
        assert get_llm_settings().PROMPT_VERSION != "v2.2.0"
