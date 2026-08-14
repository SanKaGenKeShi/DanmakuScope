"""编排层纯单测：ZIP 打包/文件名清洗/进度文件/缓存/偏好注入/简单路降级/温度校验/.env 优先级（全部离线）"""
import asyncio
import json
import os
import subprocess
import sys
import zipfile

import pytest

from danmaku_analyzer.cache_manager import CacheManager
from danmaku_analyzer.llm_client import LLMClient
from danmaku_analyzer.llm_models import LLMOutput, SentenceFunctionOutput
from danmaku_analyzer.pipeline import (
    _append_progress,
    _load_progress,
    _package_reports_zip,
    _relativize_zip_path,
    _resolve_progress_zip_path,
    _sanitize_zip_filename,
)
from danmaku_analyzer.prompt_builder import PromptComponents


@pytest.fixture
def isolated_data_root(tmp_path, monkeypatch):
    from danmaku_analyzer.config import get_settings
    monkeypatch.setattr(get_settings(), "DATA_ROOT", str(tmp_path))
    return tmp_path


class TestSanitizeZipFilename:

    def test_illegal_chars_replaced(self):
        assert _sanitize_zip_filename('a/b\\c:d*e?f"g<h>i|j') == "a_b_c_d_e_f_g_h_i_j"

    def test_trailing_dots_and_spaces_stripped(self):
        assert _sanitize_zip_filename("标题..  ") == "标题"

    def test_long_title_truncated_by_bytes(self):
        cleaned = _sanitize_zip_filename("弹" * 100)
        assert len(cleaned.encode("utf-8")) <= 150

    def test_blank_title_sanitizes_to_empty(self):
        assert _sanitize_zip_filename("   ") == ""
        assert _sanitize_zip_filename("") == ""

    def test_truncation_exposing_trailing_dot_stripped(self):
        # 截断点恰落在点号后：截断后需再次清洗尾随点/空格
        title = "a" * 149 + "." + "b" * 100
        cleaned = _sanitize_zip_filename(title)
        assert cleaned == "a" * 149
        title = "a" * 149 + " " + "b" * 100
        assert _sanitize_zip_filename(title) == "a" * 149


class TestPackageReportsZip:

    def test_valid_zip_removes_sources(self, tmp_path):
        src = tmp_path / "a.csv"
        src.write_text("x", encoding="utf-8")
        zip_path = str(tmp_path / "out.zip")
        assert _package_reports_zip({"a": str(src)}, zip_path, lambda s, m: None)
        assert zipfile.is_zipfile(zip_path)
        assert not src.exists()

    def test_all_reports_missing_returns_false(self, tmp_path):
        zip_path = str(tmp_path / "out.zip")
        reports = {"a": str(tmp_path / "missing1.csv"), "b": str(tmp_path / "missing2.csv")}
        assert not _package_reports_zip(reports, zip_path, lambda s, m: None)
        assert not os.path.exists(zip_path)

    def test_write_failure_returns_false_and_keeps_sources(self, tmp_path):
        src = tmp_path / "a.csv"
        src.write_text("x", encoding="utf-8")
        zip_path = str(tmp_path / "no_such_dir" / "out.zip")
        assert not _package_reports_zip({"a": str(src)}, zip_path, lambda s, m: None)
        assert src.exists()


class TestProgressFile:

    def test_roundtrip_stores_relative_path(self, isolated_data_root):
        zip_path = os.path.join(str(isolated_data_root), "reports", "[BV1xx]t.zip")
        _append_progress("BV1xx", "BV1xx", zip_path, reused=False)
        record = _load_progress()["BV1xx"]
        assert record["zip_path"] == os.path.join("reports", "[BV1xx]t.zip")
        assert _resolve_progress_zip_path(record["zip_path"]) == zip_path

    def test_legacy_absolute_path_resolves(self, isolated_data_root):
        abs_path = os.path.join(str(isolated_data_root), "legacy.zip")
        assert _resolve_progress_zip_path(abs_path) == abs_path

    def test_corrupt_line_skipped(self, isolated_data_root):
        path = isolated_data_root / "scheduler" / "progress.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "not-json\n" + json.dumps({"input": "BV1", "bvid": "BV1", "zip_path": "a.zip"}) + "\n",
            encoding="utf-8",
        )
        index = _load_progress()
        assert list(index) == ["BV1"]

    def test_rotation_keeps_latest_per_key(self, isolated_data_root, monkeypatch):
        import danmaku_analyzer.pipeline as pipeline_module
        monkeypatch.setattr(pipeline_module, "_PROGRESS_MAX_BYTES", 50)
        zip_path = str(isolated_data_root / "x.zip")
        for i in range(3):
            _append_progress("BV1", "BV1", zip_path, reused=(i == 2))
        index = _load_progress()
        assert len(index) == 1
        assert index["BV1"]["status"] == "reused"
        lines = (isolated_data_root / "scheduler" / "progress.jsonl").read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) < 3

    def test_cross_drive_falls_back_to_absolute(self):
        if os.name != "nt":
            pytest.skip("跨驱动器仅 Windows 存在")
        assert _relativize_zip_path("Z:\\fake\\x.zip") == "Z:\\fake\\x.zip"


class TestCacheManager:

    def test_set_get_roundtrip_no_tmp_left(self, tmp_path):
        cm = CacheManager(cache_dir=str(tmp_path))
        assert cm.set("k", {"v": 1})
        assert cm.get("k") == {"v": 1}
        assert not any(f.endswith(".tmp") for f in os.listdir(tmp_path))

    def test_corrupt_file_deleted_on_get(self, tmp_path):
        cm = CacheManager(cache_dir=str(tmp_path))
        path = cm._get_cache_path("bad")
        with open(path, "wb") as f:
            f.write(b"not-a-pickle")
        assert cm.get("bad") is None
        assert not os.path.exists(path)


class TestApplySavedPrefsValidation:

    def test_invalid_type_ignored(self, monkeypatch):
        from danmaku_analyzer import prefs
        from danmaku_analyzer.config import get_settings
        settings = get_settings()
        monkeypatch.setattr(prefs, "load_prefs", lambda: {"TOP_N": "not-an-int"})
        before = settings.TOP_N
        prefs.apply_saved_prefs()
        assert settings.TOP_N == before

    def test_valid_value_applied(self, monkeypatch):
        from danmaku_analyzer import prefs
        from danmaku_analyzer.config import get_settings
        settings = get_settings()
        monkeypatch.setattr(settings, "TOP_N", 10)
        monkeypatch.setattr(prefs, "load_prefs", lambda: {"TOP_N": 25})
        prefs.apply_saved_prefs()
        assert settings.TOP_N == 25


class _FakeClient:
    """假 LLM 后端：complete 返回 payload 的 JSON 字符串，error 非空则抛错"""

    def __init__(self, payload=None, error=None):
        self._payload = payload
        self._error = error

    async def complete(self, **kwargs):
        if self._error is not None:
            raise self._error
        return json.dumps(self._payload)


def _components():
    return PromptComponents(system_prompt="s", user_prompt="u", prompt_version="v2.3.0")


class TestSimplePathFallback:

    @pytest.fixture(autouse=True)
    def _fast_retry(self):
        from tenacity import wait_none
        retry_state = LLMClient._call_llm.retry
        original_wait = retry_state.wait
        retry_state.wait = wait_none()
        yield
        retry_state.wait = original_wait

    def test_analyze_simple_failure_returns_none(self):
        client = LLMClient()
        client.simple_client = _FakeClient(error=RuntimeError("boom"))
        assert asyncio.run(client.analyze_simple(_components())) is None

    def test_analyze_keeps_complex_sentence_function_when_simple_fails(self):
        client = LLMClient()
        client.complex_client = _FakeClient(payload=LLMOutput.default().to_dict())
        client.simple_client = _FakeClient(error=RuntimeError("boom"))
        result = asyncio.run(client.analyze(_components(), _components()))
        assert result.output.sentence_function.label == SentenceFunctionOutput().label
        assert result.weight_multiplier == 1.0

    def test_analyze_overrides_when_simple_succeeds(self):
        client = LLMClient()
        client.complex_client = _FakeClient(payload=LLMOutput.default().to_dict())
        client.simple_client = _FakeClient(payload={"sentence_function": {"label": "question", "confidence": 0.9}})
        result = asyncio.run(client.analyze(_components(), _components()))
        assert result.output.sentence_function.label == "question"


class TestTemperatureValidation:

    def test_empty_temperatures_rejected(self):
        from danmaku_analyzer.llm_config import LLMSettings
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            LLMSettings(COMPLEX_LLM_TEMPERATURES=[])


class TestEnvFilePriority:

    def test_dataroot_env_overrides_package_env(self, tmp_path):
        """DATA_ROOT/.env 必须覆盖包内 .env 同名键（pydantic-settings 元组后者胜出，需新进程验证：
        候选清单在类定义时固化）；包内 .env 含同名键时本用例才是真实优先级证明"""
        (tmp_path / ".env").write_text("COMPLEX_LLM_MODEL=probe_from_dataroot\n", encoding="utf-8")
        code = (
            "from danmaku_analyzer.llm_config import LLMSettings; "
            "print(LLMSettings().COMPLEX_LLM_MODEL)"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True,
            env={**os.environ, "DATA_ROOT": str(tmp_path)},
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "probe_from_dataroot"


class TestWriteLlmEnvPermission:

    def test_posix_chmod_600(self, tmp_path, monkeypatch):
        from danmaku_analyzer import prefs
        calls = []
        monkeypatch.setattr(prefs.os, "name", "posix")
        monkeypatch.setattr(prefs.os, "chmod", lambda p, m: calls.append((p, m)))
        prefs.write_llm_env({"COMPLEX_LLM_MODEL": "m"}, str(tmp_path / ".env"))
        assert calls and calls[0][1] == 0o600


class TestProgressHardening:

    def test_rotation_leaves_no_tmp_file(self, isolated_data_root, monkeypatch):
        import danmaku_analyzer.pipeline as pipeline_module
        monkeypatch.setattr(pipeline_module, "_PROGRESS_MAX_BYTES", 50)
        zip_path = str(isolated_data_root / "x.zip")
        for i in range(4):
            _append_progress(f"BV{i}", f"BV{i}", zip_path, reused=False)
        scheduler = isolated_data_root / "scheduler"
        assert not any(p.name.endswith(".tmp") for p in scheduler.iterdir())

    def test_deep_uplevel_falls_back_to_absolute(self, isolated_data_root, monkeypatch):
        import danmaku_analyzer.pipeline as pipeline_module
        fake_rel = (".." + os.sep) * 20 + "x.zip"
        monkeypatch.setattr(pipeline_module.os.path, "relpath", lambda p, s: fake_rel)
        assert _relativize_zip_path("Z:\\far\\x.zip") == "Z:\\far\\x.zip"


class TestFetchAllSourceMarking:

    def test_xml_fallback_marked(self, monkeypatch):
        from datetime import datetime
        from danmaku_analyzer.crawler import BilibiliCrawler, DanmakuItem, VideoMeta

        crawler = BilibiliCrawler()

        async def fake_metadata(bvid):
            return VideoMeta(bvid=bvid, title="t", tname="游戏", pubdate=datetime(2025, 1, 1), cid=7)

        async def fake_xml(bvid, cid=None):
            return [DanmakuItem(uid_hash="u1", content="x", time_sec=1.0, identity_type="real_user")]

        class FakeVideo:
            def __init__(self, bvid=None, credential=None):
                pass

            async def get_danmakus(self, cid=None):
                raise RuntimeError("protobuf 不可用")

        crawler.fetch_video_metadata = fake_metadata
        crawler._fetch_danmaku_xml_fallback = fake_xml
        monkeypatch.setattr("danmaku_analyzer.crawler.video.Video", FakeVideo)

        meta, items, source = asyncio.run(crawler.fetch_all("BV1xx"))
        assert source == "xml"
        assert len(items) == 1


class TestAnalyzeVideoContract:
    """analyze_video 全流程 mock 串联：阶段间数据契约的回归锚点"""

    def test_full_flow_with_fake_crawler_and_llm(self, tmp_path, monkeypatch):
        from datetime import datetime
        import danmaku_analyzer.pipeline as pipeline_module
        from danmaku_analyzer.config import get_settings
        from danmaku_analyzer.crawler import DanmakuItem, VideoMeta
        from danmaku_analyzer.llm_client import LLMClient
        from danmaku_analyzer.llm_models import ConsensusLevel, DualPathResult, LLMOutput

        settings = get_settings()
        monkeypatch.setattr(settings, "ENABLE_LLM_ANALYSIS_REPORT", False)
        monkeypatch.setattr(settings, "DATA_ROOT", str(tmp_path))

        async def fake_metadata(self, bvid):
            return VideoMeta(bvid=bvid, title="标题", tname="游戏", pubdate=datetime(2025, 1, 1), cid=1)

        async def fake_danmaku(self, bvid, cid=None):
            items = [
                DanmakuItem(uid_hash=f"u{i}", content=f"第{i}条弹幕内容", time_sec=float(i), identity_type="real_user")
                for i in range(40)
            ]
            return items, "protobuf"

        monkeypatch.setattr(pipeline_module.BilibiliCrawler, "fetch_video_metadata", fake_metadata)
        monkeypatch.setattr(pipeline_module.BilibiliCrawler, "fetch_danmaku", fake_danmaku)

        import danmaku_analyzer.account as account_module
        monkeypatch.setattr(account_module, "resolve_credential", lambda credential_file=None: (None, ""))

        async def fake_analyze(self, complex_prompt, simple_prompt):
            output = LLMOutput.default()
            return DualPathResult(
                output=output, consensus_level=ConsensusLevel.HIGH, jsd_score=0.0,
                weight_multiplier=1.0, raw_outputs=[output.to_dict()], prompt_version="v-test",
            )

        monkeypatch.setattr(LLMClient, "analyze", fake_analyze)

        result = asyncio.run(pipeline_module.analyze_video(
            "BV1uu4y1s7TB", output_dir=str(tmp_path / "out"), no_cache=True,
        ))
        assert result.zip_valid and os.path.exists(result.zip_path)
        with zipfile.ZipFile(result.zip_path) as z:
            assert "metadata.json" in z.namelist()
            meta = json.loads(z.read("metadata.json"))
        assert meta["bvid"] == "BV1uu4y1s7TB"
        assert meta["danmaku_source"] == "protobuf"
