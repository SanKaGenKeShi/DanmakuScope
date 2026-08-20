"""
可复现 manifest / LLM 后端协议与 adapter / YAML 脚本解析单元测试
"""

import asyncio
import json
import types

import pytest

from danmaku_analyzer.llm_factory import (
    LLMBackend,
    OllamaBackend,
    OpenAICompatibleBackend,
    create_backend,
)
from danmaku_analyzer.reproducibility import ReproManifestBuilder

# MAX_CONTEXT_TOKENS 为分析参数非凭证，故标记清单不含 TOKEN
_SENSITIVE_MARKERS = ("KEY", "SESSDATA", "JCT", "BUVID", "COOKIE", "BASE_URL")


class TestReproManifest:

    def test_whitelist_excludes_sensitive_and_env_fields(self):
        snapshot = ReproManifestBuilder.reproducible_config_snapshot()
        for key in snapshot:
            assert not any(marker in key.upper() for marker in _SENSITIVE_MARKERS), key
        assert "COMPLEX_LLM_MODEL" in snapshot
        assert "TOP_N" in snapshot
        assert "PROMPT_VERSION" in snapshot
        # 路径/凭证/提示词表一律不入快照
        for excluded in ("BILIBILI_SESSDATA", "BILIBILI_JCT", "BILIBILI_BUVID3",
                         "DATA_ROOT", "OUTPUT_DIR", "REGISTER_HINTS",
                         "COMPLEX_LLM_API_KEY", "SIMPLE_LLM_BASE_URL"):
            assert excluded not in snapshot

    def test_manifest_structure(self):
        manifest = ReproManifestBuilder().build()
        assert manifest["pipeline_version"]
        assert manifest["python_version"]
        assert manifest["platform"]
        assert manifest["package_versions"]["danmaku-analyzer"]
        assert isinstance(manifest["config_snapshot"], dict)

    def test_write_manifest_contains_no_secret_values(self, tmp_path):
        path = ReproManifestBuilder().write(str(tmp_path))
        text = open(path, encoding="utf-8").read()
        data = json.loads(text)
        assert data["config_snapshot"]
        # 占位 Key（sk-xxx 等）与凭证值绝不出现在产物文本中
        for secret in ("sk-xxx", "sk-yyy", "sk-zzz"):
            assert secret not in text


class TestBackendProtocol:

    def test_adapters_satisfy_protocol(self):
        assert isinstance(OpenAICompatibleBackend("http://x", "k"), LLMBackend)
        assert isinstance(OllamaBackend(), LLMBackend)

    def test_empty_key_selects_ollama_adapter(self):
        backend = create_backend("http://127.0.0.1:11434/v1", "")
        assert backend.backend_name == "ollama"

    def test_key_present_selects_openai_compatible(self):
        backend = create_backend("https://api.example.com/v1", "sk-real")
        assert backend.backend_name == "openai-compatible"

    def test_each_adapter_binds_own_connection_config(self):
        """双轨约束：实例各自绑定独立 base_url，无共享连接入口"""
        a = create_backend("http://host-a/v1", "key-a")
        b = create_backend("http://host-b/v1", "")
        assert a.base_url != b.base_url


class TestConnectionDefaults:
    """连接三件套默认空（v0.3.8-beta）：占位示范值不可用，初始留空由用户直接填写"""

    def test_connection_fields_default_empty(self):
        from danmaku_analyzer.llm_config import LLMSettings
        fields = ("COMPLEX_LLM_BASE_URL", "COMPLEX_LLM_API_KEY", "COMPLEX_LLM_MODEL",
                  "SIMPLE_LLM_BASE_URL", "SIMPLE_LLM_API_KEY", "SIMPLE_LLM_MODEL",
                  "ANALYSIS_REPORT_LLM_BASE_URL", "ANALYSIS_REPORT_LLM_API_KEY", "ANALYSIS_REPORT_LLM_MODEL")
        for name in fields:
            assert LLMSettings.model_fields[name].default == "", name


class TestBackendComplete:

    def _backend_with_fake_client(self, content):
        captured = {}

        class _Completions:
            async def create(self, **kwargs):
                captured.update(kwargs)
                message = types.SimpleNamespace(content=content)
                return types.SimpleNamespace(choices=[types.SimpleNamespace(message=message)])

        backend = OpenAICompatibleBackend("http://x", "k")
        backend._client = types.SimpleNamespace(chat=types.SimpleNamespace(completions=_Completions()))
        return backend, captured

    def test_complete_returns_content_and_forwards_extra_body(self):
        backend, captured = self._backend_with_fake_client('{"ok": 1}')
        content = asyncio.run(backend.complete(
            model="m", messages=[{"role": "user", "content": "hi"}],
            temperature=0.1, extra_body={"enable_thinking": False},
        ))
        assert content == '{"ok": 1}'
        assert captured["extra_body"] == {"enable_thinking": False}
        assert captured["temperature"] == 0.1
        assert "response_format" not in captured

    def test_response_format_forwarded_when_given(self):
        backend, captured = self._backend_with_fake_client("{}")
        asyncio.run(backend.complete(
            model="m", messages=[], temperature=0.0,
            response_format={"type": "json_object"},
        ))
        assert captured["response_format"] == {"type": "json_object"}

    def test_empty_content_normalized_to_empty_string(self):
        backend, _ = self._backend_with_fake_client(None)
        assert asyncio.run(backend.complete(model="m", messages=[], temperature=0.0)) == ""


class TestScriptParsing:

    def _write_yaml(self, tmp_path, text):
        path = tmp_path / "tasks.yaml"
        path.write_text(text, encoding="utf-8")
        return str(path)

    def test_valid_tasks_parsed(self, tmp_path):
        from danmaku_analyzer.cli import _parse_script_tasks
        path = self._write_yaml(tmp_path, """
tasks:
  - command: analyze
    input: BV1xx
    options:
      freq_based: true
      top_n: 15
  - command: compare
    inputs: [BV1x, BV1y]
    options:
      reuse: false
""")
        tasks = _parse_script_tasks(path)
        assert len(tasks) == 2
        assert tasks[0]["command"] == "analyze"
        assert tasks[0]["options"]["top_n"] == 15
        assert tasks[1]["inputs"] == ["BV1x", "BV1y"]
        assert tasks[1]["options"]["reuse"] is False

    def test_unknown_command_rejected(self, tmp_path):
        from danmaku_analyzer.cli import _parse_script_tasks
        path = self._write_yaml(tmp_path, "tasks:\n  - command: corpus\n    input: x.zip\n")
        with pytest.raises(ValueError, match="不支持的命令"):
            _parse_script_tasks(path)

    def test_unknown_option_rejected(self, tmp_path):
        from danmaku_analyzer.cli import _parse_script_tasks
        path = self._write_yaml(tmp_path, "tasks:\n  - command: analyze\n    input: BV1x\n    options:\n      typo_option: 1\n")
        with pytest.raises(ValueError, match="未知选项"):
            _parse_script_tasks(path)

    def test_missing_inputs_rejected(self, tmp_path):
        from danmaku_analyzer.cli import _parse_script_tasks
        path = self._write_yaml(tmp_path, "tasks:\n  - command: compare\n")
        with pytest.raises(ValueError, match="inputs"):
            _parse_script_tasks(path)

    def test_empty_tasks_rejected(self, tmp_path):
        from danmaku_analyzer.cli import _parse_script_tasks
        path = self._write_yaml(tmp_path, "tasks: []\n")
        with pytest.raises(ValueError, match="非空"):
            _parse_script_tasks(path)


class TestScriptExecution:
    """脚本任务执行路径：单任务失败（含 sys.exit）不中断后续"""

    def test_tasks_run_sequentially_and_failure_isolated(self, monkeypatch):
        import danmaku_analyzer.cli as cli_module

        executed = []

        async def fake_analyze(input_str, *args, **kwargs):
            executed.append(("analyze", input_str))

        async def fake_compare(input_list, *args, **kwargs):
            executed.append(("compare", tuple(input_list)))
            raise SystemExit(1)

        monkeypatch.setattr(cli_module, "_analyze_async", fake_analyze)
        monkeypatch.setattr(cli_module, "_compare_async", fake_compare)

        tasks = [
            {"command": "analyze", "input": "BV1a", "inputs": [], "options": {}},
            {"command": "compare", "input": "", "inputs": ["BV1b", "BV1c"], "options": {}},
            {"command": "analyze", "input": "BV1d", "inputs": [], "options": {}},
        ]
        with pytest.raises(SystemExit):
            asyncio.run(cli_module._run_script_async(tasks))
        assert executed == [("analyze", "BV1a"), ("compare", ("BV1b", "BV1c")), ("analyze", "BV1d")]
