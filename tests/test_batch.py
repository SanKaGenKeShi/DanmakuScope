"""段内批量推理测试 - 批量提示词模板 / analyze_batch 逐条对齐与条数校验"""

import asyncio
import json

import pytest

from danmaku_analyzer.llm_client import LLMClient
from danmaku_analyzer.llm_config import get_llm_settings
from danmaku_analyzer.llm_models import ConsensusLevel
from danmaku_analyzer.prompt_builder import PromptBuilder


class _FakeBackend:
    """假 LLM 后端：实现 LLMBackend 协议的 complete 接口，按序消费 payload 队列"""

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


def _item(emotion="positive", label="expression"):
    return {
        "emotion": {"label": emotion, "confidence": 0.9},
        "cooperative_principle": {"violated": False, "maxim": None},
        "interaction_type": {"label": label, "confidence": 0.8},
        "orthography": {"status": "standard", "confidence": 0.9},
    }


def _complex_payload(items):
    return json.dumps({"items": items})


def _simple_payload(n):
    return json.dumps({"items": [
        {"sentence_function": {"label": "assertion", "confidence": 0.9}} for _ in range(n)
    ]})


class TestBatchPrompts:

    def test_prompt_version_bumped(self):
        assert get_llm_settings().PROMPT_VERSION == "v2.3.0"

    def test_complex_batch_template(self):
        builder = PromptBuilder()
        pc = builder.build_complex_prompt_batch(
            "鬼畜", ["梗"], [("弹幕一", "语境一"), ("弹幕二", None)]
        )
        assert "【鬼畜】" in pc.system_prompt
        # 正字法指令与宁可误判准则保留
        assert "community_variant" in pc.system_prompt
        assert "宁可误判" in pc.system_prompt
        # 批量输出格式与条数约束
        assert '"items"' in pc.system_prompt
        assert "完全一致" in pc.system_prompt
        assert "【第1条】" in pc.user_prompt and "【第2条】" in pc.user_prompt
        assert "弹幕一" in pc.user_prompt and "语境一" in pc.user_prompt
        assert "共 2 条" in pc.user_prompt

    def test_simple_batch_template(self):
        builder = PromptBuilder()
        pc = builder.build_simple_prompt_batch(["弹幕一", "弹幕二", "弹幕三"])
        assert '"items"' in pc.user_prompt
        assert "【第3条】弹幕三" in pc.user_prompt
        assert "3 条弹幕" in pc.user_prompt


class TestAnalyzeBatch:

    def _make_client(self):
        client = LLMClient()
        client.enable_dual_path = True
        return client

    def test_per_item_alignment_dual_path(self):
        client = self._make_client()
        items = [_item(), _item(emotion="negative")]
        client.complex_client = _FakeAsyncClient([_complex_payload(items), _complex_payload(items)])
        client.simple_client = _FakeAsyncClient([_simple_payload(2)])
        complex_pc = PromptBuilder().build_complex_prompt_batch("音乐", [], [("a", None), ("b", None)])
        simple_pc = PromptBuilder().build_simple_prompt_batch(["a", "b"])
        results = asyncio.run(client.analyze_batch(complex_pc, simple_pc, 2))
        assert len(results) == 2
        # 两路完全一致 → 逐条 HIGH 共识
        assert all(r.consensus_level == ConsensusLevel.HIGH for r in results)
        assert all(r.jsd_score == pytest.approx(0.0) for r in results)
        # 句类来自简单路批量输出
        assert all(r.output.sentence_function.label == "assertion" for r in results)
        # 逐条情感标签按序对齐
        assert results[0].output.emotion.label == "positive"
        assert results[1].output.emotion.label == "negative"

    def test_count_mismatch_raises(self):
        client = self._make_client()
        client.complex_client = _FakeAsyncClient([
            _complex_payload([_item()]),  # 期望 2 条但只返回 1 条
            _complex_payload([_item(), _item()]),
        ])
        client.simple_client = _FakeAsyncClient([_simple_payload(2)])
        complex_pc = PromptBuilder().build_complex_prompt_batch("音乐", [], [("a", None), ("b", None)])
        simple_pc = PromptBuilder().build_simple_prompt_batch(["a", "b"])
        with pytest.raises(ValueError, match="条数不符"):
            asyncio.run(client.analyze_batch(complex_pc, simple_pc, 2))

    def test_single_path_batch(self):
        client = self._make_client()
        client.enable_dual_path = False
        client.complex_client = _FakeAsyncClient([_complex_payload([_item()])])
        client.simple_client = _FakeAsyncClient([_simple_payload(1)])
        complex_pc = PromptBuilder().build_complex_prompt_batch("音乐", [], [("a", None)])
        simple_pc = PromptBuilder().build_simple_prompt_batch(["a"])
        results = asyncio.run(client.analyze_batch(complex_pc, simple_pc, 1))
        assert len(results) == 1
        assert results[0].consensus_level == ConsensusLevel.HIGH
