import asyncio
import json
from types import SimpleNamespace

import pytest
from tenacity import wait_none

from danmaku_analyzer.aggregator import AggregatedData, Aggregator
from danmaku_analyzer.config import get_settings
from danmaku_analyzer.llm_client import LLMClient
from danmaku_analyzer.llm_config import get_llm_settings
from danmaku_analyzer.llm_models import ConsensusLevel, LLMOutput
from danmaku_analyzer.prompt_builder import PromptComponents


class FakeBackend:
    def __init__(self, payload=None, failing_temperature=None, error=None):
        self.payload = payload
        self.failing_temperature = failing_temperature
        self.error = error
        self.calls = 0

    async def complete(self, **kwargs):
        self.calls += 1
        if self.error or kwargs["temperature"] == self.failing_temperature:
            raise RuntimeError("offline simulated failure")
        return json.dumps(self.payload)


@pytest.fixture
def client(tmp_path, monkeypatch):
    import danmaku_analyzer.llm_client as module

    monkeypatch.setattr(get_settings(), "DATA_ROOT", str(tmp_path))
    monkeypatch.setattr(module, "complex_backend", lambda **kwargs: object())
    monkeypatch.setattr(module, "simple_backend", lambda **kwargs: object())
    monkeypatch.setattr(get_llm_settings(), "ENABLE_DUAL_PATH", True)
    monkeypatch.setattr(get_llm_settings(), "COMPLEX_LLM_TEMPERATURES", [0.1, 0.4])
    monkeypatch.setattr(get_llm_settings(), "LOW_CONSENSUS_WEIGHT", 0.2)
    monkeypatch.setattr(LLMClient._call_llm.retry, "wait", wait_none())
    return LLMClient()


@pytest.fixture
def prompt():
    return PromptComponents(system_prompt="s", user_prompt="u", prompt_version="test")


@pytest.mark.parametrize("payload", [{}, {"emotion": {"label": "INVALID"}}])
def test_invalid_response_retried_without_invented_labels(client, prompt, payload):
    client.complex_client = FakeBackend(payload)
    result = asyncio.run(client.analyze_complex(prompt))
    assert client.complex_client.calls == 6
    assert result.analysis_status == "failed"
    assert result.successful_paths == 0
    assert result.output.emotion is None
    assert result.output.orthography is None
    assert result.consensus_level == ConsensusLevel.LOW
    assert result.weight_multiplier == pytest.approx(0.2)


def test_surviving_path_is_degraded_not_high_consensus(client, prompt):
    client.complex_client = FakeBackend(LLMOutput.default().to_dict(), failing_temperature=0.4)
    result = asyncio.run(client.analyze_complex(prompt))
    assert result.analysis_status == "degraded"
    assert result.successful_paths == 1 and result.requested_paths == 2
    assert result.consensus_level == ConsensusLevel.LOW
    assert result.weight_multiplier == pytest.approx(0.2)
    assert result.output.emotion is not None


def test_empty_simple_output_does_not_overwrite_complex_label(client, prompt):
    output = LLMOutput.default()
    output.sentence_function.label = "exclamation"
    client.complex_client = FakeBackend(output.to_dict())
    client.simple_client = FakeBackend({})
    result = asyncio.run(client.analyze(prompt, prompt))
    assert result.output.sentence_function.label == "exclamation"
    assert result.sentence_function_source == "complex"


def test_successful_sentence_survives_complex_failure(client, prompt):
    client.complex_client = FakeBackend(error=True)
    client.simple_client = FakeBackend({"sentence_function": {"label": "question", "confidence": 0.9}})
    result = asyncio.run(client.analyze(prompt, prompt))
    assert result.output.emotion is None
    assert result.output.sentence_function.label == "question"
    assert result.analysis_status == "degraded"
    assert result.sentence_function_source == "simple"


def test_failed_labels_are_not_neutral_observations(client):
    failed = client.failed_result("test")
    aggregate = AggregatedData(tname="游戏", zone_type="hot_zone")
    Aggregator()._aggregate_soft_labels(aggregate, [SimpleNamespace(llm_result=failed)])
    assert aggregate.emotion_distribution == {}
    assert aggregate.orthography_status_distribution == {}
    assert aggregate.cooperative_principle_violation_rate is None
    assert aggregate.label_weight_sums["emotion"] == 0
    assert aggregate.valid_label_counts["emotion"] == 0


def test_missing_output_roundtrip_and_status_serialization(client):
    result = client.failed_result("test")
    serialized = result.to_dict()
    assert serialized["analysis_status"] == "failed"
    assert serialized["successful_paths"] == 0
    assert all(value is None for value in serialized["output"].values())
    assert LLMOutput.from_dict(serialized["output"]).to_dict() == serialized["output"]


def test_batch_failure_cancels_outstanding_requests(client, prompt, monkeypatch):
    async def run():
        started = 0
        cancelled = 0
        both_started = asyncio.Event()
        never_finishes = asyncio.Event()

        async def request(backend, *args, **kwargs):
            nonlocal started, cancelled
            if backend is client.simple_client:
                await both_started.wait()
                raise RuntimeError("batch failure")
            started += 1
            if started == 2:
                both_started.set()
            try:
                await never_finishes.wait()
            except asyncio.CancelledError:
                cancelled += 1
                raise

        monkeypatch.setattr(client, "_call_llm", request)
        with pytest.raises(RuntimeError, match="batch failure"):
            await client.analyze_batch(prompt, prompt, 1)
        assert cancelled == 2

    asyncio.run(asyncio.wait_for(run(), timeout=2))
