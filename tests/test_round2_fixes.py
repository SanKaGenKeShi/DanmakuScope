"""
v0.2.1-beta 第二轮核查修复回归测试：
#1 共识 CI 基数改用 llm_record_count / #2 多维 JSD 均值 /
#3 多维 confidence 择优 / #4 LLM 分词并发批量 / #5 缓存 schema 版本校验
"""

import asyncio
import json
import pickle

import pytest
from unittest.mock import patch

from danmaku_analyzer.cache_manager import CacheManager
from danmaku_analyzer.config import get_settings
from danmaku_analyzer.hard_metrics import HardMetricsAnalyzer
from danmaku_analyzer.llm_client import ConsensusLevel, LLMClient
from danmaku_analyzer.pipeline import _stage_aggregate

from test_core_modules import make_danmaku_record
from test_fixes import _FakeSyncClient


@pytest.fixture
def client():
    with patch("danmaku_analyzer.llm_client.AsyncOpenAI"):
        c = LLMClient()
    return c


# ========== #1：共识 CI 基数与共识率基数一致（LLM 记录数） ==========

class TestConsensusCISampleBase:

    def _run_stage(self, records):
        return asyncio.run(_stage_aggregate(records, get_settings(), lambda s, m: None))

    def test_ci_computed_from_record_count_not_segment_count(self):
        # 40 条记录分布在 2 个段：旧实现按段数 n=2 < 30 必然 insufficient_sample
        records = [make_danmaku_record(segment_id=i % 2) for i in range(40)]
        agg = self._run_stage(records)[0]
        assert agg.llm_record_count == 40
        assert agg.consensus_ci.get("method") == "wilson"
        assert agg.consensus_ci["point_estimate"] == pytest.approx(1.0)

    def test_few_records_insufficient(self):
        agg = self._run_stage([make_danmaku_record() for _ in range(5)])[0]
        assert agg.consensus_ci["status"] == "insufficient_sample"
        assert agg.consensus_ci["sample_size"] == 5

    def test_to_dict_exposes_llm_record_count(self):
        records = [make_danmaku_record() for _ in range(3)]
        agg = self._run_stage(records)[0]
        assert agg.to_dict()["consensus_stats"]["llm_record_count"] == 3


# ========== #2：JSD 为四维均值，情感单维不再一票定共识 ==========

class TestMultiDimJSD:

    _OPPOSING_OTHER_DIMS = (
        {"interaction_type": {"label": "expression", "confidence": 0.95},
         "orthography": {"status": "standard", "confidence": 0.95},
         "cooperative_principle": {"violated": False, "maxim": "quality"}},
        {"interaction_type": {"label": "mocking", "confidence": 0.95},
         "orthography": {"status": "non_standard_typo", "confidence": 0.95},
         "cooperative_principle": {"violated": True, "maxim": "manner"}},
    )

    def test_emotion_same_but_other_dims_oppose_not_high(self, client):
        a = {"emotion": {"label": "positive", "confidence": 0.95}}
        b = {"emotion": {"label": "positive", "confidence": 0.95}}
        a.update(self._OPPOSING_OTHER_DIMS[0])
        b.update(self._OPPOSING_OTHER_DIMS[1])
        jsd = client._calculate_jsd([a, b])
        assert client._determine_consensus_level(jsd) != ConsensusLevel.HIGH

    def test_emotion_oppose_but_others_same_diluted(self, client):
        base = {"interaction_type": {"label": "expression", "confidence": 0.9},
                "orthography": {"status": "standard", "confidence": 0.9},
                "cooperative_principle": {"violated": False, "maxim": "quality"}}
        a = {"emotion": {"label": "positive", "confidence": 0.95}}
        b = {"emotion": {"label": "negative", "confidence": 0.95}}
        a.update(base)
        b.update(base)
        jsd = client._calculate_jsd([a, b])
        # 单维对立被其余三维稀释：旧实现约 0.56，新均值语义降至约 0.14
        assert jsd < 0.2

    def test_all_dims_oppose_low_consensus(self, client):
        a = {"emotion": {"label": "positive", "confidence": 0.95}}
        b = {"emotion": {"label": "negative", "confidence": 0.95}}
        a.update(self._OPPOSING_OTHER_DIMS[0])
        b.update(self._OPPOSING_OTHER_DIMS[1])
        jsd = client._calculate_jsd([a, b])
        assert client._determine_consensus_level(jsd) == ConsensusLevel.LOW


# ========== #3：非高共识时按多维 confidence 总和择优 ==========

class TestMultiDimMerge:

    def test_high_emotion_confidence_does_not_win_alone(self, client):
        out_a = {"emotion": {"label": "positive", "confidence": 0.98},
                 "interaction_type": {"label": "expression", "confidence": 0.3},
                 "orthography": {"status": "standard", "confidence": 0.3},
                 "cooperative_principle": {"violated": False, "maxim": "quality"},
                 "sentence_function": {"label": "assertion", "confidence": 0.5}}
        out_b = {"emotion": {"label": "negative", "confidence": 0.7},
                 "interaction_type": {"label": "mocking", "confidence": 0.9},
                 "orthography": {"status": "community_variant", "confidence": 0.9},
                 "cooperative_principle": {"violated": True, "maxim": "manner"},
                 "sentence_function": {"label": "exclamation", "confidence": 0.8}}
        merged = client._merge_outputs([out_a, out_b], ConsensusLevel.MEDIUM)
        # B 多维总和 2.5 > A 的 1.58，即使 A 的情感自信度更高
        assert merged.emotion.label == "negative"


# ========== #4：LLM 分词批量并发 + 单条失败回退 jieba ==========

class _CountingSyncClient(_FakeSyncClient):
    def __init__(self, payload):
        super().__init__(payload)
        self.create_calls = 0
        inner_create = self.chat.completions.create

        def counting_create(**kwargs):
            self.create_calls += 1
            return inner_create(**kwargs)

        self.chat.completions.create = counting_create


class _RaisingCompletions:
    def create(self, **kwargs):
        raise RuntimeError("接口故障")


class _RaisingChat:
    completions = _RaisingCompletions()


class _RaisingSyncClient:
    chat = _RaisingChat()


class TestBatchLLMTokenize:

    def test_long_texts_all_sent_to_llm(self):
        analyzer = HardMetricsAnalyzer()
        analyzer.enable_llm_tokenizer = True
        analyzer.llm_tokenizer_min_length = 4
        analyzer.llm_tokenizer_concurrency = 4
        analyzer.llm_client = _CountingSyncClient(json.dumps([["你好", "noun"]]))
        analyzer.llm_model = "fake"
        analyzer.enable_thinking = False
        long_texts = ["这是一条足够长的弹幕文本", "另一条同样足够长的弹幕", "第三条也足够长的弹幕啊"]
        result = analyzer.analyze(long_texts + ["短"])
        assert analyzer.llm_client.create_calls == 3
        assert result.total_danmaku_count == 4
        assert result.total_word_count > 0

    def test_llm_failure_falls_back_to_jieba(self):
        analyzer = HardMetricsAnalyzer()
        analyzer.enable_llm_tokenizer = True
        analyzer.llm_tokenizer_min_length = 4
        analyzer.llm_client = _RaisingSyncClient()
        analyzer.llm_model = "fake"
        analyzer.enable_thinking = False
        result = analyzer.analyze(["这是一条足够长的弹幕文本", "短"])
        assert result.total_danmaku_count == 2
        assert result.total_word_count > 0


# ========== #5：缓存 schema 版本校验 ==========

class TestCacheSchemaVersion:

    def test_roundtrip(self, tmp_path):
        cache = CacheManager(cache_dir=str(tmp_path))
        payload = ("meta", [1, 2, 3])
        assert cache.set("k1", payload) is True
        assert cache.get("k1") == payload

    def test_legacy_raw_pickle_rejected(self, tmp_path):
        cache = CacheManager(cache_dir=str(tmp_path))
        path = cache._get_cache_path("legacy")
        with open(path, "wb") as f:
            pickle.dump(("old", "format"), f)
        assert cache.get("legacy") is None

    def test_schema_version_mismatch_rejected(self, tmp_path):
        from danmaku_analyzer import cache_manager as cm
        cache = CacheManager(cache_dir=str(tmp_path))
        path = cache._get_cache_path("stale")
        with open(path, "wb") as f:
            pickle.dump({"schema_version": cm.CACHE_SCHEMA_VERSION + 1,
                         "payload": ("x",)}, f)
        assert cache.get("stale") is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
