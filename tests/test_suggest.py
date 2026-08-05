"""
语料库补足建议单元测试 - 缺口分析（离线纯函数） + 候选解析（mock 网络）
"""

import asyncio
import sys
import types
from enum import Enum

import pytest

from danmaku_analyzer.corpus_store import CorpusStore
from danmaku_analyzer.corpus_suggester import CorpusSuggester, PARTITION_TO_TID


@pytest.fixture
def tmp_store(tmp_path):
    return CorpusStore(index_path=str(tmp_path / "corpus_index.json"))


@pytest.fixture
def suggester(tmp_store):
    return CorpusSuggester(store=tmp_store)


def _register(store, bvid, tname):
    store.register_video({"bvid": bvid, "tname": tname, "zip_path": "x.zip"})


# ========== 缺口分析 ==========

class TestAnalyzeGaps:

    def test_empty_corpus_returns_no_gaps(self, suggester):
        assert suggester.analyze_gaps() == []

    def test_gap_counts_missing(self, suggester, tmp_store):
        _register(tmp_store, "BV1a", "游戏")
        gaps = suggester.analyze_gaps()
        assert len(gaps) == 1
        g = gaps[0]
        assert g.tname == "游戏"
        assert g.have == 1
        assert g.min_required == 3
        assert g.missing == 2
        assert not g.is_sufficient

    def test_sufficient_partition_has_zero_missing(self, suggester, tmp_store):
        for i in range(3):
            _register(tmp_store, f"BV1a{i}", "音乐")
        gaps = suggester.analyze_gaps()
        assert gaps[0].missing == 0
        assert gaps[0].is_sufficient

    def test_unknown_partition_not_counted_as_gap(self, suggester, tmp_store):
        _register(tmp_store, "BV1x", "")
        gaps = suggester.analyze_gaps()
        assert gaps[0].tname == "未知"
        assert gaps[0].missing == 0
        assert gaps[0].is_sufficient

    def test_insufficient_partitions_filters_only_gaps(self, suggester, tmp_store):
        _register(tmp_store, "BV1a", "游戏")
        for i in range(3):
            _register(tmp_store, f"BV1b{i}", "音乐")
        assert suggester.insufficient_partitions() == ["游戏"]

    def test_sorted_by_count_desc(self, suggester, tmp_store):
        _register(tmp_store, "BV1a", "游戏")
        for i in range(3):
            _register(tmp_store, f"BV1b{i}", "音乐")
        gaps = suggester.analyze_gaps()
        assert [g.tname for g in gaps] == ["音乐", "游戏"]

    def test_gap_to_dict(self, suggester, tmp_store):
        _register(tmp_store, "BV1a", "知识")
        d = suggester.analyze_gaps()[0].to_dict()
        assert d == {"tname": "知识", "have": 1, "min_required": 3, "missing": 2, "is_sufficient": False}


# ========== 候选解析（mock 搜索结果） ==========

def _mock_item(bvid, title="标题", play=1000, danmaku=500, pubdate=1700000000):
    return {"bvid": bvid, "title": title, "play": play, "video_review": danmaku, "pubdate": pubdate}


class TestParseResults:

    def test_excludes_existing_bvids(self):
        raw = {"result": [_mock_item("BV1a"), _mock_item("BV1b")]}
        out = CorpusSuggester._parse_results(raw, {"BV1a"}, limit=10)
        assert [c.bvid for c in out] == ["BV1b"]

    def test_sorted_by_danmaku_desc_and_limited(self):
        raw = {"result": [
            _mock_item("BV1a", danmaku=10),
            _mock_item("BV1b", danmaku=999),
            _mock_item("BV1c", danmaku=500),
        ]}
        out = CorpusSuggester._parse_results(raw, set(), limit=2)
        assert [c.bvid for c in out] == ["BV1b", "BV1c"]

    def test_pubdate_formatted(self):
        from datetime import datetime
        raw = {"result": [_mock_item("BV1a", pubdate=1700000000)]}
        out = CorpusSuggester._parse_results(raw, set(), limit=10)
        assert out[0].pubdate == datetime.fromtimestamp(1700000000).strftime("%Y-%m-%d")

    def test_missing_bvid_skipped(self):
        raw = {"result": [{"title": "无BV"}, _mock_item("BV1a")]}
        out = CorpusSuggester._parse_results(raw, set(), limit=10)
        assert [c.bvid for c in out] == ["BV1a"]

    def test_empty_result(self):
        assert CorpusSuggester._parse_results({"result": None}, set(), limit=10) == []
        assert CorpusSuggester._parse_results({}, set(), limit=10) == []

    def test_title_highlight_tags_and_entities_cleaned(self):
        raw = {"result": [_mock_item("BV1a", title='【<em class="keyword">游戏</em>】评测&amp;试玩')]}
        out = CorpusSuggester._parse_results(raw, set(), limit=10)
        assert out[0].title == "【游戏】评测&试玩"


# ========== 在线候选获取（mock bilibili_api.search） ==========

class _FakeSearchObjectType(Enum):
    VIDEO = "video"


class _FakeOrderVideo(Enum):
    DM = "dm"


@pytest.fixture
def fake_search(monkeypatch):
    """注入假的 bilibili_api.search 模块，记录调用参数并返回可控结果"""
    calls = []

    async def search_by_type(**kwargs):
        calls.append(kwargs)
        return {"result": [
            _mock_item("BV1new1", danmaku=800),
            _mock_item("BV1new2", danmaku=900),
            _mock_item("BV1old", danmaku=999),
        ]}

    fake = types.ModuleType("bilibili_api.search")
    fake.search_by_type = search_by_type
    fake.SearchObjectType = _FakeSearchObjectType
    fake.OrderVideo = _FakeOrderVideo
    fake_bili = types.ModuleType("bilibili_api")
    fake_bili.search = fake
    monkeypatch.setitem(sys.modules, "bilibili_api", fake_bili)
    monkeypatch.setitem(sys.modules, "bilibili_api.search", fake)
    return calls


class TestFetchCandidates:

    def test_excludes_existing_and_passes_zone_filter(self, suggester, tmp_store, fake_search):
        _register(tmp_store, "BV1old", "游戏")
        result = asyncio.run(suggester.fetch_candidates(["游戏"], per_partition=10))
        assert len(fake_search) == 1
        call = fake_search[0]
        assert call["keyword"] == "游戏"
        assert call["video_zone_type"] == PARTITION_TO_TID["游戏"]
        assert call["order_type"] == _FakeOrderVideo.DM
        videos = result["游戏"]
        assert [c.bvid for c in videos] == ["BV1new2", "BV1new1"]

    def test_unknown_partition_searches_without_zone(self, suggester, fake_search):
        asyncio.run(suggester.fetch_candidates(["不存在的分区"], per_partition=5))
        assert fake_search[0]["video_zone_type"] is None

    def test_search_failure_returns_empty_not_raised(self, suggester, monkeypatch):
        fake = types.ModuleType("bilibili_api.search")

        async def search_by_type(**kwargs):
            raise RuntimeError("风控")

        fake.search_by_type = search_by_type
        fake.SearchObjectType = _FakeSearchObjectType
        fake.OrderVideo = _FakeOrderVideo
        fake_bili = types.ModuleType("bilibili_api")
        fake_bili.search = fake
        monkeypatch.setitem(sys.modules, "bilibili_api", fake_bili)
        monkeypatch.setitem(sys.modules, "bilibili_api.search", fake)

        result = asyncio.run(suggester.fetch_candidates(["游戏"], per_partition=10))
        assert result == {"游戏": []}


# ========== suggest 整体流程 ==========

class TestSuggestFlow:

    def test_default_targets_are_insufficient_partitions(self, suggester, tmp_store, fake_search):
        _register(tmp_store, "BV1old", "游戏")
        for i in range(3):
            _register(tmp_store, f"BV1m{i}", "音乐")
        result = asyncio.run(suggester.suggest(per_partition=5))
        # 默认只搜索有缺口的"游戏"分区
        assert [c["keyword"] for c in fake_search] == ["游戏"]
        assert [g.tname for g in result.gaps] == ["音乐", "游戏"]
        assert "游戏" in result.candidates

    def test_explicit_partitions_override_default(self, suggester, tmp_store, fake_search):
        _register(tmp_store, "BV1a", "游戏")
        result = asyncio.run(suggester.suggest(partitions=["知识"], per_partition=5))
        assert [c["keyword"] for c in fake_search] == ["知识"]
        assert "知识" in result.candidates

    def test_empty_corpus_no_candidates_fetched(self, suggester, fake_search):
        result = asyncio.run(suggester.suggest(per_partition=5))
        assert fake_search == []
        assert result.gaps == []
        assert result.candidates == {}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
