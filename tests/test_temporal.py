"""
历时维度端到端测试 - ENABLE_TEMPORAL_GROUPING 分桶 + 统计分层
覆盖 corpus_builder 时间分桶/聚合分层与 statistical_validator 跨分区检验的协作
"""

import json
import os
import zipfile
from io import StringIO

import pandas as pd
import pytest

import danmaku_analyzer.corpus_builder as corpus_builder_module
import danmaku_analyzer.corpus_store as corpus_store_module
from danmaku_analyzer.config import get_settings
from danmaku_analyzer.corpus_builder import CorpusBuilder
from danmaku_analyzer.statistical_validator import StatisticalValidator


def write_table_csv(zipf: zipfile.ZipFile, filename: str, rows: list):
    df = pd.DataFrame(rows)
    buffer = StringIO()
    df.to_csv(buffer, index=False, encoding='utf-8-sig')
    zipf.writestr(filename, buffer.getvalue().encode('utf-8-sig'))


def make_fake_zip(
    dir_path, bvid: str, tname: str, pubdate: str = "2025-03-15T10:00:00", density: float = 0.5,
) -> str:
    metadata = {
        "generated_at": "2025-08-05T12:00:00", "prompt_version": "v2.2.1",
        "total_videos": 1, "total_danmaku": 100, "total_segments": 1,
        "partitions": [tname],
        "bvid": bvid, "title": f"测试-{bvid}", "tname": tname, "tags": [],
        "pubdate": pubdate, "view_count": 1000, "danmaku_count": 100,
        "pipeline_version": "0.3.1-beta",
    }
    zip_path = os.path.join(str(dir_path), f"[{bvid}]test.zip")
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        zipf.writestr("metadata.json", json.dumps(metadata, ensure_ascii=False))
        write_table_csv(zipf, "table_lexical_by_partition.csv", [{
            "tname": tname, "zone_type": "hot_zone", "danmaku_count": 100,
            "avg_word_length": 2.0, "content_word_density": density, "punctuation_emoji_rate": 0.1,
        }])
        write_table_csv(zipf, "table_consensus_stats.csv", [{
            "tname": tname, "zone_type": "hot_zone", "danmaku_count": 100,
            "high_consensus_rate": 0.6, "medium_consensus_rate": 0.2,
            "low_consensus_rate": 0.2, "avg_weight_multiplier": 0.96,
        }])
        write_table_csv(zipf, "table_emotion.csv", [{
            "tname": tname, "zone_type": "hot_zone", "danmaku_count": 100,
            "cooperative_principle_violation_rate": 0.1,
        }])
    return zip_path


@pytest.fixture
def tmp_store(monkeypatch, tmp_path):
    real_store = corpus_store_module.CorpusStore
    index_path = str(tmp_path / "corpus_index.json")
    monkeypatch.setattr(corpus_builder_module, "CorpusStore", lambda: real_store(index_path=index_path))
    return real_store(index_path=index_path)


@pytest.fixture
def temporal_on(monkeypatch):
    monkeypatch.setattr(get_settings(), "ENABLE_TEMPORAL_GROUPING", True)
    monkeypatch.setattr(get_settings(), "TEMPORAL_GRANULARITY", "year")


class TestBucketPubdate:

    def test_year(self):
        assert CorpusBuilder._bucket_pubdate("2024-06-01T00:00:00", "year") == "2024"

    def test_quarter(self):
        assert CorpusBuilder._bucket_pubdate("2024-04-15T00:00:00", "quarter") == "2024-Q2"

    def test_month(self):
        assert CorpusBuilder._bucket_pubdate("2024-04-15T00:00:00", "month") == "2024-04"

    def test_invalid_pubdate_returns_unknown(self):
        assert CorpusBuilder._bucket_pubdate("not-a-date", "year") == "unknown"

    def test_empty_pubdate_returns_unknown(self):
        assert CorpusBuilder._bucket_pubdate("", "year") == "unknown"


class TestTemporalAggregation:

    def test_summary_stratified_by_time_period(self, tmp_path, tmp_store, temporal_on):
        make_fake_zip(tmp_path, "BV1a", "游戏", pubdate="2024-05-01T00:00:00", density=0.3)
        make_fake_zip(tmp_path, "BV1b", "游戏", pubdate="2025-05-01T00:00:00", density=0.5)
        zip_paths = [str(p) for p in tmp_path.glob("*.zip")]

        result = CorpusBuilder().build_from_zips(zip_paths, output_dir=str(tmp_path / "out"))
        df = pd.read_csv(result.csv_path, encoding='utf-8-sig')
        assert "time_period" in df.columns
        assert set(df["time_period"].astype(str)) == {"2024", "2025"}
        assert len(df) == 2

    def test_temporal_disabled_no_time_period(self, tmp_path, tmp_store):
        make_fake_zip(tmp_path, "BV1a", "游戏", pubdate="2024-05-01T00:00:00")
        make_fake_zip(tmp_path, "BV1b", "游戏", pubdate="2025-05-01T00:00:00")
        zip_paths = [str(p) for p in tmp_path.glob("*.zip")]

        result = CorpusBuilder().build_from_zips(zip_paths, output_dir=str(tmp_path / "out"))
        df = pd.read_csv(result.csv_path, encoding='utf-8-sig')
        assert "time_period" not in df.columns
        assert len(df) == 1
        assert df.iloc[0]["video_count"] == 2

    def test_videos_csv_keeps_per_video_observation(self, tmp_path, tmp_store, temporal_on):
        make_fake_zip(tmp_path, "BV1a", "游戏", pubdate="2024-05-01T00:00:00")
        make_fake_zip(tmp_path, "BV1b", "音乐", pubdate="2025-05-01T00:00:00")
        zip_paths = [str(p) for p in tmp_path.glob("*.zip")]

        result = CorpusBuilder().build_from_zips(zip_paths, output_dir=str(tmp_path / "out"))
        videos_df = pd.read_csv(result.videos_csv_path, encoding='utf-8-sig')
        assert len(videos_df) == 2
        assert set(videos_df["tname"]) == {"游戏", "音乐"}


class TestTemporalStatisticsIntegration:

    def test_corpus_compare_stratifies_by_tname_with_temporal_on(self, tmp_path, tmp_store, temporal_on):
        for i in range(3):
            make_fake_zip(tmp_path, f"BV1g{i}", "游戏", pubdate="2024-05-01T00:00:00", density=0.3 + i * 0.05)
        for i in range(3):
            make_fake_zip(tmp_path, f"BV1m{i}", "音乐", pubdate="2025-05-01T00:00:00", density=0.6 + i * 0.05)
        zip_paths = [str(p) for p in tmp_path.glob("*.zip")]

        result = CorpusBuilder().build_from_zips(zip_paths, output_dir=str(tmp_path / "out"))
        comparison = StatisticalValidator().corpus_compare(result.videos_csv_path)
        df = comparison.to_dataframe()

        kw_rows = df[df["test_type"] == "Kruskal-Wallis"]
        assert len(kw_rows) > 0
        assert "content_word_density" in set(kw_rows["metric"])
        status = df[df["test_type"] == "sample_status"].set_index("group1")
        assert status.loc["游戏", "note"] == "sample_sufficient"
        assert status.loc["音乐", "note"] == "sample_sufficient"

    def test_sparse_temporal_partitions_marked_insufficient(self, tmp_path, tmp_store, temporal_on):
        # 同分区拆到两个年份桶后每桶仅 1-2 个视频 → 汇总层保留但推断层按 tname 计数
        make_fake_zip(tmp_path, "BV1g0", "游戏", pubdate="2024-05-01T00:00:00", density=0.3)
        make_fake_zip(tmp_path, "BV1g1", "游戏", pubdate="2025-05-01T00:00:00", density=0.35)
        make_fake_zip(tmp_path, "BV1g2", "游戏", pubdate="2025-06-01T00:00:00", density=0.4)
        make_fake_zip(tmp_path, "BV1m0", "音乐", pubdate="2025-05-01T00:00:00", density=0.6)
        zip_paths = [str(p) for p in tmp_path.glob("*.zip")]

        result = CorpusBuilder().build_from_zips(zip_paths, output_dir=str(tmp_path / "out"))
        summary = pd.read_csv(result.csv_path, encoding='utf-8-sig')
        # 汇总层按 (tname, time_period) 分层：游戏×2 桶 + 音乐×1 桶
        assert len(summary) == 3

        comparison = StatisticalValidator().corpus_compare(result.videos_csv_path)
        df = comparison.to_dataframe()
        # 多分区场景仍按 tname 分组：游戏（3 视频）足够、音乐（1 视频）不足；有效组 < 2 未执行组间检验，样本状态行照常输出
        assert not df.empty
        assert (df["test_type"] == "sample_status").all()
        status = df.set_index("group1")["note"]
        assert status["游戏"] == "sample_sufficient"
        assert "insufficient_sample" in status["音乐"]
        assert list(df.columns) == [
            "metric", "test_type", "group1", "group2", "n1", "n2",
            "statistic", "p_value", "effect_size", "effect_magnitude", "note",
        ]


class TestTemporalObservationColumn:

    def test_videos_csv_has_time_period_column(self, tmp_path, tmp_store, temporal_on):
        make_fake_zip(tmp_path, "BV1g0", "游戏", pubdate="2024-05-01T00:00:00", density=0.3)
        zip_path = str(next(tmp_path.glob("*.zip")))
        result = CorpusBuilder().build_from_zips([zip_path], output_dir=str(tmp_path / "out"))
        df = pd.read_csv(result.videos_csv_path, encoding='utf-8-sig')
        assert "time_period" in df.columns
        # CSV 往返后年份桶可能被解析为数值 dtype，统一按字符串比较
        assert str(df.iloc[0]["time_period"]) == "2024"
