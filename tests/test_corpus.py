"""
语料库模块单元测试 - CorpusStore / CorpusBuilder
不依赖网络：构造假 ZIP 报告验证回读与跨视频聚合
"""

import json
import os
import zipfile
from io import StringIO

import pandas as pd
import pytest
from unittest.mock import patch

import danmaku_analyzer.corpus_builder as corpus_builder_module
import danmaku_analyzer.corpus_store as corpus_store_module
from danmaku_analyzer.corpus_store import CorpusStore
from danmaku_analyzer.corpus_builder import CorpusBuilder
from danmaku_analyzer.config import get_settings


# ========== 辅助工厂函数 ==========

def make_lexical_row(tname: str, zone_type: str, danmaku_count: int, density: float) -> dict:
    return {
        "tname": tname, "zone_type": zone_type, "danmaku_count": danmaku_count,
        "avg_word_length": 2.0, "content_word_density": density, "punctuation_emoji_rate": 0.1,
        "pos_n": 0.3, "pos_v": 0.7, "syllable_单音节": 1.0,
    }


def make_consensus_row(tname: str, zone_type: str, danmaku_count: int, high_rate: float) -> dict:
    return {
        "tname": tname, "zone_type": zone_type, "danmaku_count": danmaku_count,
        "high_consensus_rate": high_rate, "medium_consensus_rate": 0.0,
        "low_consensus_rate": 1.0 - high_rate, "avg_weight_multiplier": 1.0,
    }


def make_emotion_row(tname: str, zone_type: str, danmaku_count: int, positive: float) -> dict:
    return {
        "tname": tname, "zone_type": zone_type, "danmaku_count": danmaku_count,
        "positive": positive, "negative": 1.0 - positive,
    }


def write_table_csv(zipf: zipfile.ZipFile, filename: str, rows: list):
    df = pd.DataFrame(rows)
    buffer = StringIO()
    df.to_csv(buffer, index=False, encoding='utf-8-sig')
    zipf.writestr(filename, buffer.getvalue().encode('utf-8-sig'))


def make_fake_zip(
    dir_path, bvid: str, tname: str, danmaku_count: int = 100, density: float = 0.5,
    high_rate: float = 0.6, positive: float = 0.8, prompt_version: str = "v2.2.0",
    pubdate: str = "2025-03-15T10:00:00", zones: tuple = ("hot_zone",),
) -> str:
    """构造包含 metadata.json + 三张聚合表的最小假报告 ZIP"""
    metadata = {
        "generated_at": "2025-08-05T12:00:00", "prompt_version": prompt_version,
        "total_videos": 1, "total_danmaku": danmaku_count, "total_segments": 2,
        "partitions": [tname],
        "bvid": bvid, "title": f"测试-{bvid}", "tname": tname, "tags": [],
        "pubdate": pubdate, "view_count": 1000, "danmaku_count": danmaku_count,
        "pipeline_version": "0.3.0-beta",
    }
    zip_path = os.path.join(str(dir_path), f"[{bvid}]test.zip")
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        zipf.writestr("metadata.json", json.dumps(metadata, ensure_ascii=False))
        per_zone_count = danmaku_count // len(zones)
        write_table_csv(zipf, "table_lexical_by_partition.csv",
                        [make_lexical_row(tname, z, per_zone_count, density) for z in zones])
        write_table_csv(zipf, "table_consensus_stats.csv",
                        [make_consensus_row(tname, z, per_zone_count, high_rate) for z in zones])
        write_table_csv(zipf, "table_emotion.csv",
                        [make_emotion_row(tname, z, per_zone_count, positive) for z in zones])
    return zip_path


@pytest.fixture
def tmp_store(monkeypatch, tmp_path):
    """把 CorpusStore 重定向到临时索引，避免污染真实 DATA_ROOT"""
    real_store = corpus_store_module.CorpusStore
    index_path = str(tmp_path / "corpus_index.json")
    # corpus_builder 顶部导入已绑定符号，必须 patch 消费方命名空间
    monkeypatch.setattr(corpus_builder_module, "CorpusStore", lambda: real_store(index_path=index_path))
    return real_store(index_path=index_path)


# ========== CorpusStore 测试 ==========

class TestCorpusStore:

    def test_load_missing_returns_empty(self, tmp_path):
        store = CorpusStore(index_path=str(tmp_path / "nope.json"))
        index = store.load()
        assert index["schema_version"] == "1.0"
        assert index["videos"] == []

    def test_load_corrupt_returns_empty(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("not json", encoding='utf-8')
        store = CorpusStore(index_path=str(path))
        assert store.load()["videos"] == []

    def test_register_and_dedupe_by_bvid(self, tmp_path):
        store = CorpusStore(index_path=str(tmp_path / "idx.json"))
        store.register_video({"bvid": "BV1a", "tname": "游戏", "zip_path": str(tmp_path / "a.zip")})
        store.register_video({"bvid": "BV1b", "tname": "音乐", "zip_path": str(tmp_path / "b.zip")})
        store.register_video({"bvid": "BV1a", "tname": "游戏", "zip_path": str(tmp_path / "a2.zip")})
        videos = store.get_videos()
        assert len(videos) == 2
        a_entries = [v for v in videos if v["bvid"] == "BV1a"]
        assert len(a_entries) == 1
        assert a_entries[0]["zip_path"].endswith("a2.zip") or "a2" in a_entries[0]["zip_path"]

    def test_register_requires_bvid(self, tmp_path):
        store = CorpusStore(index_path=str(tmp_path / "idx.json"))
        with pytest.raises(ValueError):
            store.register_video({"tname": "游戏"})

    def test_zip_path_inside_data_root_relativized_and_roundtrip(self, tmp_store):
        data_root = get_settings().DATA_ROOT
        inner = os.path.join(data_root, "reports", "x.zip")
        entry = tmp_store.register_video({"bvid": "BV1rel", "zip_path": inner})
        assert not os.path.isabs(entry["zip_path"])
        assert tmp_store.resolve_zip_path(entry["zip_path"]) == inner

    def test_zip_path_cross_drive_falls_back_to_absolute(self, tmp_store, monkeypatch):
        # 模拟跨驱动器（relpath 抛 ValueError）
        def raise_value_error(*args, **kwargs):
            raise ValueError("path is on mount 'C:', start on mount 'E:'")
        monkeypatch.setattr("danmaku_analyzer.corpus_store.os.path.relpath", raise_value_error)
        entry = tmp_store.register_video({"bvid": "BV1abs", "zip_path": "C:/other/x.zip"})
        assert os.path.isabs(entry["zip_path"])


# ========== CorpusBuilder 测试 ==========

class TestCorpusBuilder:

    def test_end_to_end_group_by_tname(self, tmp_path, tmp_store):
        make_fake_zip(tmp_path, "BV1g1", "游戏", density=0.4)
        make_fake_zip(tmp_path, "BV1g2", "游戏", density=0.6)
        make_fake_zip(tmp_path, "BV1m1", "音乐", density=0.8)
        zip_paths = [str(p) for p in tmp_path.glob("*.zip")]

        builder = CorpusBuilder()
        result = builder.build_from_zips(zip_paths, output_dir=str(tmp_path / "out"))

        df = pd.read_csv(result.csv_path, encoding='utf-8-sig')
        assert len(df) == 2
        game = df[df["tname"] == "游戏"].iloc[0]
        assert game["video_count"] == 2
        assert game["total_danmaku"] == 200
        assert game["content_word_density_mean"] == pytest.approx(0.5)
        assert game["content_word_density_std"] == pytest.approx(0.1414, abs=0.01)
        # 视频级观测表：检验的原始观测来源，每视频一行
        videos_df = pd.read_csv(os.path.join(str(tmp_path / "out"), "corpus_videos.csv"), encoding='utf-8-sig')
        assert len(videos_df) == 3
        assert {"bvid", "tname", "content_word_density"} <= set(videos_df.columns)
        # 索引已登记全部视频
        assert len(tmp_store.get_videos()) == 3

    def test_hot_only_skips_cold_only_zip(self, tmp_path, tmp_store):
        make_fake_zip(tmp_path, "BV1hot", "游戏")
        make_fake_zip(tmp_path, "BV1cold", "游戏", zones=("cold_zone",))
        zip_paths = [str(p) for p in tmp_path.glob("*.zip")]

        builder = CorpusBuilder()
        result = builder.build_from_zips(zip_paths, output_dir=str(tmp_path / "out"))
        df = pd.read_csv(result.csv_path, encoding='utf-8-sig')
        assert df.iloc[0]["video_count"] == 1

    def test_weighted_policy_merges_zones(self, tmp_path, tmp_store, monkeypatch):
        make_fake_zip(tmp_path, "BV1w", "游戏", danmaku_count=300, zones=("hot_zone", "cold_zone"))
        zip_path = str(next(tmp_path.glob("*.zip")))

        builder = CorpusBuilder()
        monkeypatch.setattr(get_settings(), "CORPUS_ZONE_POLICY", "weighted")
        summaries = builder.summarize_video(*builder.read_zip(zip_path))
        assert len(summaries) == 1
        assert summaries[0].danmaku_count == 300

    def test_all_policy_keeps_zone_dimension(self, tmp_path, tmp_store, monkeypatch):
        make_fake_zip(tmp_path, "BV1a", "游戏", zones=("hot_zone", "cold_zone"))
        zip_path = str(next(tmp_path.glob("*.zip")))

        builder = CorpusBuilder()
        monkeypatch.setattr(get_settings(), "CORPUS_ZONE_POLICY", "all")
        summaries = builder.summarize_video(*builder.read_zip(zip_path))
        assert len(summaries) == 2
        assert {s.zone_type for s in summaries} == {"hot_zone", "cold_zone"}

    def test_prompt_version_mixed_warns(self, tmp_path, tmp_store):
        make_fake_zip(tmp_path, "BV1v1", "游戏", prompt_version="v2.2.0")
        make_fake_zip(tmp_path, "BV1v2", "游戏", prompt_version="v3.0.0")
        zip_paths = [str(p) for p in tmp_path.glob("*.zip")]

        builder = CorpusBuilder()
        with patch("danmaku_analyzer.corpus_builder.logger") as mock_logger:
            builder.build_from_zips(zip_paths, output_dir=str(tmp_path / "out"))
            warn_msgs = [str(c.args[0]) for c in mock_logger.warning.call_args_list]
            assert any("prompt_version" in m for m in warn_msgs)

    def test_missing_metadata_raises_and_skipped(self, tmp_path, tmp_store):
        bad_zip = tmp_path / "bad.zip"
        with zipfile.ZipFile(bad_zip, 'w') as zipf:
            zipf.writestr("other.txt", "x")
        make_fake_zip(tmp_path, "BV1ok", "游戏")

        builder = CorpusBuilder()
        result = builder.build_from_zips([str(bad_zip), str(next(tmp_path.glob("*ok*.zip")))],
                                           output_dir=str(tmp_path / "out"))
        df = pd.read_csv(result.csv_path, encoding='utf-8-sig')
        assert df.iloc[0]["video_count"] == 1

    def test_empty_input_raises(self, tmp_path):
        builder = CorpusBuilder()
        with pytest.raises(ValueError):
            builder.build_from_zips([], output_dir=str(tmp_path))

    def test_legacy_zip_tname_fallback_to_partitions(self, tmp_path):
        """旧版 ZIP 的 metadata 无 tname 透传字段，回退 partitions"""
        zip_path = make_fake_zip(tmp_path, "BV1old", "游戏")
        with zipfile.ZipFile(zip_path, 'r') as zipf:
            metadata = json.loads(zipf.read("metadata.json"))
        metadata.pop("tname")
        # 重写 ZIP 去掉 tname
        with zipfile.ZipFile(zip_path, 'r') as zin:
            items = {n: zin.read(n) for n in zin.namelist()}
        items["metadata.json"] = json.dumps(metadata, ensure_ascii=False).encode('utf-8')
        with zipfile.ZipFile(zip_path, 'w') as zout:
            for name, data in items.items():
                zout.writestr(name, data)

        builder = CorpusBuilder()
        meta, tables = builder.read_zip(zip_path)
        summaries = builder.summarize_video(meta, tables)
        assert summaries[0].tname == "游戏"

    def test_temporal_bucketing(self):
        assert CorpusBuilder._bucket_pubdate("2025-03-15T10:00:00", "year") == "2025"
        assert CorpusBuilder._bucket_pubdate("2025-03-15T10:00:00", "quarter") == "2025-Q1"
        assert CorpusBuilder._bucket_pubdate("2025-12-01T10:00:00", "quarter") == "2025-Q4"
        assert CorpusBuilder._bucket_pubdate("2025-03-15T10:00:00", "month") == "2025-03"
        assert CorpusBuilder._bucket_pubdate("", "year") == "unknown"
        assert CorpusBuilder._bucket_pubdate("not-a-date", "year") == "unknown"

    def test_temporal_grouping_splits_groups(self, tmp_path, tmp_store, monkeypatch):
        make_fake_zip(tmp_path, "BV1t1", "游戏", pubdate="2023-05-01T00:00:00")
        make_fake_zip(tmp_path, "BV1t2", "游戏", pubdate="2025-05-01T00:00:00")
        zip_paths = [str(p) for p in tmp_path.glob("*.zip")]

        builder = CorpusBuilder()
        monkeypatch.setattr(get_settings(), "ENABLE_TEMPORAL_GROUPING", True)
        result = builder.build_from_zips(zip_paths, output_dir=str(tmp_path / "out"))
        df = pd.read_csv(result.csv_path, encoding='utf-8-sig')
        assert len(df) == 2
        assert set(df["time_period"].astype(str)) == {"2023", "2025"}


# ========== 语料库快照打包测试 ==========

class TestCorpusPackaging:

    def test_package_snapshot_contains_all_and_keeps_source(self, tmp_path, tmp_store):
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        make_fake_zip(src_dir, "BV1p1", "游戏")
        make_fake_zip(src_dir, "BV1p2", "音乐")
        zip_paths = [str(p) for p in src_dir.glob("*.zip")]

        builder = CorpusBuilder()
        result = builder.build_from_zips(zip_paths, output_dir=str(tmp_path / "out"))
        zip_path = builder.package_snapshot(result)

        assert result.zip_valid
        assert os.path.basename(zip_path).startswith("[corpus]_2videos_")
        with zipfile.ZipFile(zip_path, 'r') as zipf:
            names = set(zipf.namelist())
        assert {"corpus_summary.csv", "corpus_videos.csv", "corpus_metadata.json"} <= names
        assert "videos/[BV1p1]test.zip" in names and "videos/[BV1p2]test.zip" in names
        # 散落源文件已清理，但源视频 ZIP 原文件一律保留
        out_files = {p.name for p in (tmp_path / "out").iterdir()}
        assert out_files == {os.path.basename(zip_path)}
        assert len(list(src_dir.glob("*.zip"))) == 2

    def test_snapshot_metadata_records_evidence_chain(self, tmp_path, tmp_store):
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        make_fake_zip(src_dir, "BV1m1", "游戏", prompt_version="v2.2.0")
        make_fake_zip(src_dir, "BV1m2", "游戏", prompt_version="v3.0.0")
        zip_paths = [str(p) for p in src_dir.glob("*.zip")]

        builder = CorpusBuilder()
        result = builder.build_from_zips(zip_paths, output_dir=str(tmp_path / "out"))
        builder.package_snapshot(result)

        zip_path = result.zip_path
        with zipfile.ZipFile(zip_path, 'r') as zipf:
            meta = json.loads(zipf.read("corpus_metadata.json"))
        assert meta["video_count"] == 2
        assert meta["bvids"] == ["BV1m1", "BV1m2"]
        assert meta["prompt_versions"] == ["v2.2.0", "v3.0.0"]
        assert meta["zone_policy"] == get_settings().CORPUS_ZONE_POLICY
        assert meta["pipeline_version"]

    def test_package_includes_extra_files(self, tmp_path, tmp_store):
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        make_fake_zip(src_dir, "BV1e1", "游戏")

        builder = CorpusBuilder()
        result = builder.build_from_zips([str(next(src_dir.glob("*.zip")))],
                                          output_dir=str(tmp_path / "out"))
        extra = os.path.join(result.output_dir, "corpus_analysis_report.md")
        with open(extra, 'w', encoding='utf-8') as f:
            f.write("# 报告正文")
        builder.package_snapshot(result, extra_files=[extra])

        assert result.zip_valid
        with zipfile.ZipFile(result.zip_path, 'r') as zipf:
            assert "corpus_analysis_report.md" in zipf.namelist()
            assert zipf.read("corpus_analysis_report.md").decode('utf-8') == "# 报告正文"


class TestCorpusReportPrompt:

    def test_corpus_user_prompt_contains_overview_and_groups(self, tmp_path, tmp_store):
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        make_fake_zip(src_dir, "BV1r1", "游戏", density=0.4)
        make_fake_zip(src_dir, "BV1r2", "音乐", density=0.8)

        builder = CorpusBuilder()
        result = builder.build_from_zips([str(p) for p in src_dir.glob("*.zip")],
                                          output_dir=str(tmp_path / "out"))
        meta = builder.build_snapshot_metadata(result)
        assert meta["video_count"] == 2

        with patch("danmaku_analyzer.llm_factory.AsyncOpenAI"):
            from danmaku_analyzer.report_generator import AnalysisReportGenerator
            gen = AnalysisReportGenerator()
        prompt = gen._build_corpus_user_prompt(result.csv_path, result.videos_csv_path, meta)
        assert "语料库概况" in prompt
        assert "组级聚合数据" in prompt
        assert "BV1r1" in prompt and "BV1r2" in prompt
