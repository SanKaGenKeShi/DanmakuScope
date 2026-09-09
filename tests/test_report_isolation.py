"""报告归档与逐样本失败标记的离线隔离回归。"""

import asyncio
import csv
import io
import json
import os
import struct
import zipfile
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest


BVID_A = "BV1aaaaaaaaa"
BVID_B = "BV1bbbbbbbbb"
CORE_FILES = (
    "metadata.json", "table_lexical_by_partition.csv", "table_orthography.csv",
    "table_sentence_function.csv", "table_emotion.csv", "table_interaction_type.csv",
    "table_consensus_stats.csv", "heatmap_data.json", "danmaku_raw.csv", "kappa_ready.csv",
)


@pytest.fixture(autouse=True)
def isolated_runtime(tmp_path, monkeypatch):
    from pydantic_settings import DotEnvSettingsSource, EnvSettingsSource

    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    monkeypatch.setattr(DotEnvSettingsSource, "_read_env_files", lambda self: {})
    monkeypatch.setattr(EnvSettingsSource, "_load_env_vars", lambda self: {})
    import danmaku_analyzer.config as config
    import danmaku_analyzer.llm_config as llm_config

    settings = config.Settings(_env_file=None, DATA_ROOT=str(tmp_path))
    llm_settings = llm_config.LLMSettings(_env_file=None)
    monkeypatch.setattr(config, "_settings", settings)
    monkeypatch.setattr(llm_config, "llm_settings", llm_settings)
    return settings


def _archive_contents(bvid=BVID_A, status="ok"):
    files = {name: "tname,zone_type,danmaku_count\n游戏,hot_zone,6\n" for name in CORE_FILES}
    files["metadata.json"] = json.dumps({
        "bvid": bvid, "analysis_status": status, "analysis_sample_count": 6,
    })
    files["heatmap_data.json"] = "{}"
    return files


def _write_archive(path, contents):
    with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as archive:
        for name, content in contents.items():
            archive.writestr(name, content)


def _write_sources(directory, contents):
    directory.mkdir()
    reports = {}
    for name, content in contents.items():
        path = directory / name
        path.write_text(content, encoding="utf-8")
        reports[name] = str(path)
    return reports


def _corrupt_member(path, name):
    with zipfile.ZipFile(path) as archive:
        offset = archive.getinfo(name).header_offset
    data = bytearray(path.read_bytes())
    filename_size, extra_size = struct.unpack_from("<HH", data, offset + 26)
    data[offset + 30 + filename_size + extra_size] ^= 0xFF
    path.write_bytes(data)


@pytest.mark.parametrize("missing", CORE_FILES)
def test_strict_archive_rejects_missing_core_file(tmp_path, missing):
    from danmaku_analyzer.report_archive import ReportArchive

    contents = _archive_contents()
    contents.pop(missing)
    path = tmp_path / "incomplete.zip"
    _write_archive(path, contents)
    assert not ReportArchive(str(path), BVID_A).validate()


@pytest.mark.parametrize("metadata", ["[]", "{broken", json.dumps({"bvid": BVID_B})])
def test_strict_archive_rejects_invalid_identity(tmp_path, metadata):
    from danmaku_analyzer.report_archive import ReportArchive

    contents = _archive_contents()
    contents["metadata.json"] = metadata
    path = tmp_path / "wrong.zip"
    _write_archive(path, contents)
    assert not ReportArchive(str(path), BVID_A).validate()


def test_archive_checks_crc_of_optional_last_member(tmp_path):
    from danmaku_analyzer.report_archive import ReportArchive

    contents = {**_archive_contents(), "optional.md": "damaged-last-member"}
    path = tmp_path / "corrupt.zip"
    _write_archive(path, contents)
    _corrupt_member(path, "optional.md")
    with zipfile.ZipFile(path) as archive:
        assert json.loads(archive.read("metadata.json"))["bvid"] == BVID_A
    assert not ReportArchive(str(path), BVID_A).validate()
    assert not ReportArchive(str(path)).validate()


def test_archive_rejects_duplicate_members(tmp_path):
    from danmaku_analyzer.report_archive import ReportArchive

    path = tmp_path / "duplicates.zip"
    _write_archive(path, _archive_contents())
    with pytest.warns(UserWarning, match="Duplicate name"):
        with zipfile.ZipFile(path, "a") as archive:
            archive.writestr("metadata.json", json.dumps({"bvid": BVID_A}))
    assert not ReportArchive(str(path), BVID_A).validate()


@pytest.mark.parametrize("missing_from_listing", [True, False])
def test_missing_report_cannot_hide_behind_existing_file_count(tmp_path, missing_from_listing):
    from danmaku_analyzer.pipeline import _package_reports_zip

    directory = tmp_path / "attempt"
    reports = _write_sources(directory, _archive_contents())
    Path(reports["metadata.json"]).unlink()
    if missing_from_listing:
        reports.pop("metadata.json")
    target = tmp_path / "out.zip"
    messages = []
    assert not _package_reports_zip(
        reports, str(target), lambda stage, message: messages.append(message),
        expected_bvid=BVID_A, source_dir=str(directory),
    )
    assert not target.exists()
    assert (directory / "danmaku_raw.csv").exists()
    assert any(str(directory) in message for message in messages)


@pytest.mark.parametrize("failure", ["write", "replace", "crc", "identity", "missing", "failed_analysis"])
def test_packaging_failure_preserves_previous_success_and_sources(tmp_path, monkeypatch, failure):
    from danmaku_analyzer.pipeline import _package_reports_zip
    from danmaku_analyzer.report_archive import ReportArchive

    target = tmp_path / "out.zip"
    _write_archive(target, _archive_contents())
    previous = target.read_bytes()
    contents = _archive_contents(BVID_B if failure == "identity" else BVID_A)
    if failure == "missing":
        contents.pop("metadata.json")
    if failure == "failed_analysis":
        contents = _archive_contents(status="failed")
    directory = tmp_path / "attempt"
    reports = _write_sources(directory, contents)

    def denied(*args, **kwargs):
        raise OSError("offline simulated disk failure")

    if failure == "write":
        monkeypatch.setattr(zipfile.ZipFile, "write", denied)
    elif failure == "replace":
        monkeypatch.setattr(os, "replace", denied)
    elif failure == "crc":
        monkeypatch.setattr(zipfile.ZipFile, "testzip", lambda self: "danmaku_raw.csv")
    messages = []
    assert not _package_reports_zip(
        reports, str(target), lambda stage, message: messages.append(message),
        expected_bvid=BVID_A, source_dir=str(directory),
    )
    assert target.read_bytes() == previous
    assert all(Path(path).exists() for path in reports.values())
    staged = list(tmp_path.glob(".out.zip.*.tmp"))
    assert len(staged) == 1
    assert any(str(staged[0]) in message for message in messages)
    if failure != "crc":
        assert ReportArchive(str(target), BVID_A).validate(require_completed=True)


def test_atomic_package_only_replaces_target_after_validation(tmp_path, monkeypatch):
    from danmaku_analyzer.pipeline import _package_reports_zip
    from danmaku_analyzer.report_archive import ReportArchive

    target = tmp_path / "out.zip"
    _write_archive(target, _archive_contents())
    previous = target.read_bytes()
    directory = tmp_path / "attempt"
    contents = {**_archive_contents(), "new.md": "new report"}
    reports = _write_sources(directory, contents)
    sibling = tmp_path / "other-attempt"
    sibling.mkdir()
    sentinel = sibling / "metadata.json"
    sentinel.write_text("other video", encoding="utf-8")
    observed = []
    original_replace = os.replace

    def checked_replace(source, destination):
        assert target.read_bytes() == previous
        assert Path(source) != target
        assert ReportArchive(str(source), BVID_A).validate(require_completed=True)
        observed.append(source)
        original_replace(source, destination)

    monkeypatch.setattr(os, "replace", checked_replace)
    assert _package_reports_zip(
        reports, str(target), lambda stage, message: None,
        expected_bvid=BVID_A, source_dir=str(directory),
    )
    assert len(observed) == 1
    assert not directory.exists()
    assert sentinel.read_text(encoding="utf-8") == "other video"
    assert not list(tmp_path.glob(".out.zip.*.tmp"))


@pytest.mark.parametrize("status", ["ok", "degraded", "failed"])
def test_resume_and_reuse_require_complete_matching_nonfailed_archive(tmp_path, status):
    from danmaku_analyzer.pipeline import (
        CompareItem, _find_reusable_zip, _restore_recovered_items, _try_resume_item,
    )
    from danmaku_analyzer.scheduler import ScheduledTask

    target = tmp_path / "out.zip"
    _write_archive(target, _archive_contents(status=status))
    stored = {"input": BVID_A, "bvid": BVID_A, "zip_path": str(target), "status": "ok"}
    index = {BVID_A: stored}
    store = SimpleNamespace(
        get_videos=lambda: [stored], resolve_zip_path=lambda path: path,
    )
    item = CompareItem(raw_input=BVID_A)
    expected = status != "failed"
    assert _try_resume_item(item, index, BVID_A, lambda *args: None, 1, 1) == expected
    assert bool(_find_reusable_zip(store, BVID_A)) == expected
    task = ScheduledTask(input=BVID_A, bvid=BVID_A, status="done", zip_path=str(target))
    restored = CompareItem(raw_input=BVID_A)
    _restore_recovered_items(SimpleNamespace(tasks=[task]), {BVID_A: restored}, lambda *args: None)
    assert restored.ok == expected
    assert task.status == ("done" if expected else "pending")


@pytest.mark.parametrize("damage", ["identity", "crc", "missing"])
def test_invalid_old_archive_is_never_restored_or_reused(tmp_path, damage):
    from danmaku_analyzer.pipeline import (
        CompareItem, _find_reusable_zip, _restore_recovered_items, _try_resume_item,
    )
    from danmaku_analyzer.scheduler import ScheduledTask

    contents = _archive_contents(BVID_B if damage == "identity" else BVID_A)
    if damage == "missing":
        contents.pop("danmaku_raw.csv")
    path = tmp_path / "bad.zip"
    _write_archive(path, contents)
    if damage == "crc":
        _corrupt_member(path, "kappa_ready.csv")
    stored = {"bvid": BVID_A, "zip_path": str(path), "status": "ok"}
    store = SimpleNamespace(get_videos=lambda: [stored], resolve_zip_path=lambda value: value)
    item = CompareItem(raw_input=BVID_A)
    assert not _try_resume_item(item, {BVID_A: stored}, BVID_A, lambda *args: None, 1, 1)
    assert _find_reusable_zip(store, BVID_A) == ""
    task = ScheduledTask(input=BVID_A, bvid=BVID_A, status="done", zip_path=str(path))
    _restore_recovered_items(SimpleNamespace(tasks=[task]), {BVID_A: item}, lambda *args: None)
    assert task.status == "pending"
    assert not item.ok


@pytest.fixture
def offline_pipeline(monkeypatch, isolated_runtime):
    import danmaku_analyzer.llm_client as llm_module
    import danmaku_analyzer.pipeline as pipeline
    from danmaku_analyzer.crawler import DanmakuItem, VideoMeta
    from danmaku_analyzer.hard_metrics import HardMetricsResult
    from danmaku_analyzer.llm_models import ConsensusLevel, DualPathResult, LLMOutput
    from danmaku_analyzer.social_variables import SocialVariables
    from danmaku_analyzer.timeline_segmenter import TimeSegment
    from danmaku_analyzer.user_deduplicator import DeduplicationResult

    monkeypatch.setattr(llm_module, "complex_backend", lambda **kwargs: object())
    monkeypatch.setattr(llm_module, "simple_backend", lambda **kwargs: object())
    isolated_runtime.ENABLE_LLM_ANALYSIS_REPORT = False

    async def crawl(bvid, options):
        items = [
            DanmakuItem(uid_hash=f"{bvid}-u{i}", content=f"{bvid}-sample-{i}",
                        time_sec=float(i), identity_type="real_user")
            for i in range(6)
        ]
        meta = VideoMeta(
            bvid=bvid, title=f"title-{bvid}", tname="游戏", tags=[bvid],
            pubdate=datetime(2025, 1, 1), cid=1,
        )
        return pipeline.CrawlOutput(meta=meta, danmaku_list=items)

    async def preprocess(crawled, progress):
        items = crawled.danmaku_list
        return pipeline.PreprocessOutput(
            social_vars=SocialVariables(tname="游戏", tags=crawled.meta.tags),
            dedup_result=DeduplicationResult(items, 6, 6, 0, 0),
            segments=[
                TimeSegment(0.0, 3.0, [0, 1, 2], 1.0, "hot_zone"),
                TimeSegment(3.0, 6.0, [3, 4, 5], 1.0, "hot_zone"),
            ],
        )

    class OfflineHardMetrics:
        async def analyze_async(self, texts):
            return HardMetricsResult(
                pos_distribution={"n": 1.0}, syllable_distribution={"single": 1.0},
                avg_word_length=2.0, content_word_density=1.0, punctuation_emoji_rate=0.0,
                orthography_hard_metrics={"uppercase_abbr_per_1000": 0.0},
                total_danmaku_count=len(texts), total_word_count=len(texts),
                total_char_count=len(texts) * 2,
            )

    class OfflineContext:
        def get_context(self, danmaku, segment, segment_danmaku):
            return SimpleNamespace(to_prompt_text=lambda: danmaku.content)

    async def analyze(self, complex_prompt, simple_prompt):
        output = LLMOutput.default()
        output.emotion.label = "positive" if BVID_A in complex_prompt.user_prompt else "negative"
        return DualPathResult(
            output, ConsensusLevel.HIGH, 0.0, 1.0, [output.to_dict()], "offline-test",
        )

    monkeypatch.setattr(pipeline, "_stage_crawl", crawl)
    monkeypatch.setattr(pipeline, "_stage_preprocess", preprocess)
    monkeypatch.setattr(pipeline, "HardMetricsAnalyzer", OfflineHardMetrics)
    monkeypatch.setattr(pipeline, "ContextProvider", OfflineContext)
    monkeypatch.setattr(llm_module.LLMClient, "analyze", analyze)
    return pipeline


def test_two_video_llm_report_awaits_cannot_cross_contaminate(tmp_path, monkeypatch, offline_pipeline):
    pipeline = offline_pipeline
    pipeline.get_settings().ENABLE_LLM_ANALYSIS_REPORT = True
    user_file = tmp_path / "metadata.json"
    user_file.write_text("pre-existing user report", encoding="utf-8")
    directories = {}

    async def exercise():
        a_waiting, b_waiting, a_packaged = asyncio.Event(), asyncio.Event(), asyncio.Event()

        async def report(self, aggregated, metadata=None):
            bvid = metadata["bvid"]
            directories[bvid] = Path(self.output_dir)
            if bvid == BVID_A:
                a_waiting.set()
                await b_waiting.wait()
            else:
                await a_waiting.wait()
                b_waiting.set()
                await a_packaged.wait()
            path = Path(self.output_dir) / "sociolinguistic_analysis_report.md"
            path.write_text(f"report-for-{bvid}", encoding="utf-8")
            return str(path)

        monkeypatch.setattr(pipeline.Reporter, "generate_llm_analysis_report", report)

        async def run_video(bvid):
            result = await pipeline.analyze_video(
                bvid, output_dir=str(tmp_path), no_cache=True, progress_callback=lambda *args: None,
            )
            if bvid == BVID_A:
                assert not directories[BVID_A].exists()
                assert directories[BVID_B].exists()
                assert json.loads((directories[BVID_B] / "metadata.json").read_text("utf-8"))["bvid"] == BVID_B
                a_packaged.set()
            return result

        return await asyncio.wait_for(asyncio.gather(run_video(BVID_A), run_video(BVID_B)), timeout=20)

    results = asyncio.run(exercise())
    assert directories[BVID_A] != directories[BVID_B]
    for result, bvid, emotion in zip(results, (BVID_A, BVID_B), ("positive", "negative")):
        assert result.zip_valid and result.analysis_status == "ok"
        assert Path(result.zip_path).parent == tmp_path
        assert Path(result.zip_path).name == f"[{bvid}]title-{bvid}.zip"
        with zipfile.ZipFile(result.zip_path) as archive:
            assert archive.testzip() is None
            metadata = json.loads(archive.read("metadata.json"))
            assert metadata["bvid"] == bvid and metadata["tags"] == [bvid]
            for name in ("danmaku_raw.csv", "kappa_ready.csv"):
                rows = list(csv.DictReader(io.StringIO(archive.read(name).decode("utf-8-sig"))))
                text_field = "content" if name == "danmaku_raw.csv" else "raw_text"
                assert [row[text_field] for row in rows] == [f"{bvid}-sample-{i}" for i in range(6)]
            assert archive.read("sociolinguistic_analysis_report.md").decode("utf-8") == f"report-for-{bvid}"
            assert emotion in archive.read("table_emotion.csv").decode("utf-8-sig")
        assert not directories[bvid].exists()
    assert user_file.read_text(encoding="utf-8") == "pre-existing user report"


@pytest.mark.parametrize("batch_mode", [False, True])
def test_per_sample_exception_preserves_other_successes_and_alignment(monkeypatch, offline_pipeline, batch_mode):
    pipeline = offline_pipeline
    pipeline.get_settings().ENABLE_BATCH_SEGMENT_ANALYSIS = batch_mode
    original_analyze = pipeline.LLMClient.analyze

    async def partial_failure(self, complex_prompt, simple_prompt):
        if f"{BVID_A}-sample-1" in simple_prompt.user_prompt:
            raise RuntimeError("offline sample failure")
        return await original_analyze(self, complex_prompt, simple_prompt)

    async def batch_failure(self, *args):
        raise ValueError("offline batch failure")

    monkeypatch.setattr(pipeline.LLMClient, "analyze", partial_failure)
    monkeypatch.setattr(pipeline.LLMClient, "analyze_batch", batch_failure)

    async def exercise():
        options = pipeline.PipelineOptions(input_str=BVID_A)
        crawl = await pipeline._stage_crawl(BVID_A, options)
        pre = await pipeline._stage_preprocess(crawl, lambda *args: None)
        return await pipeline._stage_analyze_segments(pre, options)

    analysis = asyncio.run(exercise())
    records = pipeline._build_kappa_records(analysis)
    assert len(records) == len(analysis.sample_danmaku) == 6
    assert [record["raw_text"] for record in records] == [f"{BVID_A}-sample-{i}" for i in range(6)]
    assert [record["analysis_status"] for record in records] == ["ok", "failed", "ok", "ok", "ok", "ok"]
    assert all(value is None for value in records[1]["llm_output"].values())
    assert records[1]["requested_paths"] == 2 and records[1]["successful_paths"] == 0
    assert records[1]["sentence_function_source"] == "unavailable"
    assert records[1]["weight_multiplier"] == pytest.approx(0.2)
    assert analysis.status_summary == {
        "analysis_status": "degraded", "analysis_sample_count": 6,
        "failed_sample_count": 1, "degraded_sample_count": 0, "valid_label_sample_count": 5,
    }
    assert [record.segment_id for record in analysis.records] == [0, 0, 0, 1, 1, 1]


def test_all_failed_labels_still_archive_complete_raw_and_hard_metrics(tmp_path, monkeypatch, offline_pipeline):
    pipeline = offline_pipeline
    from danmaku_analyzer.report_archive import ReportArchive

    async def failed(self, *args):
        raise RuntimeError("offline total failure")

    monkeypatch.setattr(pipeline.LLMClient, "analyze", failed)
    result = asyncio.run(pipeline.analyze_video(
        BVID_A, output_dir=str(tmp_path), top_n=2, no_cache=True, progress_callback=lambda *args: None,
    ))
    assert result.zip_valid and result.analysis_status == "failed"
    assert ReportArchive(result.zip_path, BVID_A).validate()
    assert not ReportArchive(result.zip_path, BVID_A).validate(require_completed=True)
    with zipfile.ZipFile(result.zip_path) as archive:
        metadata = json.loads(archive.read("metadata.json"))
        assert metadata["analysis_status"] == "failed"
        assert metadata["failed_sample_count"] == metadata["analysis_sample_count"] == 4
        assert metadata["degraded_sample_count"] == metadata["valid_label_sample_count"] == 0
        raw = list(csv.DictReader(io.StringIO(archive.read("danmaku_raw.csv").decode("utf-8-sig"))))
        labels = list(csv.DictReader(io.StringIO(archive.read("kappa_ready.csv").decode("utf-8-sig"))))
        hard = list(csv.DictReader(io.StringIO(archive.read("table_lexical_by_partition.csv").decode("utf-8-sig"))))
        assert len(raw) == 6 and len(labels) == 4
        assert [row["raw_text"] for row in labels] == [f"{BVID_A}-sample-{i}" for i in (0, 1, 3, 4)]
        assert all(row["analysis_status"] == "failed" and row["emotion_label"] == "" for row in labels)
        assert len(hard) == 1 and int(hard[0]["danmaku_count"]) == 6
        assert float(hard[0]["avg_word_length"]) == 2.0
        assert "sociolinguistic_analysis_report.md" not in archive.namelist()


def test_kappa_records_reject_misalignment(offline_pipeline):
    pipeline = offline_pipeline
    with pytest.raises(ValueError, match="一一对应"):
        pipeline._build_kappa_records(pipeline.SegmentAnalysisOutput(sample_danmaku=[object()]))


def test_metadata_counts_degraded_and_failed_samples(tmp_path, monkeypatch, offline_pipeline):
    pipeline = offline_pipeline
    original_analyze = pipeline.LLMClient.analyze

    async def mixed(self, complex_prompt, simple_prompt):
        if f"{BVID_A}-sample-1" in simple_prompt.user_prompt:
            raise RuntimeError("offline failed sample")
        result = await original_analyze(self, complex_prompt, simple_prompt)
        if f"{BVID_A}-sample-2" in simple_prompt.user_prompt:
            result.analysis_status = "degraded"
            result.successful_paths = 1
            result.sentence_function_source = "simple"
        return result

    monkeypatch.setattr(pipeline.LLMClient, "analyze", mixed)
    result = asyncio.run(pipeline.analyze_video(
        BVID_A, output_dir=str(tmp_path), no_cache=True, progress_callback=lambda *args: None,
    ))
    assert result.zip_valid and result.analysis_status == "degraded"
    with zipfile.ZipFile(result.zip_path) as archive:
        metadata = json.loads(archive.read("metadata.json"))
        assert metadata["analysis_status"] == "degraded"
        assert metadata["failed_sample_count"] == metadata["degraded_sample_count"] == 1
        assert metadata["valid_label_sample_count"] == 5
        rows = list(csv.DictReader(io.StringIO(archive.read("kappa_ready.csv").decode("utf-8-sig"))))
        assert rows[2]["analysis_status"] == "degraded"
        assert rows[2]["successful_paths"] == "1"
        assert rows[2]["sentence_function_source"] == "simple"


@pytest.mark.parametrize("interruption", [OSError, asyncio.CancelledError])
def test_report_generation_interruption_keeps_own_workspace(tmp_path, monkeypatch, offline_pipeline, interruption):
    pipeline = offline_pipeline
    locations = []
    messages = []

    def interrupted(self, *args):
        locations.append(Path(self.output_dir))
        raise interruption("offline interrupted report")

    monkeypatch.setattr(pipeline.Reporter, "generate_raw_danmaku", interrupted)
    with pytest.raises(interruption):
        asyncio.run(pipeline.analyze_video(
            BVID_A, output_dir=str(tmp_path), no_cache=True,
            progress_callback=lambda stage, message: messages.append(message),
        ))
    assert len(locations) == 1 and locations[0].is_dir()
    assert (locations[0] / "metadata.json").is_file()
    assert any(str(locations[0]) in message for message in messages)
    assert not list(tmp_path.glob("*.zip"))


def test_compare_failure_archive_is_not_recorded_as_completed(tmp_path, monkeypatch, offline_pipeline):
    pipeline = offline_pipeline
    from danmaku_analyzer.scheduler import ScheduledTask

    async def failed(self, *args):
        raise RuntimeError("offline total failure")

    def reject_completion(*args, **kwargs):
        pytest.fail("failed analysis must not be recorded as completed")

    monkeypatch.setattr(pipeline.LLMClient, "analyze", failed)
    monkeypatch.setattr(pipeline, "_append_progress", reject_completion)
    task = ScheduledTask(input=BVID_A)
    item = pipeline.CompareItem(raw_input=BVID_A)
    with pytest.raises(RuntimeError, match="LLM 分析失败"):
        asyncio.run(pipeline._process_compare_item(
            task, item, SimpleNamespace(), {}, 1, 1,
            reuse=False, resume=False, output_dir=str(tmp_path), progress=lambda *args: None,
        ))
    assert not item.ok and Path(item.zip_path).is_file()
    assert task.status not in {"done", "reused"}
    assert task.zip_path
