import asyncio
import json

import pandas as pd
import pytest
from click.testing import CliRunner
from pydantic import ValidationError

from danmaku_analyzer.config import Settings, get_settings
from danmaku_analyzer.corpus_visualizer import CorpusVisualizer
from danmaku_analyzer.exporter import Exporter
from danmaku_analyzer.pipeline import PipelineOptions, _write_supplementary_reports
from danmaku_analyzer.reporter import Reporter
from danmaku_analyzer.reproducibility import ReproManifestBuilder


@pytest.mark.parametrize("field,value", [("LLM_CONCURRENCY", 0), ("TOP_N", -1),
    ("SCHEDULER_WORKERS", 0), ("MIN_SEGMENT_SAMPLES", 0), ("CONFIDENCE_LEVEL", 1),
    ("MOE", 0), ("SIGNIFICANCE_ALPHA", -0.1)])
def test_invalid_runtime_settings_rejected(tmp_path, field, value):
    with pytest.raises(ValidationError):
        Settings(_env_file=None, DATA_ROOT=str(tmp_path), **{field: value})


@pytest.mark.parametrize("top_n", [0, -1, "ten"])
def test_pipeline_rejects_invalid_sampling_before_network(top_n):
    with pytest.raises(ValueError):
        PipelineOptions(input_str="BVtest", use_top_n=top_n)


def test_runtime_sampling_manifest_matches_methodology(tmp_path, monkeypatch):
    settings = get_settings()
    monkeypatch.setattr(settings, "DATA_ROOT", str(tmp_path))
    monkeypatch.setattr(settings, "TOP_N", 10)
    monkeypatch.setattr(settings, "ENABLE_FREQ_BASED_SAMPLING", False)
    options = PipelineOptions(input_str="BVtest", use_top_n=25, use_freq_based=True)
    reports = _write_supplementary_reports(Reporter(str(tmp_path)), {}, options, lambda *args: None)
    with open(reports["repro_manifest"], encoding="utf-8") as stream:
        snapshot = json.load(stream)["config_snapshot"]
    assert snapshot["TOP_N"] == 25
    assert snapshot["ENABLE_FREQ_BASED_SAMPLING"] is True
    with open(reports["methodology"], encoding="utf-8") as stream:
        text = stream.read()
    assert "25" in text and "按频次排序" in text
    assert settings.TOP_N == 10


def test_manifest_override_cannot_add_secret_fields():
    with pytest.raises(ValueError, match="白名单"):
        ReproManifestBuilder().build({"COMPLEX_LLM_API_KEY": "fake-secret"})


def test_cli_rejects_complete_label_failure_with_valid_archive(monkeypatch):
    import danmaku_analyzer.cli as module
    import danmaku_analyzer.pipeline as pipeline
    from types import SimpleNamespace

    async def failed(*args, **kwargs):
        return SimpleNamespace(zip_valid=True, zip_path="retained.zip", analysis_status="failed")

    monkeypatch.setattr(pipeline, "analyze_video", failed)
    with pytest.raises(RuntimeError, match="retained.zip"):
        asyncio.run(module._analyze_async("BVtest", None, None))


def test_cli_rejects_nonpositive_top_n():
    from danmaku_analyzer.cli import cli

    result = CliRunner().invoke(cli, ["analyze", "BVtest", "--top-n", "0"])
    assert result.exit_code == 2


def test_apa_preserves_strata_for_same_metric():
    rows = [{"metric": "content_word_density", "test_type": "Kruskal-Wallis",
             "group1": "", "group2": "", "statistic": 3.0, "p_value": 0.1,
             "effect_size": 0.2, "note": "未校正 p 值；冷热区分层：" + zone}
            for zone in ["hot_zone", "cold_zone"]]
    text = Exporter().stats_to_apa(pd.DataFrame(rows))
    assert "hot_zone" in text and "cold_zone" in text


def test_visualizers_consume_summary_and_keep_stratum_keys():
    visualizer = CorpusVisualizer()
    python = visualizer.render_python_script()
    r = visualizer.render_r_script()
    compile(python, "corpus_plots.py", "exec")
    for text in [python, r]:
        assert "corpus_summary.csv" in text
        assert "冷热区分层：" in text
    assert 'kw_labels[(row["metric"], zone)]' in python
    assert "paste(kw$metric, kw_zone" in r


def test_generated_python_visualizer_runs_offline(tmp_path):
    import os
    import subprocess
    import sys

    pytest.importorskip("matplotlib")
    pytest.importorskip("seaborn")
    videos = []
    for partition, baseline in [("game", 0.2), ("music", 0.6)]:
        for index in range(3):
            for zone, shift in [("hot_zone", 0), ("cold_zone", 0.1)]:
                videos.append({"bvid": f"{partition}{index}", "tname": partition,
                               "zone_type": zone, "danmaku_count": 30,
                               "content_word_density": baseline + index * 0.01 + shift})
    pd.DataFrame(videos).to_csv(tmp_path / "corpus_videos.csv", index=False)
    pd.DataFrame([{"tname": partition, "zone_type": zone, "positive_mean": 0.2,
                   "negative_mean": 0.8, "content_word_density_mean": 0.4}
                  for partition in ["game", "music"] for zone in ["hot_zone", "cold_zone"]]).to_csv(
                      tmp_path / "corpus_summary.csv", index=False)
    pd.DataFrame([{"metric": "content_word_density", "test_type": "Kruskal-Wallis",
                   "p_value": p, "note": "冷热区分层：" + zone}
                  for zone, p in [("hot_zone", 0.01), ("cold_zone", 0.3)]]).to_csv(
                      tmp_path / "statistical_tests.csv", index=False)
    script = CorpusVisualizer().write_python_script(str(tmp_path))
    result = subprocess.run([sys.executable, script], cwd=tmp_path,
                            env={**os.environ, "MPLCONFIGDIR": str(tmp_path / "mpl")},
                            capture_output=True, timeout=60)
    assert result.returncode == 0, result.stderr.decode("utf-8", errors="replace")
    for name in ["corpus_boxplots.png", "corpus_distributions.png"]:
        assert (tmp_path / name).stat().st_size > 0
