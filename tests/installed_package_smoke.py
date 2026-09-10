"""从源码目录外验证已安装产物：包来源、随包资源、版本还原与控制台入口。

设计为在隔离解释器（python -I）、已安装环境、源码目录外运行：断言包来自安装环境
而非源码树，lexicon 与 .env.example 随 wheel 分发，__version__ 未回退为 dev，
两个控制台脚本可用。DS 无离线 mock LLM 后端，故不跑完整分析流程（导入/资源/入口级冒烟）。
"""

import importlib.metadata as importlib_metadata
import importlib.resources as resources
import os
import subprocess
import sys
from pathlib import Path


def run_smoke() -> None:
    import danmaku_analyzer

    package_path = Path(danmaku_analyzer.__file__).resolve()
    assert package_path.is_relative_to(Path(sys.prefix).resolve()), package_path
    assert not (Path.cwd() / "danmaku_analyzer").exists(), "cwd 含源码目录，可能误导入源码树"

    assert importlib_metadata.version("danmaku-analyzer"), "danmaku-analyzer 分发元数据缺失"
    assert danmaku_analyzer.__version__ != "0.0.0-dev", "版本回退为 dev，说明包未从安装环境导入"

    lexicon = resources.files("danmaku_analyzer") / "lexicon"
    assert any(p.name.endswith(".txt") for p in lexicon.iterdir()), "lexicon/*.txt 未随包分发"
    assert (lexicon / "report_spec.md").is_file(), "lexicon/report_spec.md 未随包分发"
    assert (resources.files("danmaku_analyzer") / ".env.example").is_file(), ".env.example 未随包分发"

    # 轻量配置模块与触发重型依赖链（jieba/pandas/openai/bilibili_api）的 pipeline 均须干净导入
    from danmaku_analyzer import config, pipeline, report_schema  # noqa: F401

    scripts = Path(sys.executable).parent
    for name in ("danmaku-analyzer", "danmaku-tui"):
        executable = scripts / (f"{name}.exe" if os.name == "nt" else name)
        assert executable.is_file(), f"控制台入口缺失: {executable}"
    cli = scripts / ("danmaku-analyzer.exe" if os.name == "nt" else "danmaku-analyzer")
    subprocess.run([str(cli), "--help"], check=True, capture_output=True)

    print(f"Installed package smoke passed: {package_path}")


run_smoke()
