# -*- mode: python ; coding: utf-8 -*-
# DanmakuScope 打包规格（跨平台）：产出 CLI + TUI 双入口
#   pyinstaller scripts/danmakuscope.spec --distpath Release --workpath Release/build --noconfirm
import os
import platform
import sys

from PyInstaller.utils.hooks import collect_all, collect_data_files, collect_submodules

# SPECPATH 是 spec 文件所在目录（scripts/）
# PROJECT_ROOT 是项目根目录
PROJECT_ROOT = os.path.abspath(os.path.join(SPECPATH, ".."))
sys.path.insert(0, PROJECT_ROOT)
from danmaku_analyzer import __version__

_PLATFORM = {"win32": "windows", "darwin": "macos"}.get(sys.platform, sys.platform)
_ARCH = {"AMD64": "x64", "x86_64": "x64", "ARM64": "arm64", "aarch64": "arm64"}.get(
    platform.machine(), platform.machine().lower()
)
_NAME_PREFIX = f"DanmakuScope-{__version__}-{_PLATFORM}-{_ARCH}"

# 惰性导入重依赖：整包收集（子模块 + 数据文件 + 动态库）
datas, binaries, hiddenimports = [], [], []
for pkg in ("bilibili_api", "textual", "jieba", "tiktoken", "tiktoken_ext", "emoji", "qrcode", "certifi"):
    try:
        d, b, h = collect_all(pkg)
        datas += d
        binaries += b
        hiddenimports += h
    except Exception:
        pass

# 项目自带数据：lexicon（词典 + report_spec.md + methodology_template.md）
datas += collect_data_files("danmaku_analyzer", includes=["lexicon/**"])

# 项目全部子模块（含函数内惰性导入的模块路径）
hiddenimports += collect_submodules("danmaku_analyzer")
hiddenimports += [
    "openai", "scipy", "scipy.stats", "scipy.special",
    "yaml", "tenacity", "ruptures", "httpx", "httpcore",
    "pandas", "numpy", "click", "rich", "loguru",
    "pydantic", "pydantic_settings", "dotenv", "regex",
]

# 图标：Windows 用 .ico，macOS 需 .icns（不存在时不打图标）
if sys.platform == "darwin":
    _icon_path = os.path.join(PROJECT_ROOT, "assets", "DanmakuScope.icns")
    _ICON = _icon_path if os.path.exists(_icon_path) else None
else:
    _ICON = os.path.join(PROJECT_ROOT, "assets", "DanmakuScope.ico")


def build_exe(entry, component):
    a = Analysis(
        [os.path.join(SPECPATH, entry)],
        pathex=[PROJECT_ROOT],
        datas=datas,
        binaries=binaries,
        hiddenimports=hiddenimports,
        hookspath=[],
        runtime_hooks=[],
        excludes=[],
        noarchive=False,
    )
    pyz = PYZ(a.pure)
    return EXE(
        pyz,
        a.scripts,
        a.binaries,
        a.datas,
        [],
        name=f"{_NAME_PREFIX}-{component}",
        debug=False,
        strip=False,
        upx=False,
        console=True,
        icon=_ICON,
    )


exe_tui = build_exe("entry_tui.py", "tui")
exe_cli = build_exe("entry_cli.py", "cli")
