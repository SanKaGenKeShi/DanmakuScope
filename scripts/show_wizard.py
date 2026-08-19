r"""重置首启向导（OOBE）标记并拉起 TUI — 向导界面测试用

用法：.venv\Scripts\python.exe scripts\show_wizard.py
"""
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from danmaku_analyzer.prefs import delete_pref
from danmaku_analyzer.tui.app import run_tui

if delete_pref("wizard_completed"):
    print("已重置 wizard_completed 标记，启动后将弹出首启向导")
else:
    print("向导标记不存在（首启状态），启动后将弹出首启向导")

run_tui()
