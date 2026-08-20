"""TUI pilot 测试 - 首启向导与设置页关键路径（Textual run_test 离线覆盖）

隔离策略：DATA_ROOT 指向 tmp_path（偏好/.env 落盘不污染真实用户数据），
resolve_credential patch 为未登录（避免 bilibili_api 冷导入与真实凭证读取），
预载 worker 置空（重依赖已由其余测试导入，预载属 UX 优化非被测逻辑）。
"""

import pytest

from danmaku_analyzer.config import get_settings
from danmaku_analyzer.prefs import load_prefs, save_prefs
from danmaku_analyzer.tui.app import DanmakuTUI
from danmaku_analyzer.tui.screens.wizard import FirstRunWizardScreen

_SIZE = (120, 40)


async def _click_wizard_button(pilot, button_id: str) -> None:
    """对话框内容可能高于可视区（overflow-y 承接滚动），点击前先无动画滚到底部确保按钮可见"""
    screen = pilot.app.screen
    dialog = screen.query_one("#wizard-dialog")
    dialog.scroll_to(y=dialog.virtual_size.height, animate=False)
    await pilot.pause()
    await pilot.click(button_id)
    await pilot.pause()


@pytest.fixture(autouse=True)
def _skip_preload(monkeypatch):
    async def _noop(self):
        return None

    monkeypatch.setattr(DanmakuTUI, "_preload_heavy_modules", _noop)


@pytest.fixture
def isolated_data_root(tmp_path, monkeypatch):
    monkeypatch.setattr(get_settings(), "DATA_ROOT", str(tmp_path))
    return tmp_path


@pytest.fixture
def no_credential(monkeypatch):
    import danmaku_analyzer.account as account

    monkeypatch.setattr(account, "resolve_credential", lambda *args, **kwargs: (None, ""))


@pytest.mark.asyncio
async def test_wizard_skip_marks_completed(isolated_data_root, no_credential):
    async with DanmakuTUI().run_test(size=_SIZE) as pilot:
        assert isinstance(pilot.app.screen, FirstRunWizardScreen)
        await _click_wizard_button(pilot, "#btn-wizard-skip")
    assert load_prefs().get("wizard_completed") is True


@pytest.mark.asyncio
async def test_wizard_done_writes_llm_env(isolated_data_root, no_credential):
    async with DanmakuTUI().run_test(size=_SIZE) as pilot:
        assert isinstance(pilot.app.screen, FirstRunWizardScreen)
        await _click_wizard_button(pilot, "#btn-wizard-done")
    assert load_prefs().get("wizard_completed") is True
    assert (isolated_data_root / ".env").exists()


@pytest.mark.asyncio
async def test_wizard_not_pushed_when_completed(isolated_data_root, no_credential):
    save_prefs({"wizard_completed": True})
    async with DanmakuTUI().run_test(size=_SIZE) as pilot:
        await pilot.pause()
        assert not isinstance(pilot.app.screen, FirstRunWizardScreen)


@pytest.mark.asyncio
async def test_settings_open_and_cancel(isolated_data_root, no_credential):
    from danmaku_analyzer.tui.screens.settings import SettingsScreen

    save_prefs({"wizard_completed": True})
    async with DanmakuTUI().run_test(size=_SIZE) as pilot:
        pilot.app.action_settings()
        await pilot.pause()
        assert isinstance(pilot.app.screen, SettingsScreen)
        await pilot.click("#btn-cancel")
        await pilot.pause()
        assert not isinstance(pilot.app.screen, SettingsScreen)
