"""首启向导 - 首次启动引导完成B站登录与 LLM 连接配置

单屏两步：凭证状态检查（打开终端登录）与三套 LLM 连接填写（支持检测），
完成/跳过均标记 wizard_completed 不再弹出；LLM 配置经 write_llm_env 写回 .env。
"""

import asyncio

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, Rule, Static

from ...llm_config import get_llm_settings
from ...prefs import save_prefs, write_llm_env
from ..i18n import i18n

_LLM_PREFIXES = (
    ("simple", "SIMPLE_LLM", "settings.simple_section"),
    ("complex", "COMPLEX_LLM", "settings.complex_section"),
    ("report", "ANALYSIS_REPORT_LLM", "settings.report_section"),
)

# 终端列数低于该值时 LLM 行改三行堆叠（宽布局需约 100 列内容区才不裁切，对应终端约 106 列）
_NARROW_MAX_WIDTH = 106


class FirstRunWizardScreen(ModalScreen[bool]):
    """首启向导：登录与 LLM 配置两步引导，完成或跳过均不再弹出"""

    DEFAULT_CSS = """
    FirstRunWizardScreen {
        align: center middle;
    }

    /* 宽度随终端伸缩（窄终端不挤压折行，最大化后同步变宽），高度不足时滚动承接 */
    #wizard-dialog {
        width: 90%;
        min-width: 80;
        max-width: 120;
        height: auto;
        max-height: 92%;
        border: thick $primary;
        background: $background;
        padding: 1 2;
        overflow-y: auto;
    }

    #wizard-dialog .wizard-title {
        width: 100%;
        text-style: bold;
        color: $accent;
        padding: 0 0 1 0;
    }

    #wizard-dialog .wizard-desc {
        width: 100%;
        height: auto;
        color: $foreground 70%;
        padding: 0 0 1 0;
    }

    #wizard-dialog .wizard-step {
        width: 100%;
        text-style: bold;
        color: $accent;
        padding: 1 0 0 0;
    }

    #wizard-dialog .setting-row {
        width: 100%;
        height: 3;
        layout: horizontal;
    }

    #wizard-dialog .setting-row Label {
        width: 24;
        padding: 1 0;
        color: $foreground 80%;
    }

    #wizard-dialog .setting-row Input {
        width: 44;
    }

    #wizard-dialog .key-btn {
        width: 10;
        margin-left: 1;
    }

    #wizard-cred-status {
        width: 1fr;
        padding: 1 0;
    }

    /* LLM 字段 grid 布局（Horizontal 的 fr 分配在 Textual 8 不稳定，经探针改用 grid）：
       宽屏六列单行（输入框 21%×3 + 按钮 10×3），窄屏两列自动流式排为三行（各输入框与配套按钮同行）；
       Textual 无 grid 单项定位属性，集合 Label 移出 grid 独立成行实现两档共用同一套 DOM */
    #wizard-dialog .llm-set-label {
        width: 100%;
        padding: 1 0 0 0;
        color: $foreground 80%;
    }

    #wizard-dialog .llm-row {
        width: 100%;
        height: 3;
        layout: grid;
        grid-size: 6;
        grid-columns: 21% 10 21% 10 21% 10;
        grid-gutter: 0 1;
    }

    #wizard-dialog .llm-row.narrow {
        height: auto;
        grid-size: 2;
        grid-columns: 1fr 10;
    }

    #wizard-dialog .llm-row Input {
        width: 100%;
    }

    #wizard-dialog .llm-row .row-btn {
        width: 100%;
    }

    #wizard-buttons {
        height: 3;
        width: 100%;
        margin-top: 1;
    }

    #wizard-buttons Button {
        margin-right: 1;
    }
    """

    def compose(self) -> ComposeResult:
        llm_cfg = get_llm_settings()
        self._thinking = {
            "simple": llm_cfg.SIMPLE_LLM_ENABLE_THINKING,
            "complex": llm_cfg.COMPLEX_LLM_ENABLE_THINKING,
            "report": llm_cfg.ANALYSIS_REPORT_LLM_ENABLE_THINKING,
        }
        with Vertical(id="wizard-dialog"):
            yield Label(i18n.t("wizard.title"), classes="wizard-title")
            yield Label(i18n.t("wizard.intro"), classes="wizard-desc")

            yield Label(i18n.t("wizard.step_login"), classes="wizard-step")
            yield Label(i18n.t("wizard.login_desc"), classes="wizard-desc")
            with Static(classes="setting-row"):
                yield Label(i18n.t("settings.credential_status"))
                yield Label(i18n.t("wizard.checking"), id="wizard-cred-status")
            with Static(classes="setting-row"):
                yield Label(i18n.t("settings.account_actions"))
                yield Button(i18n.t("settings.open_terminal_login"), id="btn-wizard-login")
                yield Button(i18n.t("wizard.recheck"), id="btn-wizard-recheck")

            yield Label(i18n.t("wizard.step_llm"), classes="wizard-step")
            yield Label(i18n.t("wizard.llm_desc"), classes="wizard-desc")
            for prefix, env_prefix, section_key in _LLM_PREFIXES:
                yield Rule(line_style="dashed")
                yield Label(i18n.t(section_key), classes="llm-set-label")
                with Horizontal(classes="llm-row"):
                    yield Input(getattr(llm_cfg, f"{env_prefix}_BASE_URL"), placeholder="Base URL", id=f"wiz-{prefix}-url")
                    yield Button(i18n.t("settings.key_test"), id=f"wiz-test-{prefix}", classes="row-btn")
                    yield Input(getattr(llm_cfg, f"{env_prefix}_API_KEY"), password=True, placeholder="API Key", id=f"wiz-{prefix}-key")
                    yield Button(i18n.t("settings.key_show"), id=f"wiz-show-{prefix}", classes="row-btn")
                    yield Input(getattr(llm_cfg, f"{env_prefix}_MODEL"), placeholder=i18n.t("settings.model_name"), id=f"wiz-{prefix}-model")
                    yield Button(
                        self._thinking_label(prefix), id=f"wiz-thinking-{prefix}", classes="row-btn",
                        variant="primary" if self._thinking[prefix] else "default",
                    )

            with Horizontal(id="wizard-buttons"):
                yield Button(i18n.t("wizard.done"), variant="primary", id="btn-wizard-done")
                yield Button(i18n.t("wizard.skip"), id="btn-wizard-skip")

    def on_mount(self) -> None:
        self._apply_llm_layout()
        self.run_worker(self._refresh_credential_status(), exclusive=False)

    def on_resize(self, event) -> None:
        self._apply_llm_layout()

    def _apply_llm_layout(self) -> None:
        """按终端宽度切换 LLM 行布局：宽度不足时三字段堆叠为三行，避免单行裁切显示不全"""
        narrow = self.size.width < _NARROW_MAX_WIDTH
        for row in self.query(".llm-row"):
            row.set_class(narrow, "narrow")

    def _recheck_credential(self) -> None:
        """点击即给可见反馈（状态文案回滚检查中 + 通知），避免结果不变时看似无响应"""
        self.query_one("#wizard-cred-status", Label).update(i18n.t("wizard.checking"))
        self.notify(i18n.t("wizard.rechecking"), severity="information")
        self.run_worker(self._refresh_credential_status(), exclusive=False)

    async def _refresh_credential_status(self) -> None:
        """resolve_credential 会触发 bilibili_api 惰性导入，后台线程执行避免冻结界面"""
        from ...account import resolve_credential

        label = self.query_one("#wizard-cred-status", Label)
        try:
            credential, source = await asyncio.to_thread(resolve_credential)
        except Exception as e:
            label.update(i18n.t("wizard.check_failed", error=e))
            return
        if credential is None:
            label.update(i18n.t("wizard.not_logged_in"))
        else:
            source_text = {
                "file": i18n.t("wizard.source_file"),
                "login": i18n.t("wizard.source_login"),
                "settings": i18n.t("wizard.source_settings"),
            }.get(source, source)
            label.update(i18n.t("wizard.logged_in", source=source_text))

    def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id
        if button_id == "btn-wizard-login":
            self.app._open_system_terminal()
        elif button_id == "btn-wizard-recheck":
            self._recheck_credential()
        elif button_id.startswith("wiz-test-"):
            self._test_connection(button_id.removeprefix("wiz-test-"))
        elif button_id.startswith("wiz-show-"):
            self._toggle_key_visibility(button_id)
        elif button_id.startswith("wiz-thinking-"):
            self._toggle_thinking(button_id)
        elif button_id == "btn-wizard-done":
            self._finish(save_llm=True)
        elif button_id == "btn-wizard-skip":
            self._finish(save_llm=False)

    def _toggle_key_visibility(self, button_id: str) -> None:
        prefix = button_id.removeprefix("wiz-show-")
        key_input = self.query_one(f"#wiz-{prefix}-key", Input)
        key_input.password = not key_input.password
        self.query_one(f"#{button_id}", Button).label = (
            i18n.t("settings.key_hide") if not key_input.password else i18n.t("settings.key_show")
        )

    def _thinking_label(self, prefix: str) -> str:
        return i18n.t("settings.thinking_on") if self._thinking[prefix] else i18n.t("settings.thinking_off")

    def _toggle_thinking(self, button_id: str) -> None:
        prefix = button_id.removeprefix("wiz-thinking-")
        self._thinking[prefix] = not self._thinking[prefix]
        btn = self.query_one(f"#{button_id}", Button)
        btn.label = self._thinking_label(prefix)
        btn.variant = "primary" if self._thinking[prefix] else "default"

    def _test_connection(self, prefix: str) -> None:
        base_url = self.query_one(f"#wiz-{prefix}-url", Input).value.strip()
        api_key = self.query_one(f"#wiz-{prefix}-key", Input).value.strip()
        if not base_url:
            self.notify(i18n.t("settings.test_failed", error="Base URL 为空"), severity="error")
            return
        self.notify(i18n.t("settings.testing"), severity="information")
        self.run_worker(self._run_connection_test(base_url, api_key), exclusive=False)

    async def _run_connection_test(self, base_url: str, api_key: str) -> None:
        from openai import AsyncOpenAI

        try:
            client = AsyncOpenAI(base_url=base_url, api_key=api_key or "sk-test", timeout=15.0)
            await client.models.list()
            self.notify(i18n.t("settings.test_ok"), severity="information")
        except Exception as e:
            self.notify(i18n.t("settings.test_failed", error=e), severity="error")

    def _finish(self, save_llm: bool) -> None:
        if save_llm:
            updates = {}
            for prefix, env_prefix, _ in _LLM_PREFIXES:
                updates[f"{env_prefix}_BASE_URL"] = self.query_one(f"#wiz-{prefix}-url", Input).value.strip()
                updates[f"{env_prefix}_API_KEY"] = self.query_one(f"#wiz-{prefix}-key", Input).value.strip()
                updates[f"{env_prefix}_MODEL"] = self.query_one(f"#wiz-{prefix}-model", Input).value.strip()
                updates[f"{env_prefix}_ENABLE_THINKING"] = self._thinking[prefix]
            write_llm_env(updates)
        save_prefs({"wizard_completed": True})
        self.dismiss(save_llm)
