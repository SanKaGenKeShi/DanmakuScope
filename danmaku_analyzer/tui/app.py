"""TUI 主应用 - DanmakuScope 终端交互界面（OpenCode 风格）"""

import re
import threading

from textual import events
from textual.app import App, ComposeResult
from textual.binding import Binding, BindingsMap
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.widgets import Button, Input, Label, Static, Switch, TextArea

from ..utils.logger import get_logger
from .i18n import apply_saved_prefs, i18n, load_prefs, save_prefs
from .screens import SettingsScreen

logger = get_logger(__name__)

_MARKUP_RE = re.compile(r"\[/?[a-zA-Z#][^\]]*\]")


class CompareRequested(Message):
    """比对输入区按下 Alt+Enter 时触发"""


class CompareArea(TextArea):
    """多行比对输入区：Enter 换行，Alt+Enter 触发比对分析"""

    async def _on_key(self, event: events.Key) -> None:
        if event.key == "alt+enter":
            event.stop()
            event.prevent_default()
            self.post_message(CompareRequested())
            return
        await super()._on_key(event)


class DanmakuTUI(App):
    """DanmakuScope TUI 主应用"""

    TITLE = "DanmakuScope"
    ENABLE_COMMAND_PALETTE = False
    UPDATE_INTERVAL = 1 / 120

    CSS = """
    Screen {
        background: $background;
    }

    ToastRack {
        dock: top;
        align: right top;
        margin-bottom: 0;
        margin-top: 1;
    }

    #title-bar {
        dock: top;
        height: 3;
        background: $boost;
        color: $text;
        layout: horizontal;
    }

    #title-text {
        width: 1fr;
        content-align-vertical: middle;
        padding: 1 2;
        text-style: bold;
    }

    #mode-buttons {
        width: auto;
        align-vertical: middle;
    }

    #mode-buttons Button {
        margin-left: 1;
        min-width: 12;
    }

    #log-panel {
        margin: 1 2;
        border: round $secondary;
        background: $surface;
        padding: 0 1;
    }

    #tui-log-panel {
        display: none;
        margin: 1 2;
        border: round $secondary;
        background: $surface;
        padding: 0 1;
    }

    #bottom-bar {
        dock: bottom;
        height: 3;
        layout: horizontal;
        padding: 0 2;
    }

    #bvid-input {
        width: 1fr;
        background: $background;
    }

    #btn-analyze {
        margin-left: 1;
        min-width: 12;
    }

    #bottom-actions {
        width: auto;
    }

    #bottom-actions Button {
        margin-left: 1;
        min-width: 12;
    }

    #compare-controls {
        display: none;
        dock: bottom;
        height: 12;
        layout: horizontal;
        padding: 0 2;
    }

    #compare-input-col {
        width: 1fr;
        height: 100%;
    }

    #compare-options {
        height: 3;
        layout: horizontal;
    }

    #sw-reuse-label {
        width: auto;
        padding: 1 0;
        margin-right: 1;
    }

    #compare-buttons {
        width: auto;
        height: 100%;
        margin-left: 1;
    }

    #compare-buttons Button {
        margin-top: 1;
        min-width: 14;
    }

    #compare-area {
        height: 1fr;
        background: $background;
    }
    """

    _BASE_BINDINGS = [
        ("ctrl+a", "analyze", "binding.analyze", False),
        ("ctrl+i", "show_config", "binding.config", True),
        ("ctrl+s", "settings", "binding.settings", True),
        ("ctrl+v", "paste_input", "binding.paste", True),
        ("ctrl+c", "copy_selection", "binding.copy_selection", True),
        ("ctrl+q", "quit", "binding.quit", True),
    ]

    @property
    def SUB_TITLE(self) -> str:
        return i18n.t("app.sub_title")

    def compose(self) -> ComposeResult:
        with Horizontal(id="title-bar"):
            yield Static(self._title_text(), id="title-text")
            with Horizontal(id="mode-buttons"):
                yield Button(i18n.t("mode.single"), variant="primary", id="btn-mode-single")
                yield Button(i18n.t("mode.compare"), id="btn-mode-compare")
                yield Button(i18n.t("mode.log"), id="btn-mode-log")
        yield TextArea("", id="log-panel", read_only=True, show_line_numbers=False)
        yield TextArea("", id="tui-log-panel", read_only=True, show_line_numbers=False)
        with Horizontal(id="compare-controls"):
            with Vertical(id="compare-input-col"):
                with Horizontal(id="compare-options"):
                    yield Label(i18n.t("compare.reuse"), id="sw-reuse-label")
                    yield Switch(load_prefs().get("compare_reuse", True), id="sw-reuse")
                yield CompareArea(id="compare-area", placeholder=i18n.t("compare.placeholder"))
            with Vertical(id="compare-buttons"):
                yield Button(i18n.t("compare.start"), variant="primary", id="btn-compare")
                yield Button(i18n.t("btn.show_config"), id="btn-config-2")
                yield Button(i18n.t("btn.settings"), id="btn-settings-2")
        with Horizontal(id="bottom-bar"):
            yield Input(placeholder=i18n.t("input.placeholder"), id="bvid-input")
            yield Button(i18n.t("btn.analyze"), variant="primary", id="btn-analyze")
            with Horizontal(id="bottom-actions"):
                yield Button(i18n.t("btn.show_config"), id="btn-config")
                yield Button(i18n.t("btn.settings"), id="btn-settings")

    def _title_text(self) -> str:
        return f"{self.TITLE} — {i18n.t('app.sub_title')}"

    def on_mount(self) -> None:
        apply_saved_prefs()
        self._apply_binding_labels()
        self.query_one("#bvid-input", Input).focus()
        self._greet()
        self._tui_log_sink_id = logger.add(self._tui_log_sink, level="INFO")

    def on_unmount(self) -> None:
        sink_id = getattr(self, "_tui_log_sink_id", None)
        if sink_id is not None:
            logger.remove(sink_id)

    def _tui_log_sink(self, message) -> None:
        """loguru sink：将日志行转发到专属日志面板（任意线程触发均安全）"""
        record = message.record
        name = record["extra"].get("name", record["module"])
        line = f'{record["time"].strftime("%H:%M:%S")} | {record["level"].name:<8} | {name} | {record["message"]}'
        if threading.get_ident() == self._thread_id:
            self._append_tui_log(line)
        else:
            self.call_from_thread(self._append_tui_log, line)

    def _append_tui_log(self, line: str) -> None:
        panel = self.query_one("#tui-log-panel", TextArea)
        prefix = "\n" if panel.text else ""
        panel.move_cursor(panel.document.end)
        panel.insert(prefix + line)
        panel.move_cursor(panel.document.end)

    def _greet(self) -> None:
        from .. import __version__

        self._log_lines(
            [
                f"{i18n.t('app.welcome')}  v{__version__}",
                i18n.t("app.desc"),
                "",
                i18n.t("app.quick_start"),
                f"  1. {i18n.t('app.step_input')}",
                f"  2. {i18n.t('app.step_analyze')}",
                f"  3. {i18n.t('app.step_result')}",
                f"  4. {i18n.t('app.step_compare')}",
                "",
                i18n.t("app.hint"),
            ]
        )

    @property
    def _log_panel(self) -> TextArea:
        return self.query_one("#log-panel", TextArea)

    def _log_lines(self, lines: list[str]) -> None:
        """向日志面板追加纯文本行（TextArea 不支持 Rich markup，先剥离标签）"""
        panel = self._log_panel
        text = "\n".join(_MARKUP_RE.sub("", line) for line in lines)
        prefix = "\n" if panel.text else ""
        panel.move_cursor(panel.document.end)
        panel.insert(prefix + text)
        panel.move_cursor(panel.document.end)

    def _log_lines_threadsafe(self, lines: list[str]) -> None:
        """进度回调可能在应用线程或工作线程触发，按线程路由避免 call_from_thread 同线程报错"""
        if threading.get_ident() == self._thread_id:
            self._log_lines(lines)
        else:
            self.call_from_thread(self._log_lines, lines)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-analyze":
            self.action_analyze()
        elif event.button.id in ("btn-config", "btn-config-2"):
            self.action_show_config()
        elif event.button.id in ("btn-settings", "btn-settings-2"):
            self.action_settings()
        elif event.button.id == "btn-mode-single":
            self._set_mode("single")
        elif event.button.id == "btn-mode-compare":
            self._set_mode("compare")
        elif event.button.id == "btn-mode-log":
            self._set_mode("log")
        elif event.button.id == "btn-compare":
            self.action_compare()

    def on_compare_requested(self, event: CompareRequested) -> None:
        self.action_compare()

    def _set_mode(self, mode: str) -> None:
        """切换个体/比对/日志三种模式：内容区与底部输入区互斥显示"""
        compare = mode == "compare"
        self.query_one("#bottom-bar").styles.display = "block" if mode == "single" else "none"
        self.query_one("#compare-controls").styles.display = "block" if compare else "none"
        self.query_one("#log-panel").styles.display = "none" if mode == "log" else "block"
        self.query_one("#tui-log-panel").styles.display = "block" if mode == "log" else "none"
        for btn_mode, btn_name in (
            ("single", "#btn-mode-single"),
            ("compare", "#btn-mode-compare"),
            ("log", "#btn-mode-log"),
        ):
            self.query_one(btn_name, Button).variant = "primary" if mode == btn_mode else "default"
        if compare:
            self.query_one("#compare-area", CompareArea).focus()
        elif mode == "single":
            self.query_one("#bvid-input", Input).focus()

    def _open_system_terminal(self) -> None:
        """打开系统终端执行 danmaku-analyzer login（TUI 无法内嵌终端，降级为新窗口方案）"""
        import subprocess
        import sys

        try:
            if sys.platform == "win32":
                subprocess.Popen(
                    ["cmd", "/c", "start", "cmd", "/k", "danmaku-analyzer login"],
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
            else:
                subprocess.Popen(["x-terminal-emulator", "-e", "danmaku-analyzer login"])
            self.notify(i18n.t("terminal.opened"), severity="information")
        except OSError as e:
            logger.error(f"打开系统终端失败: {e}")
            self.notify(i18n.t("terminal.open_failed"), severity="error")

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "bvid-input":
            self.action_analyze()

    def action_analyze(self) -> None:
        input_str = self.query_one("#bvid-input", Input).value.strip()
        if not input_str:
            self.notify(i18n.t("notify.empty_input"), severity="warning")
            return
        self.run_worker(self._run_analysis(input_str), exclusive=True)

    async def _run_analysis(self, input_str: str) -> None:
        from ..pipeline import analyze_video

        self._log_lines([f"▶ {i18n.t('log.start')} {input_str}"])

        def progress_callback(stage: str, message: str):
            self._log_lines_threadsafe([f"  ✔ {stage} {message}"])

        try:
            result = await analyze_video(
                input_str=input_str,
                progress_callback=progress_callback,
            )
            if not result.zip_valid:
                raise RuntimeError(i18n.t("error.no_report", input=input_str))
            self._log_lines(
                [
                    f"✔ {i18n.t('log.done')} {i18n.t('log.video')}: {result.title}",
                    f"  {i18n.t('log.partition')}: {result.tname} | "
                    f"{i18n.t('log.segments')}: {result.segments_count} | "
                    f"{i18n.t('log.groups')}: {result.aggregated_count}",
                ]
            )
            if result.zip_path:
                self._log_lines([f"  {i18n.t('log.zip')}: {result.zip_path}"])
            self.notify(i18n.t("notify.done"), severity="information")
        except Exception as e:
            logger.error(f"TUI 分析失败: {e}", exc_info=True)
            self._log_lines([f"✘ {i18n.t('log.failed')} {e}"])
            self.notify(i18n.t("notify.failed", error=e), severity="error")

    @staticmethod
    def _parse_compare_inputs(text: str) -> list[str]:
        """切分多行/逗号/分号/空格分隔的批量输入，过滤空项"""
        return [part for part in re.split(r"[\s,;，；]+", text) if part.strip()]

    def action_compare(self) -> None:
        inputs = self._parse_compare_inputs(self.query_one("#compare-area", CompareArea).text)
        if not inputs:
            self.notify(i18n.t("compare.empty"), severity="warning")
            return
        reuse = self.query_one("#sw-reuse", Switch).value
        save_prefs({"compare_reuse": reuse})
        self.run_worker(self._run_compare(inputs, reuse), exclusive=True)

    async def _run_compare(self, inputs: list[str], reuse: bool) -> None:
        from ..pipeline import compare_videos

        self._log_lines([f"▶ {i18n.t('compare.begin', count=len(inputs))}"])

        def progress_callback(stage: str, message: str):
            self._log_lines_threadsafe([f"  ✔ {stage} {message}"])

        try:
            result = await compare_videos(inputs, reuse=reuse, progress=progress_callback)
            for item in result.items:
                if item.ok:
                    status = i18n.t("compare.item_reused") if item.reused else i18n.t("log.done")
                    self._log_lines([f"  • {item.bvid or item.raw_input}: {status}"])
                else:
                    self._log_lines([f"  • {item.raw_input}: {i18n.t('compare.item_failed')} {item.error}"])
            if result.summary_csv_path:
                self._log_lines([f"  {i18n.t('compare.summary')}: {result.summary_csv_path}"])
            if result.statistics_csv_path:
                self._log_lines([f"  {i18n.t('compare.statistics')}: {result.statistics_csv_path}"])
            if result.snapshot_path:
                self._log_lines([f"  {i18n.t('compare.snapshot')}: {result.snapshot_path}"])
            self.notify(i18n.t("compare.done"), severity="information")
        except Exception as e:
            logger.error(f"TUI 比对分析失败: {e}", exc_info=True)
            self._log_lines([f"✘ {e}"])
            self.notify(i18n.t("compare.failed", error=e), severity="error")

    def action_show_config(self) -> None:
        from ..config import get_settings
        from ..llm_config import get_llm_settings

        settings = get_settings()
        llm_cfg = get_llm_settings()
        self._remove_previous_config()

        on = i18n.t("log.enabled")
        off = i18n.t("log.disabled")

        def flag(value: bool) -> str:
            return on if value else off

        lines = [i18n.t("log.config")]
        lines.append(
            f"  [{i18n.t('cfg.section_sampling')}] "
            f"{i18n.t('log.segmentation')}: {settings.SEGMENTATION_MODE} | "
            f"{i18n.t('log.min_samples')}: {settings.MIN_SEGMENT_SAMPLES} | "
            f"{i18n.t('cfg.sampling_strategy')}: {i18n.t('settings.sampling_freq') if settings.ENABLE_FREQ_BASED_SAMPLING else i18n.t('settings.sampling_head')} | "
            f"TOP_N: {settings.TOP_N} | "
            f"{i18n.t('settings.confidence_level')}: {settings.CONFIDENCE_LEVEL} | "
            f"{i18n.t('settings.llm_tokenizer')}: {flag(settings.ENABLE_LLM_TOKENIZER)} | "
            f"{i18n.t('settings.context_window')}: {settings.CONTEXT_TIME_WINDOW}s | "
            f"{i18n.t('settings.max_context_tokens')}: {settings.MAX_CONTEXT_TOKENS}"
        )
        lines.append(
            f"  [{i18n.t('cfg.section_llm')}] "
            f"{i18n.t('log.simple_llm')}: {llm_cfg.SIMPLE_LLM_MODEL} | "
            f"{i18n.t('log.complex_llm')}: {llm_cfg.COMPLEX_LLM_MODEL} | "
            f"{i18n.t('log.dual_path')}: {flag(llm_cfg.ENABLE_DUAL_PATH)} | "
            f"JSD: {llm_cfg.JSD_THRESHOLD_LOW}/{llm_cfg.JSD_THRESHOLD_MEDIUM} | "
            f"{i18n.t('cfg.concurrency')}: {settings.LLM_CONCURRENCY} | "
            f"{i18n.t('settings.thinking')}(简/复/报): "
            f"{flag(llm_cfg.SIMPLE_LLM_ENABLE_THINKING)}/"
            f"{flag(llm_cfg.COMPLEX_LLM_ENABLE_THINKING)}/"
            f"{flag(llm_cfg.ANALYSIS_REPORT_LLM_ENABLE_THINKING)}"
        )
        lines.append(
            f"  [{i18n.t('cfg.report_llm')}] "
            f"{i18n.t('log.llm_report')}: {flag(settings.ENABLE_LLM_ANALYSIS_REPORT)} | "
            f"{i18n.t('settings.model_name')}: {llm_cfg.ANALYSIS_REPORT_LLM_MODEL} | "
            f"Base URL: {llm_cfg.ANALYSIS_REPORT_LLM_BASE_URL} | "
            f"{i18n.t('settings.report_temp')}: {llm_cfg.ANALYSIS_REPORT_LLM_TEMPERATURE} | "
            f"Prompt: {llm_cfg.PROMPT_VERSION}"
        )
        lines.append(
            f"  [{i18n.t('cfg.section_corpus')}] "
            f"{i18n.t('compare.reuse')}: {on if load_prefs().get('compare_reuse', True) else off} | "
            f"{i18n.t('settings.corpus_statistics')}: {flag(settings.ENABLE_CORPUS_STATISTICS)} | "
            f"{i18n.t('settings.corpus_min_videos')}: {settings.CORPUS_MIN_VIDEOS_PER_PARTITION} | "
            f"{i18n.t('settings.corpus_zone_policy')}: {settings.CORPUS_ZONE_POLICY} | "
            f"{i18n.t('settings.corpus_temporal')}: {flag(settings.ENABLE_TEMPORAL_GROUPING)} | "
            f"{i18n.t('settings.corpus_granularity')}: {settings.TEMPORAL_GRANULARITY}"
        )
        lines.append(
            f"  [{i18n.t('cfg.section_general')}] "
            f"{i18n.t('settings.credential_status')}: {self._credential_status()}"
        )
        lines.append(
            f"  [{i18n.t('cfg.section_paths')}] "
            f"DATA_ROOT: {settings.DATA_ROOT}"
        )
        lines.append(
            f"  [{i18n.t('cfg.section_interface')}] "
            f"{i18n.t('settings.theme')}: {self.theme}"
        )
        self._log_lines(lines)
        self._config_line_count = len(lines)

    @staticmethod
    def _credential_status() -> str:
        from ..account import resolve_credential

        credential, source = resolve_credential()
        if credential is None:
            return i18n.t("settings.credential_none")
        if source == "login":
            return i18n.t("settings.credential_login")
        if source == "settings":
            return i18n.t("settings.credential_env")
        return i18n.t("settings.credential_none")

    def _remove_previous_config(self) -> None:
        """连续点击查看配置时，移除上一次的配置输出，避免刷屏"""
        previous = getattr(self, "_config_line_count", 0)
        if not previous:
            return
        panel = self._log_panel
        doc = panel.document
        start_row = max(0, doc.line_count - previous - 1)
        panel.delete((start_row, 0), doc.end)
        self._config_line_count = 0

    def action_settings(self) -> None:
        self.push_screen(SettingsScreen())

    def action_paste_input(self) -> None:
        """从系统剪贴板粘贴文本到当前模式的输入控件（Textual 无法读取 OS 剪贴板，用 Win API 兜底）"""
        text = self._read_system_clipboard()
        if not text:
            return
        if self.query_one("#compare-controls").display:
            area = self.query_one("#compare-area", CompareArea)
            area.insert(text.strip())
            area.focus()
        else:
            input_widget = self.query_one("#bvid-input", Input)
            input_widget.insert_text_at_cursor(text.strip())
            input_widget.focus()

    def action_copy_selection(self) -> None:
        """将当前选中的文本复制到系统剪贴板"""
        panel = self._log_panel
        selected = panel.selected_text if panel.selection else ""
        if not selected:
            selected = self.screen.get_selected_text()
        if not selected:
            self.notify(i18n.t("notify.copy_empty"), severity="warning")
            return
        if self._write_system_clipboard(selected):
            self.notify(i18n.t("notify.copy_done"), severity="information")
        else:
            self.notify(i18n.t("notify.copy_failed"), severity="error")

    def _write_system_clipboard(self, text: str) -> bool:
        try:
            import ctypes

            CF_UNICODETEXT = 13
            GMEM_MOVEABLE = 0x0002
            user32 = ctypes.windll.user32
            kernel32 = ctypes.windll.kernel32

            user32.OpenClipboard.argtypes = [ctypes.c_void_p]
            user32.OpenClipboard.restype = ctypes.c_bool
            user32.EmptyClipboard.restype = ctypes.c_bool
            user32.SetClipboardData.argtypes = [ctypes.c_uint, ctypes.c_void_p]
            user32.SetClipboardData.restype = ctypes.c_void_p
            user32.CloseClipboard.restype = ctypes.c_bool
            kernel32.GlobalAlloc.argtypes = [ctypes.c_uint, ctypes.c_size_t]
            kernel32.GlobalAlloc.restype = ctypes.c_void_p
            kernel32.GlobalLock.argtypes = [ctypes.c_void_p]
            kernel32.GlobalLock.restype = ctypes.c_void_p
            kernel32.GlobalUnlock.argtypes = [ctypes.c_void_p]

            data = (text + "\0").encode("utf-16-le")
            if not user32.OpenClipboard(None):
                return False
            try:
                user32.EmptyClipboard()
                handle = kernel32.GlobalAlloc(GMEM_MOVEABLE, len(data))
                if not handle:
                    return False
                ptr = kernel32.GlobalLock(handle)
                if not ptr:
                    return False
                ctypes.memmove(ptr, data, len(data))
                kernel32.GlobalUnlock(handle)
                user32.SetClipboardData(CF_UNICODETEXT, handle)
                return True
            finally:
                user32.CloseClipboard()
        except Exception:
            return False

    def _read_system_clipboard(self) -> str:
        try:
            import ctypes

            CF_UNICODETEXT = 13
            GMEM_MOVEABLE = 0x0002
            user32 = ctypes.windll.user32
            kernel32 = ctypes.windll.kernel32

            user32.OpenClipboard.argtypes = [ctypes.c_void_p]
            user32.OpenClipboard.restype = ctypes.c_bool
            user32.GetClipboardData.argtypes = [ctypes.c_uint]
            user32.GetClipboardData.restype = ctypes.c_void_p
            user32.CloseClipboard.restype = ctypes.c_bool
            kernel32.GlobalLock.argtypes = [ctypes.c_void_p]
            kernel32.GlobalLock.restype = ctypes.c_void_p
            kernel32.GlobalUnlock.argtypes = [ctypes.c_void_p]

            if not user32.OpenClipboard(None):
                return ""
            try:
                handle = user32.GetClipboardData(CF_UNICODETEXT)
                if not handle:
                    return ""
                ptr = kernel32.GlobalLock(handle)
                if not ptr:
                    return ""
                try:
                    return ctypes.wstring_at(ptr)
                finally:
                    kernel32.GlobalUnlock(handle)
            finally:
                user32.CloseClipboard()
        except Exception:
            return ""

    def action_quit(self) -> None:
        self._quit()

    def _quit(self) -> None:
        """退出应用：取消所有 worker 后立即退出，避免后台任务阻塞"""
        for worker in self.workers:
            worker.cancel()
        self.exit()

    def _apply_binding_labels(self) -> None:
        """以当前语言重建绑定映射并刷新 Footer"""
        self._bindings = BindingsMap(
            [
                Binding(key, action, i18n.t(i18n_key), show=True, priority=priority)
                for key, action, i18n_key, priority in self._BASE_BINDINGS
            ]
        )
        self.refresh_bindings()


def run_tui() -> None:
    """TUI 入口（pyproject.toml [project.scripts] 指向此处）：移除控制台日志输出，第三方 stdlib 日志桥接入 loguru"""
    import logging

    from loguru import logger as loguru_logger

    from ..utils.logger import setup_tui_logger

    setup_tui_logger()

    class _InterceptHandler(logging.Handler):
        def emit(self, record):
            try:
                level = loguru_logger.level(record.levelname).name
            except ValueError:
                level = record.levelno
            loguru_logger.opt(depth=6, exception=record.exc_info).log(level, record.getMessage())

    logging.basicConfig(handlers=[_InterceptHandler()], level=logging.INFO, force=True)

    DanmakuTUI().run()
