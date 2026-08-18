"""TUI 主应用 - DanmakuScope 终端交互界面（OpenCode 风格）"""

import asyncio
import os
import re
import sys
import threading

from textual import events
from textual.app import App, ComposeResult
from textual.binding import Binding, BindingsMap
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual.widgets import Button, Input, Label, Static, Switch, TextArea

from ..utils.logger import get_logger
from ..prefs import apply_saved_prefs, load_prefs, save_prefs
from .i18n import i18n
from .screens import SettingsScreen
from .themes import CUSTOM_THEMES, DEFAULT_THEME

logger = get_logger(__name__)

_MARKUP_RE = re.compile(r"\[/?[a-zA-Z#][^\]]*\]")

_LOGO_LETTERS = {
    "D": (
        "██████╗ ",
        "██╔══██╗",
        "██║  ██║",
        "██║  ██║",
        "██████╔╝",
        "╚═════╝ ",
    ),
    "A": (
        " █████╗ ",
        "██╔══██╗",
        "███████║",
        "██╔══██║",
        "██║  ██║",
        "╚═╝  ╚═╝",
    ),
    "N": (
        "███╗   ██╗",
        "████╗  ██║",
        "██╔██╗ ██║",
        "██║╚██╗██║",
        "██║ ╚████║",
        "╚═╝  ╚═══╝",
    ),
    "M": (
        "███╗   ███╗",
        "████╗ ████║",
        "██╔████╔██║",
        "██║╚██╔╝██║",
        "██║ ╚═╝ ██║",
        "╚═╝     ╚═╝",
    ),
    "K": (
        "██╗  ██╗",
        "██║ ██╔╝",
        "█████╔╝ ",
        "██╔═██╗ ",
        "██║  ██╗",
        "╚═╝  ╚═╝",
    ),
    "U": (
        "██╗   ██╗",
        "██║   ██║",
        "██║   ██║",
        "██║   ██║",
        "╚██████╔╝",
        " ╚═════╝ ",
    ),
}


def _logo_text() -> str:
    """逐行拼接 ANSI Shadow 字体字母，生成 DANMAKU 巨型徽标"""
    rows = []
    for row in range(6):
        rows.append(" ".join(_LOGO_LETTERS[ch][row] for ch in "DANMAKU"))
    return "\n".join(rows)


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
    ALLOW_SELECT = False
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
        height: 100%;
        padding: 0 2;
        border: none;
        transition: color 0.25s in_out_cubic, background 0.25s in_out_cubic;
    }

    /* Textual 默认按钮上下各一道 tall 边框，primary 切换时边框明暗反转，
       观感如按钮错位一格；模式按钮压平并以主题色加粗文字表达选中态 */
    #mode-buttons Button.-primary {
        color: $accent;
        text-style: bold;
    }

    /* 标题栏设置小按钮：与模式按钮拉开间距，宽度收窄（双 id 压过 #mode-buttons Button 规则） */
    #mode-buttons #btn-settings {
        min-width: 8;
        margin-left: 2;
    }

    #home-panel {
        margin: 1 2;
        padding: 1 6;
        background: $background;
        align-vertical: middle;
    }

    #home-content {
        width: 100%;
        height: auto;
    }

    .home-logo {
        text-align: center;
        color: $accent;
        text-style: bold;
        margin-top: 1;
    }

    .home-tagline {
        text-align: center;
        color: $text-muted;
        margin: 1 0;
    }

    .home-section {
        border: round $primary;
        padding: 1 2;
        margin-bottom: 1;
        height: auto;
    }

    .home-section-title {
        color: $accent;
        text-style: bold;
        margin-bottom: 1;
    }

    .home-keys {
        height: auto;
    }

    .home-keys-col {
        width: 1fr;
        height: auto;
    }

    .home-hint {
        text-align: center;
        color: $text-muted;
        margin: 1 0;
        transition: color 0.8s in_out_sine;
    }

    .anim-on .home-hint.-pulse {
        color: $accent;
    }

    #log-panel {
        display: none;
        margin: 1 2;
        border: round $secondary;
        background: $surface;
        padding: 0 1;
    }

    #compare-log-panel {
        display: none;
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
        display: none;
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
        ("ctrl+s", "settings", "binding.settings", True),
        ("ctrl+v", "paste_input", "binding.paste", True),
        ("ctrl+c", "copy_selection", "binding.copy_selection", True),
        ("ctrl+q", "quit", "binding.quit", True),
    ]

    _mode = "home"
    _task_worker = None
    _animations = True

    _MODE_PANEL = {
        "single": "#log-panel",
        "compare": "#compare-log-panel",
        "log": "#tui-log-panel",
    }

    @property
    def SUB_TITLE(self) -> str:
        return i18n.t("app.sub_title")

    def compose(self) -> ComposeResult:
        with Horizontal(id="title-bar"):
            yield Static(self._title_text(), id="title-text")
            with Horizontal(id="mode-buttons"):
                yield Button(i18n.t("mode.home"), variant="primary", id="btn-mode-home")
                yield Button(i18n.t("mode.single"), id="btn-mode-single")
                yield Button(i18n.t("mode.compare"), id="btn-mode-compare")
                yield Button(i18n.t("mode.log"), id="btn-mode-log")
                yield Button(i18n.t("btn.settings"), id="btn-settings")
        with VerticalScroll(id="home-panel"):
            with Vertical(id="home-content"):
                yield Static(_logo_text(), classes="home-logo", markup=False)
                yield Static(self._home_tagline(), classes="home-tagline")
                with Vertical(classes="home-section"):
                    yield Static(f"01 · {i18n.t('app.home_section_start')}", classes="home-section-title")
                    yield Static(self._home_steps_markup())
                with Vertical(classes="home-section"):
                    yield Static(f"02 · {i18n.t('app.home_section_keys')}", classes="home-section-title")
                    with Horizontal(classes="home-keys"):
                        with Vertical(classes="home-keys-col"):
                            yield Static(self._home_key_line("Ctrl+A", "app.shortcut_analyze"))
                            yield Static(self._home_key_line("Ctrl+S", "app.shortcut_settings"))
                            yield Static(self._home_key_line("Ctrl+V", "app.shortcut_paste"))
                        with Vertical(classes="home-keys-col"):
                            yield Static(self._home_key_line("Ctrl+C", "app.shortcut_copy"))
                            yield Static(self._home_key_line("Alt+Enter", "app.shortcut_compare"))
                            yield Static(self._home_key_line("Ctrl+Q", "app.shortcut_quit"))
                yield Static(i18n.t("app.hint"), classes="home-hint")
        yield TextArea("", id="log-panel", read_only=True, show_line_numbers=False)
        yield TextArea("", id="compare-log-panel", read_only=True, show_line_numbers=False)
        yield TextArea("", id="tui-log-panel", read_only=True, show_line_numbers=False)
        with Horizontal(id="compare-controls"):
            with Vertical(id="compare-input-col"):
                with Horizontal(id="compare-options"):
                    yield Label(i18n.t("compare.reuse"), id="sw-reuse-label")
                    yield Switch(load_prefs().get("compare_reuse", True), id="sw-reuse")
                yield CompareArea(id="compare-area", placeholder=i18n.t("compare.placeholder"))
            with Vertical(id="compare-buttons"):
                yield Button(i18n.t("compare.start"), variant="primary", id="btn-compare")
                yield Button(i18n.t("btn.cancel_task"), variant="warning", id="btn-cancel-task-2", disabled=True)
                yield Button(i18n.t("btn.clear"), id="btn-clear-2")
        with Horizontal(id="bottom-bar"):
            yield Input(placeholder=i18n.t("input.placeholder"), id="bvid-input")
            yield Button(i18n.t("btn.analyze"), variant="primary", id="btn-analyze")
            with Horizontal(id="bottom-actions"):
                yield Button(i18n.t("btn.cancel_task"), variant="warning", id="btn-cancel-task", disabled=True)
                yield Button(i18n.t("btn.clear"), id="btn-clear")

    def _title_text(self) -> str:
        return f"{self.TITLE} — {i18n.t('app.sub_title')}"

    def _home_tagline(self) -> str:
        from .. import __version__

        return f"{i18n.t('app.desc')}  ·  v{__version__}"

    def _home_steps_markup(self) -> str:
        steps = [
            i18n.t("app.step_input"),
            i18n.t("app.step_analyze"),
            i18n.t("app.step_result"),
            i18n.t("app.step_compare"),
        ]
        return "\n".join(f"[$accent bold]{idx}[/] {step}" for idx, step in enumerate(steps, 1))

    @staticmethod
    def _home_key_line(key: str, i18n_key: str) -> str:
        return f"[$accent bold] {key} [/]  {i18n.t(i18n_key)}"

    def on_mount(self) -> None:
        apply_saved_prefs()
        for theme in CUSTOM_THEMES.values():
            self.register_theme(theme)
        self._main_screen = self.screen
        self._restore_appearance_prefs()
        self._apply_binding_labels()
        self._disable_decorative_focus()
        self._tui_log_sink_id = logger.add(self._tui_log_sink, level="INFO")
        self._animate_home_entrance()
        self.run_worker(self._preload_heavy_modules(), exclusive=False)

    async def _preload_heavy_modules(self) -> None:
        """后台线程预载重依赖：bilibili_api 全程惰性导入（首开设置页才触发，冷导入约 1.2s），
        首次分析会同步导入 pipeline（拖入 jieba/pandas），均造成界面卡顿，预载后移出关键路径"""
        import importlib

        for name in ("bilibili_api", "danmaku_analyzer.account", "danmaku_analyzer.pipeline", "openai"):
            await asyncio.to_thread(importlib.import_module, name)

    def _restore_appearance_prefs(self) -> None:
        """恢复主题与动画偏好（Textual 不自动持久化主题，未知主题名回退默认主题）"""
        prefs = load_prefs()
        theme = prefs.get("theme", DEFAULT_THEME)
        if theme in self.available_themes:
            self.theme = theme
        self._set_animations(bool(prefs.get("animations", True)))

    def _set_animations(self, enabled: bool) -> None:
        """动画总开关：以主屏 anim-on 类门控脉动色规则并启停脉动定时器，关闭即实时清除"""
        self._animations = enabled
        if enabled:
            self._main_screen.add_class("anim-on")
            if getattr(self, "_pulse_timer", None) is None:
                self._pulse_timer = self.set_interval(0.8, self._toggle_hint_pulse)
        else:
            self._main_screen.remove_class("anim-on")
            if getattr(self, "_pulse_timer", None) is not None:
                self._pulse_timer.stop()
                self._pulse_timer = None
            self.query_one(".home-hint").remove_class("-pulse")

    def _toggle_hint_pulse(self) -> None:
        """主页底部提示呼吸脉动：定时切换 -pulse 类，颜色经 CSS transition 平滑过渡"""
        if self._mode == "home":
            self.query_one(".home-hint").toggle_class("-pulse")

    def _animate_home_entrance(self) -> None:
        """主页元素交错入场：透明度渐显，delay 递增"""
        if not self._animations or self._mode != "home":
            return
        targets = [
            self.query_one(".home-logo"),
            self.query_one(".home-tagline"),
            *self.query(".home-section"),
            self.query_one(".home-hint"),
        ]
        for index, widget in enumerate(targets):
            widget.styles.opacity = 0.0
            widget.styles.animate("opacity", 1.0, duration=0.4, delay=index * 0.07, easing="out_quart")

    def _animate_panel_in(self, mode: str) -> None:
        """模式切换时目标面板渐显（动画关闭时跳过）"""
        if not self._animations:
            return
        panel = self.query_one(self._MODE_PANEL.get(mode, "#home-panel"))
        panel.styles.opacity = 0.0
        panel.styles.animate("opacity", 1.0, duration=0.25, easing="out_quart")

    def _disable_decorative_focus(self) -> None:
        """只读面板与操作按钮不可聚焦，消除点击后默认焦点样式的白底；输入控件不受影响"""
        for widget in self.screen.query(Button):
            widget.can_focus = False
        for widget in self.screen.query(TextArea):
            if widget.read_only:
                widget.can_focus = False
        self.query_one("#home-panel").can_focus = False
        # 挂载期自动聚焦可能已落在上述控件上，需主动清除
        if self.focused is not None and not self.focused.can_focus:
            self.screen.set_focus(None)

    def on_resize(self, event: events.Resize) -> None:
        """窄屏下隐藏巨型徽标（68 列），避免自动折行破坏字形并挤占分区空间"""
        self.query_one(".home-logo", Static).display = event.size.width >= 86

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

    @property
    def _visible_panel(self) -> TextArea:
        return self.query_one(self._MODE_PANEL[self._mode], TextArea)

    def _log_lines(self, lines: list[str], panel_id: str = "#log-panel") -> None:
        """向指定输出面板追加纯文本行（TextArea 不支持 Rich markup，先剥离标签）"""
        panel = self.query_one(panel_id, TextArea)
        text = "\n".join(_MARKUP_RE.sub("", line) for line in lines)
        prefix = "\n" if panel.text else ""
        panel.move_cursor(panel.document.end)
        panel.insert(prefix + text)
        panel.move_cursor(panel.document.end)

    def _log_lines_threadsafe(self, lines: list[str], panel_id: str = "#log-panel") -> None:
        """进度回调可能在应用线程或工作线程触发，按线程路由避免 call_from_thread 同线程报错"""
        if threading.get_ident() == self._thread_id:
            self._log_lines(lines, panel_id)
        else:
            self.call_from_thread(self._log_lines, lines, panel_id)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-analyze":
            self.action_analyze()
        elif event.button.id in ("btn-clear", "btn-clear-2"):
            self.action_clear()
        elif event.button.id in ("btn-cancel-task", "btn-cancel-task-2"):
            self.action_cancel_task()
        elif event.button.id == "btn-settings":
            self.action_settings()
        elif event.button.id == "btn-mode-home":
            self._set_mode("home")
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
        """切换主页/个体/比对/实时日志四种模式：内容区与底部输入区互斥显示，输出面板各自独立"""
        self._mode = mode
        self.query_one("#bottom-bar").styles.display = "block" if mode == "single" else "none"
        self.query_one("#compare-controls").styles.display = "block" if mode == "compare" else "none"
        self.query_one("#home-panel").styles.display = "block" if mode == "home" else "none"
        self.query_one("#log-panel").styles.display = "block" if mode == "single" else "none"
        self.query_one("#compare-log-panel").styles.display = "block" if mode == "compare" else "none"
        self.query_one("#tui-log-panel").styles.display = "block" if mode == "log" else "none"
        for btn_mode, btn_name in (
            ("home", "#btn-mode-home"),
            ("single", "#btn-mode-single"),
            ("compare", "#btn-mode-compare"),
            ("log", "#btn-mode-log"),
        ):
            self.query_one(btn_name, Button).variant = "primary" if mode == btn_mode else "default"
        self._animate_panel_in(mode)
        if mode == "compare":
            self.query_one("#compare-area", CompareArea).focus()
        elif mode == "single":
            self.query_one("#bvid-input", Input).focus()

    @staticmethod
    def _cli_script_path() -> str:
        """CLI 入口绝对路径：优先当前解释器同目录的 danmaku-analyzer（新终端窗口 PATH 未必含当前环境），无则回退命令名"""
        name = "danmaku-analyzer.exe" if sys.platform == "win32" else "danmaku-analyzer"
        candidate = os.path.join(os.path.dirname(sys.executable), name)
        return candidate if os.path.exists(candidate) else "danmaku-analyzer"

    def _open_system_terminal(self) -> None:
        """打开系统终端执行 danmaku-analyzer login（TUI 无法内嵌终端，降级为新窗口方案）"""
        import shlex
        import subprocess

        login_cmd = f"{shlex.quote(self._cli_script_path())} login"
        try:
            if sys.platform == "win32":
                subprocess.Popen(
                    ["cmd", "/c", "start", "cmd", "/k", "danmaku-analyzer login"],
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
            elif sys.platform == "darwin":
                # `open -a Terminal <参数>` 把参数当作待打开的文件，执行命令须经 AppleScript do script
                subprocess.Popen(
                    ["osascript", "-e", f'tell application "Terminal" to do script "{login_cmd}"']
                )
            else:
                subprocess.Popen(["x-terminal-emulator", "-e", login_cmd])
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
        self._task_worker = self.run_worker(self._run_analysis(input_str), exclusive=True)
        self._set_cancel_buttons_enabled(True)

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
        finally:
            self._on_task_finished()

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
        self._task_worker = self.run_worker(self._run_compare(inputs, reuse), exclusive=True)
        self._set_cancel_buttons_enabled(True)

    async def _run_compare(self, inputs: list[str], reuse: bool) -> None:
        from ..pipeline import compare_videos

        panel = "#compare-log-panel"
        self._log_lines([f"▶ {i18n.t('compare.begin', count=len(inputs))}"], panel)

        def progress_callback(stage: str, message: str):
            self._log_lines_threadsafe([f"  ✔ {stage} {message}"], panel)

        try:
            result = await compare_videos(inputs, reuse=reuse, progress=progress_callback)
            for item in result.items:
                if item.ok:
                    status = i18n.t("compare.item_reused") if item.reused else i18n.t("log.done")
                    self._log_lines([f"  • {item.bvid or item.raw_input}: {status}"], panel)
                else:
                    self._log_lines([f"  • {item.raw_input}: {i18n.t('compare.item_failed')} {item.error}"], panel)
            if result.summary_csv_path:
                self._log_lines([f"  {i18n.t('compare.summary')}: {result.summary_csv_path}"], panel)
            if result.statistics_csv_path:
                self._log_lines([f"  {i18n.t('compare.statistics')}: {result.statistics_csv_path}"], panel)
            if result.snapshot_path:
                if result.snapshot_valid:
                    self._log_lines([f"  {i18n.t('compare.snapshot')}: {result.snapshot_path}"], panel)
                else:
                    self._log_lines([f"  {i18n.t('compare.snapshot_invalid')}"], panel)
            self.notify(i18n.t("compare.done"), severity="information")
        except Exception as e:
            logger.error(f"TUI 比对分析失败: {e}", exc_info=True)
            self._log_lines([f"✘ {e}"], panel)
            self.notify(i18n.t("compare.failed", error=e), severity="error")
        finally:
            self._on_task_finished()

    def action_clear(self) -> None:
        """清空当前模式对应的输出面板（主页内容为固定布局，不参与清屏）"""
        if self._mode not in self._MODE_PANEL:
            return
        self._visible_panel.clear()

    def action_cancel_task(self) -> None:
        """中断当前分析/比对任务（取消后台 worker，取消信号经 asyncio 传播至 HTTP 请求层）"""
        worker = self._task_worker
        if worker is None or not worker.is_running:
            self.notify(i18n.t("notify.cancel_none"), severity="warning")
            return
        worker.cancel()
        self._set_cancel_buttons_enabled(False)
        panel_id = self._MODE_PANEL.get(self._mode, "#log-panel")
        self._log_lines([f"✘ {i18n.t('notify.cancel_done')}"], panel_id)
        self.notify(i18n.t("notify.cancel_done"), severity="information")

    def _set_cancel_buttons_enabled(self, enabled: bool) -> None:
        self.query_one("#btn-cancel-task", Button).disabled = not enabled
        self.query_one("#btn-cancel-task-2", Button).disabled = not enabled

    def _on_task_finished(self) -> None:
        """任务结束恢复中断按钮；新任务运行中（旧任务被顶替取消）时不误禁用"""
        worker = self._task_worker
        if worker is not None and worker.is_running:
            return
        self._set_cancel_buttons_enabled(False)

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
        """将当前选中的文本复制到系统剪贴板（优先聚焦控件选区，其次当前输出面板，最后屏幕级选择）"""
        selected = self._focused_selected_text()
        if not selected and self._mode in self._MODE_PANEL:
            panel = self._visible_panel
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

    def _focused_selected_text(self) -> str:
        """聚焦控件的选中文本：Input 选区（screen 级选择不覆盖 Input 内部选区）与 TextArea 选区"""
        focused = self.focused
        if isinstance(focused, Input):
            return focused.selected_text or ""
        if isinstance(focused, TextArea):
            return focused.selected_text if focused.selection else ""
        return ""

    def _write_system_clipboard(self, text: str) -> bool:
        if sys.platform != "win32":
            return self._write_clipboard_posix(text)
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
        if sys.platform != "win32":
            return self._read_clipboard_posix()
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

    # POSIX/macOS 降级：Textual 无跨平台剪贴板能力，依次探测 pbcopy(macOS)/wl-clipboard/xclip/xsel（均缺失时静默失败）
    @staticmethod
    def _run_clipboard_tool(cmd: list[str], input_bytes: bytes | None = None) -> bytes | None:
        import shutil
        import subprocess

        if shutil.which(cmd[0]) is None:
            return None
        try:
            result = subprocess.run(cmd, input=input_bytes, capture_output=True, timeout=5)
        except (OSError, subprocess.SubprocessError):
            return None
        return result.stdout if result.returncode == 0 else None

    def _write_clipboard_posix(self, text: str) -> bool:
        data = text.encode("utf-8")
        for cmd in (["pbcopy"], ["wl-copy"], ["xclip", "-selection", "clipboard"], ["xsel", "--clipboard", "--input"]):
            if self._run_clipboard_tool(cmd, input_bytes=data) is not None:
                return True
        return False

    def _read_clipboard_posix(self) -> str:
        for cmd in (
            ["pbpaste"],
            ["wl-paste", "--no-newline"],
            ["xclip", "-selection", "clipboard", "-o"],
            ["xsel", "--clipboard", "--output"],
        ):
            output = self._run_clipboard_tool(cmd)
            if output is not None:
                return output.decode("utf-8", errors="replace")
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
    # httpx/httpcore 的逐请求 INFO 日志属传输层噪声，压到 WARNING 避免刷屏实时日志面板
    for noisy in ("httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    DanmakuTUI().run()
