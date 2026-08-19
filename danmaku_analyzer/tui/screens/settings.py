"""TUI 设置中心 - 界面/分析/路径/LLM/语料库/快捷键 全配置标签页，支持编辑与恢复默认"""

import asyncio

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import (
    Button,
    DataTable,
    Input,
    Label,
    OptionList,
    Static,
    Switch,
    TabbedContent,
    TabPane,
)
from textual.widgets.option_list import Option

from ...config import get_settings
from ...llm_config import get_llm_settings
from ...prefs import (
    ENV_LLM_KEYS,
    PERSIST_SETTINGS_KEYS,
    load_prefs,
    save_prefs,
    write_llm_env,
)
from ..i18n import i18n
from ..themes import DEFAULT_THEME, THEME_IDS

# 分析参数默认值（与 config.py 保持一致）
_ANALYSIS_DEFAULTS = {
    "SEGMENTATION_MODE": "dynamic",
    "MIN_SEGMENT_SAMPLES": 30,
    "ENABLE_FREQ_BASED_SAMPLING": False,
    "ENABLE_BATCH_SEGMENT_ANALYSIS": False,
    "TOP_N": 10,
    "MOE": 0.05,
    "CONFIDENCE_LEVEL": 0.95,
    "ENABLE_LLM_ANALYSIS_REPORT": True,
    "LLM_CONCURRENCY": 5,
    "ENABLE_LLM_TOKENIZER": False,
    "LLM_TOKENIZER_MIN_LENGTH": 20,
    "CONTEXT_TIME_WINDOW": 5.0,
    "MAX_CONTEXT_TOKENS": 200,
}

# LLM 参数默认值（与 llm_config.py 保持一致，仅可编辑项）
_LLM_DEFAULTS = {
    "ENABLE_DUAL_PATH": True,
    "JSD_THRESHOLD_LOW": 0.2,
    "JSD_THRESHOLD_MEDIUM": 0.6,
    "SIMPLE_LLM_ENABLE_THINKING": False,
    "COMPLEX_LLM_ENABLE_THINKING": False,
    "ANALYSIS_REPORT_LLM_ENABLE_THINKING": False,
    "ANALYSIS_REPORT_LLM_TEMPERATURE": 0.3,
    "COMPLEX_LLM_TIMEOUT": 120.0,
    "SIMPLE_LLM_TIMEOUT": 120.0,
    "ANALYSIS_REPORT_LLM_TIMEOUT": 180.0,
}

# 语料库参数默认值（与 config.py 保持一致）
_CORPUS_DEFAULTS = {
    "CORPUS_MIN_VIDEOS_PER_PARTITION": 3,
    "CORPUS_ZONE_POLICY": "hot_only",
    "ENABLE_TEMPORAL_GROUPING": False,
    "TEMPORAL_GRANULARITY": "year",
    "ENABLE_CORPUS_STATISTICS": True,
    "SCHEDULER_WORKERS": 2,
    "VISUALIZATION_BACKEND": "python",
}

# 分析报告 LLM 未配置时界面直接展示配置现值（占位默认值），不再预填复杂任务值

# 专家级参数在参数说明表中的条目名（与 expert-only 控件分级保持一致，基础模式下不展示）
_EXPERT_HELP_KEYS = {
    "切分模式", "最小段样本数", "抽样误差 MOE", "置信水平",
    "LLM 并发上限", "LLM 辅助分词", "分词触发长度", "微语境窗口", "微语境最大 token",
    "每分区最少视频数", "冷热区策略", "按发布时间分桶", "分桶粒度", "调度器并发数", "可视化脚本后端",
    "双路推理", "JSD 低阈值", "JSD 中阈值", "报告生成温度", "请求超时",
}

# sessdata → (uname, mid) 会话内缓存，避免每次打开设置页都请求 nav 接口
_ACCOUNT_CACHE: dict[str, tuple[str, str]] = {}


class SettingsScreen(ModalScreen[bool]):
    """设置中心：主题/分析参数/LLM/语料库/关于，可编辑并支持恢复默认"""

    DEFAULT_CSS = """
    SettingsScreen {
        align: center middle;
    }

    OptionList .option-list--option-highlighted {
        background: $boost;
    }

    OptionList .option-list--option-hover {
        background: $boost;
    }

    #settings-dialog {
        width: 100%;
        height: 100%;
        border: thick $primary;
        background: $background;
        padding: 1 2;
    }

    #settings-dialog TabbedContent {
        height: 1fr;
    }

    #theme-options {
        height: 1fr;
        margin: 1 0;
    }

    .setting-row {
        width: 100%;
        height: 4;
        layout: horizontal;
    }

    .setting-row-tall {
        height: auto;
    }

    .setting-row Label {
        width: 32;
        padding: 1 0;
        color: $foreground 80%;
    }

    .setting-row Input {
        width: 30;
    }

    .setting-row Switch {
        margin: 0;
    }

    .setting-row OptionList {
        width: 40;
        height: auto;
    }

    .key-btn {
        width: 10;
        margin-left: 1;
    }

    #btn-logout {
        margin-left: 1;
    }

    .section-label {
        width: 100%;
        padding: 1 0 0 0;
        text-style: bold;
        color: $accent;
    }

    .help-text {
        width: 100%;
        height: auto;
        padding: 0 0 1 0;
        color: $foreground 80%;
    }

    .help-table {
        height: auto;
        margin: 0 0 1 0;
    }

    #credential-status-label {
        width: 1fr;
        padding: 1 0;
    }

    #mode-hint {
        width: 1fr;
        padding: 1 0;
        color: $foreground 60%;
    }

    #tab-llm .setting-row Input {
        width: 50;
    }

    #about-desc {
        height: auto;
        margin: 1 0;
        color: $foreground 80%;
    }

    #about-info {
        height: auto;
        margin: 1 0;
    }

    .about-links {
        height: 3;
        layout: horizontal;
        margin: 1 0;
    }

    .link-btn {
        margin-right: 1;
        min-width: 20;
    }

    #settings-buttons {
        height: 3;
        width: 100%;
    }

    #settings-buttons .btn-spacer {
        width: 1fr;
    }

    #settings-buttons Button {
        margin: 0 1;
    }
    """

    BINDINGS = [
        Binding("escape", "cancel", "取消", show=False),
        Binding("ctrl+v", "paste_clipboard", "粘贴", show=False, priority=True),
    ]

    def compose(self) -> ComposeResult:
        settings = get_settings()
        llm_cfg = get_llm_settings()
        self._thinking = {
            "simple": llm_cfg.SIMPLE_LLM_ENABLE_THINKING,
            "complex": llm_cfg.COMPLEX_LLM_ENABLE_THINKING,
            "report": llm_cfg.ANALYSIS_REPORT_LLM_ENABLE_THINKING,
        }
        self._ui_mode = str(load_prefs().get("ui_mode", "basic"))
        with Vertical(id="settings-dialog"):
            with Static(classes="setting-row"):
                yield Label(i18n.t("settings.ui_mode"))
                yield Button(
                    i18n.t("settings.mode_basic"), id="btn-mode-basic",
                    variant="primary" if self._ui_mode != "expert" else "default",
                )
                yield Button(
                    i18n.t("settings.mode_expert"), id="btn-mode-expert",
                    variant="primary" if self._ui_mode == "expert" else "default",
                )
                yield Label(i18n.t("settings.mode_hint"), id="mode-hint")
            with TabbedContent(initial="tab-display"):
                with TabPane(i18n.t("settings.tab_display"), id="tab-display"):
                    yield OptionList(
                        *[Option(self._theme_label(theme_id), id=theme_id) for theme_id in THEME_IDS],
                        id="theme-options",
                    )
                    with Static(classes="setting-row"):
                        yield Label(i18n.t("settings.animations"))
                        yield Switch(self.app._animations, id="sw-animations")
                with TabPane(i18n.t("settings.tab_general"), id="tab-general"):
                    with Static(classes="setting-row"):
                        yield Label(i18n.t("settings.credential_status"))
                        yield Label("", id="credential-status-label")
                    with Static(classes="setting-row"):
                        yield Label(i18n.t("settings.sessdata"))
                        yield Input("", password=True, id="inp-cred-sessdata")
                        yield Button(i18n.t("settings.key_show"), id="btn-show-cred-sessdata", classes="key-btn")
                        yield Button(i18n.t("settings.key_copy"), id="btn-copy-cred-sessdata", classes="key-btn")
                        yield Button(i18n.t("settings.key_paste"), id="btn-paste-cred-sessdata", classes="key-btn")
                    with Static(classes="setting-row"):
                        yield Label(i18n.t("settings.bili_jct"))
                        yield Input("", password=True, id="inp-cred-jct")
                        yield Button(i18n.t("settings.key_show"), id="btn-show-cred-jct", classes="key-btn")
                        yield Button(i18n.t("settings.key_copy"), id="btn-copy-cred-jct", classes="key-btn")
                        yield Button(i18n.t("settings.key_paste"), id="btn-paste-cred-jct", classes="key-btn")
                    with Static(classes="setting-row"):
                        yield Label(i18n.t("settings.buvid3"))
                        yield Input("", password=True, id="inp-cred-buvid3")
                        yield Button(i18n.t("settings.key_show"), id="btn-show-cred-buvid3", classes="key-btn")
                        yield Button(i18n.t("settings.key_copy"), id="btn-copy-cred-buvid3", classes="key-btn")
                        yield Button(i18n.t("settings.key_paste"), id="btn-paste-cred-buvid3", classes="key-btn")
                    with Static(classes="setting-row"):
                        yield Label(i18n.t("settings.account_actions"))
                        yield Button(i18n.t("settings.open_terminal_login"), id="btn-open-terminal-login")
                        yield Button(i18n.t("settings.logout"), variant="error", id="btn-logout")
                with TabPane(i18n.t("settings.tab_analysis"), id="tab-analysis"):
                    with Static(classes="setting-row setting-row-tall expert-only"):
                        yield Label(i18n.t("settings.seg_mode"))
                        yield OptionList(
                            Option(i18n.t("settings.dynamic"), id="dynamic"),
                            Option(i18n.t("settings.fixed"), id="fixed"),
                            id="seg-mode-options",
                        )
                    with Static(classes="setting-row expert-only"):
                        yield Label(i18n.t("settings.min_samples"))
                        yield Input(str(settings.MIN_SEGMENT_SAMPLES), type="integer", id="inp-min-samples")
                    with Static(classes="setting-row"):
                        yield Label(i18n.t("settings.sampling_freq"))
                        yield Switch(settings.ENABLE_FREQ_BASED_SAMPLING, id="sw-freq-sampling")
                    with Static(classes="setting-row"):
                        yield Label(i18n.t("settings.top_n"))
                        yield Input(str(settings.TOP_N), type="integer", id="inp-top-n")
                    with Static(classes="setting-row expert-only"):
                        yield Label(i18n.t("settings.moe"))
                        yield Input(str(settings.MOE), type="number", id="inp-moe")
                    with Static(classes="setting-row"):
                        yield Label(i18n.t("settings.batch_analysis"))
                        yield Switch(settings.ENABLE_BATCH_SEGMENT_ANALYSIS, id="sw-batch-analysis")
                    with Static(classes="setting-row expert-only"):
                        yield Label(i18n.t("settings.confidence_level"))
                        yield Input(str(settings.CONFIDENCE_LEVEL), type="number", id="inp-confidence-level")
                    with Static(classes="setting-row"):
                        yield Label(i18n.t("settings.llm_report"))
                        yield Switch(settings.ENABLE_LLM_ANALYSIS_REPORT, id="sw-llm-report")
                    with Static(classes="setting-row expert-only"):
                        yield Label(i18n.t("settings.llm_concurrency"))
                        yield Input(str(settings.LLM_CONCURRENCY), type="integer", id="inp-concurrency")
                    with Static(classes="setting-row expert-only"):
                        yield Label(i18n.t("settings.llm_tokenizer"))
                        yield Switch(settings.ENABLE_LLM_TOKENIZER, id="sw-llm-tokenizer")
                    with Static(classes="setting-row expert-only"):
                        yield Label(i18n.t("settings.tokenizer_min_len"))
                        yield Input(str(settings.LLM_TOKENIZER_MIN_LENGTH), type="integer", id="inp-tokenizer-min-len")
                    with Static(classes="setting-row expert-only"):
                        yield Label(i18n.t("settings.context_window"))
                        yield Input(str(settings.CONTEXT_TIME_WINDOW), type="number", id="inp-context-window")
                    with Static(classes="setting-row expert-only"):
                        yield Label(i18n.t("settings.max_context_tokens"))
                        yield Input(str(settings.MAX_CONTEXT_TOKENS), type="integer", id="inp-max-context-tokens")
                    yield Label(i18n.t("settings.section_corpus"), classes="section-label expert-only")
                    with Static(classes="setting-row"):
                        yield Label(i18n.t("compare.reuse"))
                        yield Switch(load_prefs().get("compare_reuse", True), id="sw-compare-reuse")
                    with Static(classes="setting-row"):
                        yield Label(i18n.t("settings.corpus_statistics"))
                        yield Switch(settings.ENABLE_CORPUS_STATISTICS, id="sw-corpus-statistics")
                    with Static(classes="setting-row expert-only"):
                        yield Label(i18n.t("settings.corpus_min_videos"))
                        yield Input(str(settings.CORPUS_MIN_VIDEOS_PER_PARTITION), type="integer", id="inp-corpus-min-videos")
                    with Static(classes="setting-row setting-row-tall expert-only"):
                        yield Label(i18n.t("settings.corpus_zone_policy"))
                        yield OptionList(
                            Option(i18n.t("settings.zone_hot_only"), id="hot_only"),
                            Option(i18n.t("settings.zone_all"), id="all"),
                            Option(i18n.t("settings.zone_weighted"), id="weighted"),
                            id="zone-policy-options",
                        )
                    with Static(classes="setting-row expert-only"):
                        yield Label(i18n.t("settings.corpus_temporal"))
                        yield Switch(settings.ENABLE_TEMPORAL_GROUPING, id="sw-temporal-grouping")
                    with Static(classes="setting-row setting-row-tall expert-only"):
                        yield Label(i18n.t("settings.corpus_granularity"))
                        yield OptionList(
                            Option(i18n.t("settings.gran_year"), id="year"),
                            Option(i18n.t("settings.gran_quarter"), id="quarter"),
                            Option(i18n.t("settings.gran_month"), id="month"),
                            id="granularity-options",
                        )
                    with Static(classes="setting-row expert-only"):
                        yield Label(i18n.t("settings.scheduler_workers"))
                        yield Input(str(settings.SCHEDULER_WORKERS), type="integer", id="inp-scheduler-workers")
                    with Static(classes="setting-row setting-row-tall expert-only"):
                        yield Label(i18n.t("settings.viz_backend"))
                        yield OptionList(
                            Option(i18n.t("settings.viz_python"), id="python"),
                            Option(i18n.t("settings.viz_r"), id="r"),
                            id="viz-backend-options",
                        )
                with TabPane(i18n.t("settings.tab_llm"), id="tab-llm"):
                    with Static(classes="setting-row expert-only"):
                        yield Label(i18n.t("settings.dual_path"))
                        yield Switch(llm_cfg.ENABLE_DUAL_PATH, id="sw-dual-path")
                    with Static(classes="setting-row expert-only"):
                        yield Label(i18n.t("settings.jsd_low"))
                        yield Input(str(llm_cfg.JSD_THRESHOLD_LOW), type="number", id="inp-jsd-low")
                    with Static(classes="setting-row expert-only"):
                        yield Label(i18n.t("settings.jsd_medium"))
                        yield Input(str(llm_cfg.JSD_THRESHOLD_MEDIUM), type="number", id="inp-jsd-medium")
                    yield Label(i18n.t("settings.simple_section"), classes="section-label")
                    with Static(classes="setting-row"):
                        yield Label(i18n.t("settings.base_url"))
                        yield Input(llm_cfg.SIMPLE_LLM_BASE_URL, id="inp-simple-url")
                        yield Button(i18n.t("settings.key_test"), id="btn-test-simple-url", classes="key-btn")
                    with Static(classes="setting-row"):
                        yield Label(i18n.t("settings.api_key"))
                        yield Input(llm_cfg.SIMPLE_LLM_API_KEY, password=True, id="inp-simple-key")
                        yield Button(i18n.t("settings.key_show"), id="btn-show-simple-key", classes="key-btn")
                    with Static(classes="setting-row"):
                        yield Label(i18n.t("settings.model_name"))
                        yield Input(llm_cfg.SIMPLE_LLM_MODEL, id="inp-simple-model")
                        yield Button(
                            self._thinking_label("simple"), id="btn-thinking-simple", classes="key-btn",
                            variant="primary" if self._thinking["simple"] else "default",
                        )
                    with Static(classes="setting-row expert-only"):
                        yield Label(i18n.t("settings.timeout"))
                        yield Input(str(llm_cfg.SIMPLE_LLM_TIMEOUT), type="number", id="inp-simple-timeout")
                    yield Label(i18n.t("settings.complex_section"), classes="section-label")
                    with Static(classes="setting-row"):
                        yield Label(i18n.t("settings.base_url"))
                        yield Input(llm_cfg.COMPLEX_LLM_BASE_URL, id="inp-complex-url")
                        yield Button(i18n.t("settings.key_test"), id="btn-test-complex-url", classes="key-btn")
                    with Static(classes="setting-row"):
                        yield Label(i18n.t("settings.api_key"))
                        yield Input(llm_cfg.COMPLEX_LLM_API_KEY, password=True, id="inp-complex-key")
                        yield Button(i18n.t("settings.key_show"), id="btn-show-complex-key", classes="key-btn")
                    with Static(classes="setting-row"):
                        yield Label(i18n.t("settings.model_name"))
                        yield Input(llm_cfg.COMPLEX_LLM_MODEL, id="inp-complex-model")
                        yield Button(
                            self._thinking_label("complex"), id="btn-thinking-complex", classes="key-btn",
                            variant="primary" if self._thinking["complex"] else "default",
                        )
                    with Static(classes="setting-row expert-only"):
                        yield Label(i18n.t("settings.timeout"))
                        yield Input(str(llm_cfg.COMPLEX_LLM_TIMEOUT), type="number", id="inp-complex-timeout")
                    yield Label(i18n.t("settings.report_section"), classes="section-label")
                    with Static(classes="setting-row"):
                        yield Label(i18n.t("settings.base_url"))
                        yield Input(llm_cfg.ANALYSIS_REPORT_LLM_BASE_URL, id="inp-report-url")
                        yield Button(i18n.t("settings.key_test"), id="btn-test-report-url", classes="key-btn")
                    with Static(classes="setting-row"):
                        yield Label(i18n.t("settings.api_key"))
                        yield Input(llm_cfg.ANALYSIS_REPORT_LLM_API_KEY, password=True, id="inp-report-key")
                        yield Button(i18n.t("settings.key_show"), id="btn-show-report-key", classes="key-btn")
                    with Static(classes="setting-row"):
                        yield Label(i18n.t("settings.model_name"))
                        yield Input(llm_cfg.ANALYSIS_REPORT_LLM_MODEL, id="inp-report-model")
                        yield Button(
                            self._thinking_label("report"), id="btn-thinking-report", classes="key-btn",
                            variant="primary" if self._thinking["report"] else "default",
                        )
                    with Static(classes="setting-row expert-only"):
                        yield Label(i18n.t("settings.report_temp"))
                        yield Input(str(llm_cfg.ANALYSIS_REPORT_LLM_TEMPERATURE), type="number", id="inp-report-temp")
                    with Static(classes="setting-row expert-only"):
                        yield Label(i18n.t("settings.timeout"))
                        yield Input(str(llm_cfg.ANALYSIS_REPORT_LLM_TIMEOUT), type="number", id="inp-report-timeout")
                with TabPane(i18n.t("settings.tab_about"), id="tab-about"):
                    yield Label(i18n.t("settings.about_desc"), id="about-desc")
                    yield Label(i18n.t("settings.tab_analysis"), classes="section-label")
                    yield DataTable(id="help-analysis-table", classes="help-table")
                    yield Label(i18n.t("settings.tab_llm"), classes="section-label")
                    yield DataTable(id="help-llm-table", classes="help-table")
                    yield Label(i18n.t("settings.tab_general"), classes="section-label")
                    yield DataTable(id="help-general-table", classes="help-table")
                    yield Label(i18n.t("settings.paths_section"), classes="section-label")
                    yield Label("", id="paths-info", classes="help-text")
                    yield Label("", id="about-info")
                    with Static(classes="about-links"):
                        yield Button(i18n.t("settings.about_project"), id="btn-open-project", classes="link-btn")
                        yield Button(i18n.t("settings.about_author_home"), id="btn-open-author", classes="link-btn")
            with Horizontal(id="settings-buttons"):
                yield Button(i18n.t("settings.reset"), variant="warning", id="btn-reset-current")
                yield Static(classes="btn-spacer")
                yield Button(i18n.t("settings.save"), variant="primary", id="btn-save")
                yield Button(i18n.t("settings.cancel"), id="btn-cancel")

    def on_mount(self) -> None:
        self._apply_ui_mode()
        self._constrain_tab_panes()
        self._disable_decorative_focus()
        settings = get_settings()
        self._select_current_theme()
        self._highlight_option("#seg-mode-options", settings.SEGMENTATION_MODE)
        self._highlight_option("#zone-policy-options", settings.CORPUS_ZONE_POLICY)
        self._highlight_option("#granularity-options", settings.TEMPORAL_GRANULARITY)
        self._highlight_option("#viz-backend-options", settings.VISUALIZATION_BACKEND)
        self._fill_paths_info()
        self._fill_help_tables()
        self._fill_about_info()
        self._sync_reset_button()
        self.run_worker(self._load_credential_fields(), exclusive=False)

    def _constrain_tab_panes(self) -> None:
        """TabPane 默认 height:auto 会撑破小窗口，强制限高并启用滚动；
        显示页签例外：其 OptionList 自带滚动，页签再滚会嵌套，改由列表独占滚动"""
        for pane in self.query(TabPane):
            pane.styles.height = "1fr"
            pane.styles.overflow_y = "hidden" if pane.id == "tab-display" else "auto"

    def _disable_decorative_focus(self) -> None:
        """按钮与只读表格不可聚焦，消除点击后默认焦点样式的白底；输入控件不受影响"""
        for widget in self.query(Button):
            widget.can_focus = False
        for widget in self.query(DataTable):
            widget.can_focus = False

    def _highlight_option(self, selector: str, option_id: str) -> None:
        """将 OptionList 高亮定位到指定 id 的选项，未命中时保持首位"""
        options = self.query_one(selector, OptionList)
        for index in range(options.option_count):
            if str(options.get_option_at_index(index).id) == option_id:
                options.highlighted = index
                return

    def _select_theme_option(self, theme_id: str) -> None:
        theme_options = self.query_one("#theme-options", OptionList)
        theme_options.highlighted = THEME_IDS.index(theme_id) if theme_id in THEME_IDS else 0

    def _select_current_theme(self) -> None:
        self._select_theme_option(self.app.theme)

    def _theme_label(self, theme_id: str) -> str:
        return i18n.raw("settings.theme_names").get(theme_id, theme_id)

    def _fill_paths_info(self) -> None:
        """实时解析并刷新路径说明，保证路径变更后显示始终准确"""
        settings = get_settings()
        info = self.query_one("#paths-info", Label)
        info.update(
            f"{i18n.t('settings.output_dir')}: {settings.resolve_data_path(settings.OUTPUT_DIR)}\n"
            f"{i18n.t('settings.cache_dir')}: {settings.resolve_data_path(settings.CACHE_DIR)}\n"
            f"{i18n.t('settings.log_dir')}: {settings.resolve_data_path(settings.LOG_DIR)}"
        )

    def _fill_help_tables(self) -> None:
        """填充关于页参数说明表（无表头，两列：参数名/说明）；基础模式过滤专家级条目与控件显隐分级一致"""
        show_expert = self._ui_mode == "expert"
        for table_id, help_key in (
            ("#help-analysis-table", "settings.help_analysis"),
            ("#help-llm-table", "settings.help_llm"),
            ("#help-general-table", "settings.help_general"),
        ):
            table = self.query_one(table_id, DataTable)
            table.clear(columns=True)
            table.show_header = False
            table.add_columns("参数", "说明")
            for name, desc in i18n.raw(help_key).items():
                if not show_expert and name in _EXPERT_HELP_KEYS:
                    continue
                table.add_row(name, desc)

    async def _load_credential_fields(self) -> None:
        """后台线程解析凭证并回填输入框与状态文案：compose 内同步调用会触发 bilibili_api 首次导入冻结界面；
        回填完成后再串行刷新用户名（并行会因缓存命中时本 worker 晚到而把用户名覆盖回来源文案）"""
        fields, status = await asyncio.to_thread(lambda: (self._current_credential(), self._credential_status_text()))
        self.query_one("#inp-cred-sessdata", Input).value = fields["sessdata"]
        self.query_one("#inp-cred-jct", Input).value = fields["bili_jct"]
        self.query_one("#inp-cred-buvid3", Input).value = fields["buvid3"]
        self.query_one("#credential-status-label", Label).update(status)
        await self._refresh_credential_account(fields["sessdata"])

    @staticmethod
    def _current_credential() -> dict:
        from ...account import resolve_credential

        credential, _ = resolve_credential()
        if credential is None:
            return {"sessdata": "", "bili_jct": "", "buvid3": ""}
        return {
            "sessdata": credential.sessdata or "",
            "bili_jct": credential.bili_jct or "",
            "buvid3": credential.buvid3 or "",
        }

    @staticmethod
    def _credential_status_text() -> str:
        from ...account import resolve_credential

        credential, source = resolve_credential()
        if credential is None:
            return i18n.t("settings.credential_none")
        if source == "login":
            return i18n.t("settings.credential_login")
        if source == "settings":
            return i18n.t("settings.credential_env")
        return i18n.t("settings.credential_none")

    async def _refresh_credential_account(self, sessdata: str = "") -> None:
        """异步拉取登录用户名与 UID，替换凭证状态括号内的来源说明；失败时保持原文案"""
        from ...account import fetch_account_info, resolve_credential

        if not sessdata:
            credential, _ = await asyncio.to_thread(resolve_credential)
            sessdata = (credential.sessdata if credential else "") or ""
        if not sessdata:
            return
        account = _ACCOUNT_CACHE.get(sessdata)
        if account is None:
            try:
                info = await fetch_account_info(sessdata)
            except Exception:
                return
            if not info.get("is_login"):
                return
            account = (info["uname"], info["mid"])
            _ACCOUNT_CACHE[sessdata] = account
        if account[0]:
            self.query_one("#credential-status-label", Label).update(
                i18n.t("settings.credential_login_user", uname=account[0], mid=account[1])
            )

    def _fill_about_info(self) -> None:
        from ... import __version__

        llm_cfg = get_llm_settings()
        info = self.query_one("#about-info", Label)
        info.update(
            f"[bold]{i18n.t('settings.about_version')}:[/bold] {__version__}\n"
            f"[bold]{i18n.t('settings.about_prompt_version')}:[/bold] {llm_cfg.PROMPT_VERSION}\n"
            f"[bold]{i18n.t('settings.about_author')}:[/bold] SanKaGenKeShi"
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id
        if button_id == "btn-save":
            self._save()
        elif button_id == "btn-cancel":
            self.dismiss(False)
        elif button_id == "btn-reset-current":
            self._reset_current_tab()
        elif button_id.startswith("btn-show-"):
            self._toggle_key_visibility(button_id)
        elif button_id.startswith("btn-copy-"):
            self._copy_field(button_id)
        elif button_id.startswith("btn-paste-"):
            self._paste_field(button_id)
        elif button_id.startswith("btn-thinking-"):
            self._toggle_thinking(button_id)
        elif button_id.startswith("btn-test-"):
            self._test_connection(button_id)
        elif button_id == "btn-mode-basic":
            self._switch_ui_mode("basic")
        elif button_id == "btn-mode-expert":
            self._switch_ui_mode("expert")
        elif button_id == "btn-open-project":
            self._open_url("https://github.com/SanKaGenKeShi/DanmakuScope")
        elif button_id == "btn-open-author":
            self._open_url("https://github.com/SanKaGenKeShi")
        elif button_id == "btn-open-terminal-login":
            self.app._open_system_terminal()
        elif button_id == "btn-logout":
            self.app.push_screen(_LogoutConfirmScreen(), self._on_logout_confirmed)

    def _apply_ui_mode(self) -> None:
        """基础模式隐藏 expert-only 控件降低认知负担，专家模式全量展示；两档共用同一批控件，保存/恢复默认逻辑不分档"""
        show_expert = self._ui_mode == "expert"
        for widget in self.query(".expert-only"):
            widget.display = show_expert
        self.query_one("#btn-mode-basic", Button).variant = "default" if show_expert else "primary"
        self.query_one("#btn-mode-expert", Button).variant = "primary" if show_expert else "default"

    def _switch_ui_mode(self, mode: str) -> None:
        """模式切换会话内即时生效并落盘（界面偏好，与主题/动画同性质，不受保存/取消约束）；参数说明表同步重填"""
        self._ui_mode = mode
        self._apply_ui_mode()
        self._fill_help_tables()
        save_prefs({"ui_mode": mode})

    def _reset_current_tab(self) -> None:
        """根据当前激活标签页执行对应的恢复默认逻辑（仅可设置页）"""
        tabs = self.query_one(TabbedContent)
        active = tabs.active
        if active == "tab-display":
            self._reset_appearance()
        elif active == "tab-analysis":
            self._reset_analysis()
        elif active == "tab-llm":
            self._reset_llm()

    _RESETTABLE_TABS = {"tab-display", "tab-analysis", "tab-llm"}

    def on_tabbed_content_tab_activated(self, event: TabbedContent.TabActivated) -> None:
        """切换到只读页时隐藏恢复默认按钮，可设置页时显示；进入关于页时实时刷新路径"""
        self._sync_reset_button()
        if str(event.pane.id) == "tab-about":
            self._fill_paths_info()

    def _sync_reset_button(self) -> None:
        tabs = self.query_one(TabbedContent)
        reset_btn = self.query_one("#btn-reset-current", Button)
        active = str(tabs.active) if tabs.active else ""
        reset_btn.display = active in self._RESETTABLE_TABS
        reset_btn.refresh()

    def _reset_appearance(self) -> None:
        """仅重置界面控件，不触碰运行时状态：点保存才生效落盘，取消则丢弃（主题选中态经 _select_theme_option 同步）"""
        self._select_theme_option(DEFAULT_THEME)
        self.query_one("#sw-animations", Switch).value = True
        self.notify(i18n.t("settings.reset_done"), severity="information")

    def _open_url(self, url: str) -> None:
        import webbrowser

        webbrowser.open(url)

    @staticmethod
    def _field_input_id(button_id: str) -> str:
        """btn-<动作>-<字段> → inp-<字段>（如 btn-copy-simple-url → inp-simple-url）"""
        return "inp-" + "-".join(button_id.split("-")[2:])

    def _toggle_key_visibility(self, button_id: str) -> None:
        key_input = self.query_one(f"#{self._field_input_id(button_id)}", Input)
        toggle_btn = self.query_one(f"#{button_id}", Button)
        key_input.password = not key_input.password
        toggle_btn.label = i18n.t("settings.key_hide") if not key_input.password else i18n.t("settings.key_show")

    def _copy_field(self, button_id: str) -> None:
        value = self.query_one(f"#{self._field_input_id(button_id)}", Input).value
        self.app.copy_to_clipboard(value)
        self.notify(i18n.t("settings.key_copied"), severity="information")

    def _thinking_label(self, prefix: str) -> str:
        return i18n.t("settings.thinking_on") if self._thinking[prefix] else i18n.t("settings.thinking_off")

    def _apply_thinking_button(self, prefix: str) -> None:
        """同步思考按钮文案与背景（开启时蓝色 primary）"""
        btn = self.query_one(f"#btn-thinking-{prefix}", Button)
        btn.label = self._thinking_label(prefix)
        btn.variant = "primary" if self._thinking[prefix] else "default"

    def _toggle_thinking(self, button_id: str) -> None:
        """思考模式按钮开关：点击切换状态并更新按钮外观"""
        prefix = button_id.split("-")[-1]
        self._thinking[prefix] = not self._thinking[prefix]
        self._apply_thinking_button(prefix)

    def _test_connection(self, button_id: str) -> None:
        """检测 Base URL 对应 LLM 是否可正常通讯（读取当前输入框未保存值）"""
        prefix = "complex" if "complex" in button_id else ("report" if "report" in button_id else "simple")
        base_url = self.query_one(f"#inp-{prefix}-url", Input).value.strip()
        api_key = self.query_one(f"#inp-{prefix}-key", Input).value.strip()
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

    def action_paste_clipboard(self) -> None:
        """终端粘贴事件不可靠，Ctrl+V 走系统剪贴板 API 向当前聚焦输入框插入"""
        focused = self.focused
        if not isinstance(focused, Input):
            return
        self._insert_clipboard_into(focused)

    def _paste_field(self, button_id: str) -> None:
        target = self.query_one(f"#{self._field_input_id(button_id)}", Input)
        text = self.app._read_system_clipboard().strip()
        if not text:
            return
        if target.disabled:
            target.value = text
        else:
            target.focus()
            target.insert_text_at_cursor(text)

    def _insert_clipboard_into(self, target: Input) -> None:
        text = self.app._read_system_clipboard()
        if text:
            target.focus()
            target.insert_text_at_cursor(text)

    def _reset_analysis(self) -> None:
        """仅重置界面控件不修改配置单例：点保存才生效落盘，取消则丢弃"""
        self._highlight_option("#seg-mode-options", _ANALYSIS_DEFAULTS["SEGMENTATION_MODE"])
        self.query_one("#inp-min-samples", Input).value = str(_ANALYSIS_DEFAULTS["MIN_SEGMENT_SAMPLES"])
        self.query_one("#sw-freq-sampling", Switch).value = _ANALYSIS_DEFAULTS["ENABLE_FREQ_BASED_SAMPLING"]
        self.query_one("#sw-batch-analysis", Switch).value = _ANALYSIS_DEFAULTS["ENABLE_BATCH_SEGMENT_ANALYSIS"]
        self.query_one("#inp-top-n", Input).value = str(_ANALYSIS_DEFAULTS["TOP_N"])
        self.query_one("#inp-moe", Input).value = str(_ANALYSIS_DEFAULTS["MOE"])
        self.query_one("#inp-confidence-level", Input).value = str(_ANALYSIS_DEFAULTS["CONFIDENCE_LEVEL"])
        self.query_one("#sw-llm-report", Switch).value = _ANALYSIS_DEFAULTS["ENABLE_LLM_ANALYSIS_REPORT"]
        self.query_one("#inp-concurrency", Input).value = str(_ANALYSIS_DEFAULTS["LLM_CONCURRENCY"])
        self.query_one("#sw-llm-tokenizer", Switch).value = _ANALYSIS_DEFAULTS["ENABLE_LLM_TOKENIZER"]
        self.query_one("#inp-tokenizer-min-len", Input).value = str(_ANALYSIS_DEFAULTS["LLM_TOKENIZER_MIN_LENGTH"])
        self.query_one("#inp-context-window", Input).value = str(_ANALYSIS_DEFAULTS["CONTEXT_TIME_WINDOW"])
        self.query_one("#inp-max-context-tokens", Input).value = str(_ANALYSIS_DEFAULTS["MAX_CONTEXT_TOKENS"])
        self.query_one("#inp-corpus-min-videos", Input).value = str(_CORPUS_DEFAULTS["CORPUS_MIN_VIDEOS_PER_PARTITION"])
        self._highlight_option("#zone-policy-options", _CORPUS_DEFAULTS["CORPUS_ZONE_POLICY"])
        self.query_one("#sw-temporal-grouping", Switch).value = _CORPUS_DEFAULTS["ENABLE_TEMPORAL_GROUPING"]
        self._highlight_option("#granularity-options", _CORPUS_DEFAULTS["TEMPORAL_GRANULARITY"])
        self.query_one("#sw-corpus-statistics", Switch).value = _CORPUS_DEFAULTS["ENABLE_CORPUS_STATISTICS"]
        self.query_one("#inp-scheduler-workers", Input).value = str(_CORPUS_DEFAULTS["SCHEDULER_WORKERS"])
        self._highlight_option("#viz-backend-options", _CORPUS_DEFAULTS["VISUALIZATION_BACKEND"])
        self.query_one("#sw-compare-reuse", Switch).value = True
        self.notify(i18n.t("settings.reset_done"), severity="information")

    def _reset_llm(self) -> None:
        """仅重置界面控件与思考状态缓存不修改配置单例：点保存才生效落盘，取消则丢弃"""
        self.query_one("#sw-dual-path", Switch).value = _LLM_DEFAULTS["ENABLE_DUAL_PATH"]
        self.query_one("#inp-jsd-low", Input).value = str(_LLM_DEFAULTS["JSD_THRESHOLD_LOW"])
        self.query_one("#inp-jsd-medium", Input).value = str(_LLM_DEFAULTS["JSD_THRESHOLD_MEDIUM"])
        for prefix, key in (
            ("simple", "SIMPLE_LLM_ENABLE_THINKING"),
            ("complex", "COMPLEX_LLM_ENABLE_THINKING"),
            ("report", "ANALYSIS_REPORT_LLM_ENABLE_THINKING"),
        ):
            self._thinking[prefix] = _LLM_DEFAULTS[key]
            self._apply_thinking_button(prefix)
        self.query_one("#inp-report-temp", Input).value = str(_LLM_DEFAULTS["ANALYSIS_REPORT_LLM_TEMPERATURE"])
        self.query_one("#inp-simple-timeout", Input).value = str(_LLM_DEFAULTS["SIMPLE_LLM_TIMEOUT"])
        self.query_one("#inp-complex-timeout", Input).value = str(_LLM_DEFAULTS["COMPLEX_LLM_TIMEOUT"])
        self.query_one("#inp-report-timeout", Input).value = str(_LLM_DEFAULTS["ANALYSIS_REPORT_LLM_TIMEOUT"])
        self.notify(i18n.t("settings.reset_done"), severity="information")

    def _save(self) -> None:
        settings = get_settings()
        llm_cfg = get_llm_settings()

        theme_options = self.query_one("#theme-options", OptionList)
        theme = theme_options.get_option_at_index(theme_options.highlighted)
        if theme and str(theme.id) != self.app.theme:
            self.app.theme = str(theme.id)
        animations = self.query_one("#sw-animations", Switch).value
        self.app._set_animations(animations)

        seg_options = self.query_one("#seg-mode-options", OptionList)
        seg_mode = seg_options.get_option_at_index(seg_options.highlighted)
        if seg_mode:
            settings.SEGMENTATION_MODE = str(seg_mode.id)
        settings.MIN_SEGMENT_SAMPLES = self._int_value("#inp-min-samples", settings.MIN_SEGMENT_SAMPLES)
        settings.ENABLE_FREQ_BASED_SAMPLING = self.query_one("#sw-freq-sampling", Switch).value
        settings.ENABLE_BATCH_SEGMENT_ANALYSIS = self.query_one("#sw-batch-analysis", Switch).value
        settings.TOP_N = self._int_value("#inp-top-n", settings.TOP_N)
        settings.MOE = self._float_value("#inp-moe", settings.MOE)
        settings.CONFIDENCE_LEVEL = self._float_value("#inp-confidence-level", settings.CONFIDENCE_LEVEL)
        settings.ENABLE_LLM_ANALYSIS_REPORT = self.query_one("#sw-llm-report", Switch).value
        settings.LLM_CONCURRENCY = self._int_value("#inp-concurrency", settings.LLM_CONCURRENCY)
        settings.ENABLE_LLM_TOKENIZER = self.query_one("#sw-llm-tokenizer", Switch).value
        settings.LLM_TOKENIZER_MIN_LENGTH = self._int_value("#inp-tokenizer-min-len", settings.LLM_TOKENIZER_MIN_LENGTH)
        settings.CONTEXT_TIME_WINDOW = self._float_value("#inp-context-window", settings.CONTEXT_TIME_WINDOW)
        settings.MAX_CONTEXT_TOKENS = self._int_value("#inp-max-context-tokens", settings.MAX_CONTEXT_TOKENS)

        llm_cfg.ENABLE_DUAL_PATH = self.query_one("#sw-dual-path", Switch).value
        llm_cfg.SIMPLE_LLM_ENABLE_THINKING = self._thinking["simple"]
        llm_cfg.COMPLEX_LLM_ENABLE_THINKING = self._thinking["complex"]
        llm_cfg.ANALYSIS_REPORT_LLM_ENABLE_THINKING = self._thinking["report"]
        llm_cfg.JSD_THRESHOLD_LOW = self._float_value("#inp-jsd-low", llm_cfg.JSD_THRESHOLD_LOW)
        llm_cfg.JSD_THRESHOLD_MEDIUM = self._float_value("#inp-jsd-medium", llm_cfg.JSD_THRESHOLD_MEDIUM)

        llm_cfg.COMPLEX_LLM_BASE_URL = self._text_value("#inp-complex-url")
        llm_cfg.COMPLEX_LLM_API_KEY = self._text_value("#inp-complex-key")
        llm_cfg.COMPLEX_LLM_MODEL = self._text_value("#inp-complex-model")
        llm_cfg.SIMPLE_LLM_BASE_URL = self._text_value("#inp-simple-url")
        llm_cfg.SIMPLE_LLM_API_KEY = self._text_value("#inp-simple-key")
        llm_cfg.SIMPLE_LLM_MODEL = self._text_value("#inp-simple-model")
        llm_cfg.ANALYSIS_REPORT_LLM_BASE_URL = self._text_value("#inp-report-url")
        llm_cfg.ANALYSIS_REPORT_LLM_API_KEY = self._text_value("#inp-report-key")
        llm_cfg.ANALYSIS_REPORT_LLM_MODEL = self._text_value("#inp-report-model")
        llm_cfg.ANALYSIS_REPORT_LLM_TEMPERATURE = self._float_value("#inp-report-temp", llm_cfg.ANALYSIS_REPORT_LLM_TEMPERATURE)
        llm_cfg.SIMPLE_LLM_TIMEOUT = self._float_value("#inp-simple-timeout", llm_cfg.SIMPLE_LLM_TIMEOUT)
        llm_cfg.COMPLEX_LLM_TIMEOUT = self._float_value("#inp-complex-timeout", llm_cfg.COMPLEX_LLM_TIMEOUT)
        llm_cfg.ANALYSIS_REPORT_LLM_TIMEOUT = self._float_value("#inp-report-timeout", llm_cfg.ANALYSIS_REPORT_LLM_TIMEOUT)

        settings.CORPUS_MIN_VIDEOS_PER_PARTITION = self._int_value("#inp-corpus-min-videos", settings.CORPUS_MIN_VIDEOS_PER_PARTITION)
        zone_options = self.query_one("#zone-policy-options", OptionList)
        zone_policy = zone_options.get_option_at_index(zone_options.highlighted)
        if zone_policy:
            settings.CORPUS_ZONE_POLICY = str(zone_policy.id)
        settings.ENABLE_TEMPORAL_GROUPING = self.query_one("#sw-temporal-grouping", Switch).value
        settings.ENABLE_CORPUS_STATISTICS = self.query_one("#sw-corpus-statistics", Switch).value
        gran_options = self.query_one("#granularity-options", OptionList)
        granularity = gran_options.get_option_at_index(gran_options.highlighted)
        if granularity:
            settings.TEMPORAL_GRANULARITY = str(granularity.id)
        settings.SCHEDULER_WORKERS = self._int_value("#inp-scheduler-workers", settings.SCHEDULER_WORKERS)
        viz_options = self.query_one("#viz-backend-options", OptionList)
        viz_backend = viz_options.get_option_at_index(viz_options.highlighted)
        if viz_backend:
            settings.VISUALIZATION_BACKEND = str(viz_backend.id)

        self._save_credentials()

        prefs = {key: getattr(settings, key) for key in PERSIST_SETTINGS_KEYS}
        prefs["compare_reuse"] = self.query_one("#sw-compare-reuse", Switch).value
        prefs["theme"] = self.app.theme
        prefs["animations"] = animations
        prefs["ui_mode"] = self._ui_mode
        save_prefs(prefs)
        # LLM 配置写回 .env（.env 为 LLM 配置唯一数据源，CLI/TUI 启动时直接加载）
        write_llm_env({key: getattr(llm_cfg, key) for key in ENV_LLM_KEYS})

        self.notify(i18n.t("settings.saved"), severity="information")

    def _save_credentials(self) -> None:
        """手动填写的凭证写入登录凭证文件（credential.json，解析优先级高于 .env）"""
        from ...account import save_credential

        sessdata = self.query_one("#inp-cred-sessdata", Input).value.strip()
        bili_jct = self.query_one("#inp-cred-jct", Input).value.strip()
        buvid3 = self.query_one("#inp-cred-buvid3", Input).value.strip()
        if not sessdata:
            return
        current = self._current_credential()
        if (sessdata, bili_jct, buvid3) == (current["sessdata"], current["bili_jct"], current["buvid3"]):
            return
        save_credential({"sessdata": sessdata, "bili_jct": bili_jct, "buvid3": buvid3})
        self.query_one("#credential-status-label", Label).update(i18n.t("settings.credential_login"))
        self.run_worker(self._refresh_credential_account(sessdata), exclusive=False)

    def _on_logout_confirmed(self, confirmed: bool) -> None:
        if not confirmed:
            return
        self.run_worker(self._logout_async(), exclusive=False)

    async def _logout_async(self) -> None:
        """先通知B站服务端失效会话，再删除本地凭证；服务端失败不阻断本地删除"""
        from ...account import load_credential, logout, remote_logout

        credential = load_credential()
        if not credential:
            self.notify(i18n.t("settings.logout_none"), severity="warning")
            return
        try:
            remote_ok = await remote_logout(credential)
        except Exception:
            remote_ok = False
        logout()
        _ACCOUNT_CACHE.clear()
        for field_id in ("#inp-cred-sessdata", "#inp-cred-jct", "#inp-cred-buvid3"):
            self.query_one(field_id, Input).value = ""
        self.query_one("#credential-status-label", Label).update(self._credential_status_text())
        key = "settings.logout_done" if remote_ok else "settings.logout_done_remote_failed"
        self.notify(i18n.t(key), severity="information")

    def _int_value(self, selector: str, fallback: int) -> int:
        try:
            return int(self.query_one(selector, Input).value)
        except (ValueError, TypeError):
            return fallback

    def _float_value(self, selector: str, fallback: float) -> float:
        try:
            return float(self.query_one(selector, Input).value)
        except (ValueError, TypeError):
            return fallback

    def _text_value(self, selector: str) -> str:
        """清空输入框属合法操作（如移除 API Key），直接返回输入值不做旧值回填"""
        return self.query_one(selector, Input).value.strip()

    def action_cancel(self) -> None:
        self.dismiss(False)


class _LogoutConfirmScreen(ModalScreen[bool]):
    """退出登录确认对话框（删除 credential.json 不可逆，需显式确认）"""

    DEFAULT_CSS = """
    _LogoutConfirmScreen {
        align: center middle;
    }

    #logout-confirm-dialog {
        width: 74;
        height: auto;
        border: thick $error;
        background: $background;
        padding: 1 2;
    }

    #logout-confirm-dialog Label {
        width: 100%;
        height: auto;
    }

    #logout-confirm-buttons {
        height: 3;
        layout: horizontal;
        align: right middle;
        margin-top: 1;
    }

    #logout-confirm-buttons Button {
        margin-left: 1;
    }
    """

    def compose(self) -> ComposeResult:
        with Vertical(id="logout-confirm-dialog"):
            yield Label(i18n.t("settings.logout_confirm"))
            with Horizontal(id="logout-confirm-buttons"):
                yield Button(i18n.t("settings.logout"), variant="error", id="btn-confirm-logout")
                yield Button(i18n.t("settings.cancel"), id="btn-cancel-logout")

    def on_mount(self) -> None:
        for widget in self.query(Button):
            widget.can_focus = False

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id == "btn-confirm-logout")
