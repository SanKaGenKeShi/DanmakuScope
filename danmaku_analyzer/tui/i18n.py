"""TUI 文案模块 - 中文文案与偏好持久化"""

import json
import os

from ..utils.logger import get_logger

logger = get_logger(__name__)

_STRINGS = {
    "app.sub_title": "B站弹幕社会语言学分析",
    "app.welcome": "欢迎使用 DanmakuScope",
    "app.desc": "B站弹幕社会语言学分析工具，为实证研究提供可溯源、可复核的语料数据。",
    "app.quick_start": "快速开始：",
    "app.step_input": "个体分析：在下方输入框粘贴 BV/AV 号或视频链接，按 Enter 或点击“开始分析”",
    "app.step_analyze": "比对分析：标题栏切换“比对分析”，批量粘贴视频标识，点“开始比对”或 Alt+Enter",
    "app.step_result": "分析进度与比对结果在此实时展示，报告自动打包为 ZIP",
    "app.step_compare": "按 Ctrl+S 打开设置，可调整采样/LLM/语料库/通用参数并查阅参数说明",
    "app.hint": "提示：输出面板支持鼠标选中后 Ctrl+C 复制；设置页含参数说明与凭证查看。",
    "input.placeholder": "请输入BV/AV号或链接",
    "btn.analyze": "开始分析",
    "btn.show_config": "查看配置",
    "btn.settings": "设置",
    "log.start": "开始分析",
    "log.done": "分析完成",
    "log.video": "视频",
    "log.partition": "分区",
    "log.segments": "段数",
    "log.groups": "聚合组",
    "log.zip": "报告 ZIP",
    "log.failed": "分析失败",
    "log.config": "当前配置",
    "log.segmentation": "切分模式",
    "log.min_samples": "最小段样本",
    "log.complex_llm": "复杂LLM",
    "log.simple_llm": "简单LLM",
    "log.dual_path": "双路推理",
    "log.llm_report": "LLM报告",
    "log.enabled": "开启",
    "log.disabled": "关闭",
    "cfg.section_sampling": "采样",
    "cfg.section_llm": "LLM",
    "cfg.section_corpus": "语料库",
    "cfg.section_general": "通用",
    "cfg.section_paths": "路径",
    "cfg.section_interface": "界面",
    "cfg.sampling_strategy": "采样策略",
    "cfg.concurrency": "并发",
    "cfg.report_llm": "报告LLM",
    "notify.empty_input": "请输入 BV 号 / AV 号 / URL",
    "notify.copy_empty": "没有选中的文本可复制",
    "notify.copy_done": "输出已复制到剪贴板",
    "notify.copy_failed": "复制失败",
    "notify.done": "分析完成",
    "notify.failed": "分析失败: {error}",
    "settings.save": "保存",
    "settings.cancel": "取消",
    "settings.saved": "设置已保存（重启后仍生效）",
    "settings.tab_display": "显示",
    "settings.tab_general": "通用",
    "settings.tab_analysis": "分析",
    "settings.tab_llm": "LLM",
    "settings.section_corpus": "语料库比对",
    "settings.help_analysis": {
        "切分模式": "动态按时序弹幕密度突变点自动切分；固定按最小段样本数等量切分。",
        "最小段样本数": "每段弹幕数量下限，不足的段自动并入相邻段。",
        "频次排序": "开启后按出现频次取 TOP_N 弹幕采样；关闭则取每段前 N 条。",
        "采样条数 TOP_N": "每段送入 LLM 分析的弹幕条数。",
        "置信水平": "共识率置信区间（Wilson 区间）的置信水平。",
        "LLM 分析报告": "开启后分析完成自动生成社会语言学分析报告并打包。",
        "LLM 并发上限": "LLM API 请求的最大并发数。",
        "LLM 辅助分词": "开启后长文本由简单任务 LLM 分词；关闭则纯 jieba 分词。",
        "分词触发长度": "达到该长度的文本才交由 LLM 分词。",
        "微语境窗口": "为每条弹幕构建语境参照的时间范围（秒）。",
        "微语境最大 token": "语境注入提示词的 token 上限。",
        "复用过往分析数据": "比对分析时，索引中已有报告 ZIP 的视频直接复用；关闭则全部重新分析。",
        "每分区最少视频数": "语料库比较时视频数少于此值的分区结果仅供参考。",
        "冷热区策略": "仅热区：只聚合热区；两区各保留：冷热区各保留一行；加权合并：按弹幕数加权合并。",
        "按发布时间分桶": "开启后语料库聚合增加历时维度分组。",
        "分桶粒度": "时间分桶的粒度（年/季度/月）。",
    },
    "settings.help_llm": {
        "双路推理": "复杂任务用两个不同温度并行推理，两路分歧用归一化 JSD 度量并判定共识水平。",
        "JSD 低阈值": "低于此值判为高共识（权重 1.0）。",
        "JSD 中阈值": "低/中阈值之间为中共识；达到中阈值判低共识（权重 0.2，按零丢弃原则保留）。",
        "简单任务 LLM": "负责句类判断与 LLM 辅助分词。",
        "复杂任务 LLM": "负责情感、合作原则、互动类型、正字法四类软标签。",
        "分析报告 LLM": "生成单视频与语料库比对分析报告，需独立配置。",
        "思考模式": "控制对应模型是否启用深度思考，三者独立；按钮蓝色背景表示已开启。",
        "报告生成温度": "分析报告生成的采样温度，越低越稳定。",
        "Base URL / API Key / 模型": "均支持粘贴、复制；Key 额外支持显示/隐藏；Base URL 支持连接检测。",
    },
    "settings.help_general": {
        "凭证状态": "当前生效的 B站凭证来源：登录凭证文件（DATA_ROOT/credential.json）优先，其次 .env 环境变量；无凭证时仅能获取少量弹幕。",
        "SESSDATA / bili_jct / buvid3": "B站凭证核心字段，支持显示/隐藏、复制与粘贴；未登录时可点击“打开终端登录”按钮扫码获取。",
        "打开终端登录": "在系统终端中执行 danmaku-analyzer login，扫码登录后凭证自动保存至 credential.json。",
    },
    "settings.credential_status": "凭证状态",
    "settings.credential_login": "已登录（credential.json）",
    "settings.credential_env": ".env 环境变量",
    "settings.credential_none": "无凭证（仅能获取少量弹幕）",
    "settings.open_terminal_login": "打开终端登录",
    "settings.sessdata": "SESSDATA",
    "settings.bili_jct": "bili_jct",
    "settings.buvid3": "buvid3",
    "settings.theme": "主题",
    "settings.seg_mode": "切分模式",
    "settings.min_samples": "最小段样本数",
    "settings.sampling_freq": "频次排序",
    "settings.sampling_head": "每段前 N 条",
    "settings.top_n": "采样条数 TOP_N",
    "settings.confidence_level": "置信水平",
    "settings.llm_report": "LLM 分析报告",
    "settings.llm_concurrency": "LLM 并发上限",
    "settings.llm_tokenizer": "LLM 辅助分词",
    "settings.tokenizer_min_len": "分词触发长度",
    "settings.context_window": "微语境窗口（秒）",
    "settings.max_context_tokens": "微语境最大 token",
    "settings.thinking": "思考模式",
    "settings.thinking_on": "思考:开",
    "settings.thinking_off": "思考:关",
    "settings.dual_path": "双路推理",
    "settings.output_dir": "输出目录",
    "settings.cache_dir": "缓存目录",
    "settings.log_dir": "日志目录",
    "settings.jsd_low": "JSD 低阈值",
    "settings.jsd_medium": "JSD 中阈值",
    "settings.complex_section": "复杂任务 LLM",
    "settings.simple_section": "简单任务 LLM",
    "settings.report_section": "分析报告 LLM",
    "settings.report_temp": "报告生成温度",
    "settings.base_url": "Base URL",
    "settings.api_key": "API Key",
    "settings.model_name": "模型名",
    "settings.key_show": "显示",
    "settings.key_hide": "隐藏",
    "settings.key_copy": "复制",
    "settings.key_paste": "粘贴",
    "settings.key_test": "检测",
    "settings.testing": "正在检测 LLM 连接...",
    "settings.test_ok": "LLM 连接正常",
    "settings.test_failed": "LLM 连接失败: {error}",
    "settings.key_copied": "内容已复制到剪贴板",
    "settings.corpus_min_videos": "每分区最少视频数",
    "settings.corpus_zone_policy": "冷热区策略",
    "settings.corpus_temporal": "按发布时间分桶",
    "settings.corpus_granularity": "分桶粒度",
    "settings.zone_hot_only": "仅热区",
    "settings.zone_all": "两区各保留",
    "settings.zone_weighted": "加权合并",
    "settings.gran_year": "年",
    "settings.gran_quarter": "季度",
    "settings.gran_month": "月",
    "settings.theme_names": {
        "textual-dark": "Textual 深色",
        "textual-light": "Textual 浅色",
        "tokyo-night": "东京之夜",
        "dracula": "德古拉",
        "nord": "北欧",
        "gruvbox": "复古盒",
        "monokai": "莫奈",
        "catppuccin-mocha": "卡布奇诺摩卡",
        "catppuccin-latte": "卡布奇诺拿铁",
        "catppuccin-frappe": "卡布奇诺冰沙",
        "catppuccin-macchiato": "卡布奇诺玛奇朵",
        "solarized-dark": "曝光深色",
        "solarized-light": "曝光浅色",
        "rose-pine": "玫瑰松",
        "rose-pine-moon": "玫瑰松月",
        "atom-one-dark": "Atom 深色",
    },
    "settings.tab_about": "关于此软件",
    "settings.about_version": "版本",
    "settings.about_prompt_version": "Prompt 版本",
    "settings.about_author": "作者",
    "settings.about_project": "项目主页",
    "settings.about_author_home": "作者 GitHub",
    "settings.about_desc": "B站弹幕社会语言学分析工具，为社会语言学/语料库语言学实证研究提供可溯源、可复核的语料数据。",
    "settings.paths_section": "路径说明",
    "settings.reset": "恢复默认",
    "settings.reset_done": "已恢复默认设置",
    "settings.fixed": "固定等分",
    "settings.dynamic": "动态密度",
    "binding.analyze": "开始分析",
    "binding.config": "查看配置",
    "binding.settings": "设置",
    "binding.paste": "粘贴",
    "binding.copy_selection": "复制选中",
    "binding.quit": "退出",
    "mode.single": "个体分析",
    "mode.compare": "比对分析",
    "mode.log": "日志",
    "terminal.opened": "已打开系统终端，请在新窗口扫码登录",
    "terminal.open_failed": "打开系统终端失败，请手动在终端运行 danmaku-analyzer login",
    "compare.placeholder": "粘贴多个 BV / AV / 链接，一行一条（或用逗号、空格分隔）\nEnter 换行，Alt+Enter 开始比对",
    "compare.reuse": "复用过往分析数据",
    "compare.start": "开始比对",
    "compare.empty": "请至少输入一个 BV / AV / 链接",
    "compare.begin": "开始比对分析，共 {count} 个视频",
    "compare.done": "比对分析完成",
    "compare.failed": "比对分析失败: {error}",
    "compare.item_reused": "已复用过往报告",
    "compare.item_failed": "失败",
    "compare.summary": "比对表",
    "compare.snapshot": "语料库快照",
    "error.no_report": "分析未产生有效报告: {input}",
}

# 可持久化到 tui_prefs.json 的设置（本地用户数据，已 gitignore 不入版本库）
PERSIST_SETTINGS_KEYS = (
    "SEGMENTATION_MODE", "MIN_SEGMENT_SAMPLES", "ENABLE_FREQ_BASED_SAMPLING",
    "TOP_N", "CONFIDENCE_LEVEL", "ENABLE_LLM_ANALYSIS_REPORT", "LLM_CONCURRENCY",
    "ENABLE_LLM_TOKENIZER", "LLM_TOKENIZER_MIN_LENGTH",
    "CONTEXT_TIME_WINDOW", "MAX_CONTEXT_TOKENS",
    "CORPUS_MIN_VIDEOS_PER_PARTITION", "CORPUS_ZONE_POLICY",
    "ENABLE_TEMPORAL_GROUPING", "TEMPORAL_GRANULARITY",
)
PERSIST_LLM_KEYS = (
    "ENABLE_DUAL_PATH", "JSD_THRESHOLD_LOW", "JSD_THRESHOLD_MEDIUM",
    "SIMPLE_LLM_ENABLE_THINKING", "COMPLEX_LLM_ENABLE_THINKING",
    "ANALYSIS_REPORT_LLM_ENABLE_THINKING", "ANALYSIS_REPORT_LLM_TEMPERATURE",
    "SIMPLE_LLM_BASE_URL", "SIMPLE_LLM_API_KEY", "SIMPLE_LLM_MODEL",
    "COMPLEX_LLM_BASE_URL", "COMPLEX_LLM_API_KEY", "COMPLEX_LLM_MODEL",
    "ANALYSIS_REPORT_LLM_BASE_URL", "ANALYSIS_REPORT_LLM_API_KEY", "ANALYSIS_REPORT_LLM_MODEL",
)


class I18n:
    def t(self, key: str, **kwargs) -> str:
        """按键取文案，缺失时回退键名本身"""
        text = _STRINGS.get(key, key)
        return text.format(**kwargs) if kwargs else text

    def raw(self, key: str):
        """按键取原始值（用于快捷键映射等非字符串条目）"""
        return _STRINGS.get(key)


def _prefs_path() -> str:
    from ..config import get_settings

    return get_settings().resolve_data_path("tui_prefs.json")


def load_prefs() -> dict:
    """读取 TUI 偏好（已持久化设置），文件缺失/损坏时返回空 dict"""
    try:
        path = _prefs_path()
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError) as e:
        logger.warning(f"TUI 偏好读取失败，使用默认值: {e}")
    return {}


def save_prefs(updates: dict) -> None:
    """合并写入 TUI 偏好（不覆盖已有键）"""
    try:
        prefs = load_prefs()
        prefs.update(updates)
        path = _prefs_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(prefs, f, ensure_ascii=False, indent=2)
    except OSError as e:
        logger.warning(f"TUI 偏好保存失败（仅本次会话生效）: {e}")


def apply_saved_prefs() -> None:
    """启动时将已持久化的设置应用到配置单例"""
    from ..config import get_settings
    from ..llm_config import get_llm_settings

    prefs = load_prefs()
    settings = get_settings()
    llm_cfg = get_llm_settings()
    for key in PERSIST_SETTINGS_KEYS:
        if key in prefs:
            setattr(settings, key, prefs[key])
    for key in PERSIST_LLM_KEYS:
        if key in prefs:
            setattr(llm_cfg, key, prefs[key])


i18n = I18n()
