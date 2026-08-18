"""偏好持久化模块 - 业务偏好读写（tui_prefs.json）与 LLM 配置写回 .env，CLI 与 TUI 共用

数据源分工：业务/分析参数存 tui_prefs.json；LLM 配置以 .env 为唯一数据源，
TUI 保存时经 `write_llm_env` 写回 .env，启动时由 pydantic-settings 直接加载，
避免双存储漂移（此前 LLM 配置同时存两处，偏好文件覆盖 .env 导致手改 .env 不生效）。
"""

import json
import os
from typing import Optional

from .config import get_settings
from .utils.logger import get_logger

logger = get_logger(__name__)

PERSIST_SETTINGS_KEYS = (
    "SEGMENTATION_MODE", "MIN_SEGMENT_SAMPLES", "ENABLE_FREQ_BASED_SAMPLING",
    "ENABLE_BATCH_SEGMENT_ANALYSIS",
    "TOP_N", "MOE", "CONFIDENCE_LEVEL", "ENABLE_LLM_ANALYSIS_REPORT", "LLM_CONCURRENCY",
    "ENABLE_LLM_TOKENIZER", "LLM_TOKENIZER_MIN_LENGTH",
    "CONTEXT_TIME_WINDOW", "MAX_CONTEXT_TOKENS",
    "CORPUS_MIN_VIDEOS_PER_PARTITION", "CORPUS_ZONE_POLICY",
    "ENABLE_TEMPORAL_GROUPING", "TEMPORAL_GRANULARITY",
    "ENABLE_CORPUS_STATISTICS", "SCHEDULER_WORKERS", "VISUALIZATION_BACKEND",
)
ENV_LLM_KEYS = (
    "ENABLE_DUAL_PATH", "JSD_THRESHOLD_LOW", "JSD_THRESHOLD_MEDIUM",
    "SIMPLE_LLM_ENABLE_THINKING", "COMPLEX_LLM_ENABLE_THINKING",
    "ANALYSIS_REPORT_LLM_ENABLE_THINKING", "ANALYSIS_REPORT_LLM_TEMPERATURE",
    "SIMPLE_LLM_BASE_URL", "SIMPLE_LLM_API_KEY", "SIMPLE_LLM_MODEL",
    "COMPLEX_LLM_BASE_URL", "COMPLEX_LLM_API_KEY", "COMPLEX_LLM_MODEL",
    "ANALYSIS_REPORT_LLM_BASE_URL", "ANALYSIS_REPORT_LLM_API_KEY", "ANALYSIS_REPORT_LLM_MODEL",
    "COMPLEX_LLM_TIMEOUT", "SIMPLE_LLM_TIMEOUT", "ANALYSIS_REPORT_LLM_TIMEOUT",
)


def _prefs_path() -> str:
    return get_settings().resolve_data_path("tui_prefs.json")


def _env_path() -> str:
    """LLM 配置写回目标：DATA_ROOT/.env（与读取侧优先级一致；正式安装下包目录不可写）"""
    return get_settings().resolve_data_path(".env")


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


def write_llm_env(updates: dict, path: Optional[str] = None) -> None:
    """LLM 配置写回 .env：原位替换对应 KEY=VALUE 行（保留注释与其余行），缺失键追加至末尾"""
    path = path or _env_path()
    try:
        with open(path, encoding="utf-8") as f:
            lines = f.readlines()
    except OSError:
        lines = []
    remaining = dict(updates)
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key = stripped.split("=", 1)[0].strip()
        if key in remaining:
            lines[i] = f"{key}={remaining.pop(key)}\n"
    for key, value in remaining.items():
        lines.append(f"{key}={value}\n")
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.writelines(lines)
        # .env 承载 LLM API Key（及可能的 B 站凭证），与 credential.json 同标准收紧权限
        if os.name == "posix":
            try:
                os.chmod(path, 0o600)
            except OSError as e:
                logger.warning(f".env 权限收紧失败（建议手动 chmod 600）: {e}")
    except OSError as e:
        logger.warning(f"LLM 配置写回 .env 失败（仅本次会话生效）: {e}")


def apply_saved_prefs() -> None:
    """将已持久化业务设置应用到配置单例（CLI 与 TUI 入口均调用，重启后仍生效）；
    LLM 配置不经此路径，启动时由 pydantic-settings 直接加载 .env"""
    prefs = load_prefs()
    settings = get_settings()
    for key in PERSIST_SETTINGS_KEYS:
        if key in prefs:
            try:
                setattr(settings, key, prefs[key])
            except Exception as e:
                logger.warning(f"偏好键 {key} 赋值校验失败（手改或旧版残留），忽略该值: {e}")
