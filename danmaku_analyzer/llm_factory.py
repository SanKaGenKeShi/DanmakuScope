"""LLM 客户端统一工厂 - 所有调用方的 OpenAI 客户端构建收口于此

调用方不再各自实例化 OpenAI/AsyncOpenAI，避免凭证/超时/占位符检测逻辑分散。
"""

from openai import AsyncOpenAI

from .llm_config import get_llm_settings
from .utils.logger import get_logger

logger = get_logger(__name__)

_PLACEHOLDER_KEYS = ("sk-xxx", "sk-yyy", "sk-zzz", "")


def _check_api_key(name: str, api_key: str) -> None:
    """API Key 占位符检测：未配置 .env 时提前提示，避免运行时报 401 无从排查"""
    if api_key in _PLACEHOLDER_KEYS:
        logger.warning(f"{name} 未配置（当前为占位值），请在 .env 中设置真实 Key")


def create_async_client(base_url: str, api_key: str, timeout: float = 60.0) -> AsyncOpenAI:
    return AsyncOpenAI(base_url=base_url, api_key=api_key, timeout=timeout)


def complex_async_client(timeout: float = 60.0) -> AsyncOpenAI:
    """复杂任务双路推理客户端（COMPLEX_LLM 配置）"""
    cfg = get_llm_settings()
    _check_api_key("COMPLEX_LLM_API_KEY", cfg.COMPLEX_LLM_API_KEY)
    return create_async_client(cfg.COMPLEX_LLM_BASE_URL, cfg.COMPLEX_LLM_API_KEY, timeout)


def simple_async_client(timeout: float = 60.0) -> AsyncOpenAI:
    """简单任务客户端（SIMPLE_LLM 配置）"""
    cfg = get_llm_settings()
    _check_api_key("SIMPLE_LLM_API_KEY", cfg.SIMPLE_LLM_API_KEY)
    return create_async_client(cfg.SIMPLE_LLM_BASE_URL, cfg.SIMPLE_LLM_API_KEY, timeout)


def analysis_report_async_client(timeout: float = 120.0) -> AsyncOpenAI:
    """分析报告客户端（ANALYSIS_REPORT_LLM 独立配置；空值以占位符构造，实际调用将失败并已告警）"""
    cfg = get_llm_settings()
    _check_api_key("ANALYSIS_REPORT_LLM_API_KEY", cfg.ANALYSIS_REPORT_LLM_API_KEY)
    return create_async_client(
        cfg.ANALYSIS_REPORT_LLM_BASE_URL or "https://api.openai.com/v1",
        cfg.ANALYSIS_REPORT_LLM_API_KEY or "sk-zzz",
        timeout,
    )
