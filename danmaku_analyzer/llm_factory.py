"""LLM 后端统一工厂 - Protocol 定义 + adapter 实现 + 按用途构建收口于此

插件化后端：每个 adapter 实例绑定一个固定连接配置（双轨约束在 Protocol 层保持，
不提供任何"共享连接"接口）；调用方不再各自实例化 AsyncOpenAI。
"""

from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from openai import AsyncOpenAI

from .llm_config import get_llm_settings
from .utils.logger import get_logger

logger = get_logger(__name__)

_PLACEHOLDER_KEYS = ("sk-xxx", "sk-yyy", "sk-zzz")


def _check_api_key(name: str, api_key: str) -> None:
    """API Key 占位符检测：未配置 .env 时提前提示，避免运行时报 401 无从排查"""
    if api_key in _PLACEHOLDER_KEYS:
        logger.warning(f"{name} 未配置（当前为占位值），请在 .env 中设置真实 Key")


def create_async_client(base_url: str, api_key: str, timeout: float = 60.0) -> AsyncOpenAI:
    """空 key（本地推理后端如 Ollama）以 ollama 约定哑值构造，openai SDK 拒绝空串凭证"""
    return AsyncOpenAI(base_url=base_url, api_key=api_key or "ollama", timeout=timeout)


@runtime_checkable
class LLMBackend(Protocol):
    """后端协议：单次补全 + 模型列表探测；实例与连接配置一对一绑定"""

    backend_name: str
    base_url: str
    timeout: float

    async def complete(
        self,
        *,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float,
        response_format: Optional[Dict[str, str]] = None,
        extra_body: Optional[Dict[str, Any]] = None,
    ) -> str: ...

    async def list_models(self) -> List[str]: ...


class OpenAICompatibleBackend:
    """OpenAI 兼容后端（远程 API 与 llama.cpp 系本地服务均走此 adapter）"""

    backend_name = "openai-compatible"

    def __init__(self, base_url: str, api_key: str, timeout: float = 60.0):
        self.base_url = base_url
        self.timeout = timeout
        self._client = create_async_client(base_url, api_key, timeout)

    async def complete(
        self,
        *,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float,
        response_format: Optional[Dict[str, str]] = None,
        extra_body: Optional[Dict[str, Any]] = None,
    ) -> str:
        kwargs: Dict[str, Any] = {"model": model, "messages": messages, "temperature": temperature}
        if response_format is not None:
            kwargs["response_format"] = response_format
        if extra_body is not None:
            kwargs["extra_body"] = extra_body
        response = await self._client.chat.completions.create(**kwargs)
        return response.choices[0].message.content or ""

    async def list_models(self) -> List[str]:
        result = await self._client.models.list()
        return [model.id for model in result.data]


class OllamaBackend(OpenAICompatibleBackend):
    """本地推理后端（Ollama 约定）：以独立 base_url 区分，api_key 允许为空（哑值构造）"""

    backend_name = "ollama"

    def __init__(self, base_url: str = "http://127.0.0.1:11434/v1", api_key: str = "", timeout: float = 60.0):
        super().__init__(base_url, api_key or "ollama", timeout)


def create_backend(base_url: str, api_key: str, timeout: float = 60.0) -> LLMBackend:
    """空 Key 按本地后端约定走 Ollama adapter，其余走 OpenAI 兼容 adapter"""
    if not api_key:
        return OllamaBackend(base_url, api_key, timeout)
    return OpenAICompatibleBackend(base_url, api_key, timeout)


def complex_backend(timeout: float = 60.0) -> LLMBackend:
    """复杂任务双路推理后端（COMPLEX_LLM 独立连接配置）"""
    cfg = get_llm_settings()
    _check_api_key("COMPLEX_LLM_API_KEY", cfg.COMPLEX_LLM_API_KEY)
    return create_backend(cfg.COMPLEX_LLM_BASE_URL, cfg.COMPLEX_LLM_API_KEY, timeout)


def simple_backend(timeout: float = 60.0) -> LLMBackend:
    """简单任务后端（SIMPLE_LLM 独立连接配置）"""
    cfg = get_llm_settings()
    _check_api_key("SIMPLE_LLM_API_KEY", cfg.SIMPLE_LLM_API_KEY)
    return create_backend(cfg.SIMPLE_LLM_BASE_URL, cfg.SIMPLE_LLM_API_KEY, timeout)


def analysis_report_backend(timeout: float = 120.0) -> LLMBackend:
    """分析报告后端（ANALYSIS_REPORT_LLM 独立连接配置）"""
    cfg = get_llm_settings()
    _check_api_key("ANALYSIS_REPORT_LLM_API_KEY", cfg.ANALYSIS_REPORT_LLM_API_KEY)
    return create_backend(
        cfg.ANALYSIS_REPORT_LLM_BASE_URL or "https://api.openai.com/v1",
        cfg.ANALYSIS_REPORT_LLM_API_KEY,
        timeout,
    )
