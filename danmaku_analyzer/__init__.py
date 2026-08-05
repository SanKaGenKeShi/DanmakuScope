"""
DanmakuScope - B站弹幕社会语言学分析工具
"""

from importlib import import_module as _import_module
from importlib.metadata import version as _pkg_version, PackageNotFoundError
import re as _re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .config import get_settings, Settings
    from .crawler import BilibiliCrawler, VideoMeta, DanmakuItem
    from .social_variables import SocialVariables
    from .user_deduplicator import UserDeduplicator, DeduplicationResult
    from .timeline_segmenter import TimelineSegmenter, TimeSegment
    from .hard_metrics import HardMetricsAnalyzer, HardMetricsResult
    from .context_provider import ContextProvider, ContextWindow
    from .statistical_validator import StatisticalValidator, ConfidenceInterval, DescriptiveStats
    from .prompt_builder import PromptBuilder, PromptComponents
    from .llm_client import (
        LLMClient, DualPathResult, ConsensusLevel, LLMOutput,
        EmotionOutput, CooperativePrincipleOutput, InteractionTypeOutput,
        SentenceFunctionOutput, OrthographyOutput,
    )
    from .aggregator import Aggregator, AggregatedData, DanmakuRecord
    from .reporter import Reporter
    from .report_generator import AnalysisReportGenerator
    from .cache_manager import CacheManager, get_cache_manager
    from .corpus_store import CorpusStore
    from .corpus_builder import CorpusBuilder
    from .corpus_suggester import CorpusSuggester
    from .corpus_visualizer import CorpusVisualizer

try:
    __version__ = _pkg_version("danmaku-analyzer")
    # importlib.metadata 会把预发布段规范化为 PEP 440 格式（如 0.2.1-beta → 0.2.1b0），还原为 pyproject.toml 原始写法
    _m = _re.match(r'^(\d+\.\d+\.\d+)b(\d+)$', __version__)
    if _m:
        suffix = "beta" if _m.group(2) == "0" else f"beta{_m.group(2)}"
        __version__ = f"{_m.group(1)}-{suffix}"
except PackageNotFoundError:
    __version__ = "0.0.0-dev"  # 未安装时的回退值

__author__ = "DanmakuScope"

__all__ = [
    "__version__",
    "get_settings", "Settings",
    "BilibiliCrawler", "VideoMeta", "DanmakuItem",
    "SocialVariables",
    "UserDeduplicator", "DeduplicationResult",
    "TimelineSegmenter", "TimeSegment",
    "HardMetricsAnalyzer", "HardMetricsResult",
    "ContextProvider", "ContextWindow",
    "StatisticalValidator", "ConfidenceInterval", "DescriptiveStats",
    "PromptBuilder", "PromptComponents",
    "LLMClient", "DualPathResult", "ConsensusLevel", "LLMOutput",
    "EmotionOutput", "CooperativePrincipleOutput", "InteractionTypeOutput",
    "SentenceFunctionOutput", "OrthographyOutput",
    "Aggregator", "AggregatedData", "DanmakuRecord",
    "Reporter", "AnalysisReportGenerator",
    "CacheManager", "get_cache_manager",
    "CorpusStore", "CorpusBuilder", "CorpusSuggester", "CorpusVisualizer",
]

# 符号 → 所在子模块（PEP 562 懒加载：首次访问才导入，避免仅读版本号时拉起全部重型依赖）
_LAZY_EXPORTS = {
    "get_settings": ".config", "Settings": ".config",
    "BilibiliCrawler": ".crawler", "VideoMeta": ".crawler", "DanmakuItem": ".crawler",
    "SocialVariables": ".social_variables",
    "UserDeduplicator": ".user_deduplicator", "DeduplicationResult": ".user_deduplicator",
    "TimelineSegmenter": ".timeline_segmenter", "TimeSegment": ".timeline_segmenter",
    "HardMetricsAnalyzer": ".hard_metrics", "HardMetricsResult": ".hard_metrics",
    "ContextProvider": ".context_provider", "ContextWindow": ".context_provider",
    "StatisticalValidator": ".statistical_validator",
    "ConfidenceInterval": ".statistical_validator", "DescriptiveStats": ".statistical_validator",
    "PromptBuilder": ".prompt_builder", "PromptComponents": ".prompt_builder",
    "LLMClient": ".llm_client", "DualPathResult": ".llm_client",
    "ConsensusLevel": ".llm_client", "LLMOutput": ".llm_client",
    "EmotionOutput": ".llm_client", "CooperativePrincipleOutput": ".llm_client",
    "InteractionTypeOutput": ".llm_client", "SentenceFunctionOutput": ".llm_client",
    "OrthographyOutput": ".llm_client",
    "Aggregator": ".aggregator", "AggregatedData": ".aggregator", "DanmakuRecord": ".aggregator",
    "Reporter": ".reporter",
    "AnalysisReportGenerator": ".report_generator",
    "CacheManager": ".cache_manager", "get_cache_manager": ".cache_manager",
    "CorpusStore": ".corpus_store",
    "CorpusBuilder": ".corpus_builder",
    "CorpusSuggester": ".corpus_suggester",
    "CorpusVisualizer": ".corpus_visualizer",
}


def __getattr__(name):
    module_path = _LAZY_EXPORTS.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(_import_module(module_path, __name__), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(_LAZY_EXPORTS))
