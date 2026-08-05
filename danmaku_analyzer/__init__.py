"""
DanmakuScope - B站弹幕社会语言学分析工具
"""

from importlib.metadata import version as _pkg_version, PackageNotFoundError
import re as _re

try:
    __version__ = _pkg_version("danmaku-analyzer")
    # importlib.metadata 会把预发布段规范化为 PEP 440 格式（如 0.2.0-beta → 0.2.0b0），还原为 pyproject.toml 原始写法
    _m = _re.match(r'^(\d+\.\d+\.\d+)b(\d+)$', __version__)
    if _m:
        suffix = "beta" if _m.group(2) == "0" else f"beta{_m.group(2)}"
        __version__ = f"{_m.group(1)}-{suffix}"
except PackageNotFoundError:
    __version__ = "0.0.0-dev"  # 未安装时的回退值

__author__ = "DanmakuScope"

from .config import get_settings, Settings
from .crawler import BilibiliCrawler, VideoMeta, DanmakuItem
from .social_variables import SocialVariableExtractor, SocialVariables
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

__all__ = [
    "__version__",
    "get_settings", "Settings",
    "BilibiliCrawler", "VideoMeta", "DanmakuItem",
    "SocialVariableExtractor", "SocialVariables",
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
