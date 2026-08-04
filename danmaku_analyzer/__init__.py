"""
DanmakuScope - B站弹幕社会语言学分析工具
"""

from importlib.metadata import version as _pkg_version, PackageNotFoundError

try:
    __version__ = _pkg_version("danmaku-analyzer")
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
]
