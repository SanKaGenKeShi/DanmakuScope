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
    # 版本
    "__version__",
    # 配置
    "get_settings", "Settings",
    # 爬虫
    "BilibiliCrawler", "VideoMeta", "DanmakuItem",
    # 社会变量
    "SocialVariableExtractor", "SocialVariables",
    # 去重
    "UserDeduplicator", "DeduplicationResult",
    # 时序切分
    "TimelineSegmenter", "TimeSegment",
    # 硬统计
    "HardMetricsAnalyzer", "HardMetricsResult",
    # 微语境
    "ContextProvider", "ContextWindow",
    # 统计验证
    "StatisticalValidator", "ConfidenceInterval", "DescriptiveStats",
    # Prompt
    "PromptBuilder", "PromptComponents",
    # LLM
    "LLMClient", "DualPathResult", "ConsensusLevel", "LLMOutput",
    "EmotionOutput", "CooperativePrincipleOutput", "InteractionTypeOutput",
    "SentenceFunctionOutput", "OrthographyOutput",
    # 聚合
    "Aggregator", "AggregatedData", "DanmakuRecord",
    # 报告
    "Reporter", "AnalysisReportGenerator",
    # 缓存
    "CacheManager", "get_cache_manager",
]
