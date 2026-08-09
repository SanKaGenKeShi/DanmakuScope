"""
配置中心 - Pydantic Settings 管理业务配置参数
包含抽样统计参数、社会变量配置、输出配置等
LLM 配置已独立到 llm_config.py
"""

import os
from pathlib import Path
from typing import Literal, Optional
from pydantic_settings import BaseSettings
from pydantic import Field

from .llm_config import get_llm_settings, LLMSettings, SETTINGS_MODEL_CONFIG  # noqa: F401


class Settings(BaseSettings):
    """应用配置中心，从环境变量或 .env 文件加载配置"""

    MOE: float = Field(default=0.05, description="抽样误差")
    CONFIDENCE_LEVEL: float = Field(default=0.95, description="置信水平")

    ENABLE_FREQ_BASED_SAMPLING: bool = Field(default=False, description="是否按频次排序采样，默认False使用每段前N条")
    TOP_N: int = Field(default=10, description="频次排序采样时取前N条弹幕")

    ENABLE_SIGNIFICANCE_TESTING: bool = Field(default=False, description="是否启用显著性检验，默认关闭，仅探索")
    SIGNIFICANCE_ALPHA: float = Field(default=0.05, description="显著性检验的 alpha 水平")

    CORPUS_MIN_VIDEOS_PER_PARTITION: int = Field(default=3, description="语料库比较时每分区最少视频数，少于此不参与比较")
    CORPUS_ZONE_POLICY: Literal["hot_only", "all", "weighted"] = Field(default="hot_only", description="语料库聚合冷热区策略：hot_only 仅热区 / all 两区各保留 / weighted 按弹幕数加权合并")
    ENABLE_TEMPORAL_GROUPING: bool = Field(default=False, description="语料库聚合时是否按发布时间分桶（历时维度）")
    TEMPORAL_GRANULARITY: Literal["year", "quarter", "month"] = Field(default="year", description="时间分桶粒度")

    SEGMENTATION_MODE: Literal["fixed", "dynamic"] = Field(default="dynamic", description="切分模式：fixed 或 dynamic")
    DYNAMIC_SEGMENT_METHOD: Literal["density"] = Field(default="density", description="动态切分方法：density")
    MIN_SEGMENT_SAMPLES: int = Field(default=30, description="每个动态分段最少弹幕数")

    ENABLE_LLM_ANALYSIS_REPORT: bool = Field(default=True, description="是否启用LLM分析报告生成")
    LLM_CONCURRENCY: int = Field(default=5, description="LLM 并发调用上限")

    ENABLE_LLM_TOKENIZER: bool = Field(default=False, description="是否启用LLM辅助分词，复用SIMPLE_LLM")
    LLM_TOKENIZER_MIN_LENGTH: int = Field(default=20, description="触发LLM分词的最小文本长度")

    CONTEXT_TIME_WINDOW: float = Field(default=5.0, description="微语境时间窗口（秒）")
    MAX_CONTEXT_TOKENS: int = Field(default=200, description="微语境最大 token 数")

    REGISTER_HINTS: dict = Field(
        default={
            "鬼畜": "反讽、解构、重复造梗、多义性",
            "动画": "厨力释放、角色崇拜、剧情讨论",
            "番剧": "剧情探讨、CP讨论、名场面引用",
            "国创": "国漫支持、剧情分析、角色讨论",
            "音乐": "技术点评、氛围渲染、主观感受",
            "舞蹈": "表现力评价、美感欣赏、编舞讨论",
            "游戏": "操作点评、胜负情绪、策略交流",
            "知识": "信息求证、补充说明、理性讨论",
            "科技": "原理探讨、产品评价、前沿科普",
            "运动": "赛况解说、队伍声援、情绪宣泄",
            "汽车": "车型评价、参数讨论、驾驶体验分享",
            "生活": "日常分享、体验记录、轻松互动",
            "美食": "馋点互动、做法请教、地方风味讨论",
            "动物圈": "萌点评论、拟人化表达、养宠经验分享",
            "时尚": "穿搭点评、产品种草、潮流讨论",
            "娱乐": "八卦调侃、明星崇拜、轻松消遣",
            "影视": "情节分析、角色共情、悬念推测",
            "纪录片": "知识补充、史实讨论、情感共鸣",
            "电影": "剧情分析、镜头点评、主题解读",
            "电视剧": "追剧讨论、CP磕糖、角色命运共情",
        },
        description="社会语域提示映射表"
    )
    DEFAULT_REGISTER_HINT: str = Field(default="一般网络交流、情绪表达", description="默认语域提示")

    @property
    def llm(self) -> LLMSettings:
        return get_llm_settings()

    OUTPUT_DIR: str = Field(default="reports", description="报告输出目录（相对 DATA_ROOT 或绝对路径）")
    CACHE_DIR: str = Field(default="cache", description="缓存目录（相对 DATA_ROOT 或绝对路径）")
    LOG_DIR: str = Field(default="logs", description="日志目录（相对 DATA_ROOT 或绝对路径）")

    DEBUG: bool = Field(default=False, description="调试模式")
    LOG_LEVEL: str = Field(default="INFO", description="日志级别")

    BILIBILI_SESSDATA: str = Field(default="", description="B站SESSDATA")
    BILIBILI_JCT: str = Field(default="", description="B站JCT")
    BILIBILI_BUVID3: str = Field(default="", description="B站BUVID3")

    DATA_ROOT: str = Field(default="", description="用户数据根目录（报告/缓存/日志），默认 ~/.danmaku-scope")
    PROJECT_ROOT: str = Field(default="", description="包安装目录（只读资源：词典等）")
    LEXICON_DIR: str = Field(default="", description="词典目录")
    
    model_config = SETTINGS_MODEL_CONFIG
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if not self.PROJECT_ROOT:
            self.PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
        if not self.LEXICON_DIR:
            self.LEXICON_DIR = os.path.join(self.PROJECT_ROOT, "lexicon")
        if not self.DATA_ROOT:
            self.DATA_ROOT = str(Path.home() / ".danmaku-scope")
        # 相对 DATA_ROOT 基于 PROJECT_ROOT 的父目录解析，不依赖 CWD
        self.DATA_ROOT = os.path.expanduser(self.DATA_ROOT)
        if not os.path.isabs(self.DATA_ROOT):
            project_parent = os.path.dirname(self.PROJECT_ROOT)
            self.DATA_ROOT = os.path.normpath(os.path.join(project_parent, self.DATA_ROOT))
        os.makedirs(self.DATA_ROOT, exist_ok=True)

    def resolve_data_path(self, raw_path: str) -> str:
        """解析数据路径：相对路径基于 DATA_ROOT，绝对路径原样返回"""
        if os.path.isabs(raw_path):
            return raw_path
        return os.path.join(self.DATA_ROOT, raw_path)


_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """懒加载单例：首次调用才实例化，避免 import 即创建 DATA_ROOT 等副作用"""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reload_settings() -> Settings:
    global _settings
    _settings = Settings()
    return _settings
