"""
配置中心 - Pydantic Settings 管理业务配置参数
包含抽样统计参数、社会变量配置、输出配置等
LLM 配置已独立到 llm_config.py
"""

import os
from pathlib import Path
from typing import Literal
from pydantic_settings import BaseSettings
from pydantic import Field

# 导入 LLM 配置（保持向后兼容）
from .llm_config import get_llm_settings, LLMSettings  # noqa: F401


class Settings(BaseSettings):
    """
    应用配置中心
    从环境变量或 .env 文件加载配置
    """
    
    # ========== 抽样与统计核心 ==========
    MOE: float = Field(default=0.05, description="抽样误差")
    CONFIDENCE_LEVEL: float = Field(default=0.95, description="置信水平")
    
    # ========== 弹幕采样策略 ==========
    ENABLE_FREQ_BASED_SAMPLING: bool = Field(default=False, description="是否按频次排序采样，默认False使用每段前N条")
    TOP_N: int = Field(default=10, description="频次排序采样时取前N条弹幕")
    
    # ========== 社会变量配置（仅使用官方分区 tname 作为硬分组） ==========
    ENABLE_SIGNIFICANCE_TESTING: bool = Field(default=False, description="是否启用显著性检验，默认关闭，仅探索")
    SIGNIFICANCE_ALPHA: float = Field(default=0.05, description="显著性检验的 alpha 水平")
    
    # ========== 时序动态切分 ==========
    SEGMENTATION_MODE: Literal["fixed", "dynamic"] = Field(default="dynamic", description="切分模式：fixed 或 dynamic")
    DYNAMIC_SEGMENT_METHOD: Literal["density"] = Field(default="density", description="动态切分方法：density")
    MIN_SEGMENT_SAMPLES: int = Field(default=30, description="每个动态分段最少弹幕数")
    
    # ========== LLM 分析报告开关 ==========
    ENABLE_LLM_ANALYSIS_REPORT: bool = Field(default=False, description="是否启用LLM分析报告生成")
    LLM_CONCURRENCY: int = Field(default=5, description="LLM 并发调用上限")
    
    # ========== 分词策略 ==========
    ENABLE_LLM_TOKENIZER: bool = Field(default=False, description="是否启用LLM辅助分词，复用SIMPLE_LLM")
    LLM_TOKENIZER_MIN_LENGTH: int = Field(default=20, description="触发LLM分词的最小文本长度")
    
    # ========== 微语境 ==========
    CONTEXT_TIME_WINDOW: float = Field(default=5.0, description="微语境时间窗口（秒）")
    MAX_CONTEXT_TOKENS: int = Field(default=200, description="微语境最大 token 数")
    
    # ========== 社会语域提示映射表 ==========
    REGISTER_HINTS: dict = Field(
        default={
            "鬼畜": "反讽、解构、重复造梗、多义性",
            "动画": "厨力释放、角色崇拜、剧情讨论",
            "音乐": "技术点评、氛围渲染、主观感受",
            "知识": "信息求证、补充说明、理性讨论",
            "生活": "日常分享、体验记录、轻松互动",
            "游戏": "操作点评、胜负情绪、策略交流",
            "影视": "情节分析、角色共情、悬念推测",
            "娱乐": "八卦调侃、明星崇拜、轻松消遣",
            "科技": "原理探讨、产品评价、前沿科普",
            "舞蹈": "表现力评价、美感欣赏、编舞讨论",
        },
        description="社会语域提示映射表"
    )
    DEFAULT_REGISTER_HINT: str = Field(default="一般网络交流、情绪表达", description="默认语域提示")
    
    # ========== LLM 配置代理（向后兼容） ==========
    @property
    def llm(self) -> LLMSettings:
        """获取 LLM 配置实例"""
        return get_llm_settings()
    
    # ========== 输出配置 ==========
    OUTPUT_DIR: str = Field(default="reports", description="报告输出目录（相对 DATA_ROOT 或绝对路径）")
    CACHE_DIR: str = Field(default="cache", description="缓存目录（相对 DATA_ROOT 或绝对路径）")
    LOG_DIR: str = Field(default="logs", description="日志目录（相对 DATA_ROOT 或绝对路径）")
    
    # ========== 调试 ==========
    DEBUG: bool = Field(default=False, description="调试模式")
    LOG_LEVEL: str = Field(default="INFO", description="日志级别")
    
    # ========== B站登录凭证 ==========
    BILIBILI_SESSDATA: str = Field(default="", description="B站SESSDATA")
    BILIBILI_JCT: str = Field(default="", description="B站JCT")
    BILIBILI_BUVID3: str = Field(default="", description="B站BUVID3")
    
    # ========== 路径配置 ==========
    DATA_ROOT: str = Field(default="", description="用户数据根目录（报告/缓存/日志），默认 ~/.danmaku-scope")
    PROJECT_ROOT: str = Field(default="", description="包安装目录（只读资源：词典等）")
    LEXICON_DIR: str = Field(default="", description="词典目录")
    
    model_config = {
        "env_file": os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"),
        "env_file_encoding": "utf-8",
        "case_sensitive": True,
        "extra": "ignore",
    }
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 包安装目录（只读资源：词典、规范文档等）
        if not self.PROJECT_ROOT:
            self.PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
        if not self.LEXICON_DIR:
            self.LEXICON_DIR = os.path.join(self.PROJECT_ROOT, "lexicon")
        # 用户数据根目录（可写：报告/缓存/日志）
        if not self.DATA_ROOT:
            self.DATA_ROOT = str(Path.home() / ".danmaku-scope")
        # 展开 ~ 并解析相对路径（基于项目根目录，即 PROJECT_ROOT 的父目录，禁止依赖 CWD）
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


# 全局配置实例
settings = Settings()


def get_settings() -> Settings:
    """获取配置实例"""
    return settings


def reload_settings() -> Settings:
    """重新加载配置"""
    global settings
    settings = Settings()
    return settings
