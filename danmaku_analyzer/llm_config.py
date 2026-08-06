"""
LLM 配置模块 - 集中管理所有 LLM 相关配置
包含复杂任务、简单任务、分析报告三个 LLM 的配置
"""

import os
from typing import List, Optional
from pydantic_settings import BaseSettings
from pydantic import Field

# 两个 Settings 类共用的 .env 加载配置（单一定义，避免两处漂移）
SETTINGS_MODEL_CONFIG = {
    "env_file": os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"),
    "env_file_encoding": "utf-8",
    "case_sensitive": True,
    "extra": "ignore",
}


class LLMSettings(BaseSettings):
    """LLM 配置中心，从环境变量或 .env 文件加载 LLM 配置"""

    # 复杂语用任务（情感、合作原则、互动类型、正字法）
    COMPLEX_LLM_BASE_URL: str = Field(
        default="https://api.openai.com/v1", 
        description="复杂任务 LLM 基础 URL"
    )
    COMPLEX_LLM_API_KEY: str = Field(
        default="sk-xxx", 
        description="复杂任务 LLM API Key"
    )
    COMPLEX_LLM_MODEL: str = Field(
        default="gpt-4o-mini", 
        description="复杂任务 LLM 模型"
    )
    COMPLEX_LLM_TEMPERATURES: List[float] = Field(
        default=[0.1, 0.4], 
        description="复杂任务双路温度"
    )
    
    # 简单句类任务（sentence_function）
    SIMPLE_LLM_BASE_URL: str = Field(
        default="https://api.deepseek.com/v1", 
        description="简单任务 LLM 基础 URL"
    )
    SIMPLE_LLM_API_KEY: str = Field(
        default="sk-yyy", 
        description="简单任务 LLM API Key"
    )
    SIMPLE_LLM_MODEL: str = Field(
        default="deepseek-chat", 
        description="简单任务 LLM 模型"
    )
    SIMPLE_LLM_TEMPERATURE: float = Field(
        default=0.0, 
        description="简单任务温度（单路，确定性）"
    )
    
    # 分析报告生成任务，留空则自动复用 COMPLEX_LLM 配置
    ANALYSIS_REPORT_LLM_BASE_URL: str = Field(
        default="", 
        description="分析报告 LLM 基础 URL，留空复用COMPLEX_LLM"
    )
    ANALYSIS_REPORT_LLM_API_KEY: str = Field(
        default="", 
        description="分析报告 LLM API Key，留空复用COMPLEX_LLM"
    )
    ANALYSIS_REPORT_LLM_MODEL: str = Field(
        default="", 
        description="分析报告 LLM 模型，留空复用COMPLEX_LLM"
    )
    ANALYSIS_REPORT_LLM_TEMPERATURE: float = Field(
        default=0.3, 
        description="分析报告 LLM 温度"
    )
    
    ENABLE_DUAL_PATH: bool = Field(
        default=True, 
        description="是否启用双路推理"
    )
    JSD_THRESHOLD_LOW: float = Field(
        default=0.2, 
        description="JSD 低阈值（归一化后相对散度，1.0=完全分歧）"
    )
    JSD_THRESHOLD_MEDIUM: float = Field(
        default=0.6, 
        description="JSD 中阈值（归一化后相对散度，1.0=完全分歧）"
    )
    LOW_CONSENSUS_WEIGHT: float = Field(
        default=0.2, 
        description="低共识权重"
    )
    
    ENABLE_THINKING: bool = Field(
        default=False,
        description="是否启用模型思考模式（Qwen等模型的reasoning/thinking）。关闭可避免content为空导致的JSON解析重试"
    )
    
    PROMPT_VERSION: str = Field(
        default="v2.2.1", 
        description="Prompt 版本"
    )
    
    model_config = SETTINGS_MODEL_CONFIG
    
    @property
    def effective_analysis_report_base_url(self) -> str:
        return self.ANALYSIS_REPORT_LLM_BASE_URL or self.COMPLEX_LLM_BASE_URL
    
    @property
    def effective_analysis_report_api_key(self) -> str:
        return self.ANALYSIS_REPORT_LLM_API_KEY or self.COMPLEX_LLM_API_KEY
    
    @property
    def effective_analysis_report_model(self) -> str:
        return self.ANALYSIS_REPORT_LLM_MODEL or self.COMPLEX_LLM_MODEL


llm_settings: Optional[LLMSettings] = None


def get_llm_settings() -> LLMSettings:
    """懒加载单例：首次调用才实例化，避免 import 副作用"""
    global llm_settings
    if llm_settings is None:
        llm_settings = LLMSettings()
    return llm_settings


def reload_llm_settings() -> LLMSettings:
    global llm_settings
    llm_settings = LLMSettings()
    return llm_settings
