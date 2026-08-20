"""
LLM 配置模块 - 集中管理所有 LLM 相关配置
包含复杂任务、简单任务、分析报告三个 LLM 的配置
"""

import os
from typing import List, Optional
from pydantic_settings import BaseSettings
from pydantic import Field, field_validator

# 与 config.py 的 DATA_ROOT 默认值保持一致
_DEFAULT_DATA_ROOT = os.path.join("~", ".danmaku-scope")


def _package_env_path() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")


def _read_env_file_value(path: str, key: str) -> str:
    """直读 .env 中的 KEY=VALUE（配置加载阶段尚不能依赖 config 单例，避免循环导入）"""
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if not stripped or stripped.startswith("#") or "=" not in stripped:
                    continue
                k, v = stripped.split("=", 1)
                if k.strip() == key:
                    return v.strip()
    except OSError:
        pass
    return ""


def _candidate_env_files() -> tuple:
    """.env 候选清单：包内 .env 在前、DATA_ROOT/.env 在后——pydantic-settings 对 env_file 元组为“后者覆盖前者”，
    故后位的 DATA_ROOT/.env 实际优先（实测 2.14.x 语义），包内 .env 兼容存量；缺失文件由 pydantic-settings 静默跳过。
    相对 DATA_ROOT 与 config.py 同规则基于包目录父目录解析，不依赖 CWD"""
    data_root = os.environ.get("DATA_ROOT", "").strip()
    if not data_root:
        data_root = _read_env_file_value(_package_env_path(), "DATA_ROOT")
    data_root = os.path.expanduser(data_root or _DEFAULT_DATA_ROOT)
    if not os.path.isabs(data_root):
        project_parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_root = os.path.normpath(os.path.join(project_parent, data_root))
    data_root_env = os.path.join(data_root, ".env")
    return tuple(dict.fromkeys([_package_env_path(), data_root_env]))


# 两个 Settings 类共用的 .env 加载配置（单一定义，避免两处漂移）
SETTINGS_MODEL_CONFIG = {
    "env_file": _candidate_env_files(),
    "env_file_encoding": "utf-8",
    "case_sensitive": True,
    "extra": "ignore",
    "validate_assignment": True,
}


class LLMSettings(BaseSettings):
    """LLM 配置中心，从环境变量或 .env 文件加载 LLM 配置"""

    # 复杂语用任务（情感、合作原则、互动类型、正字法）
    # 连接三件套默认空：占位示范值不可用且需先清空再输入，改空值由首启向导/设置页直接填写
    COMPLEX_LLM_BASE_URL: str = Field(
        default="",
        description="复杂任务 LLM 基础 URL（默认空，需经 .env 或设置界面填写）"
    )
    COMPLEX_LLM_API_KEY: str = Field(
        default="",
        description="复杂任务 LLM API Key（本地推理后端可留空）"
    )
    COMPLEX_LLM_MODEL: str = Field(
        default="",
        description="复杂任务 LLM 模型",
        json_schema_extra={"reproducible": True}
    )
    COMPLEX_LLM_TEMPERATURES: List[float] = Field(
        default=[0.1, 0.4], 
        description="复杂任务双路温度",
        json_schema_extra={"reproducible": True}
    )
    COMPLEX_LLM_ENABLE_THINKING: bool = Field(
        default=False,
        description="复杂任务 LLM 是否启用思考模式（独立配置）",
        json_schema_extra={"reproducible": True}
    )
    COMPLEX_LLM_TIMEOUT: float = Field(
        default=120.0,
        description="复杂任务 LLM 请求超时（秒）",
        json_schema_extra={"reproducible": True}
    )
    
    # 简单句类任务（sentence_function）
    SIMPLE_LLM_BASE_URL: str = Field(
        default="",
        description="简单任务 LLM 基础 URL（默认空，需经 .env 或设置界面填写）"
    )
    SIMPLE_LLM_API_KEY: str = Field(
        default="",
        description="简单任务 LLM API Key（本地推理后端可留空）"
    )
    SIMPLE_LLM_MODEL: str = Field(
        default="",
        description="简单任务 LLM 模型",
        json_schema_extra={"reproducible": True}
    )
    SIMPLE_LLM_TEMPERATURE: float = Field(
        default=0.0, 
        description="简单任务温度（单路，确定性）",
        json_schema_extra={"reproducible": True}
    )
    SIMPLE_LLM_ENABLE_THINKING: bool = Field(
        default=False,
        description="简单任务 LLM 是否启用思考模式（独立配置）",
        json_schema_extra={"reproducible": True}
    )
    SIMPLE_LLM_TIMEOUT: float = Field(
        default=120.0,
        description="简单任务 LLM 请求超时（秒）",
        json_schema_extra={"reproducible": True}
    )
    
    # 分析报告生成任务，独立配置（不再留空复用 COMPLEX_LLM）
    ANALYSIS_REPORT_LLM_BASE_URL: str = Field(
        default="",
        description="分析报告 LLM 基础 URL（独立配置，默认空，需经 .env 或设置界面填写）"
    )
    ANALYSIS_REPORT_LLM_API_KEY: str = Field(
        default="",
        description="分析报告 LLM API Key（独立配置，本地推理后端可留空）"
    )
    ANALYSIS_REPORT_LLM_MODEL: str = Field(
        default="",
        description="分析报告 LLM 模型（独立配置）",
        json_schema_extra={"reproducible": True}
    )
    ANALYSIS_REPORT_LLM_TEMPERATURE: float = Field(
        default=0.3, 
        description="分析报告 LLM 温度",
        json_schema_extra={"reproducible": True}
    )
    ANALYSIS_REPORT_LLM_ENABLE_THINKING: bool = Field(
        default=False,
        description="分析报告 LLM 是否启用思考模式（独立配置）",
        json_schema_extra={"reproducible": True}
    )
    ANALYSIS_REPORT_LLM_TIMEOUT: float = Field(
        default=180.0,
        description="分析报告 LLM 请求超时（秒，报告生成耗时长，默认高于分析任务）",
        json_schema_extra={"reproducible": True}
    )
    
    ENABLE_DUAL_PATH: bool = Field(
        default=True, 
        description="是否启用双路推理",
        json_schema_extra={"reproducible": True}
    )
    JSD_THRESHOLD_LOW: float = Field(
        default=0.2, 
        description="JSD 低阈值（归一化后相对散度，1.0=完全分歧）",
        json_schema_extra={"reproducible": True}
    )
    JSD_THRESHOLD_MEDIUM: float = Field(
        default=0.6, 
        description="JSD 中阈值（归一化后相对散度，1.0=完全分歧）",
        json_schema_extra={"reproducible": True}
    )
    LOW_CONSENSUS_WEIGHT: float = Field(
        default=0.2, 
        description="低共识权重",
        json_schema_extra={"reproducible": True}
    )
    
    PROMPT_VERSION: str = Field(
        default="v2.3.0", 
        description="Prompt 版本",
        json_schema_extra={"reproducible": True}
    )
    
    model_config = SETTINGS_MODEL_CONFIG

    @field_validator("COMPLEX_LLM_TEMPERATURES")
    @classmethod
    def _validate_complex_temperatures(cls, v):
        if not v:
            raise ValueError("COMPLEX_LLM_TEMPERATURES 不能为空列表（至少需要一个推理温度）")
        return v


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
