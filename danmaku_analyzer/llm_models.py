"""LLM 输出数据模型 - 四维分析输出类型定义与序列化"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Literal, Optional

from pydantic import BaseModel, Field, model_validator


class EmotionOutput(BaseModel):
    """情感分析输出"""
    label: Literal["positive", "neutral", "negative"] = "neutral"
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class CooperativePrincipleOutput(BaseModel):
    """合作原则输出；maxim 仅在 violated=True 时有意义，未违反时为 None"""
    violated: bool = False
    maxim: Optional[Literal["quality", "quantity", "relation", "manner"]] = None

    @model_validator(mode="after")
    def _normalize_maxim(self):
        # 模型在未违反时仍会按 Prompt 约定填充 maxim 占位值，此处统一归一化
        if not self.violated:
            self.maxim = None
        return self


class InteractionTypeOutput(BaseModel):
    """互动类型输出"""
    label: Literal["check_in", "identity_claim", "mocking", "info_request", "expression", "other"] = "other"
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class SentenceFunctionOutput(BaseModel):
    """句类判断输出"""
    label: Literal["assertion", "question", "exclamation", "directive", "fragment"] = "fragment"
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class OrthographyOutput(BaseModel):
    """正字法状态输出"""
    status: Literal["standard", "community_variant", "non_standard_typo"] = "standard"
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class ConsensusLevel(Enum):
    """共识水平"""
    HIGH = "high"  # 归一化 JSD < JSD_THRESHOLD_LOW
    MEDIUM = "medium"  # JSD_THRESHOLD_LOW <= 归一化 JSD < JSD_THRESHOLD_MEDIUM
    LOW = "low"  # 归一化 JSD >= JSD_THRESHOLD_MEDIUM


@dataclass
class LLMOutput:
    """LLM 输出结果（类型化）"""
    emotion: EmotionOutput | None
    cooperative_principle: CooperativePrincipleOutput | None
    interaction_type: InteractionTypeOutput | None
    sentence_function: SentenceFunctionOutput | None
    orthography: OrthographyOutput | None

    def to_dict(self) -> dict:
        return {
            name: value.model_dump() if value is not None else None
            for name, value in vars(self).items()
        }

    @classmethod
    def missing(cls) -> "LLMOutput":
        return cls(None, None, None, None, None)

    @classmethod
    def from_response(cls, data: dict) -> "LLMOutput":
        """验证实际模型响应，缺失必需维度不得由默认分类补齐。"""
        if not isinstance(data, dict):
            raise ValueError("LLM 输出必须是对象")
        required = {
            "emotion": ("label", "confidence"),
            "cooperative_principle": ("violated",),
            "interaction_type": ("label", "confidence"),
            "orthography": ("status", "confidence"),
        }
        for name, fields in required.items():
            value = data.get(name)
            if not isinstance(value, dict) or any(key not in value for key in fields):
                raise ValueError(f"LLM 输出缺少有效维度: {name}")
        validated = dict(data)
        validated["sentence_function"] = None
        if data.get("sentence_function") is not None:
            validated["sentence_function"] = cls.parse_sentence(data).model_dump()
        return cls.from_dict(validated)

    @staticmethod
    def parse_sentence(data: dict) -> SentenceFunctionOutput:
        value = data.get("sentence_function") if isinstance(data, dict) else None
        if not isinstance(value, dict) or "label" not in value or "confidence" not in value:
            raise ValueError("LLM 输出缺少有效句类")
        return SentenceFunctionOutput.model_validate(value)

    @classmethod
    def default(cls) -> "LLMOutput":
        """默认类别对象，仅供显式构造；请求失败使用 missing。"""
        return cls(
            emotion=EmotionOutput(),
            cooperative_principle=CooperativePrincipleOutput(),
            interaction_type=InteractionTypeOutput(),
            sentence_function=SentenceFunctionOutput(),
            orthography=OrthographyOutput(),
        )

    @classmethod
    def from_dict(cls, data: Dict) -> "LLMOutput":
        """兼容历史字典；显式 null 表示未得到标注。"""
        models = {
            "emotion": EmotionOutput,
            "cooperative_principle": CooperativePrincipleOutput,
            "interaction_type": InteractionTypeOutput,
            "sentence_function": SentenceFunctionOutput,
            "orthography": OrthographyOutput,
        }
        return cls(**{
            name: model.model_validate(data.get(name, {})) if data.get(name, {}) is not None else None
            for name, model in models.items()
        })


@dataclass
class DualPathResult:
    """双路推理结果"""
    output: LLMOutput
    consensus_level: ConsensusLevel
    jsd_score: float
    weight_multiplier: float  # 低共识时为 0.2
    raw_outputs: List[Optional[Dict]]
    prompt_version: str
    analysis_status: Literal["ok", "degraded", "failed"] = "ok"
    requested_paths: int = 2
    successful_paths: int = 2
    sentence_function_source: Literal["simple", "complex", "unavailable"] = "complex"

    def to_dict(self) -> dict:
        return {
            "output": self.output.to_dict(),
            "consensus_level": self.consensus_level.value,
            "jsd_score": round(self.jsd_score, 4),
            "weight_multiplier": self.weight_multiplier,
            "prompt_version": self.prompt_version,
            "analysis_status": self.analysis_status,
            "requested_paths": self.requested_paths,
            "successful_paths": self.successful_paths,
            "sentence_function_source": self.sentence_function_source,
        }
