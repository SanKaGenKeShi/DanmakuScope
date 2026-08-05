"""LLM 输出数据模型 - 四维分析输出类型定义与序列化"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Literal

from pydantic import BaseModel, Field


class EmotionOutput(BaseModel):
    """情感分析输出"""
    label: Literal["positive", "neutral", "negative"] = "neutral"
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class CooperativePrincipleOutput(BaseModel):
    """合作原则输出"""
    violated: bool = False
    maxim: Literal["quality", "quantity", "relation", "manner"] = "quality"


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
    HIGH = "high"  # JSD < 0.15
    MEDIUM = "medium"  # 0.15 <= JSD < 0.4
    LOW = "low"  # JSD >= 0.4


@dataclass
class LLMOutput:
    """LLM 输出结果（类型化）"""
    emotion: EmotionOutput
    cooperative_principle: CooperativePrincipleOutput
    interaction_type: InteractionTypeOutput
    sentence_function: SentenceFunctionOutput
    orthography: OrthographyOutput
    
    def to_dict(self) -> dict:
        return {
            "emotion": self.emotion.model_dump(),
            "cooperative_principle": self.cooperative_principle.model_dump(),
            "interaction_type": self.interaction_type.model_dump(),
            "sentence_function": self.sentence_function.model_dump(),
            "orthography": self.orthography.model_dump(),
        }


@dataclass
class DualPathResult:
    """双路推理结果"""
    output: LLMOutput
    consensus_level: ConsensusLevel
    jsd_score: float
    weight_multiplier: float  # 低共识时为 0.2
    raw_outputs: List[Dict]
    prompt_version: str
    
    def to_dict(self) -> dict:
        return {
            "output": self.output.to_dict(),
            "consensus_level": self.consensus_level.value,
            "jsd_score": round(self.jsd_score, 4),
            "weight_multiplier": self.weight_multiplier,
            "prompt_version": self.prompt_version,
        }


def default_llm_output() -> LLMOutput:
    """全默认输出（推理失败兜底）"""
    return LLMOutput(
        emotion=EmotionOutput(),
        cooperative_principle=CooperativePrincipleOutput(),
        interaction_type=InteractionTypeOutput(),
        sentence_function=SentenceFunctionOutput(),
        orthography=OrthographyOutput(),
    )


def dict_to_llm_output(data: Dict) -> LLMOutput:
    """dict → 类型化 LLMOutput（容错）"""
    return LLMOutput(
        emotion=EmotionOutput.model_validate(data.get("emotion", {})),
        cooperative_principle=CooperativePrincipleOutput.model_validate(data.get("cooperative_principle", {})),
        interaction_type=InteractionTypeOutput.model_validate(data.get("interaction_type", {})),
        sentence_function=SentenceFunctionOutput.model_validate(data.get("sentence_function", {})),
        orthography=OrthographyOutput.model_validate(data.get("orthography", {})),
    )
