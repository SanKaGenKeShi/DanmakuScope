"""
LLM 客户端模块 - 双路推理 + 四维输出（含正字法状态）
包含复杂任务双路推理和简单任务单路推理
"""

import json
import asyncio
from typing import List, Dict, Any, Optional, Literal
from dataclasses import dataclass
from enum import Enum

import numpy as np
import regex
from openai import AsyncOpenAI
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential

from .llm_config import get_llm_settings
from .prompt_builder import PromptComponents
from .utils.logger import get_logger

logger = get_logger(__name__)


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


class LLMClient:
    
    def __init__(self):
        llm_cfg = get_llm_settings()
        
        self.complex_client = AsyncOpenAI(
            base_url=llm_cfg.COMPLEX_LLM_BASE_URL,
            api_key=llm_cfg.COMPLEX_LLM_API_KEY,
            timeout=60.0,
        )
        self.complex_model = llm_cfg.COMPLEX_LLM_MODEL
        self.complex_temperatures = llm_cfg.COMPLEX_LLM_TEMPERATURES
        
        self.simple_client = AsyncOpenAI(
            base_url=llm_cfg.SIMPLE_LLM_BASE_URL,
            api_key=llm_cfg.SIMPLE_LLM_API_KEY,
            timeout=60.0,
        )
        self.simple_model = llm_cfg.SIMPLE_LLM_MODEL
        self.simple_temperature = llm_cfg.SIMPLE_LLM_TEMPERATURE
        

        
        self.jsd_threshold_low = llm_cfg.JSD_THRESHOLD_LOW
        self.jsd_threshold_medium = llm_cfg.JSD_THRESHOLD_MEDIUM
        self.low_consensus_weight = llm_cfg.LOW_CONSENSUS_WEIGHT
        
        self.enable_dual_path = llm_cfg.ENABLE_DUAL_PATH
        self.enable_thinking = llm_cfg.ENABLE_THINKING
        
        # API Key 占位符检测：未配置 .env 时提前提示，避免运行时报 401 无从排查
        for name, key in (
            ("COMPLEX_LLM_API_KEY", llm_cfg.COMPLEX_LLM_API_KEY),
            ("SIMPLE_LLM_API_KEY", llm_cfg.SIMPLE_LLM_API_KEY),
        ):
            if key in ("sk-xxx", "sk-yyy", ""):
                logger.warning(f"{name} 未配置（当前为占位值），请在 .env 中设置真实 Key")
        
        logger.info(
            f"LLM 客户端初始化完成，"
            f"复杂模型: {self.complex_model}，"
            f"简单模型: {self.simple_model}，"
            f"双路: {'开启' if self.enable_dual_path else '关闭'}，"
            f"思考模式: {'开启' if self.enable_thinking else '关闭'}"
        )
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def _call_llm(
        self, 
        client: AsyncOpenAI,
        model: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float
    ) -> Dict[str, Any]:
        try:
            extra_body = {"enable_thinking": self.enable_thinking} if not self.enable_thinking else None
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                response_format={"type": "json_object"},
                **({"extra_body": extra_body} if extra_body is not None else {}),
            )
            
            content = response.choices[0].message.content
            
            try:
                result = json.loads(content)
                return result
            except json.JSONDecodeError as e:
                logger.warning(f"JSON 解析失败: {e}，尝试提取 JSON")
                json_match = regex.search(r'\{(?:[^{}]|(?R))*\}', content)
                if json_match:
                    return json.loads(json_match.group())
                else:
                    raise ValueError(f"无法从响应中提取 JSON: {content}")
            
        except Exception as e:
            logger.error(f"LLM 调用失败: {e}")
            raise
    
    async def analyze_complex(
        self, 
        prompt_components: PromptComponents
    ) -> DualPathResult:
        """双路推理（两个温度并行）+ JSD 共识判定；ENABLE_DUAL_PATH 关闭时单路"""
        logger.info("开始复杂任务分析（双路推理）" if self.enable_dual_path else "开始复杂任务分析（单路推理）")
        
        system_prompt = prompt_components.system_prompt
        user_prompt = prompt_components.user_prompt
        
        temperatures = self.complex_temperatures if self.enable_dual_path else self.complex_temperatures[:1]
        
        # 并行执行各路推理，失败返回 None（不再用默认值混入 JSD 计算）
        async def call_with_temp(temp):
            try:
                return await self._call_llm(
                    self.complex_client,
                    self.complex_model,
                    system_prompt,
                    user_prompt,
                    temp
                )
            except Exception as e:
                logger.error(f"复杂任务推理失败 (temp={temp}): {e}")
                return None
        
        results = await asyncio.gather(*[call_with_temp(temp) for temp in temperatures])
        outputs = [r for r in results if r is not None]
        
        if not outputs:
            # 全部路径失败：保留默认输出并强制低共识（权重 0.2）
            logger.warning("复杂任务全部推理路径失败，使用默认输出并标记为低共识")
            return DualPathResult(
                output=self._default_llm_output(),
                consensus_level=ConsensusLevel.LOW,
                jsd_score=1.0,
                weight_multiplier=self.low_consensus_weight,
                raw_outputs=list(results),
                prompt_version=prompt_components.prompt_version,
            )
        
        # 计算 JSD 和共识水平（仅基于成功路径）
        jsd_score = self._calculate_jsd(outputs)
        consensus_level = self._determine_consensus_level(jsd_score)
        
        merged_output = self._merge_outputs(outputs, consensus_level)
        weight_multiplier = self._calculate_weight_multiplier(consensus_level)
        
        result = DualPathResult(
            output=merged_output,
            consensus_level=consensus_level,
            jsd_score=jsd_score,
            weight_multiplier=weight_multiplier,
            raw_outputs=outputs,
            prompt_version=prompt_components.prompt_version,
        )
        
        logger.info(
            f"复杂任务分析完成，共识水平: {consensus_level.value}，"
            f"JSD: {jsd_score:.4f}，权重: {weight_multiplier}"
        )
        
        return result
    
    async def analyze_simple(
        self, 
        prompt_components: PromptComponents
    ) -> SentenceFunctionOutput:
        """单路推理，仅判断句类"""
        logger.info("开始简单任务分析（单路推理）")
        
        try:
            output = await self._call_llm(
                self.simple_client,
                self.simple_model,
                prompt_components.system_prompt,
                prompt_components.user_prompt,
                self.simple_temperature
            )
            
            sf_data = output.get("sentence_function", {})
            result = SentenceFunctionOutput.model_validate(sf_data)
            
            logger.info(f"简单任务分析完成: {result}")
            return result
            
        except Exception as e:
            logger.error(f"简单任务分析失败: {e}")
            return SentenceFunctionOutput()
    
    async def analyze(
        self, 
        complex_prompt: PromptComponents,
        simple_prompt: PromptComponents
    ) -> DualPathResult:
        """复杂+简单并行，用简单路结果覆盖句类"""
        complex_task = self.analyze_complex(complex_prompt)
        simple_task = self.analyze_simple(simple_prompt)
        
        complex_result, sentence_function = await asyncio.gather(
            complex_task, simple_task
        )
        
        complex_result.output.sentence_function = sentence_function
        
        return complex_result
    
    def _calculate_jsd(self, outputs: List[Dict]) -> float:
        """情感分布的 Jensen-Shannon 散度"""
        if len(outputs) < 2:
            return 0.0
        
        try:
            emotion_labels = ["positive", "neutral", "negative"]
            
            distributions = []
            for output in outputs:
                emotion = output.get("emotion", {})
                label = emotion.get("label", "neutral")
                confidence = emotion.get("confidence", 0.5)
                
                dist = np.zeros(len(emotion_labels))
                if label in emotion_labels:
                    idx = emotion_labels.index(label)
                    dist[idx] = confidence
                    remaining = (1 - confidence) / (len(emotion_labels) - 1)
                    for i in range(len(emotion_labels)):
                        if i != idx:
                            dist[i] = remaining
                else:
                    dist = np.ones(len(emotion_labels)) / len(emotion_labels)
                
                distributions.append(dist)
            
            # epsilon 平滑避免零概率除零
            eps = 1e-10
            distributions = [d + eps for d in distributions]
            distributions = [d / d.sum() for d in distributions]
            
            avg_dist = np.mean(distributions, axis=0)
            jsd = 0.0
            for dist in distributions:
                kl = np.sum(dist * np.log(dist / avg_dist))
                jsd += kl
            jsd /= len(distributions)
            
            return float(jsd)
            
        except Exception as e:
            logger.warning(f"JSD 计算失败: {e}")
            return 0.0
    
    def _determine_consensus_level(self, jsd_score: float) -> ConsensusLevel:
        if jsd_score < self.jsd_threshold_low:
            return ConsensusLevel.HIGH
        elif jsd_score < self.jsd_threshold_medium:
            return ConsensusLevel.MEDIUM
        else:
            return ConsensusLevel.LOW
    
    def _merge_outputs(
        self, 
        outputs: List[Dict], 
        consensus_level: ConsensusLevel
    ) -> LLMOutput:
        if not outputs:
            return self._default_llm_output()
        
        if consensus_level == ConsensusLevel.HIGH:
            return self._dict_to_llm_output(outputs[0])
        
        best_output = max(outputs, key=lambda x: x.get("emotion", {}).get("confidence", 0))
        return self._dict_to_llm_output(best_output)
    
    def _calculate_weight_multiplier(self, consensus_level: ConsensusLevel) -> float:
        if consensus_level == ConsensusLevel.LOW:
            return self.low_consensus_weight
        else:
            return 1.0
    
    def _default_llm_output(self) -> LLMOutput:
        return LLMOutput(
            emotion=EmotionOutput(),
            cooperative_principle=CooperativePrincipleOutput(),
            interaction_type=InteractionTypeOutput(),
            sentence_function=SentenceFunctionOutput(),
            orthography=OrthographyOutput(),
        )
    
    def _dict_to_llm_output(self, data: Dict) -> LLMOutput:
        """dict → 类型化 LLMOutput（容错）"""
        return LLMOutput(
            emotion=EmotionOutput.model_validate(data.get("emotion", {})),
            cooperative_principle=CooperativePrincipleOutput.model_validate(data.get("cooperative_principle", {})),
            interaction_type=InteractionTypeOutput.model_validate(data.get("interaction_type", {})),
            sentence_function=SentenceFunctionOutput.model_validate(data.get("sentence_function", {})),
            orthography=OrthographyOutput.model_validate(data.get("orthography", {})),
        )

