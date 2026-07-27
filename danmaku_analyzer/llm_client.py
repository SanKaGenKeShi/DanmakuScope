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


# ========== LLM 输出 Schema（Pydantic 类型约束） ==========

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
    output: LLMOutput  # 最终输出
    consensus_level: ConsensusLevel  # 共识水平
    jsd_score: float  # JSD 分数
    weight_multiplier: float  # 权重乘数（低共识时为 0.2）
    raw_outputs: List[Dict]  # 原始输出列表
    prompt_version: str  # Prompt 版本
    
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
        
        # 初始化复杂任务客户端
        self.complex_client = AsyncOpenAI(
            base_url=llm_cfg.COMPLEX_LLM_BASE_URL,
            api_key=llm_cfg.COMPLEX_LLM_API_KEY,
            timeout=60.0,  # 60秒超时
        )
        self.complex_model = llm_cfg.COMPLEX_LLM_MODEL
        self.complex_temperatures = llm_cfg.COMPLEX_LLM_TEMPERATURES
        
        # 初始化简单任务客户端
        self.simple_client = AsyncOpenAI(
            base_url=llm_cfg.SIMPLE_LLM_BASE_URL,
            api_key=llm_cfg.SIMPLE_LLM_API_KEY,
            timeout=60.0,  # 60秒超时
        )
        self.simple_model = llm_cfg.SIMPLE_LLM_MODEL
        self.simple_temperature = llm_cfg.SIMPLE_LLM_TEMPERATURE
        

        
        # JSD 阈值
        self.jsd_threshold_low = llm_cfg.JSD_THRESHOLD_LOW
        self.jsd_threshold_medium = llm_cfg.JSD_THRESHOLD_MEDIUM
        self.low_consensus_weight = llm_cfg.LOW_CONSENSUS_WEIGHT
        
        # 是否启用双路
        self.enable_dual_path = llm_cfg.ENABLE_DUAL_PATH
        
        logger.info(
            f"LLM 客户端初始化完成，"
            f"复杂模型: {self.complex_model}，"
            f"简单模型: {self.simple_model}，"
            f"双路: {'开启' if self.enable_dual_path else '关闭'}"
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
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                response_format={"type": "json_object"},
            )
            
            content = response.choices[0].message.content
            
            # 解析 JSON
            try:
                result = json.loads(content)
                return result
            except json.JSONDecodeError as e:
                logger.warning(f"JSON 解析失败: {e}，尝试提取 JSON")
                # 尝试从文本中提取 JSON
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
        """双路推理（两个温度并行）+ JSD 共识判定"""
        logger.info("开始复杂任务分析（双路推理）")
        
        system_prompt = prompt_components.system_prompt
        user_prompt = prompt_components.user_prompt
        
        # 双路推理（并行执行）
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
                return self._default_output()
        
        # 并行执行两个温度的推理
        results = await asyncio.gather(*[call_with_temp(temp) for temp in self.complex_temperatures])
        outputs = list(results)
        
        # 计算 JSD 和共识水平
        jsd_score = self._calculate_jsd(outputs)
        consensus_level = self._determine_consensus_level(jsd_score)
        
        # 合并输出（取平均或选择高共识的）
        merged_output = self._merge_outputs(outputs, consensus_level)
        
        # 计算权重乘数
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
            
            # 提取句类判断
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
        # 并行执行复杂任务和简单任务
        complex_task = self.analyze_complex(complex_prompt)
        simple_task = self.analyze_simple(simple_prompt)
        
        complex_result, sentence_function = await asyncio.gather(
            complex_task, simple_task
        )
        
        # 用简单任务的结果替换句类判断
        complex_result.output.sentence_function = sentence_function
        
        return complex_result
    
    def _calculate_jsd(self, outputs: List[Dict]) -> float:
        """情感分布的 Jensen-Shannon 散度"""
        if len(outputs) < 2:
            return 0.0
        
        try:
            # 提取情感标签的概率分布
            emotion_labels = ["positive", "neutral", "negative"]
            
            distributions = []
            for output in outputs:
                emotion = output.get("emotion", {})
                label = emotion.get("label", "neutral")
                confidence = emotion.get("confidence", 0.5)
                
                # 构建概率分布
                dist = np.zeros(len(emotion_labels))
                if label in emotion_labels:
                    idx = emotion_labels.index(label)
                    dist[idx] = confidence
                    # 剩余概率均匀分配
                    remaining = (1 - confidence) / (len(emotion_labels) - 1)
                    for i in range(len(emotion_labels)):
                        if i != idx:
                            dist[i] = remaining
                else:
                    dist = np.ones(len(emotion_labels)) / len(emotion_labels)
                
                distributions.append(dist)
            
            # 计算 JSD（添加 epsilon 平滑避免零概率除零）
            eps = 1e-10
            distributions = [d + eps for d in distributions]  # 平滑
            # 重新归一化
            distributions = [d / d.sum() for d in distributions]
            
            avg_dist = np.mean(distributions, axis=0)
            jsd = 0.0
            for dist in distributions:
                # KL 散度
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
        
        # 如果是高共识，取第一个输出
        if consensus_level == ConsensusLevel.HIGH:
            return self._dict_to_llm_output(outputs[0])
        
        # 否则，取置信度最高的输出
        best_output = max(outputs, key=lambda x: x.get("emotion", {}).get("confidence", 0))
        return self._dict_to_llm_output(best_output)
    
    def _calculate_weight_multiplier(self, consensus_level: ConsensusLevel) -> float:
        if consensus_level == ConsensusLevel.LOW:
            return self.low_consensus_weight
        else:
            return 1.0
    
    def _default_output(self) -> Dict[str, Any]:
        """默认输出（原始 dict，用于 JSD 计算等内部流程）"""
        return {
            "emotion": {"label": "neutral", "confidence": 0.5},
            "cooperative_principle": {"violated": False, "maxim": "quality"},
            "interaction_type": {"label": "other", "confidence": 0.5},
            "sentence_function": {"label": "fragment", "confidence": 0.5},
            "orthography": {"status": "standard", "confidence": 0.5},
        }
    
    def _default_llm_output(self) -> LLMOutput:
        """默认 LLM 输出（类型化）"""
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

