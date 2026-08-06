"""
LLM 客户端模块 - 双路推理 + 四维输出（含正字法状态）
职责：API 调用 + 重试 + 双路/简单任务编排
数据模型见 llm_models.py，JSD 共识度量与合并策略见 llm_consensus.py
"""

import json
import asyncio
from typing import List, Dict, Any

import regex
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from .llm_config import get_llm_settings
from .llm_factory import complex_async_client, simple_async_client
from .prompt_builder import PromptComponents
from .utils.logger import get_logger
from .llm_models import (
    EmotionOutput, CooperativePrincipleOutput, InteractionTypeOutput,
    SentenceFunctionOutput, OrthographyOutput,
    ConsensusLevel, LLMOutput, DualPathResult,
)
from .llm_consensus import (
    calculate_jsd, determine_consensus_level, merge_outputs, calculate_weight_multiplier,
)

logger = get_logger(__name__)

# 向后兼容：历史导入方（aggregator/pipeline/tests/__init__）均从本模块取符号
__all__ = [
    "LLMClient",
    "EmotionOutput", "CooperativePrincipleOutput", "InteractionTypeOutput",
    "SentenceFunctionOutput", "OrthographyOutput",
    "ConsensusLevel", "LLMOutput", "DualPathResult",
]


class LLMClient:
    
    def __init__(self):
        llm_cfg = get_llm_settings()
        
        self.complex_client = complex_async_client()
        self.complex_model = llm_cfg.COMPLEX_LLM_MODEL
        self.complex_temperatures = llm_cfg.COMPLEX_LLM_TEMPERATURES
        
        self.simple_client = simple_async_client()
        self.simple_model = llm_cfg.SIMPLE_LLM_MODEL
        self.simple_temperature = llm_cfg.SIMPLE_LLM_TEMPERATURE
        
        self.jsd_threshold_low = llm_cfg.JSD_THRESHOLD_LOW
        self.jsd_threshold_medium = llm_cfg.JSD_THRESHOLD_MEDIUM
        self.low_consensus_weight = llm_cfg.LOW_CONSENSUS_WEIGHT
        
        self.enable_dual_path = llm_cfg.ENABLE_DUAL_PATH
        self.enable_thinking = llm_cfg.ENABLE_THINKING
        
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
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                response_format={"type": "json_object"},
                extra_body={"enable_thinking": self.enable_thinking},
            )
            
            content = response.choices[0].message.content
            
            try:
                result = json.loads(content)
                return result
            except json.JSONDecodeError as e:
                # 模型常用 markdown 包裹 JSON，首次直解必然失败走提取兜底，属预期路径
                logger.debug(f"JSON 直解失败: {e}，尝试提取 JSON")
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
        temperatures = self.complex_temperatures if self.enable_dual_path else self.complex_temperatures[:1]
        logger.info(
            f"开始复杂任务分析（{'双路' if self.enable_dual_path else '单路'}推理），"
            f"模型: {self.complex_model}，温度: {temperatures}"
        )
        
        system_prompt = prompt_components.system_prompt
        user_prompt = prompt_components.user_prompt
        
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
                output=LLMOutput.default(),
                consensus_level=ConsensusLevel.LOW,
                jsd_score=1.0,
                weight_multiplier=self.low_consensus_weight,
                raw_outputs=list(results),
                prompt_version=prompt_components.prompt_version,
            )
        
        # 计算 JSD 和共识水平（仅基于成功路径）
        jsd_score = calculate_jsd(outputs)
        consensus_level = determine_consensus_level(
            jsd_score, self.jsd_threshold_low, self.jsd_threshold_medium
        )
        
        merged_output = merge_outputs(outputs, consensus_level)
        weight_multiplier = calculate_weight_multiplier(consensus_level, self.low_consensus_weight)
        
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
        logger.info(f"开始简单任务分析（单路推理），模型: {self.simple_model}")
        
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
