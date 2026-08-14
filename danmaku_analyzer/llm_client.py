"""
LLM 客户端模块 - 双路推理 + 四维输出（含正字法状态）
职责：API 调用 + 重试 + 双路/简单任务编排
数据模型见 llm_models.py，JSD 共识度量与合并策略见 llm_consensus.py
"""

import json
import asyncio
from typing import Dict, Any, List, Optional

import regex
from tenacity import retry, stop_after_attempt, wait_exponential

from .config import get_settings
from .llm_config import get_llm_settings
from .llm_factory import complex_backend, simple_backend
from .prompt_builder import PromptComponents
from .utils.logger import get_logger
from .llm_models import ConsensusLevel, LLMOutput, DualPathResult, SentenceFunctionOutput
from .llm_consensus import (
    calculate_jsd, determine_consensus_level, merge_outputs, calculate_weight_multiplier,
)

logger = get_logger(__name__)

# 数据模型符号由 llm_models.py 提供，消费方直接从该模块导入
__all__ = ["LLMClient"]


class LLMClient:
    
    def __init__(self):
        llm_cfg = get_llm_settings()
        
        self.semaphore = asyncio.Semaphore(get_settings().LLM_CONCURRENCY)
        self.complex_client = complex_backend(timeout=llm_cfg.COMPLEX_LLM_TIMEOUT)
        self.complex_model = llm_cfg.COMPLEX_LLM_MODEL
        self.complex_temperatures = llm_cfg.COMPLEX_LLM_TEMPERATURES
        
        self.simple_client = simple_backend(timeout=llm_cfg.SIMPLE_LLM_TIMEOUT)
        self.simple_model = llm_cfg.SIMPLE_LLM_MODEL
        self.simple_temperature = llm_cfg.SIMPLE_LLM_TEMPERATURE
        
        self.jsd_threshold_low = llm_cfg.JSD_THRESHOLD_LOW
        self.jsd_threshold_medium = llm_cfg.JSD_THRESHOLD_MEDIUM
        self.low_consensus_weight = llm_cfg.LOW_CONSENSUS_WEIGHT
        
        self.enable_dual_path = llm_cfg.ENABLE_DUAL_PATH
        self.enable_thinking = llm_cfg.COMPLEX_LLM_ENABLE_THINKING
        self.simple_enable_thinking = llm_cfg.SIMPLE_LLM_ENABLE_THINKING
        
        logger.info(
            f"LLM 客户端初始化完成，"
            f"复杂模型: {self.complex_model}，"
            f"简单模型: {self.simple_model}，"
            f"双路: {'开启' if self.enable_dual_path else '关闭'}，"
            f"思考模式(复杂/简单): {'开启' if self.enable_thinking else '关闭'}/{'开启' if self.simple_enable_thinking else '关闭'}"
        )
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def _call_llm(
        self, 
        client,
        model: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        enable_thinking: bool = None,
    ) -> Dict[str, Any]:
        async with self.semaphore:
            return await self._call_llm_once(
                client, model, system_prompt, user_prompt, temperature, enable_thinking
            )
    
    async def _call_llm_once(
        self,
        client,
        model: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        enable_thinking: bool = None,
    ) -> Dict[str, Any]:
        thinking = self.enable_thinking if enable_thinking is None else enable_thinking
        try:
            content = await client.complete(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                response_format={"type": "json_object"},
                # chat_template_kwargs 为 llama.cpp 系后端关闭思考的必需参数，其他后端忽略
                extra_body={
                    "enable_thinking": thinking,
                    "chat_template_kwargs": {"enable_thinking": thinking},
                },
            )
                
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
                    temp,
                    self.enable_thinking,
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
    ) -> Optional[SentenceFunctionOutput]:
        """单路推理，仅判断句类；失败返回 None 由调用方保留复杂路自身句类，避免默认标签静默混入统计"""
        logger.info(f"开始简单任务分析（单路推理），模型: {self.simple_model}")
        
        try:
            output = await self._call_llm(
                self.simple_client,
                self.simple_model,
                prompt_components.system_prompt,
                prompt_components.user_prompt,
                self.simple_temperature,
                self.simple_enable_thinking,
            )
            
            sf_data = output.get("sentence_function", {})
            result = SentenceFunctionOutput.model_validate(sf_data)
            
            logger.info(f"简单任务分析完成: {result}")
            return result
            
        except Exception as e:
            logger.error(f"简单任务分析失败: {e}")
            return None
    
    async def analyze(
        self, 
        complex_prompt: PromptComponents,
        simple_prompt: PromptComponents
    ) -> DualPathResult:
        """复杂+简单并行，简单路成功时覆盖句类，失败时保留复杂路自身句类输出"""
        complex_task = self.analyze_complex(complex_prompt)
        simple_task = self.analyze_simple(simple_prompt)
        
        complex_result, sentence_function = await asyncio.gather(
            complex_task, simple_task
        )
        
        if sentence_function is not None:
            complex_result.output.sentence_function = sentence_function
        else:
            logger.warning("简单路句类判断失败，保留复杂路自身句类输出")
        
        return complex_result

    @staticmethod
    def _extract_batch_items(result: Dict[str, Any], expected_count: int) -> List[Dict]:
        """批量输出校验：items 存在且条数与输入一致，否则抛错触发逐条回退"""
        items = result.get("items")
        if not isinstance(items, list) or len(items) != expected_count:
            got = len(items) if isinstance(items, list) else 0
            raise ValueError(f"批量输出条数不符: 期望 {expected_count}，实际 {got}")
        return items

    async def analyze_batch(
        self,
        complex_prompt: PromptComponents,
        simple_prompt: PromptComponents,
        expected_count: int,
    ) -> List[DualPathResult]:
        """段内批量推理：双路各对整个批次推理一次，逐条比对两路结果算 JSD 共识；
        简单路单次批量请求；任一路条数不符/失败即抛错，由调用方回退逐条模式"""
        temperatures = self.complex_temperatures if self.enable_dual_path else self.complex_temperatures[:1]
        logger.info(
            f"开始批量分析（{expected_count} 条/批，{'双路' if self.enable_dual_path else '单路'}推理），"
            f"模型: {self.complex_model}"
        )

        async def call_with_temp(temp):
            return await self._call_llm(
                self.complex_client,
                self.complex_model,
                complex_prompt.system_prompt,
                complex_prompt.user_prompt,
                temp,
                self.enable_thinking,
            )

        complex_task = asyncio.gather(*[call_with_temp(t) for t in temperatures])
        simple_task = self._call_llm(
            self.simple_client,
            self.simple_model,
            simple_prompt.system_prompt,
            simple_prompt.user_prompt,
            self.simple_temperature,
            self.simple_enable_thinking,
        )
        path_results, simple_result = await asyncio.gather(complex_task, simple_task)

        path_items = [self._extract_batch_items(r, expected_count) for r in path_results]
        simple_items = self._extract_batch_items(simple_result, expected_count)

        results = []
        for i in range(expected_count):
            outputs = [path_items[p][i] for p in range(len(path_items))]
            if len(outputs) == 1:
                merged_output = merge_outputs(outputs, ConsensusLevel.HIGH)
                consensus_level = ConsensusLevel.HIGH
                jsd_score = 0.0
            else:
                jsd_score = calculate_jsd(outputs)
                consensus_level = determine_consensus_level(
                    jsd_score, self.jsd_threshold_low, self.jsd_threshold_medium
                )
                merged_output = merge_outputs(outputs, consensus_level)
            weight_multiplier = calculate_weight_multiplier(consensus_level, self.low_consensus_weight)
            sf_data = (simple_items[i] or {}).get("sentence_function", {})
            if sf_data:
                merged_output.sentence_function = SentenceFunctionOutput.model_validate(sf_data)
            else:
                logger.warning(f"批量模式第 {i} 条简单路句类缺失，保留复杂路自身句类输出")
            results.append(DualPathResult(
                output=merged_output,
                consensus_level=consensus_level,
                jsd_score=jsd_score,
                weight_multiplier=weight_multiplier,
                raw_outputs=outputs,
                prompt_version=complex_prompt.prompt_version,
            ))
        logger.info(f"批量分析完成，共 {expected_count} 条")
        return results
