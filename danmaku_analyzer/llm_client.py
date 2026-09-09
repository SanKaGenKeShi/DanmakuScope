"""LLM 请求校验、重试与多路编排，失败样本以缺失标注保留。"""

import asyncio
import json

import regex
from tenacity import retry, stop_after_attempt, wait_exponential

from .config import get_settings
from .llm_config import get_llm_settings
from .llm_consensus import (
    calculate_jsd,
    calculate_weight_multiplier,
    determine_consensus_level,
    merge_outputs,
)
from .llm_factory import complex_backend, simple_backend
from .llm_models import ConsensusLevel, DualPathResult, LLMOutput, SentenceFunctionOutput
from .prompt_builder import PromptComponents
from .utils.logger import get_logger

logger = get_logger(__name__)

__all__ = ["LLMClient"]


class LLMClient:
    def __init__(self):
        cfg = get_llm_settings()
        self.semaphore = asyncio.Semaphore(get_settings().LLM_CONCURRENCY)
        self.complex_client = complex_backend(timeout=cfg.COMPLEX_LLM_TIMEOUT)
        self.simple_client = simple_backend(timeout=cfg.SIMPLE_LLM_TIMEOUT)
        self.complex_model = cfg.COMPLEX_LLM_MODEL
        self.simple_model = cfg.SIMPLE_LLM_MODEL
        self.complex_temperatures = cfg.COMPLEX_LLM_TEMPERATURES
        self.simple_temperature = cfg.SIMPLE_LLM_TEMPERATURE
        self.jsd_threshold_low = cfg.JSD_THRESHOLD_LOW
        self.jsd_threshold_medium = cfg.JSD_THRESHOLD_MEDIUM
        self.low_consensus_weight = cfg.LOW_CONSENSUS_WEIGHT
        self.enable_dual_path = cfg.ENABLE_DUAL_PATH
        self.enable_thinking = cfg.COMPLEX_LLM_ENABLE_THINKING
        self.simple_enable_thinking = cfg.SIMPLE_LLM_ENABLE_THINKING
        logger.info(
            f"LLM 客户端初始化完成，复杂模型: {self.complex_model}，"
            f"简单模型: {self.simple_model}，双路: {self.enable_dual_path}"
        )

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10), reraise=True)
    async def _call_llm(
        self, client, model: str, system_prompt: str, user_prompt: str,
        temperature: float, enable_thinking: bool | None = None,
        *, response_kind: str | None = None,
    ) -> dict:
        async with self.semaphore:
            result = await self._call_llm_once(
                client, model, system_prompt, user_prompt, temperature, enable_thinking,
            )
            if response_kind is not None:
                self._validate_response(result, response_kind)
            return result

    async def _call_llm_once(
        self, client, model: str, system_prompt: str, user_prompt: str,
        temperature: float, enable_thinking: bool | None = None,
    ) -> dict:
        thinking = self.enable_thinking if enable_thinking is None else enable_thinking
        content = await client.complete(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            response_format={"type": "json_object"},
            extra_body={
                "enable_thinking": thinking,
                "chat_template_kwargs": {"enable_thinking": thinking},
            },
        )
        try:
            result = json.loads(content)
        except json.JSONDecodeError as exc:
            match = regex.search(r'\{(?:[^{}]|(?R))*\}', content)
            if not match:
                raise ValueError("LLM 响应不包含可解析的 JSON 对象") from exc
            result = json.loads(match.group())
        if not isinstance(result, dict):
            raise ValueError("LLM 响应必须是 JSON 对象")
        return result

    @staticmethod
    def _validate_response(result: dict, kind: str) -> None:
        if kind.endswith("_batch"):
            items = result.get("items")
            if not isinstance(items, list) or not items:
                raise ValueError("批量输出缺少非空 items 数组")
        else:
            items = [result]
        for item in items:
            if kind.startswith("complex"):
                LLMOutput.from_response(item)
            else:
                LLMOutput.parse_sentence(item)

    def failed_result(self, prompt_version: str) -> DualPathResult:
        requested = len(self.complex_temperatures) if self.enable_dual_path else 1
        return DualPathResult(
            output=LLMOutput.missing(), consensus_level=ConsensusLevel.LOW,
            jsd_score=1.0, weight_multiplier=self.low_consensus_weight,
            raw_outputs=[None] * requested, prompt_version=prompt_version,
            analysis_status="failed", requested_paths=requested, successful_paths=0,
            sentence_function_source="unavailable",
        )

    def _combine_paths(self, paths: list[dict | None], prompt_version: str) -> DualPathResult:
        outputs = [output for output in paths if output is not None]
        if not outputs:
            return self.failed_result(prompt_version)
        degraded = len(outputs) < len(paths)
        jsd_score = 1.0 if degraded else calculate_jsd(outputs)
        level = determine_consensus_level(jsd_score, self.jsd_threshold_low, self.jsd_threshold_medium)
        merged = merge_outputs(outputs, level)
        return DualPathResult(
            output=merged, consensus_level=level, jsd_score=jsd_score,
            weight_multiplier=calculate_weight_multiplier(level, self.low_consensus_weight),
            raw_outputs=paths, prompt_version=prompt_version,
            analysis_status="degraded" if degraded else "ok",
            requested_paths=len(paths), successful_paths=len(outputs),
            sentence_function_source="complex" if merged.sentence_function is not None else "unavailable",
        )

    async def analyze_complex(self, prompt_components: PromptComponents) -> DualPathResult:
        temperatures = self.complex_temperatures if self.enable_dual_path else self.complex_temperatures[:1]

        async def call_with_temp(temp):
            try:
                result = await self._call_llm(
                    self.complex_client, self.complex_model,
                    prompt_components.system_prompt, prompt_components.user_prompt,
                    temp, self.enable_thinking, response_kind="complex",
                )
                return LLMOutput.from_response(result).to_dict()
            except Exception as exc:
                logger.warning(f"复杂任务路径失败，温度 {temp}，异常类型 {type(exc).__name__}")
                return None

        results = await asyncio.gather(*(call_with_temp(temp) for temp in temperatures))
        return self._combine_paths(results, prompt_components.prompt_version)

    async def analyze_simple(self, prompt_components: PromptComponents) -> SentenceFunctionOutput | None:
        try:
            output = await self._call_llm(
                self.simple_client, self.simple_model,
                prompt_components.system_prompt, prompt_components.user_prompt,
                self.simple_temperature, self.simple_enable_thinking, response_kind="simple",
            )
            return LLMOutput.parse_sentence(output)
        except Exception as exc:
            logger.warning(f"简单任务失败，异常类型 {type(exc).__name__}")
            return None

    @staticmethod
    def _apply_sentence(result: DualPathResult, sentence: SentenceFunctionOutput | None) -> None:
        if sentence is not None:
            result.output.sentence_function = sentence
            result.sentence_function_source = "simple"
            if result.analysis_status == "failed":
                result.analysis_status = "degraded"
        else:
            result.sentence_function_source = "complex" if result.output.sentence_function is not None else "unavailable"
            if result.analysis_status == "ok":
                result.analysis_status = "degraded"

    async def analyze(self, complex_prompt: PromptComponents, simple_prompt: PromptComponents) -> DualPathResult:
        result, sentence = await asyncio.gather(
            self.analyze_complex(complex_prompt), self.analyze_simple(simple_prompt),
        )
        self._apply_sentence(result, sentence)
        return result

    @staticmethod
    def _extract_batch_items(result: dict, expected_count: int) -> list[dict]:
        items = result.get("items")
        if not isinstance(items, list) or len(items) != expected_count:
            got = len(items) if isinstance(items, list) else 0
            raise ValueError(f"批量输出条数不符: 期望 {expected_count}，实际 {got}")
        return items

    async def analyze_batch(
        self, complex_prompt: PromptComponents, simple_prompt: PromptComponents, expected_count: int,
    ) -> list[DualPathResult]:
        """批量请求失败时取消并回收其余任务，再由调用方逐条回退。"""
        temperatures = self.complex_temperatures if self.enable_dual_path else self.complex_temperatures[:1]
        tasks = [asyncio.create_task(self._call_llm(
            self.complex_client, self.complex_model,
            complex_prompt.system_prompt, complex_prompt.user_prompt,
            temp, self.enable_thinking, response_kind="complex_batch",
        )) for temp in temperatures]
        tasks.append(asyncio.create_task(self._call_llm(
            self.simple_client, self.simple_model,
            simple_prompt.system_prompt, simple_prompt.user_prompt,
            self.simple_temperature, self.simple_enable_thinking, response_kind="simple_batch",
        )))
        try:
            responses = await asyncio.gather(*tasks)
            batches = [self._extract_batch_items(response, expected_count) for response in responses]
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

        results = []
        for index in range(expected_count):
            paths = [LLMOutput.from_response(batch[index]).to_dict() for batch in batches[:-1]]
            result = self._combine_paths(paths, complex_prompt.prompt_version)
            self._apply_sentence(result, LLMOutput.parse_sentence(batches[-1][index]))
            results.append(result)
        return results
