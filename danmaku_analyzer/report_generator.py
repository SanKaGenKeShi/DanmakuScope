"""
LLM 分析报告生成器 - 基于聚合数据生成社会语言学语料分析报告
"""

import asyncio
import json
import os
from typing import List, Dict, Any, Optional

import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential

from .config import get_settings
from .llm_config import get_llm_settings
from .llm_factory import analysis_report_backend
from .utils.logger import get_logger

logger = get_logger(__name__)

# 提示词容量上限：超出则截断并告警（单视频报告按分区×冷热区分组，组数天然少）
SINGLE_REPORT_MAX_GROUPS = 3
CORPUS_REPORT_MAX_GROUPS = 12


class AnalysisReportGenerator:
    """社会语言学分析报告生成器"""

    def __init__(self):
        """初始化报告生成器（使用 ANALYSIS_REPORT_LLM 独立配置）"""
        llm_cfg = get_llm_settings()

        self.client = analysis_report_backend(timeout=llm_cfg.ANALYSIS_REPORT_LLM_TIMEOUT)
        self.model = llm_cfg.ANALYSIS_REPORT_LLM_MODEL
        self.temperature = llm_cfg.ANALYSIS_REPORT_LLM_TEMPERATURE
        self.enable_thinking = llm_cfg.ANALYSIS_REPORT_LLM_ENABLE_THINKING
        # 报告生成同样受全局 LLM 并发上限约束（规范：所有 LLM 调用经 Semaphore 限速）
        self.llm_semaphore = asyncio.Semaphore(get_settings().LLM_CONCURRENCY)

        logger.info(f"分析报告生成器初始化完成，模型: {self.model}")

    async def generate(
        self,
        aggregated_data: List[Dict[str, Any]],
        metadata: Dict[str, Any]
    ) -> Optional[str]:
        logger.info(
            f"开始生成社会语言学语料分析报告，模型: {self.model}，聚合组数: {len(aggregated_data)}"
        )

        system_prompt = self._build_system_prompt(metadata)
        user_prompt = self._build_user_prompt(aggregated_data, metadata)

        try:
            report_content = await self._call_llm(system_prompt, user_prompt)
            logger.info(f"社会语言学语料分析报告生成完成，长度 {len(report_content)} 字符")
            return report_content
        except Exception as e:
            logger.error(f"分析报告生成失败: {e}")
            return None

    async def generate_corpus_report(
        self,
        summary_csv_path: str,
        videos_csv_path: str,
        corpus_metadata: Dict[str, Any],
    ) -> Optional[str]:
        """语料库级比较分析报告：输入为组级聚合表 + 视频级观测表 + 快照元数据"""
        logger.info(f"开始生成语料库级社会语言学比较分析报告，模型: {self.model}")

        try:
            user_prompt = self._build_corpus_user_prompt(summary_csv_path, videos_csv_path, corpus_metadata)
        except Exception as e:
            logger.error(f"语料库报告输入构建失败: {e}")
            return None

        try:
            report_content = await self._call_llm(self._build_corpus_system_prompt(), user_prompt)
            logger.info(f"语料库级比较分析报告生成完成，长度 {len(report_content)} 字符")
            return report_content
        except Exception as e:
            logger.error(f"语料库报告生成失败: {e}")
            return None

    def _build_corpus_system_prompt(self) -> str:
        spec_content = self._load_report_spec()
        return f"""你是一位资深的社会语言学家，专注于网络语言和社交媒体语料分析。

当前分析对象：B站弹幕跨视频语料库（多个视频、可能跨多个分区的聚合比较数据）。

你的任务是基于提供的语料库级聚合数据，撰写一份严谨、专业的跨分区/跨视频比较分析报告。
重点在于组间差异的语言学解释（而非单视频描述），并明确指出统计检验结论需以 Kruskal-Wallis/Dunn 等后续验证为准，本报告仅为描述性解读。

【重要】你必须严格遵循以下规范文档的要求，确保报告的学术规范性和一致性：

{spec_content}

请严格按照上述规范的术语定义、报告结构、写作规范和质量检查清单生成报告。"""

    def _build_corpus_user_prompt(
        self,
        summary_csv_path: str,
        videos_csv_path: str,
        corpus_metadata: Dict[str, Any],
    ) -> str:
        summary_df = pd.read_csv(summary_csv_path, encoding='utf-8-sig')
        videos_df = pd.read_csv(videos_csv_path, encoding='utf-8-sig')

        max_groups = CORPUS_REPORT_MAX_GROUPS
        if len(summary_df) > max_groups:
            logger.warning(
                f"语料库组数 {len(summary_df)} 超过提示词容量上限，仅取前 {max_groups} 组送入 LLM 报告生成，"
                f"其余 {len(summary_df) - max_groups} 组未纳入报告分析"
            )

        group_rows = []
        for record in summary_df.head(max_groups).to_dict(orient='records'):
            group_rows.append({
                k: (round(v, 4) if isinstance(v, float) else v) for k, v in record.items()
            })

        data_summary = {
            "语料库概况": {
                "视频数": int(corpus_metadata.get("video_count", 0)),
                "纳入视频bvid": corpus_metadata.get("bvids", []),
                "prompt版本": corpus_metadata.get("prompt_versions", []),
                "冷热区策略": corpus_metadata.get("zone_policy", ""),
                "视频总弹幕数": int(videos_df["danmaku_count"].sum()) if "danmaku_count" in videos_df.columns else 0,
                "聚合告警": corpus_metadata.get("warnings", []),
            },
            "组级聚合数据": group_rows,
        }

        return f"""请根据以下语料库级结构化数据，撰写跨分区/跨视频的社会语言学比较分析报告：

## 输入数据
```json
{json.dumps(data_summary, ensure_ascii=False, indent=2)}
```

请按照报告要求，生成一份完整、专业的社会语言学比较分析报告。"""

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        """单次 API 调用（含重试）；空内容与瞬时故障抛错触发 tenacity 重试"""
        async with self.llm_semaphore:
            report_content = await self.client.complete(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self.temperature,
                extra_body={
                    "enable_thinking": self.enable_thinking,
                    "chat_template_kwargs": {"enable_thinking": self.enable_thinking},
                },
            )
        if not report_content:
            raise ValueError("模型返回内容为空")
        return report_content

    def _build_system_prompt(self, metadata: Dict[str, Any]) -> str:
        """构建分析报告的系统提示词（加载规范文档）"""
        tname = metadata.get("tname", "未知分区")
        spec_content = self._load_report_spec()

        return f"""你是一位资深的社会语言学家，专注于网络语言和社交媒体语料分析。

当前分析对象：B站【{tname}】分区的弹幕语料。

你的任务是根据提供的结构化分析数据，撰写一份严谨、专业的社会语言学语料分析报告。

【重要】你必须严格遵循以下规范文档的要求，确保报告的学术规范性和一致性：

{spec_content}

请严格按照上述规范的术语定义、报告结构、写作规范和质量检查清单生成报告。"""

    def _load_report_spec(self) -> str:
        spec_path = os.path.join(get_settings().LEXICON_DIR, "report_spec.md")

        try:
            with open(spec_path, 'r', encoding='utf-8') as f:
                content = f.read()
            logger.info(f"报告规范文档已加载: {spec_path}")
            return content
        except FileNotFoundError:
            logger.warning(f"报告规范文档未找到: {spec_path}，使用默认规范")
            return self._get_default_report_spec()
        except Exception as e:
            logger.error(f"加载报告规范文档失败: {e}，使用默认规范")
            return self._get_default_report_spec()

    def _get_default_report_spec(self) -> str:
        """获取默认报告规范（当规范文档不存在时使用）"""
        return """## 报告规范（默认）

### 术语规范
- 词汇密度：实词占总词数的比例
- 社群变体：特定社群内通用的非标准书写形式
- 言语行为类型：语句的交际功能分类
- 情感极性：语料表达的情感倾向
- 共识水平：双路推理结果的一致程度

### 报告结构
1. 语料概况
2. 描写层分析（词汇构成、密度、正字法状态）
3. 语用层分析（情感分布、言语行为类型、互动类型）
4. 社会变异层分析（共识水平、分区特征）
5. 结论与讨论（主要发现、研究局限、后续建议）

### 写作要求
- 数据驱动：所有结论必须基于提供的数据
- 可溯源：引用具体数据支持论点
- 百分比保留一位小数
- 使用Markdown格式"""

    def _build_user_prompt(
        self,
        aggregated_data: List[Dict[str, Any]],
        metadata: Dict[str, Any]
    ) -> str:
        data_summary = {
            "视频信息": {
                "bvid": metadata.get("bvid", ""),
                "标题": metadata.get("title", ""),
                "分区": metadata.get("tname", ""),
                "标签": metadata.get("tags", [])[:5],
            },
            "聚合数据摘要": [],
        }

        if len(aggregated_data) > SINGLE_REPORT_MAX_GROUPS:
            logger.warning(
                f"聚合组数 {len(aggregated_data)} 超过提示词容量上限，仅取前 {SINGLE_REPORT_MAX_GROUPS} 组送入 LLM 报告生成，"
                f"其余 {len(aggregated_data) - SINGLE_REPORT_MAX_GROUPS} 组未纳入报告分析"
            )

        for i, data in enumerate(aggregated_data[:SINGLE_REPORT_MAX_GROUPS]):
            summary = {
                "分组": f"{data.get('tname', '')}_{data.get('zone_type', '')}",
                "弹幕数量": data.get("danmaku_count", 0),
                "情感分布": data.get("emotion_distribution", {}),
                "句类分布": data.get("sentence_function_distribution", {}),
                "互动类型分布": data.get("interaction_type_distribution", {}),
                "合作原则违反率": data.get("cooperative_principle_violation_rate", 0),
                "正字法状态分布": data.get("orthography_status_distribution", {}),
                "共识水平": {
                    "高共识率": data.get("high_consensus_rate", 0),
                    "中共识率": data.get("medium_consensus_rate", 0),
                    "低共识率": data.get("low_consensus_rate", 0),
                },
                "硬指标": {
                    "平均词长": data.get("avg_word_length", 0),
                    "实词密度": data.get("content_word_density", 0),
                    "标点表情率": data.get("punctuation_emoji_rate", 0),
                },
            }
            data_summary["聚合数据摘要"].append(summary)

        user_prompt = f"""请根据以下结构化数据，撰写社会语言学语料分析报告：

## 输入数据
```json
{json.dumps(data_summary, ensure_ascii=False, indent=2)}
```

请按照报告要求，生成一份完整、专业的社会语言学分析报告。"""

        return user_prompt
