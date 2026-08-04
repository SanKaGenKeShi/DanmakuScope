"""
LLM 分析报告生成器 - 基于聚合数据生成社会语言学语料分析报告
"""

import json
import os
from typing import List, Dict, Any

from openai import AsyncOpenAI

from .llm_config import get_llm_settings
from .utils.logger import get_logger

logger = get_logger(__name__)


class AnalysisReportGenerator:
    """社会语言学分析报告生成器"""

    def __init__(self):
        """初始化报告生成器（使用 ANALYSIS_REPORT_LLM 配置，留空则复用 COMPLEX_LLM）"""
        llm_cfg = get_llm_settings()

        self.client = AsyncOpenAI(
            base_url=llm_cfg.effective_analysis_report_base_url,
            api_key=llm_cfg.effective_analysis_report_api_key,
            timeout=120.0,  # 报告生成可能需要更长时间
        )
        self.model = llm_cfg.effective_analysis_report_model
        self.temperature = llm_cfg.ANALYSIS_REPORT_LLM_TEMPERATURE
        self.enable_thinking = llm_cfg.ENABLE_THINKING

        logger.info(f"分析报告生成器初始化完成，模型: {self.model}")

    async def generate(
        self,
        aggregated_data: List[Dict[str, Any]],
        metadata: Dict[str, Any],
        prompt_version: str
    ) -> str:
        logger.info("开始生成社会语言学语料分析报告")

        system_prompt = self._build_system_prompt(metadata)
        user_prompt = self._build_user_prompt(aggregated_data, metadata)

        try:
            kwargs = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": self.temperature,
            }
            if not self.enable_thinking:
                kwargs["extra_body"] = {"enable_thinking": False}
            response = await self.client.chat.completions.create(**kwargs)

            report_content = response.choices[0].message.content
            logger.info("社会语言学语料分析报告生成完成")
            return report_content

        except Exception as e:
            logger.error(f"分析报告生成失败: {e}")
            return f"分析报告生成失败: {e}"

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
        spec_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "lexicon",
            "report_spec.md"
        )

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

        for i, data in enumerate(aggregated_data[:3]):  # 最多3组数据
            summary = {
                "分组": f"{data.get('tname', '')}_{data.get('zone_type', '')}",
                "弹幕数量": data.get("danmaku_count", 0),
                "情感分布": data.get("emotion_distribution", {}),
                "句类分布": data.get("sentence_function_distribution", {}),
                "互动类型分布": data.get("interaction_type_distribution", {}),
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
