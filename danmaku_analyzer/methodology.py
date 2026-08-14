"""
方法论描述生成模块 - 渲染可直接引用的 methodology.md
纯 f-string + 条件拼接（无模板引擎）；lexicon/methodology_template.md 仅为格式参考
"""

import os
from typing import Dict, Optional

from . import __version__
from .config import get_settings
from .llm_config import get_llm_settings
from .utils.logger import get_logger

logger = get_logger(__name__)

METHODOLOGY_FILENAME = "methodology.md"


class MethodologyGenerator:

    def __init__(self, output_dir: str):
        self.output_dir = output_dir

    def render(self, metadata: Dict, sampling: Optional[Dict] = None) -> str:
        """video 元数据 + 采样参数 → 方法论文本；条件块按实际配置展开"""
        settings = get_settings()
        llm = get_llm_settings()
        sampling = sampling or {}

        lines = [
            "# 方法描述",
            "",
            f"> 本文件由 DanmakuScope v{__version__} 自动生成，可作为论文方法节的引用底稿；"
            "参数值为本次分析的实际运行配置。",
            "",
            "## 1. 数据收集",
            "",
            f"- 语料来源：哔哩哔哩（bilibili.com）视频弹幕，视频《{metadata.get('title', '')}》"
            f"（{metadata.get('bvid', '')}），官方分区「{metadata.get('tname', '')}」。",
            f"- 发布时间：{metadata.get('pubdate', '')}；播放量：{metadata.get('view_count', '')}；"
            f"获取弹幕：{metadata.get('danmaku_count', '')} 条。",
            f"- 弹幕接口：{'protobuf 分段接口' if metadata.get('danmaku_source') != 'xml' else 'XML 兜底接口（存在数量上限，样本为截断数据，结论外推需谨慎）'}。",
        ]
        tags = metadata.get("tags") or []
        if tags:
            lines.append(f"- 视频标签（前 5）：{'、'.join(tags[:5])}。")

        lines += [
            "",
            "## 2. 预处理",
            "",
            "- 用户级去重：按弹幕 CRC32 用户哈希去重，同一用户的重复弹幕仅保留一条。",
        ]
        if settings.SEGMENTATION_MODE == "dynamic":
            lines.append(
                f"- 时序切分：基于弹幕密度的动态分段（ruptures PELT，penalty 自动选择），"
                f"段内弹幕数低于 {settings.MIN_SEGMENT_SAMPLES} 时自动与相邻段合并。"
            )
        else:
            lines.append(f"- 时序切分：固定等分模式，小段（< {settings.MIN_SEGMENT_SAMPLES} 条）自动合并。")

        strategy = "按频次排序取唯一弹幕" if sampling.get("freq_based", settings.ENABLE_FREQ_BASED_SAMPLING) else "每段前 N 条"
        top_n = sampling.get("top_n", settings.TOP_N)
        lines += [
            "",
            "## 3. 采样策略",
            "",
            f"- LLM 标注样本：{strategy}，每段至多 {top_n} 条。",
        ]
        if metadata.get("batch_segment_analysis"):
            lines.append("- 段内批量推理：同段采样弹幕合并为单次 LLM 请求，请求数 = 段数 × 3。")

        tokenizer = "LLM 辅助分词（长文本经 SIMPLE_LLM 标注词性）" if settings.ENABLE_LLM_TOKENIZER else "jieba 分词（加载项目自定义词典：B站用语与外来语）"
        lines += [
            "",
            "## 4. 硬统计（描写层）",
            "",
            f"- 分词：{tokenizer}。",
            "- 指标：词类分布、音节结构、实词密度、标点/表情符号占比、正字法正则变体率。",
            "- 句类判断不由硬统计承担，全部交由 LLM 软标签（见下节）。",
            "",
            "## 5. LLM 软标签（语用层）",
            "",
            f"- Prompt 版本：{llm.PROMPT_VERSION}。",
        ]
        if llm.ENABLE_DUAL_PATH:
            temps = "/".join(str(t) for t in llm.COMPLEX_LLM_TEMPERATURES)
            lines.append(
                f"- 复杂任务（情感、合作原则、互动类型、正字法）：模型 {llm.COMPLEX_LLM_MODEL}，"
                f"双温度路径（T = {temps}）独立推理后经归一化 Jensen-Shannon 散度判定共识水平"
                f"（阈值 {llm.JSD_THRESHOLD_LOW}/{llm.JSD_THRESHOLD_MEDIUM}），低共识样本保留并降权"
                f"（权重 {llm.LOW_CONSENSUS_WEIGHT}），零丢弃。"
            )
        else:
            lines.append(f"- 复杂任务（情感、合作原则、互动类型、正字法）：模型 {llm.COMPLEX_LLM_MODEL}，单路推理。")
        lines.append(f"- 简单任务（句类）：模型 {llm.SIMPLE_LLM_MODEL}，单路推理，成功时覆盖复杂路句类输出。")
        lines += [
            "- 正字法三分类：standard / community_variant / non_standard_typo，判别准则为“宁可误判为社区变体”。",
            "- System Prompt 注入官方分区名与视频标签作为社会语境锚点。",
            "",
            "## 6. 统计方法",
            "",
            f"- 共识率高/低比例的置信区间：Wilson 区间，置信水平 {settings.CONFIDENCE_LEVEL:.0%}；"
            f"样本量 < {settings.MIN_SEGMENT_SAMPLES} 时跳过区间并标记 insufficient_sample。",
            "- 语料库级跨分区比较（如适用）：Kruskal-Wallis H + 逐对 Mann-Whitney U + Cliff's delta；"
            "p 值均为未校正值（本项目不实施多重比较校正）。",
            "",
            "## 7. 工具版本",
            "",
            f"- 分析流水线：DanmakuScope v{metadata.get('pipeline_version', __version__)}。",
            "- 产物清单与运行参数快照见同包 metadata.json。",
            "",
        ]
        return "\n".join(lines)

    def write(self, metadata: Dict, sampling: Optional[Dict] = None) -> str:
        filepath = os.path.join(self.output_dir, METHODOLOGY_FILENAME)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(self.render(metadata, sampling))
        logger.info(f"方法论描述已保存: {filepath}")
        return filepath
