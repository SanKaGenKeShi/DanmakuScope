"""
语料库级方法论描述生成模块 - 渲染可直接引用的 corpus_methodology.md
与单视频 methodology.md 同构的 f-string 条件拼接；按实际发生的分析模式（合并/比对、历时、配对）展开
"""

import os

import pandas as pd

from . import __version__
from .config import get_settings
from .utils.logger import get_logger

logger = get_logger(__name__)

CORPUS_METHODOLOGY_FILENAME = "corpus_methodology.md"

_ZONE_POLICY_LABELS = {
    "hot_only": "仅保留热区（冷区弹幕稀疏、信号质量低，整视频跳过并告警）",
    "all": "双区分别保留（观测表增加冷热区维度，供视频内配对检验消费）",
    "weighted": "冷热区按弹幕数加权合并为单行观测",
}


class CorpusMethodologyGenerator:

    def __init__(self, output_dir: str):
        self.output_dir = output_dir

    def render(self, build_result, comparison=None) -> str:
        settings = get_settings()
        tnames = [t for t in build_result.tnames if t]
        single_partition = len(tnames) <= 1
        partition_text = "、".join(tnames) if tnames else "未知"

        video_count = len(build_result.source_zip_paths)
        total_danmaku = 0
        prompt_versions = []
        try:
            videos_df = pd.read_csv(build_result.videos_csv_path, encoding='utf-8-sig')
            if "bvid" in videos_df.columns:
                video_count = int(videos_df["bvid"].nunique())
            if "danmaku_count" in videos_df.columns:
                total_danmaku = int(videos_df["danmaku_count"].sum())
            if "prompt_version" in videos_df.columns:
                prompt_versions = sorted(
                    v for v in videos_df["prompt_version"].dropna().astype(str).unique()
                    if v and v != "nan"
                )
        except Exception as e:
            logger.warning(f"语料库方法论读取观测表失败，规模信息降级: {e}")

        lines = [
            "# 语料库级方法描述",
            "",
            f"> 本文件由 DanmakuScope v{__version__} 自动生成，可作为论文方法节的引用底稿；"
            "参数值为本次语料库聚合的实际运行配置。单视频层面的采集、预处理、采样与标注方法"
            "见各源视频报告 ZIP 内的 methodology.md。",
            "",
            "## 1. 语料库构成",
            "",
            f"- 纳入视频：{video_count} 个（源视频报告 ZIP 随本快照 `videos/` 前缀入包，可逐一回溯）。",
            f"- 覆盖官方分区：{'单一分区「' + partition_text + '」' if single_partition else partition_text + f'（{len(tnames)} 个分区）'}。",
            f"- 弹幕总量：{total_danmaku:,} 条（视频级观测表 danmaku_count 求和）。",
            f"- 冷热区策略：{_ZONE_POLICY_LABELS.get(settings.CORPUS_ZONE_POLICY, settings.CORPUS_ZONE_POLICY)}。",
        ]
        if build_result.warnings:
            lines.append(f"- 聚合告警：{len(build_result.warnings)} 条（分区样本不足等，详见 corpus_metadata.json）。")

        lines += ["", "## 2. 分析模式", ""]
        if single_partition:
            lines.append("- 模式判定：纳入视频同属单一官方分区，执行**合并分析**（聚焦分区整体特征与视频间内部变异，不做跨分区比较）。")
        else:
            lines.append(f"- 模式判定：纳入视频跨 {len(tnames)} 个官方分区，执行**比对分析**（跨分区组间比较）。")
        if settings.ENABLE_TEMPORAL_GROUPING:
            lines.append(f"- 时间分桶：按视频发布时间以「{settings.TEMPORAL_GRANULARITY}」为粒度分桶（观测表 time_period 列），支撑历时比较。")
        else:
            lines.append("- 时间分桶：未启用。")
        lines.append("- 统计单位：视频级观测（corpus_videos.csv 每行一个视频或一个视频×冷热区组合），避免段级/弹幕级伪重复。")

        lines += ["", "## 3. 推断统计方法", ""]
        test_types = set()
        if comparison is not None and getattr(comparison, "enabled", False) and comparison.rows:
            test_types = {r.get("test_type") for r in comparison.rows}
        meaningful_tests = test_types & {"Kruskal-Wallis", "Mann-Whitney U", "Wilcoxon 符号秩（配对）"}
        if meaningful_tests:
            lines.append(f"- 执行门槛：每组（分区/时段）视频数 ≥ {settings.CORPUS_MIN_VIDEOS_PER_PARTITION}，未达标组标记 insufficient_sample 并排除出检验（样本状态行照常输出）。")
            if "Kruskal-Wallis" in meaningful_tests:
                axis = "时段" if any("检验轴：时段" in str(r.get("note", "")) for r in comparison.rows) else "分区"
                lines.append(f"- 组间总检验：按{axis}分组的 Kruskal-Wallis H 检验（scipy，附 ε² 效应量）。")
            if "Mann-Whitney U" in meaningful_tests:
                lines.append("- 逐对比较：Mann-Whitney U（two-sided）+ Cliff's delta 效应量（纯算术实现，Romano et al. 2006 阈值分级）。")
            if "Wilcoxon 符号秩（配对）" in meaningful_tests:
                lines.append("- 情境变异轴：冷热区视频内配对 Wilcoxon 符号秩检验（每视频热/冷区一对观测，附 Cliff's delta）。")
            lines.append("- 全部 p 值为**未校正值**：本项目不实施多重比较校正（设计搁置项），显著性结论仅作探索性参考。")
        else:
            lines.append("- 未执行推断检验（未启用语料库级统计，或无满足门槛的比较轴；详见 statistical_tests.csv 注记行）。")

        lines += [
            "",
            "## 4. 工具版本",
            "",
            f"- 聚合流水线：DanmakuScope v{__version__}。",
            f"- 源视频标注 Prompt 版本：{'、'.join(prompt_versions) or '见各源报告 metadata.json'}（混版时软标签可比性以该版本为准）。",
            "- 产物清单：corpus_summary.csv（组级聚合）/ corpus_videos.csv（视频级观测）/ statistical_tests.csv（推断检验）"
            "/ corpus_metadata.json（快照元数据，经 schema 校验）/ danmaku_corpus.csv（合并弹幕总表，供语料附录）"
            "/ corpus_report.html（可视化报告）与可选可视化脚本、LLM 比较分析报告。",
            "",
        ]
        return "\n".join(lines)

    def write(self, build_result, comparison=None) -> str:
        filepath = os.path.join(self.output_dir, CORPUS_METHODOLOGY_FILENAME)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(self.render(build_result, comparison))
        logger.info(f"语料库方法论描述已保存: {filepath}")
        return filepath
