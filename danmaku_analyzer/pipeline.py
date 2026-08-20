"""
核心分析流程模块 - 独立于CLI的分析逻辑
供 cli.py 和 full_pipeline_test.py 共同调用

数据流为显式阶段链：每个阶段声明输入与产出类型，阶段间通过返回值传递，
不再使用跨阶段共享的可变上下文对象。
"""

import asyncio
import functools
import json
import os
import zipfile
from collections import Counter
from datetime import datetime
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field

from .config import get_settings, Settings
from . import __version__
from .crawler import BilibiliCrawler, VideoMeta, DanmakuItem
from .social_variables import SocialVariables
from .user_deduplicator import UserDeduplicator, DeduplicationResult
from .timeline_segmenter import TimelineSegmenter, TimeSegment
from .hard_metrics import HardMetricsAnalyzer
from .context_provider import ContextProvider
from .prompt_builder import PromptBuilder
from .llm_client import LLMClient
from .aggregator import Aggregator, DanmakuRecord, AggregatedData
from .reporter import Reporter
from .reproducibility import ReproManifestBuilder
from .statistical_validator import StatisticalValidator
from .cache_manager import get_cache_manager
from .corpus_builder import METADATA_FILENAME, CorpusBuilder, validate_zip_archive
from .corpus_store import CorpusStore
from .utils.input_parser import InputParser, InputType
from .utils.logger import get_logger

logger = get_logger(__name__)

# 进度回调类型：(阶段名称, 进度信息)
ProgressCallback = Callable[[str, str], None]


def _default_progress(stage: str, message: str):
    print(f"[{stage}] {message}")


@dataclass
class AnalysisResult:
    bvid: str
    title: str
    tname: str
    tags: List[str]
    segments_count: int
    aggregated_count: int
    reports: dict
    zip_path: Optional[str] = None
    zip_valid: bool = False


@dataclass
class PipelineOptions:
    """流水线输入配置（只读参数集，由 analyze_video 一次性解析）"""
    input_str: str
    credential_file: Optional[str] = None
    no_cache: bool = False
    progress: ProgressCallback = field(default=_default_progress)
    settings: Settings = field(default_factory=get_settings)
    use_freq_based: bool = False
    use_top_n: int = 10
    use_output_dir: str = ""


@dataclass
class CrawlOutput:
    """阶段 2 产出：视频元数据 + 弹幕（含数据来源标记）"""
    meta: VideoMeta
    danmaku_list: List[DanmakuItem]
    danmaku_source: str = "protobuf"


@dataclass
class PreprocessOutput:
    """阶段 3 产出：社会变量 + 去重结果 + 时序分段"""
    social_vars: SocialVariables
    dedup_result: DeduplicationResult
    segments: List[TimeSegment]


@dataclass
class SegmentAnalysisOutput:
    """阶段 4 产出：LLM 分析记录与对应样本（一一对应，供 kappa_ready 使用）"""
    records: List[DanmakuRecord] = field(default_factory=list)
    sample_danmaku: List[DanmakuItem] = field(default_factory=list)
    sample_segments: List[TimeSegment] = field(default_factory=list)


@dataclass
class ReportOutput:
    """阶段 6 产出：报告清单 + ZIP 打包结果"""
    reports: dict = field(default_factory=dict)
    zip_path: Optional[str] = None
    zip_valid: bool = False


async def analyze_video(
    input_str: str,
    output_dir: Optional[str] = None,
    credential_file: Optional[str] = None,
    freq_based: bool = False,
    top_n: Optional[int] = None,
    progress_callback: Optional[ProgressCallback] = None,
    no_cache: bool = False
) -> AnalysisResult:
    settings = get_settings()

    options = PipelineOptions(
        input_str=input_str,
        credential_file=credential_file,
        no_cache=no_cache,
        progress=progress_callback or _default_progress,
        settings=settings,
        use_freq_based=freq_based or settings.ENABLE_FREQ_BASED_SAMPLING,
        use_top_n=top_n if top_n is not None else settings.TOP_N,
        use_output_dir=_resolve_output_dir(output_dir, settings),
    )

    bvid = await _stage_resolve_input(options.input_str, options.progress)
    crawl = await _stage_crawl(bvid, options)
    pre = await _stage_preprocess(crawl, options.progress)
    analysis = await _stage_analyze_segments(pre, options)

    if not analysis.records:
        options.progress("弹幕分析", "警告：所有 LLM 分析均失败，无有效记录")
        return AnalysisResult(
            bvid=bvid, title=crawl.meta.title, tname=pre.social_vars.tname,
            tags=pre.social_vars.tags, segments_count=len(pre.segments),
            aggregated_count=0, reports={}, zip_path=None, zip_valid=False
        )

    aggregated = await _stage_aggregate(analysis.records, settings, options.progress)
    report = await _stage_report(bvid, crawl, pre, analysis, aggregated, options)

    return AnalysisResult(
        bvid=bvid, title=crawl.meta.title, tname=pre.social_vars.tname,
        tags=pre.social_vars.tags, segments_count=len(pre.segments),
        aggregated_count=len(aggregated), reports=report.reports,
        zip_path=report.zip_path if report.zip_valid else None, zip_valid=report.zip_valid
    )


def _resolve_output_dir(output_dir: Optional[str], settings: Settings) -> str:
    """解析输出目录：相对路径基于 DATA_ROOT（用户可写目录）"""
    raw = output_dir or settings.OUTPUT_DIR
    resolved = settings.resolve_data_path(raw)
    os.makedirs(resolved, exist_ok=True)
    return resolved


async def _stage_resolve_input(input_str: str, progress: ProgressCallback) -> str:
    """阶段 1：解析输入，统一转为 BV 号"""
    progress("输入解析", f"正在解析: {input_str}")
    parser = InputParser()
    parsed = parser.parse(input_str)
    if parsed.input_type == InputType.UNKNOWN:
        raise ValueError(f"无法解析输入: {input_str}")
    bvid = parsed.bvid if parsed.bvid else await parser.resolve_to_bvid(parsed)
    progress("输入解析", f"解析成功: {bvid}")
    return bvid


async def _stage_crawl(bvid: str, options: PipelineOptions) -> CrawlOutput:
    """阶段 2：爬取视频元数据 + 弹幕（带缓存）"""
    from .account import resolve_credential

    credential, source = resolve_credential(options.credential_file)
    if source == "file":
        options.progress("凭证加载", "已加载凭证文件")
    elif source == "login":
        options.progress("凭证加载", "已加载登录凭证（DATA_ROOT/credential.json）")
    elif source == "settings":
        options.progress("凭证加载", "已加载B站登录凭证")

    crawler = BilibiliCrawler(credential=credential)
    options.progress("数据获取", "正在获取视频数据...")

    cache = get_cache_manager()
    cache_key = f"crawl:{bvid}"
    cached_data = None if options.no_cache else cache.get(cache_key, max_age_hours=12)

    if cached_data:
        meta, danmaku_list = cached_data
        danmaku_source = "protobuf"
        options.progress("数据获取", f"缓存命中: {meta.title} ({len(danmaku_list)} 条弹幕)")
    else:
        if options.no_cache:
            options.progress("数据获取", "已禁用缓存，强制重新爬取")
        meta, danmaku_list, danmaku_source = await crawler.fetch_all(bvid)
        # no_cache/无凭证/XML 兜底（约 1000 条上限的截断样本）均跳过写入，避免污染 12h 缓存
        if not options.no_cache and credential is not None and danmaku_source != "xml":
            cache.set(cache_key, (meta, danmaku_list))
        if danmaku_source == "xml":
            logger.warning(f"XML 兜底数据存在数量上限，结果不写入缓存: {bvid}")
            options.progress("数据获取", "警告：protobuf 失败，XML 兜底数据有数量上限，本次结果不缓存")
        options.progress("数据获取", f"获取成功: {meta.title} ({len(danmaku_list)} 条弹幕)")

    return CrawlOutput(meta=meta, danmaku_list=danmaku_list, danmaku_source=danmaku_source)


async def _stage_preprocess(crawl: CrawlOutput, progress: ProgressCallback) -> PreprocessOutput:
    """阶段 3：社会变量提取 + 用户去重 + 时序切分（CPU 密集，使用 executor 避免阻塞事件循环）"""
    loop = asyncio.get_running_loop()

    social_vars = SocialVariables(tname=crawl.meta.tname, tags=list(crawl.meta.tags))
    progress("社会变量", f"分区: {social_vars.tname}")

    deduplicator = UserDeduplicator()
    dedup_result = await loop.run_in_executor(
        None, functools.partial(deduplicator.deduplicate, crawl.danmaku_list)
    )
    progress("用户去重", f"去重完成: {dedup_result.unique_real_user_count} 用户")

    segmenter = TimelineSegmenter()
    segments = await loop.run_in_executor(
        None, functools.partial(segmenter.segment, dedup_result.deduplicated_danmaku)
    )
    progress("时序切分", f"切分完成: {len(segments)} 段")

    return PreprocessOutput(social_vars=social_vars, dedup_result=dedup_result, segments=segments)


async def _stage_analyze_segments(pre: PreprocessOutput, options: PipelineOptions) -> SegmentAnalysisOutput:
    """阶段 4：对每个分段执行硬统计 + LLM 分析（段间并行，LLM 调用经客户端实例级信号量限速）"""
    settings = options.settings
    progress = options.progress
    social_vars = pre.social_vars
    segments = pre.segments
    dedup_result = pre.dedup_result

    progress("弹幕分析", f"开始分析 {len(segments)} 段...")

    hard_analyzer = HardMetricsAnalyzer()
    context_provider = ContextProvider()
    prompt_builder = PromptBuilder()
    llm_client = LLMClient()

    # 各段弹幕预计算，硬统计任务一次性全部提交（LLM 分词由内部信号量限速，纯 jieba 时委托 executor）
    segment_danmaku_lists = [
        [dedup_result.deduplicated_danmaku[idx] for idx in seg.danmaku_indices]
        for seg in segments
    ]
    hard_metrics_list = await asyncio.gather(*[
        hard_analyzer.analyze_async([d.content for d in seg_dms])
        for seg_dms in segment_danmaku_lists
    ])

    async def _analyze_one(danmaku, segment, segment_danmaku, hard_metrics, segment_idx):
        context = context_provider.get_context(danmaku, segment, segment_danmaku)
        context_text = context.to_prompt_text()
        complex_prompt = prompt_builder.build_complex_prompt(
            social_vars.tname, social_vars.tags, danmaku.content, context_text
        )
        simple_prompt = prompt_builder.build_simple_prompt(danmaku.content)
        llm_result = await llm_client.analyze(complex_prompt, simple_prompt)
        return DanmakuRecord(
            tname=social_vars.tname, zone_type=segment.zone_type,
            tags=social_vars.tags, hard_metrics=hard_metrics,
            llm_result=llm_result, segment_id=segment_idx,
        )

    async def _analyze_segment_batch(sample_danmaku, segment, segment_danmaku, hard_metrics, segment_idx):
        """段内批量：采样弹幕合并为一次请求；失败抛错由调用方回退逐条"""
        items = []
        for d in sample_danmaku:
            context = context_provider.get_context(d, segment, segment_danmaku)
            items.append((d.content, context.to_prompt_text()))
        complex_prompt = prompt_builder.build_complex_prompt_batch(
            social_vars.tname, social_vars.tags, items
        )
        simple_prompt = prompt_builder.build_simple_prompt_batch([d.content for d in sample_danmaku])
        llm_results = await llm_client.analyze_batch(complex_prompt, simple_prompt, len(sample_danmaku))
        return [
            DanmakuRecord(
                tname=social_vars.tname, zone_type=segment.zone_type,
                tags=social_vars.tags, hard_metrics=hard_metrics,
                llm_result=llm_result, segment_id=segment_idx,
            )
            for _, llm_result in zip(sample_danmaku, llm_results)
        ]

    async def _analyze_segment(sample_danmaku, segment, segment_danmaku, hard_metrics, segment_idx):
        """批量模式入口：优先段内批量，批量失败（解析/条数不符/请求错误）回退逐条重跑"""
        try:
            return await _analyze_segment_batch(sample_danmaku, segment, segment_danmaku, hard_metrics, segment_idx)
        except Exception as e:
            logger.warning(f"段 {segment_idx} 批量推理失败，回退逐条模式: {e}")
            results = await asyncio.gather(*[
                _analyze_one(d, segment, segment_danmaku, hard_metrics, segment_idx)
                for d in sample_danmaku
            ], return_exceptions=True)
            records = []
            for result in results:
                if isinstance(result, Exception):
                    raise result
                records.append(result)
            return records

    # 采样：频次去重 TOP_N 或每段前 N 条
    segment_samples = []
    for i, segment in enumerate(segments):
        segment_danmaku = segment_danmaku_lists[i]
        if options.use_freq_based:
            content_counter = Counter(d.content for d in segment_danmaku)
            top_contents = content_counter.most_common(options.use_top_n)
            content_to_danmaku = {}
            for d in segment_danmaku:
                if d.content not in content_to_danmaku:
                    content_to_danmaku[d.content] = d
            sample_danmaku = [content_to_danmaku[content] for content, _ in top_contents]
        else:
            sample_danmaku = segment_danmaku[:options.use_top_n]
        segment_samples.append(sample_danmaku)

    # 所有段的采样与 LLM 任务一次性提交，由 LLMClient 实例级信号量限速，避免段间串行空等
    tasks = []
    task_samples = []  # 与 tasks 一一对应：(样本弹幕列表, 所属段)，批量模式下列表含整段样本
    batch_mode = settings.ENABLE_BATCH_SEGMENT_ANALYSIS
    for i, segment in enumerate(segments):
        sample_danmaku = segment_samples[i]
        hard_metrics = hard_metrics_list[i]
        if batch_mode and sample_danmaku:
            tasks.append(_analyze_segment(
                sample_danmaku, segment, segment_danmaku_lists[i], hard_metrics, i
            ))
            task_samples.append((sample_danmaku, segment))
        else:
            for d in sample_danmaku:
                tasks.append(_analyze_one(d, segment, segment_danmaku_lists[i], hard_metrics, i))
                task_samples.append(([d], segment))

    request_unit = "段" if batch_mode else "条"
    progress("弹幕分析", f"已提交 {len(tasks)} 个 LLM 分析任务（按{request_unit}计，并发上限 {settings.LLM_CONCURRENCY}）")
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # 同步过滤：仅登记成功结果对应的样本，保证 records 与样本一一对应（防止 kappa_ready 错位）
    output = SegmentAnalysisOutput()
    fail_count = 0
    for result, (sample_list, segment) in zip(results, task_samples):
        if isinstance(result, Exception):
            fail_count += len(sample_list)
            logger.error(f"LLM 分析失败: {result}")
        elif isinstance(result, list):
            output.records.extend(result)
            output.sample_danmaku.extend(sample_list)
            output.sample_segments.extend([segment] * len(sample_list))
        else:
            output.records.append(result)
            output.sample_danmaku.append(sample_list[0])
            output.sample_segments.append(segment)

    if fail_count:
        progress("弹幕分析", f"分析完成: {len(output.records)} 成功，{fail_count} 失败")
    else:
        progress("弹幕分析", f"分析完成: {len(output.records)} 条全部成功")

    return output


async def _stage_aggregate(
    records: List[DanmakuRecord],
    settings: Settings,
    progress: ProgressCallback,
) -> List[AggregatedData]:
    """阶段 5：聚合 + 统计验证"""
    progress("数据聚合", "正在聚合数据...")
    aggregated = Aggregator().aggregate(records)
    progress("数据聚合", f"聚合完成: {len(aggregated)} 组")

    progress("统计验证", "正在计算置信区间...")
    validator = StatisticalValidator()
    min_samples = settings.MIN_SEGMENT_SAMPLES
    for agg in aggregated:
        # 共识率基于 LLM 记录数计算，CI 的 n 必须与之一致（段数会导致区间失真）
        total = agg.llm_record_count
        is_sufficient, msg = validator.validate_sample_size(total, min_samples)
        if is_sufficient:
            high_ci = validator.wilson_confidence_interval(
                int(agg.high_consensus_rate * total), total
            )
            agg.consensus_ci = high_ci.to_dict()
        else:
            agg.consensus_ci = {"status": "insufficient_sample", "sample_size": total, "min_required": min_samples}
    progress("统计验证", "置信区间计算完成")
    return aggregated


# Windows/通用文件系统文件名非法字符
_UNSAFE_FILENAME_CHARS = '/\\:*?"<>|'
# ZIP 文件名标题部分 UTF-8 字节上限（文件系统 255 字节限制下预留 [BV号] 前缀与 .zip 后缀）
_ZIP_TITLE_MAX_BYTES = 150


def _sanitize_zip_filename(title: str) -> str:
    """清洗非法字符、首尾空白与 Windows 禁止的尾随点/空格，并按 UTF-8 字节截断；空结果由调用方兜底"""
    cleaned = title.translate({ord(c): "_" for c in _UNSAFE_FILENAME_CHARS})
    cleaned = cleaned.strip()
    while cleaned.endswith((".", " ")):
        cleaned = cleaned[:-1]
    encoded = cleaned.encode("utf-8")
    if len(encoded) > _ZIP_TITLE_MAX_BYTES:
        cleaned = encoded[:_ZIP_TITLE_MAX_BYTES].decode("utf-8", errors="ignore")
        # 截断后可能再次以点/空格结尾，需重新清洗
        while cleaned.endswith((".", " ")):
            cleaned = cleaned[:-1]
    return cleaned


def _build_kappa_records(analysis: SegmentAnalysisOutput) -> List[Dict]:
    """构建 kappa_ready 记录（records 与样本已在阶段 4 同步过滤，一一对应）"""
    records = []
    for record, danmaku, seg in zip(analysis.records, analysis.sample_danmaku, analysis.sample_segments):
        records.append({
            "uid_hash": danmaku.uid_hash,
            "time_segment": f"{seg.start_time:.1f}-{seg.end_time:.1f}",
            "raw_text": danmaku.content,
            "tname": record.tname,
            "zone_type": record.zone_type,
            "consensus_level": record.llm_result.consensus_level.value,
            "weight_multiplier": record.llm_result.weight_multiplier,
            "llm_output": record.llm_result.output.to_dict(),
        })
    return records


# 双路一致性量化维度：(raw_outputs 维度键, 类别字段)，句类为单路不参与
_KAPPA_DIMENSIONS = [
    ("emotion", "label"),
    ("interaction_type", "label"),
    ("orthography", "status"),
    ("cooperative_principle", "violated"),
]


def _write_supplementary_reports(
    reporter: Reporter,
    video_metadata: Dict,
    options: PipelineOptions,
    progress: ProgressCallback,
) -> Dict[str, str]:
    """方法论描述与可复现 manifest 写出：失败仅告警降级，不在全部分析完成后拖崩整条流水线"""
    reports: Dict[str, str] = {}
    try:
        reports["methodology"] = reporter.generate_methodology(
            video_metadata, sampling={"freq_based": options.use_freq_based, "top_n": options.use_top_n}
        )
    except OSError as e:
        logger.warning(f"methodology.md 写出失败，跳过该产出: {e}")
        progress("报告生成", f"警告：methodology.md 写出失败（{e}），已跳过")
    try:
        reports["repro_manifest"] = ReproManifestBuilder().write(reporter.output_dir)
    except OSError as e:
        logger.warning(f"repro_manifest.json 写出失败，跳过该产出: {e}")
        progress("报告生成", f"警告：repro_manifest.json 写出失败（{e}），已跳过")
    return reports


def _build_quality_metrics(records: List[DanmakuRecord]) -> Dict:
    """双温度路径间逐维 Cohen's Kappa（阶段 6 一次性聚合：此时 raw_outputs 仍在内存存活，落盘后不可恢复）；
    与人工复核素材链（kappa_ready.csv）语义不同，仅作全局质控指标"""
    from .statistical_validator import cohen_kappa

    paired = [
        r.llm_result.raw_outputs[:2] for r in records
        if len(r.llm_result.raw_outputs) >= 2
        and all(isinstance(o, dict) for o in r.llm_result.raw_outputs[:2])
    ]
    if not paired:
        return {}

    kappas = {}
    for dim_key, field_name in _KAPPA_DIMENSIONS:
        labels_a, labels_b = [], []
        for first, second in paired:
            value_a = (first.get(dim_key) or {}).get(field_name)
            value_b = (second.get(dim_key) or {}).get(field_name)
            if value_a is None or value_b is None:
                continue
            labels_a.append(value_a)
            labels_b.append(value_b)
        kappa = cohen_kappa(labels_a, labels_b)
        kappas[f"{dim_key}.{field_name}"] = round(kappa, 4) if kappa is not None else None
    return {"dual_path_samples": len(paired), "cohen_kappa": kappas}


def _package_reports_zip(reports: dict, zip_path: str, progress: ProgressCallback) -> bool:
    """ZIP 打包 + 完整性校验；校验通过后清理散落源文件，写入异常/报告缺失时保留源文件返回失败"""
    missing = [name for name, path in reports.items() if not os.path.exists(path)]
    if missing:
        logger.warning(f"报告文件缺失，仍尝试打包已有文件: {missing}")
    if len(missing) == len(reports):
        logger.error(f"全部报告文件缺失，放弃打包: {zip_path}")
        progress("报告打包", f"打包失败: {os.path.basename(zip_path)}（报告文件缺失，源文件保留）")
        return False
    try:
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for name, path in reports.items():
                if os.path.exists(path):
                    zipf.write(path, os.path.basename(path))
    except OSError as e:
        logger.error(f"报告打包失败（源文件保留）: {zip_path} - {e}")
        progress("报告打包", f"打包失败: {os.path.basename(zip_path)}（源文件保留）")
        return False

    zip_valid = validate_zip_archive(
        zip_path,
        sum(1 for path in reports.values() if os.path.exists(path)),
    )
    zip_filename = os.path.basename(zip_path)
    if zip_valid:
        deleted_count = 0
        for name, path in reports.items():
            if os.path.exists(path):
                try:
                    os.remove(path)
                    deleted_count += 1
                except Exception as e:
                    logger.warning(f"删除失败: {os.path.basename(path)} - {e}")
        progress("报告打包", f"打包完成: {zip_filename} (已删除{deleted_count}个源文件)")
    else:
        progress("报告打包", f"打包完成: {zip_filename} (源文件保留)")
    return zip_valid


async def _stage_report(
    bvid: str,
    crawl: CrawlOutput,
    pre: PreprocessOutput,
    analysis: SegmentAnalysisOutput,
    aggregated: List[AggregatedData],
    options: PipelineOptions,
) -> ReportOutput:
    """阶段 6：报告生成 + LLM 报告 + ZIP 打包"""
    settings = options.settings
    progress = options.progress

    progress("报告生成", "正在生成报告...")
    reporter = Reporter(output_dir=options.use_output_dir)

    video_metadata = {
        "bvid": bvid, "title": crawl.meta.title,
        "tname": pre.social_vars.tname, "tags": pre.social_vars.tags,
        "pubdate": crawl.meta.pubdate.isoformat(),
        "view_count": crawl.meta.view_count,
        "danmaku_count": len(crawl.danmaku_list),
        "danmaku_source": crawl.danmaku_source,
        "pipeline_version": __version__,
        "batch_segment_analysis": options.settings.ENABLE_BATCH_SEGMENT_ANALYSIS,
    }
    quality_metrics = _build_quality_metrics(analysis.records)
    if quality_metrics:
        video_metadata["quality_metrics"] = quality_metrics
    reports = reporter.generate_reports(
        aggregated, kappa_records=_build_kappa_records(analysis), metadata=video_metadata
    )
    try:
        reports["danmaku_raw"] = reporter.generate_raw_danmaku(crawl.danmaku_list)
    except OSError as e:
        logger.warning(f"原始弹幕表写出失败，跳过该产出: {e}")
        progress("报告生成", f"警告：原始弹幕表写出失败（{e}），已跳过")
    reports.update(_write_supplementary_reports(reporter, video_metadata, options, progress))
    progress("报告生成", "报告生成完成")

    llm_report_md = None
    if settings.ENABLE_LLM_ANALYSIS_REPORT:
        progress("LLM报告", "正在生成社会语言学分析报告...")
        llm_report_path = await reporter.generate_llm_analysis_report(aggregated, metadata=video_metadata)
        if llm_report_path:
            reports["sociolinguistic_analysis_report"] = llm_report_path
            progress("LLM报告", "社会语言学分析报告生成完成")
            try:
                with open(llm_report_path, encoding='utf-8') as f:
                    llm_report_md = f.read()
            except OSError as e:
                logger.warning(f"LLM 报告回读失败，HTML 报告不嵌入解读文本: {e}")
        else:
            progress("LLM报告", "LLM分析报告生成失败或未启用")

    try:
        reports["html_report"] = reporter.generate_html_report(aggregated, video_metadata, llm_report_md)
        progress("报告生成", "HTML 可视化报告生成完成")
    except OSError as e:
        logger.warning(f"HTML 可视化报告写出失败，跳过该产出: {e}")
        progress("报告生成", f"警告：HTML 报告写出失败（{e}），已跳过")
    reports.update(reporter.zh_reports)

    progress("报告打包", "正在打包报告...")
    title_part = _sanitize_zip_filename(crawl.meta.title) or bvid
    zip_filename = f"[{bvid}]{title_part}.zip"
    zip_path = os.path.join(options.use_output_dir, zip_filename)
    zip_valid = _package_reports_zip(reports, zip_path, progress)

    return ReportOutput(reports=reports, zip_path=zip_path, zip_valid=zip_valid)


@dataclass
class CompareItem:
    """比对清单中单个视频的处置结果"""
    raw_input: str
    bvid: str = ""
    title: str = ""
    zip_path: str = ""
    reused: bool = False
    ok: bool = False
    error: str = ""


@dataclass
class CompareResult:
    """批量比对产物：逐项处置结果 + 语料库快照"""
    items: List[CompareItem] = field(default_factory=list)
    snapshot_path: Optional[str] = None
    summary_csv_path: Optional[str] = None
    statistics_csv_path: Optional[str] = None
    snapshot_valid: bool = False


STATISTICAL_TESTS_FILENAME = "statistical_tests.csv"
PROGRESS_RELPATH = os.path.join("scheduler", "progress.jsonl")
# 进度文件字节上限，超出后轮转为按键去重的最新记录
_PROGRESS_MAX_BYTES = 512 * 1024


def _progress_file_path() -> str:
    path = get_settings().resolve_data_path(PROGRESS_RELPATH)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def _relativize_zip_path(zip_path: str) -> str:
    """优先存相对 DATA_ROOT 的路径（OUTPUT_DIR 迁移后 resume 仍有效）；跨驱动器或越层过深时回退绝对路径"""
    data_root = get_settings().DATA_ROOT
    try:
        rel = os.path.relpath(zip_path, data_root)
    except ValueError:
        return zip_path
    if rel.startswith(".." + os.sep):
        depth = rel.count(".." + os.sep)
        if depth > data_root.rstrip(os.sep).count(os.sep):
            return zip_path
    return rel


def _resolve_progress_zip_path(stored: str) -> str:
    if not stored or os.path.isabs(stored):
        return stored
    return get_settings().resolve_data_path(stored)


def _rotate_progress_if_needed() -> None:
    """进度文件只增不减，超阈值时按键去重重写（仅保留每视频最新记录）；临时文件 + os.replace 原子写防中断丢全部记录"""
    path = _progress_file_path()
    if not os.path.exists(path) or os.path.getsize(path) <= _PROGRESS_MAX_BYTES:
        return
    index = _load_progress()
    tmp_path = path + ".tmp"
    with open(tmp_path, 'w', encoding='utf-8') as f:
        for record in index.values():
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    os.replace(tmp_path, path)
    logger.info(f"进度文件超阈值，轮转为 {len(index)} 条最新记录")


def _append_progress(raw_input: str, bvid: str, zip_path: str, reused: bool):
    """单项成功即追加 JSON Lines 记录，供中断后 --resume 无损继续"""
    record = {
        "input": raw_input, "bvid": bvid,
        "status": "reused" if reused else "ok",
        "zip_path": _relativize_zip_path(zip_path),
        "timestamp": datetime.now().isoformat(timespec='seconds'),
    }
    _rotate_progress_if_needed()
    with open(_progress_file_path(), 'a', encoding='utf-8') as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _load_progress() -> Dict[str, Dict]:
    """读取进度文件，按 input/bvid 双键索引（后记录覆盖先记录）；文件缺失或行损坏仅跳过"""
    path = _progress_file_path()
    index: Dict[str, Dict] = {}
    if not os.path.exists(path):
        return index
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                logger.warning(f"进度文件行损坏，跳过: {line[:80]}")
                continue
            for key in (record.get("input"), record.get("bvid")):
                if key:
                    index[key] = record
    return index


def _try_resume_item(item: CompareItem, index: Dict[str, Dict], key: str, progress: ProgressCallback, total: int, idx: int) -> bool:
    """命中进度文件且报告 ZIP 仍存在时直接复用，返回是否命中"""
    record = index.get(key)
    if not record:
        return False
    zip_path = _resolve_progress_zip_path(record.get("zip_path", ""))
    if not zip_path or not os.path.exists(zip_path):
        return False
    item.bvid = record.get("bvid", "") or item.bvid
    item.zip_path = zip_path
    item.reused = True
    item.ok = True
    progress("复数分析", f"[{idx}/{total}] 断点续传跳过（进度文件已完成）: {item.bvid or item.raw_input}")
    return True


async def _process_compare_item(
    task,
    item: CompareItem,
    store: CorpusStore,
    progress_index: Dict[str, Dict],
    idx: int,
    total: int,
    *,
    reuse: bool,
    resume: bool,
    output_dir: Optional[str],
    progress: ProgressCallback,
) -> None:
    """单个比对任务的完整处置：进度文件跳过/索引复用/重新分析，终态回写 task 供持久化；
    调度器恢复的终态任务在 run() 前已由 _restore_recovered_items 处置，不会进入本函数"""
    raw = task.input

    if resume and _try_resume_item(item, progress_index, raw, progress, total, idx):
        task.status = "reused"
        task.bvid, task.zip_path = item.bvid, _relativize_zip_path(item.zip_path)
        return

    bvid = await _stage_resolve_input(raw, progress)
    item.bvid = bvid
    task.bvid = bvid

    if resume and _try_resume_item(item, progress_index, bvid, progress, total, idx):
        task.status = "reused"
        task.zip_path = _relativize_zip_path(item.zip_path)
        return

    if reuse:
        zip_path = _find_reusable_zip(store, bvid)
        if zip_path:
            item.zip_path = zip_path
            item.reused = True
            item.ok = True
            _append_progress(raw, bvid, zip_path, reused=True)
            task.status = "reused"
            task.zip_path = _relativize_zip_path(zip_path)
            progress("复数分析", f"[{idx}/{total}] 复用已有报告: {bvid}")
            return

    analysis = await analyze_video(
        raw,
        output_dir=output_dir,
        progress_callback=progress,
        no_cache=not reuse,
    )
    if not (analysis.zip_valid and analysis.zip_path):
        raise RuntimeError(f"分析未产生有效报告: {bvid}")
    item.zip_path = analysis.zip_path
    item.title = analysis.title
    item.ok = True
    _append_progress(raw, bvid, analysis.zip_path, reused=False)
    task.status = "done"
    task.zip_path = _relativize_zip_path(analysis.zip_path)
    progress("复数分析", f"[{idx}/{total}] 分析完成: {bvid}")


def _find_reusable_zip(store: CorpusStore, bvid: str) -> str:
    """索引中查找该 bvid 已登记的可用报告 ZIP（非空且含 metadata.json），不可用返回空串"""
    for video in store.get_videos():
        if video.get("bvid") != bvid:
            continue
        zip_path = store.resolve_zip_path(video.get("zip_path", ""))
        if not os.path.exists(zip_path) or os.path.getsize(zip_path) == 0:
            continue
        try:
            with zipfile.ZipFile(zip_path, 'r') as zipf:
                if METADATA_FILENAME not in zipf.namelist():
                    continue
        except (OSError, zipfile.BadZipFile):
            continue
        return zip_path
    return ""


def _restore_recovered_items(scheduler, items: Dict[str, "CompareItem"], progress: ProgressCallback) -> None:
    """调度器终态任务不进执行队列，须在 run() 前回填对应 CompareItem（否则恢复跳过的视频缺席聚合）；
    产物 ZIP 已不存在时重置为 pending 重新执行"""
    from .scheduler import TERMINAL_STATUSES

    for task in scheduler.tasks:
        if task.status not in TERMINAL_STATUSES:
            continue
        item = items[task.input]
        zip_path = _resolve_progress_zip_path(task.zip_path)
        if zip_path and os.path.exists(zip_path):
            item.bvid, item.zip_path = task.bvid, zip_path
            item.reused = task.status == "reused"
            item.ok = True
            progress("复数分析", f"中断恢复跳过（调度器状态已完成）: {task.bvid or task.input}")
        else:
            task.status = "pending"
            progress("复数分析", f"历史产物缺失，重新执行: {task.input}")


async def compare_videos(
    input_list: List[str],
    reuse: bool = True,
    output_dir: Optional[str] = None,
    progress: ProgressCallback = _default_progress,
    resume: bool = False,
) -> CompareResult:
    """批量比对：任务调度器并发执行个体分析（中断后按任务状态无损恢复；reuse 时索引中已有可用报告则复用，
    resume 时命中进度文件则跳过），随后语料库级聚合、推断统计并打包快照；单项成功即落盘 progress.jsonl"""
    from .scheduler import TaskScheduler

    result = CompareResult()
    store = CorpusStore()
    progress_index = _load_progress() if resume else {}

    items: Dict[str, CompareItem] = {}
    for raw in input_list:
        if raw not in items:
            items[raw] = CompareItem(raw_input=raw)
    ordered_inputs = list(items.keys())
    if len(ordered_inputs) < len(input_list):
        progress("复数分析", f"输入含 {len(input_list) - len(ordered_inputs)} 个重复项，已合并为同一任务")
    result.items = [items[raw] for raw in ordered_inputs]

    scheduler = TaskScheduler()
    # --no-reuse 且非 resume 时用户意图为全量重分析，不得被历史任务状态静默跳过
    scheduler.submit(ordered_inputs, recover=bool(resume or reuse))
    _restore_recovered_items(scheduler, items, progress)
    total = len(ordered_inputs)

    async def _handle_task(task) -> None:
        item = items[task.input]
        idx = ordered_inputs.index(task.input) + 1
        try:
            await _process_compare_item(
                task, item, store, progress_index, idx, total,
                reuse=reuse, resume=resume, output_dir=output_dir, progress=progress,
            )
        except Exception as e:
            item.error = str(e)
            logger.error(f"复数分析单项失败: {task.input} - {e}")
            progress("复数分析", f"[{idx}/{total}] 分析失败: {task.input} - {e}")
            raise

    await scheduler.run(_handle_task)

    zip_paths = [item.zip_path for item in result.items if item.ok and item.zip_path]
    if not zip_paths:
        raise RuntimeError("没有任何视频产生有效报告，无法进行复数分析")
    # 不同输入可能解析到同一 bvid（URL 与 BV 号并存），聚合前去重避免重复计数与配对错位
    deduped_paths = list(dict.fromkeys(zip_paths))
    if len(deduped_paths) < len(zip_paths):
        progress("复数分析", f"检测到 {len(zip_paths) - len(deduped_paths)} 个重复解析的视频（不同输入指向同一 BV 号），聚合已去重")
    zip_paths = deduped_paths

    progress("语料库聚合", f"开始聚合 {len(zip_paths)} 个视频报告...")
    builder = CorpusBuilder()
    build_result = builder.build_from_zips(zip_paths, output_dir)
    result.summary_csv_path = build_result.csv_path
    for warning in build_result.warnings:
        progress("语料库聚合", warning)
    mode_label = "合并分析（单一分区）" if len(build_result.tnames) <= 1 else f"比对分析（{len(build_result.tnames)} 个分区）"
    progress("语料库聚合", f"识别完成，执行{mode_label}")

    settings = get_settings()
    extra_files: List[str] = []
    try:
        extra_files.append(ReproManifestBuilder().write(build_result.output_dir))
    except OSError as e:
        logger.warning(f"repro_manifest.json 写出失败，跳过该产出: {e}")
        progress("语料库聚合", f"警告：repro_manifest.json 写出失败（{e}），已跳过")

    comparison = None
    if settings.ENABLE_CORPUS_STATISTICS:
        comparison = StatisticalValidator().corpus_compare(build_result.videos_csv_path)
        if comparison.enabled:
            stats_csv = comparison.to_csv(os.path.join(build_result.output_dir, STATISTICAL_TESTS_FILENAME))
            extra_files.append(stats_csv)
            result.statistics_csv_path = stats_csv
            progress("语料库聚合", f"推断检验结果已落盘: {STATISTICAL_TESTS_FILENAME}（{len(comparison.rows)} 行，未校正 p 值）")

    from .reporter import Reporter

    try:
        html_path = Reporter(output_dir=build_result.output_dir).generate_corpus_html_report(build_result, comparison)
        extra_files.append(html_path)
        progress("语料库聚合", "语料库 HTML 可视化报告已生成")
    except (OSError, ValueError) as e:
        logger.warning(f"语料库 HTML 报告写出失败，跳过该产出: {e}")
        progress("语料库聚合", f"警告：语料库 HTML 报告写出失败（{e}），已跳过")

    try:
        meth_path = Reporter(output_dir=build_result.output_dir).generate_corpus_methodology(build_result, comparison)
        extra_files.append(meth_path)
        progress("语料库聚合", "语料库方法论描述已生成")
    except OSError as e:
        logger.warning(f"语料库方法论描述写出失败，跳过该产出: {e}")
        progress("语料库聚合", f"警告：语料库方法论描述写出失败（{e}），已跳过")

    if settings.ENABLE_LLM_ANALYSIS_REPORT:
        corpus_metadata = builder.build_snapshot_metadata(build_result)
        llm_report_path = await Reporter(output_dir=build_result.output_dir).generate_corpus_analysis_report(
            build_result.csv_path, build_result.videos_csv_path, corpus_metadata, result.statistics_csv_path
        )
        if llm_report_path:
            extra_files.append(llm_report_path)
            progress("语料库聚合", "语料库 LLM 比较分析报告已生成")
        else:
            progress("语料库聚合", "语料库 LLM 分析报告生成失败（不影响聚合产物）")

    result.snapshot_path = builder.package_snapshot(build_result, extra_files)
    result.snapshot_valid = build_result.zip_valid
    progress("语料库聚合", f"聚合完成: {build_result.csv_path}")
    return result
