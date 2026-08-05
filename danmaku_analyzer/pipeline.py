"""
核心分析流程模块 - 独立于CLI的分析逻辑
供 cli.py 和 full_pipeline_test.py 共同调用

数据流为显式阶段链：每个阶段声明输入与产出类型，阶段间通过返回值传递，
不再使用跨阶段共享的可变上下文对象。
"""

import asyncio
import functools
import os
import zipfile
from collections import Counter
from typing import List, Optional, Callable
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
from .statistical_validator import StatisticalValidator
from .cache_manager import get_cache_manager
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
    """阶段 2 产出：视频元数据 + 弹幕"""
    meta: VideoMeta
    danmaku_list: List[DanmakuItem]


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
        options.progress("数据获取", f"缓存命中: {meta.title} ({len(danmaku_list)} 条弹幕)")
    else:
        if options.no_cache:
            options.progress("数据获取", "已禁用缓存，强制重新爬取")
        meta, danmaku_list = await crawler.fetch_all(bvid)
        # no_cache 或无凭证时跳过写入，避免不完整爬取结果污染缓存
        if not options.no_cache and credential is not None:
            cache.set(cache_key, (meta, danmaku_list))
        options.progress("数据获取", f"获取成功: {meta.title} ({len(danmaku_list)} 条弹幕)")

    return CrawlOutput(meta=meta, danmaku_list=danmaku_list)


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
    """阶段 4：对每个分段执行硬统计 + LLM 分析（段间并行，全局信号量限速）"""
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
    llm_semaphore = asyncio.Semaphore(settings.LLM_CONCURRENCY)
    loop = asyncio.get_running_loop()

    # 各段弹幕预计算，硬统计任务一次性全部提交（段间并行，CPU 密集走 executor）
    segment_danmaku_lists = [
        [dedup_result.deduplicated_danmaku[idx] for idx in seg.danmaku_indices]
        for seg in segments
    ]
    hard_metrics_list = await asyncio.gather(*[
        loop.run_in_executor(
            None, functools.partial(hard_analyzer.analyze, [d.content for d in seg_dms])
        )
        for seg_dms in segment_danmaku_lists
    ])

    async def _analyze_one(danmaku, segment, segment_danmaku, hard_metrics, segment_idx):
        async with llm_semaphore:
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

    # 所有段的采样与 LLM 任务一次性提交，由信号量全局限速，避免段间串行空等
    tasks = []
    task_samples = []  # 与 tasks 一一对应：(样本弹幕, 所属段)
    for i, segment in enumerate(segments):
        segment_danmaku = segment_danmaku_lists[i]
        hard_metrics = hard_metrics_list[i]

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

        for d in sample_danmaku:
            tasks.append(_analyze_one(d, segment, segment_danmaku, hard_metrics, i))
            task_samples.append((d, segment))

    progress("弹幕分析", f"已提交 {len(tasks)} 条 LLM 分析任务（并发上限 {settings.LLM_CONCURRENCY}）")
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # 同步过滤：仅登记成功结果对应的样本，保证 records 与样本一一对应（防止 kappa_ready 错位）
    output = SegmentAnalysisOutput()
    fail_count = 0
    for result, (d, segment) in zip(results, task_samples):
        if isinstance(result, Exception):
            fail_count += 1
            logger.error(f"LLM 分析失败: {result}")
        else:
            output.records.append(result)
            output.sample_danmaku.append(d)
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

    # 构建 kappa_ready 记录（records 与样本已在阶段 4 同步过滤，一一对应）
    kappa_records = []
    for record, danmaku, seg in zip(analysis.records, analysis.sample_danmaku, analysis.sample_segments):
        kappa_records.append({
            "uid_hash": danmaku.uid_hash,
            "time_segment": f"{seg.start_time:.1f}-{seg.end_time:.1f}",
            "raw_text": danmaku.content,
            "tname": record.tname,
            "zone_type": record.zone_type,
            "consensus_level": record.llm_result.consensus_level.value,
            "weight_multiplier": record.llm_result.weight_multiplier,
            "llm_output": record.llm_result.output.to_dict(),
        })

    video_metadata = {
        "bvid": bvid, "title": crawl.meta.title,
        "tname": pre.social_vars.tname, "tags": pre.social_vars.tags,
        "pubdate": crawl.meta.pubdate.isoformat(),
        "view_count": crawl.meta.view_count,
        "danmaku_count": len(crawl.danmaku_list),
        "pipeline_version": __version__,
    }
    reports = reporter.generate_reports(aggregated, kappa_records=kappa_records, metadata=video_metadata)
    progress("报告生成", "报告生成完成")

    if settings.ENABLE_LLM_ANALYSIS_REPORT:
        progress("LLM报告", "正在生成社会语言学分析报告...")
        llm_report_path = await reporter.generate_llm_analysis_report(aggregated, metadata=video_metadata)
        if llm_report_path:
            reports["sociolinguistic_analysis_report"] = llm_report_path
            progress("LLM报告", "社会语言学分析报告生成完成")
        else:
            progress("LLM报告", "LLM分析报告生成失败或未启用")

    progress("报告打包", "正在打包报告...")
    safe_title = crawl.meta.title.replace('/', '_').replace('\\', '_').replace(':', '_').replace('*', '_').replace('?', '_').replace('"', '_').replace('<', '_').replace('>', '_').replace('|', '_')
    zip_filename = f"[{bvid}]{safe_title}.zip"
    zip_path = os.path.join(options.use_output_dir, zip_filename)

    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for name, path in reports.items():
            if os.path.exists(path):
                zipf.write(path, os.path.basename(path))

    zip_valid = _validate_zip(zip_path, reports)
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

    return ReportOutput(reports=reports, zip_path=zip_path, zip_valid=zip_valid)


def _validate_zip(zip_path: str, reports: dict) -> bool:
    if not os.path.exists(zip_path) or os.path.getsize(zip_path) == 0:
        return False
    try:
        with zipfile.ZipFile(zip_path, 'r') as zipf:
            file_list = zipf.namelist()
            expected_count = sum(1 for path in reports.values() if os.path.exists(path))
            if len(file_list) != expected_count:
                return False
            if file_list:
                zipf.read(file_list[0])
            return True
    except Exception:
        return False
