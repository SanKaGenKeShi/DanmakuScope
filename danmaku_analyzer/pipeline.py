"""
核心分析流程模块 - 独立于CLI的分析逻辑
供 cli.py 和 full_pipeline_test.py 共同调用
"""

import asyncio
import functools
import os
import zipfile
from collections import Counter
from typing import List, Optional, Callable, Any
from dataclasses import dataclass, field

from .config import get_settings, Settings
from .crawler import BilibiliCrawler, VideoMeta, DanmakuItem
from .social_variables import SocialVariableExtractor, SocialVariables
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
from .utils.input_parser import parse_input, resolve_to_bvid, InputType
from .utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class AnalysisResult:
    """分析结果数据类"""
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
class PipelineContext:
    """流水线上下文 - 统一携带各阶段中间状态，避免松散参数传递"""
    # 输入参数
    input_str: str
    output_dir: Optional[str] = None
    credential_file: Optional[str] = None
    freq_based: bool = False
    top_n: Optional[int] = None
    progress: Callable[[str, str], None] = field(default=lambda s, m: print(f"[{s}] {m}"))

    # 阶段 1 产出
    bvid: str = ""

    # 阶段 2 产出
    meta: Optional[VideoMeta] = None
    danmaku_list: List[DanmakuItem] = field(default_factory=list)

    # 阶段 3 产出
    social_vars: Optional[SocialVariables] = None
    dedup_result: Optional[DeduplicationResult] = None
    segments: List[TimeSegment] = field(default_factory=list)

    # 阶段 4 产出
    records: List[DanmakuRecord] = field(default_factory=list)
    all_sample_danmaku: List[DanmakuItem] = field(default_factory=list)
    all_sample_segments: List[TimeSegment] = field(default_factory=list)

    # 阶段 5 产出
    aggregated: List[AggregatedData] = field(default_factory=list)

    # 阶段 6 产出
    reports: dict = field(default_factory=dict)
    zip_path: Optional[str] = None
    zip_valid: bool = False

    # 解析后的配置
    settings: Optional[Settings] = None
    use_freq_based: bool = False
    use_top_n: int = 10
    use_output_dir: str = ""


# 进度回调类型：(阶段名称, 进度信息)
ProgressCallback = Callable[[str, str], None]


def _default_progress(stage: str, message: str):
    """默认进度回调：打印到控制台"""
    print(f"[{stage}] {message}")


async def analyze_video(
    input_str: str,
    output_dir: Optional[str] = None,
    credential_file: Optional[str] = None,
    freq_based: bool = False,
    top_n: Optional[int] = None,
    progress_callback: Optional[ProgressCallback] = None
) -> AnalysisResult:
    """核心分析流程编排器"""
    settings = get_settings()
    progress = progress_callback or _default_progress

    # 构建流水线上下文
    ctx = PipelineContext(
        input_str=input_str,
        output_dir=output_dir,
        credential_file=credential_file,
        freq_based=freq_based,
        top_n=top_n,
        progress=progress,
        settings=settings,
        use_freq_based=freq_based or settings.ENABLE_FREQ_BASED_SAMPLING,
        use_top_n=top_n if top_n is not None else settings.TOP_N,
        use_output_dir=_resolve_output_dir(output_dir, settings),
    )

    # 阶段 1：输入解析
    await _stage_resolve_input(ctx)

    # 阶段 2：数据爬取
    await _stage_crawl(ctx)

    # 阶段 3：社会变量 + 去重 + 切分（CPU 密集，使用 executor）
    await _stage_preprocess(ctx)

    # 阶段 4：弹幕分析（硬统计 + LLM）
    await _stage_analyze_segments(ctx)

    # 空结果保护
    if not ctx.records:
        progress("弹幕分析", "警告：所有 LLM 分析均失败，无有效记录")
        return AnalysisResult(
            bvid=ctx.bvid, title=ctx.meta.title, tname=ctx.social_vars.tname,
            tags=ctx.social_vars.tags, segments_count=len(ctx.segments),
            aggregated_count=0, reports={}, zip_path=None, zip_valid=False
        )

    # 阶段 5：聚合 + 统计验证
    await _stage_aggregate(ctx)

    # 阶段 6：报告生成 + 打包
    await _stage_report(ctx)

    return AnalysisResult(
        bvid=ctx.bvid, title=ctx.meta.title, tname=ctx.social_vars.tname,
        tags=ctx.social_vars.tags, segments_count=len(ctx.segments),
        aggregated_count=len(ctx.aggregated), reports=ctx.reports,
        zip_path=ctx.zip_path if ctx.zip_valid else None, zip_valid=ctx.zip_valid
    )


# ========== 阶段性私有方法 ==========

def _resolve_output_dir(output_dir: Optional[str], settings: Settings) -> str:
    """解析输出目录：相对路径基于 DATA_ROOT（用户可写目录）"""
    raw = output_dir or settings.OUTPUT_DIR
    resolved = settings.resolve_data_path(raw)
    os.makedirs(resolved, exist_ok=True)
    return resolved


async def _stage_resolve_input(ctx: PipelineContext):
    """阶段 1：解析输入，统一转为 BV 号"""
    ctx.progress("输入解析", f"正在解析: {ctx.input_str}")
    parsed = parse_input(ctx.input_str)
    if parsed.input_type == InputType.UNKNOWN:
        raise ValueError(f"无法解析输入: {ctx.input_str}")
    ctx.bvid = parsed.bvid if parsed.bvid else await resolve_to_bvid(ctx.input_str)
    ctx.progress("输入解析", f"解析成功: {ctx.bvid}")


async def _stage_crawl(ctx: PipelineContext):
    """阶段 2：爬取视频元数据 + 弹幕（带缓存）"""
    settings = ctx.settings
    credential = None
    if ctx.credential_file:
        pass  # TODO: 加载凭证文件

    logger.info(f"检查凭证: SESSDATA={'有' if settings.BILIBILI_SESSDATA else '无'}")
    if not credential and settings.BILIBILI_SESSDATA:
        from bilibili_api import Credential
        credential = Credential(
            sessdata=settings.BILIBILI_SESSDATA,
            bili_jct=settings.BILIBILI_JCT,
            buvid3=settings.BILIBILI_BUVID3
        )
        ctx.progress("凭证加载", "已加载B站登录凭证")

    crawler = BilibiliCrawler(credential=credential)
    ctx.progress("数据获取", "正在获取视频数据...")

    cache = get_cache_manager()
    cache_key = f"crawl:{ctx.bvid}"
    cached_data = cache.get(cache_key, max_age_hours=12)

    if cached_data:
        ctx.meta, ctx.danmaku_list = cached_data
        ctx.progress("数据获取", f"缓存命中: {ctx.meta.title} ({len(ctx.danmaku_list)} 条弹幕)")
    else:
        ctx.meta, ctx.danmaku_list = await crawler.fetch_all(ctx.bvid)
        cache.set(cache_key, (ctx.meta, ctx.danmaku_list))
        ctx.progress("数据获取", f"获取成功: {ctx.meta.title} ({len(ctx.danmaku_list)} 条弹幕)")


async def _stage_preprocess(ctx: PipelineContext):
    """阶段 3：社会变量提取 + 用户去重 + 时序切分（CPU 密集，使用 executor 避免阻塞事件循环）"""
    loop = asyncio.get_running_loop()

    # 社会变量提取
    ctx.social_vars = SocialVariableExtractor().extract(ctx.meta)
    ctx.progress("社会变量", f"分区: {ctx.social_vars.tname}")

    # 用户去重（CPU 密集）
    deduplicator = UserDeduplicator()
    ctx.dedup_result = await loop.run_in_executor(
        None, functools.partial(deduplicator.deduplicate, ctx.danmaku_list)
    )
    ctx.progress("用户去重", f"去重完成: {ctx.dedup_result.unique_real_user_count} 用户")

    # 时序切分（CPU 密集）
    segmenter = TimelineSegmenter()
    ctx.segments = await loop.run_in_executor(
        None, functools.partial(segmenter.segment, ctx.dedup_result.deduplicated_danmaku)
    )
    ctx.progress("时序切分", f"切分完成: {len(ctx.segments)} 段")


async def _stage_analyze_segments(ctx: PipelineContext):
    """阶段 4：对每个分段执行硬统计 + LLM 分析"""
    settings = ctx.settings
    progress = ctx.progress
    social_vars = ctx.social_vars
    segments = ctx.segments
    dedup_result = ctx.dedup_result

    progress("弹幕分析", f"开始分析 {len(segments)} 段...")

    hard_analyzer = HardMetricsAnalyzer()
    context_provider = ContextProvider()
    prompt_builder = PromptBuilder()
    llm_client = LLMClient()
    llm_semaphore = asyncio.Semaphore(settings.LLM_CONCURRENCY)
    loop = asyncio.get_running_loop()

    records: List[DanmakuRecord] = []
    all_sample_danmaku = []
    all_sample_segments = []

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

    for i, segment in enumerate(segments):
        progress("弹幕分析", f"分析段 {i+1}/{len(segments)}...")
        segment_danmaku = [dedup_result.deduplicated_danmaku[idx] for idx in segment.danmaku_indices]

        # 硬统计（CPU 密集，使用 executor）
        contents = [d.content for d in segment_danmaku]
        hard_metrics = await loop.run_in_executor(
            None, functools.partial(hard_analyzer.analyze, contents)
        )

        # 采样策略
        if ctx.use_freq_based:
            content_counter = Counter(d.content for d in segment_danmaku)
            top_contents = content_counter.most_common(ctx.use_top_n)
            content_to_danmaku = {}
            for d in segment_danmaku:
                if d.content not in content_to_danmaku:
                    content_to_danmaku[d.content] = d
            sample_danmaku = [content_to_danmaku[content] for content, _ in top_contents]
        else:
            sample_danmaku = segment_danmaku[:ctx.use_top_n]

        # 并发 LLM 分析
        tasks = [_analyze_one(d, segment, segment_danmaku, hard_metrics, i) for d in sample_danmaku]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, Exception):
                logger.error(f"LLM 分析失败: {result}")
            else:
                records.append(result)

        all_sample_danmaku.extend(sample_danmaku)
        all_sample_segments.extend([segment] * len(sample_danmaku))

    ctx.records = records
    ctx.all_sample_danmaku = all_sample_danmaku
    ctx.all_sample_segments = all_sample_segments


async def _stage_aggregate(ctx: PipelineContext):
    """阶段 5：聚合 + 统计验证"""
    settings = ctx.settings
    progress = ctx.progress

    progress("数据聚合", "正在聚合数据...")
    ctx.aggregated = Aggregator().aggregate(ctx.records)
    progress("数据聚合", f"聚合完成: {len(ctx.aggregated)} 组")

    # Wilson 置信区间
    progress("统计验证", "正在计算置信区间...")
    validator = StatisticalValidator()
    min_samples = settings.MIN_SEGMENT_SAMPLES
    for agg in ctx.aggregated:
        total = agg.segment_count
        is_sufficient, msg = validator.validate_sample_size(total, min_samples)
        if is_sufficient:
            high_ci = validator.wilson_confidence_interval(
                int(agg.high_consensus_rate * total), total
            )
            agg.consensus_ci = high_ci.to_dict()
        else:
            agg.consensus_ci = {"status": "insufficient_sample", "sample_size": total, "min_required": min_samples}
    progress("统计验证", "置信区间计算完成")


async def _stage_report(ctx: PipelineContext):
    """阶段 6：报告生成 + LLM 报告 + ZIP 打包"""
    settings = ctx.settings
    progress = ctx.progress

    progress("报告生成", "正在生成报告...")
    reporter = Reporter(output_dir=ctx.use_output_dir)

    # 构建 kappa_ready 记录
    kappa_records = []
    for i, (record, danmaku) in enumerate(zip(ctx.records, ctx.all_sample_danmaku)):
        seg = ctx.all_sample_segments[i] if i < len(ctx.all_sample_segments) else ctx.segments[0]
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
        "bvid": ctx.bvid, "title": ctx.meta.title,
        "tname": ctx.social_vars.tname, "tags": ctx.social_vars.tags
    }
    ctx.reports = reporter.generate_reports(ctx.aggregated, kappa_records=kappa_records, metadata=video_metadata)
    progress("报告生成", "报告生成完成")

    # LLM 分析报告（可选）
    if settings.ENABLE_LLM_ANALYSIS_REPORT:
        progress("LLM报告", "正在生成社会语言学分析报告...")
        llm_report_path = await reporter.generate_llm_analysis_report(ctx.aggregated, metadata=video_metadata)
        if llm_report_path:
            ctx.reports["sociolinguistic_analysis_report"] = llm_report_path
            progress("LLM报告", "社会语言学分析报告生成完成")
        else:
            progress("LLM报告", "LLM分析报告生成失败或未启用")

    # ZIP 打包
    progress("报告打包", "正在打包报告...")
    safe_title = ctx.meta.title.replace('/', '_').replace('\\', '_').replace(':', '_').replace('*', '_').replace('?', '_').replace('"', '_').replace('<', '_').replace('>', '_').replace('|', '_')
    zip_filename = f"[{ctx.bvid}]{safe_title}.zip"
    zip_path = os.path.join(ctx.use_output_dir, zip_filename)

    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for name, path in ctx.reports.items():
            if os.path.exists(path):
                zipf.write(path, os.path.basename(path))

    # 验证 + 清理源文件
    ctx.zip_valid = _validate_zip(zip_path, ctx.reports)
    ctx.zip_path = zip_path
    if ctx.zip_valid:
        deleted_count = 0
        for name, path in ctx.reports.items():
            if os.path.exists(path):
                try:
                    os.remove(path)
                    deleted_count += 1
                except Exception as e:
                    logger.warning(f"删除失败: {os.path.basename(path)} - {e}")
        progress("报告打包", f"打包完成: {zip_filename} (已删除{deleted_count}个源文件)")
    else:
        progress("报告打包", f"打包完成: {zip_filename} (源文件保留)")


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

