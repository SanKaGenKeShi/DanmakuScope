"""
CLI 入口模块 - 命令行界面
"""

import asyncio
import os
import sys
from typing import List, Optional

import click
from rich.console import Console
from rich.table import Table

from .config import get_settings
from .llm_config import get_llm_settings
from .utils.logger import get_logger, setup_logger
from . import __version__

logger = get_logger(__name__)
console = Console()


@click.group()
@click.option('--debug', is_flag=True, help='启用调试模式')
@click.option('--log-level', default='INFO', help='日志级别')
def cli(debug: bool, log_level: str):
    """DanmakuScope - B站弹幕社会语言学分析工具"""
    if debug:
        log_level = 'DEBUG'
    
    setup_logger(level=log_level)
    
    if debug:
        console.print("[yellow]调试模式已启用[/yellow]")


@cli.command()
@click.argument('input_str')
@click.option('--output', '-o', default=None, help='输出目录')
@click.option('--credential', '-c', default=None, help='B站登录凭证文件（JSON：sessdata/bili_jct/buvid3）')
@click.option('--freq-based', is_flag=True, default=False, help='启用按频次排序采样（默认使用每段前N条）')
@click.option('--top-n', default=None, type=int, help='频次排序时取前N条（默认10）')
@click.option('--no-cache', is_flag=True, default=False, help='禁用爬取缓存，强制重新获取')
def analyze(input_str: str, output: Optional[str], credential: Optional[str], freq_based: bool, top_n: Optional[int], no_cache: bool):
    """分析单个视频"""
    try:
        asyncio.run(_analyze_async(input_str, output, credential, freq_based, top_n, no_cache))
    except Exception as e:
        console.print(f"[red]分析失败: {e}[/red]")
        logger.error(f"分析失败: {e}", exc_info=True)
        sys.exit(1)


async def _analyze_async(
    input_str: str, 
    output: Optional[str],
    credential_file: Optional[str],
    freq_based: bool = False,
    top_n: Optional[int] = None,
    no_cache: bool = False
):
    """异步分析单个视频，失败时向上抛出异常"""
    # pipeline 拖入 jieba/pandas/openai 等重型依赖，延迟到实际分析时才导入，
    # 使 config/login/account/corpus/suggest 等轻量命令免于全量加载
    from .pipeline import analyze_video

    def rich_progress_callback(stage: str, message: str):
        console.print(f"[cyan]{stage}[/cyan]: {message}")
    
    result = await analyze_video(
        input_str=input_str,
        output_dir=output,
        credential_file=credential_file,
        freq_based=freq_based,
        top_n=top_n,
        progress_callback=rich_progress_callback,
        no_cache=no_cache
    )
    
    # 空结果视为失败（如全部 LLM 调用失败），以退出码反映真实状态
    if not result.zip_valid:
        raise RuntimeError(f"分析未产生有效报告: {input_str}")
    
    _show_summary(result)
    return result


def _show_summary(result):
    console.print("\n" + "="*60)
    console.print("[bold green]分析完成！[/bold green]")
    console.print("="*60)
    
    console.print(f"\n[bold]视频信息[/bold]")
    console.print(f"  BV号: {result.bvid}")
    console.print(f"  标题: {result.title}")
    console.print(f"  分区: {result.tname}")
    console.print(f"  标签: {', '.join(result.tags[:5])}")
    
    console.print(f"\n[bold]分析统计[/bold]")
    console.print(f"  时间段数: {result.segments_count}")
    console.print(f"  聚合组数: {result.aggregated_count}")
    
    if result.zip_path and os.path.exists(result.zip_path):
        console.print(f"\n[bold]报告打包[/bold]")
        console.print(f"  ZIP文件: {result.zip_path}")
        console.print(f"  文件大小: {os.path.getsize(result.zip_path) / 1024:.1f} KB")
    else:
        console.print(f"\n[bold]报告文件[/bold]")
        for name, filepath in result.reports.items():
            console.print(f"  {name}: {filepath}")
    
    console.print("\n" + "="*60)


@cli.command()
@click.argument('input_list', nargs=-1)
@click.option('--output', '-o', default=None, help='输出目录')
@click.option('--credential', '-c', default=None, help='B站登录凭证文件（JSON：sessdata/bili_jct/buvid3）')
@click.option('--freq-based', is_flag=True, default=False, help='启用按频次排序采样（默认使用每段前N条）')
@click.option('--top-n', default=None, type=int, help='频次排序时取前N条（默认10）')
@click.option('--no-cache', is_flag=True, default=False, help='禁用爬取缓存，强制重新获取')
def batch(input_list: tuple, output: Optional[str], credential: Optional[str], freq_based: bool, top_n: Optional[int], no_cache: bool):
    """批量分析多个视频"""
    if not input_list:
        console.print("[red]请提供至少一个输入[/red]")
        sys.exit(1)
    
    asyncio.run(_batch_async(list(input_list), output, credential, freq_based, top_n, no_cache))


async def _batch_async(
    input_list: List[str],
    output: Optional[str],
    credential_file: Optional[str],
    freq_based: bool = False,
    top_n: Optional[int] = None,
    no_cache: bool = False
):
    """异步批量分析：单个失败不中断后续任务，最终以退出码反映失败数"""
    console.print(f"[bold]开始批量分析 {len(input_list)} 个视频[/bold]")
    
    fail_count = 0
    for i, input_str in enumerate(input_list):
        console.print(f"\n[bold blue]视频 {i+1}/{len(input_list)}[/bold blue]")
        try:
            await _analyze_async(input_str, output, credential_file, freq_based, top_n, no_cache)
        except Exception as e:
            fail_count += 1
            console.print(f"[red]分析失败: {input_str} - {e}[/red]")
            logger.error(f"分析失败: {input_str} - {e}", exc_info=True)
    
    if fail_count:
        console.print(f"[red]批量分析完成，{fail_count}/{len(input_list)} 个失败[/red]")
        sys.exit(1)


@cli.command()
@click.argument('zip_list', nargs=-1)
@click.option('--output', '-o', default=None, help='语料库聚合表输出目录')
@click.option('--from-index', is_flag=True, default=False, help='从语料库索引登记的全部视频聚合（无需列出 ZIP）')
@click.option('--with-r', is_flag=True, default=False, help='同时生成 R 可视化脚本模板（corpus_plots.R）')
def corpus(zip_list: tuple, output: Optional[str], from_index: bool, with_r: bool):
    """跨视频语料库级聚合（回读单视频 ZIP 报告，按分区输出比较表并打包快照）"""
    from .config import get_settings
    from .corpus_builder import CorpusBuilder

    if not from_index and not zip_list:
        console.print("[red]请提供至少一个 ZIP 文件，或使用 --from-index 从语料库索引聚合[/red]")
        sys.exit(1)

    try:
        builder = CorpusBuilder()
        if from_index:
            result = builder.build_from_index(output)
        else:
            result = builder.build_from_zips(list(zip_list), output)
        console.print(f"[bold green]语料库聚合完成[/bold green]")
        console.print(f"  聚合表: {result.csv_path}")

        extra_files = []
        if with_r:
            from .corpus_visualizer import CorpusVisualizer
            r_path = CorpusVisualizer().write_r_script(result.output_dir)
            extra_files.append(r_path)
            console.print(f"  R 可视化脚本: {r_path}")
            console.print(f"  [dim]运行: Rscript {os.path.basename(r_path)}（需安装 R 与 ggplot2/dplyr/tidyr）[/dim]")

        if get_settings().ENABLE_LLM_ANALYSIS_REPORT:
            report_path = asyncio.run(_generate_corpus_llm_report(builder, result))
            if report_path:
                extra_files.append(report_path)
                console.print(f"  LLM 比较分析报告: {report_path}")
            else:
                console.print("[yellow]语料库 LLM 分析报告生成失败（不影响聚合产物）[/yellow]")

        zip_path = builder.package_snapshot(result, extra_files)
        if result.zip_valid:
            console.print(f"  语料库快照: {zip_path}")
            console.print(f"  [dim]已收录 {len(result.source_zip_paths)} 个源视频 ZIP（原文件保留），散落源文件已清理[/dim]")
        else:
            console.print(f"[yellow]快照打包校验失败，散落文件已保留: {zip_path}[/yellow]")
    except Exception as e:
        console.print(f"[red]语料库聚合失败: {e}[/red]")
        logger.error(f"语料库聚合失败: {e}", exc_info=True)
        sys.exit(1)


async def _generate_corpus_llm_report(builder, result) -> Optional[str]:
    """生成语料库级 LLM 比较分析报告并落盘，返回文件路径（失败返回 None）"""
    from .report_generator import AnalysisReportGenerator

    corpus_metadata = builder.build_snapshot_metadata(result)
    content = await AnalysisReportGenerator().generate_corpus_report(
        result.csv_path, result.videos_csv_path, corpus_metadata
    )
    if not content:
        return None
    report_path = os.path.join(result.output_dir, "corpus_analysis_report.md")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(content)
    return report_path


@cli.command()
@click.argument('partitions', nargs=-1)
@click.option('--per-partition', default=10, help='每个分区的候选视频数（默认10）')
@click.option('--gaps-only', is_flag=True, default=False, help='仅显示语料库缺口，不联网拉取候选')
def suggest(partitions: tuple, per_partition: int, gaps_only: bool):
    """语料库补足建议：各分区缺口 + 候选视频（BV号/标题/播放量/弹幕数/发布日期）"""
    from .corpus_suggester import CorpusSuggester

    try:
        suggester = CorpusSuggester()
        if gaps_only:
            gaps = suggester.analyze_gaps()
            candidates = {}
        else:
            result = asyncio.run(suggester.suggest(list(partitions) or None, per_partition))
            gaps, candidates = result.gaps, result.candidates

        _show_gaps(gaps)
        if not gaps_only:
            _show_candidates(candidates)
    except Exception as e:
        console.print(f"[red]建议生成失败: {e}[/red]")
        logger.error(f"建议生成失败: {e}", exc_info=True)
        sys.exit(1)


def _show_gaps(gaps):
    table = Table(title="语料库分区缺口")
    table.add_column("分区", style="cyan")
    table.add_column("已有视频数", justify="right")
    table.add_column("最小要求", justify="right")
    table.add_column("缺口", justify="right")
    for g in gaps:
        style = "green" if g.is_sufficient else "red"
        table.add_row(g.tname, str(g.have), str(g.min_required), f"[{style}]{g.missing}[/{style}]")
    if not gaps:
        table.add_row("（语料库为空）", "0", "-", "-")
    console.print(table)


def _show_candidates(candidates):
    for tname, videos in candidates.items():
        console.print(f"\n[bold]分区「{tname}」候选视频（按弹幕数降序）[/bold]")
        if not videos:
            console.print("  [yellow]未获取到候选（搜索失败或无结果）[/yellow]")
            continue
        table = Table()
        table.add_column("BV号", style="cyan")
        table.add_column("标题", max_width=50, overflow="ellipsis")
        table.add_column("播放量", justify="right")
        table.add_column("弹幕数", justify="right")
        table.add_column("发布日期")
        for c in videos:
            table.add_row(c.bvid, c.title, f"{c.play:,}", f"{c.danmaku_count:,}", c.pubdate)
        console.print(table)
        console.print(f"  [dim]挑选后可执行: danmaku-analyzer analyze <BV号>[/dim]")


@cli.command()
@click.option('--output', '-o', default=None, help='凭证保存路径（默认 DATA_ROOT/credential.json）')
def login(output: Optional[str]):
    """扫码登录B站，自动获取并保存凭证"""
    try:
        asyncio.run(_login_async(output))
    except Exception as e:
        console.print(f"[red]登录失败: {e}[/red]")
        logger.error(f"登录失败: {e}")
        sys.exit(1)


async def _login_async(output: Optional[str]):
    from .account import qr_login, save_credential

    def status_callback(event: str, message: str):
        if event == "qr_ready":
            _render_qrcode(message)
            console.print("\n[yellow]请使用B站手机客户端扫描二维码登录（二维码有效期约 3 分钟）[/yellow]")
            console.print(f"[dim]若终端无法显示二维码，可在浏览器打开:\n{message}[/dim]")
        elif event == "scanned":
            console.print(f"[cyan]{message}[/cyan]")

    credential = await qr_login(status_callback=status_callback)
    path = save_credential(credential, output)
    console.print(f"[bold green]登录成功，凭证已保存: {path}[/bold green]")
    console.print("[dim]analyze/batch 未指定 --credential 时将自动使用该凭证[/dim]")


def _render_qrcode(url: str):
    """终端渲染二维码（qrcode 库缺失时静默降级，仅提示链接）"""
    try:
        import qrcode
        qr = qrcode.QRCode(border=1)
        qr.add_data(url)
        qr.print_ascii(invert=True)
    except ImportError:
        logger.warning("qrcode 库未安装，跳过终端二维码渲染")


@cli.command()
def account():
    """显示已保存的B站凭证状态"""
    from .account import load_credential, fetch_account_info, default_credential_path

    credential = load_credential()
    if not credential:
        console.print(f"[yellow]未找到有效凭证（{default_credential_path()}），请先执行 login[/yellow]")
        sys.exit(1)

    info = asyncio.run(fetch_account_info(credential["sessdata"]))
    if info["is_login"]:
        console.print(f"[bold green]凭证有效[/bold green]  用户: {info['uname']} (mid: {info['mid']})")
    else:
        console.print("[red]凭证已失效，请重新执行 login[/red]")
        sys.exit(1)


@cli.command()
def version():
    """显示版本信息"""
    console.print(f"[bold]DanmakuScope[/bold] v{__version__}")
    console.print(f"Prompt 版本: {get_llm_settings().PROMPT_VERSION}")
    console.print(f"Python 版本: {sys.version}")


@cli.command()
def config():
    """显示当前配置"""
    settings = get_settings()
    llm_cfg = get_llm_settings()
    
    table = Table(title="当前配置")
    table.add_column("配置项", style="cyan")
    table.add_column("值", style="green")
    
    table.add_row("MOE", str(settings.MOE))
    table.add_row("置信水平", str(settings.CONFIDENCE_LEVEL))
    table.add_row("切分模式", settings.SEGMENTATION_MODE)
    table.add_row("最小段样本", str(settings.MIN_SEGMENT_SAMPLES))
    table.add_row("弹幕采样策略", "频次排序" if settings.ENABLE_FREQ_BASED_SAMPLING else "每段前N条")
    table.add_row("TOP_N", str(settings.TOP_N))
    table.add_row("复杂LLM模型", llm_cfg.COMPLEX_LLM_MODEL)
    table.add_row("简单LLM模型", llm_cfg.SIMPLE_LLM_MODEL)
    table.add_row("双路推理", "开启" if llm_cfg.ENABLE_DUAL_PATH else "关闭")
    table.add_row("JSD低阈值", str(llm_cfg.JSD_THRESHOLD_LOW))
    table.add_row("JSD中阈值", str(llm_cfg.JSD_THRESHOLD_MEDIUM))
    table.add_row("微语境窗口", f"{settings.CONTEXT_TIME_WINDOW}s")
    table.add_row("最大Context Tokens", str(settings.MAX_CONTEXT_TOKENS))
    table.add_row("LLM分析报告", "开启" if settings.ENABLE_LLM_ANALYSIS_REPORT else "关闭")
    if settings.ENABLE_LLM_ANALYSIS_REPORT:
        table.add_row("分析报告模型", llm_cfg.effective_analysis_report_model)
    
    console.print(table)


def main():
    """主入口（pyproject.toml [project.scripts] 指向此处）"""
    cli()
