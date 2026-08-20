"""HTML 可视化报告生成器 - 聚合数据渲染为离线单文件 report.html

内联 CSS + 内联 SVG 条形图，零外部依赖与网络请求；消费聚合数据与元数据，
可选嵌入 LLM 分析报告 markdown（极简规则转 HTML）；供双击直读的学者向产出。
"""

import html
import math
import os
import re
from typing import Dict, List, Optional

from .aggregator import AggregatedData
from .utils.logger import get_logger

logger = get_logger(__name__)

HTML_REPORT_FILENAME = "report.html"
CORPUS_REPORT_FILENAME = "corpus_report.html"

_ZONE_LABELS = {"hot_zone": "热区", "cold_zone": "冷区"}
_EMOTION_LABELS = {"positive": "正面", "neutral": "中性", "negative": "负面"}
_SENTENCE_LABELS = {
    "assertion": "陈述", "question": "疑问", "exclamation": "感叹",
    "directive": "祈使", "fragment": "碎片",
}
_INTERACTION_LABELS = {
    "check_in": "打卡报到", "identity_claim": "身份声明", "mocking": "调侃嘲讽",
    "info_request": "信息询问", "expression": "情感表达", "other": "其他",
}
_ORTHOGRAPHY_LABELS = {
    "standard": "规范书写", "community_variant": "社区变体", "non_standard_typo": "非规范错字",
}

_CSS = """
body { font-family: 'Segoe UI', 'Microsoft YaHei', sans-serif; margin: 0; background: #f5f6f8; color: #2c3e50; }
.container { width: 94%; margin: 0 auto; padding: 24px 0 64px; }
header.hero { background: linear-gradient(135deg, #fb7299, #23ade5); color: #fff; border-radius: 12px; padding: 28px 32px; }
header.hero h1 { margin: 0 0 8px; font-size: 22px; }
header.hero .meta { opacity: 0.92; font-size: 13px; line-height: 1.7; }
.cards { display: flex; flex-wrap: wrap; gap: 12px; margin: 20px 0; }
.card { flex: 1 1 160px; background: #fff; border-radius: 10px; padding: 14px 18px; box-shadow: 0 1px 3px rgba(0,0,0,0.06); }
.card .num { font-size: 22px; font-weight: 600; color: #fb7299; }
.card .lbl { font-size: 12px; color: #7f8c9b; margin-top: 4px; }
section { background: #fff; border-radius: 10px; padding: 20px 24px; margin-top: 16px; box-shadow: 0 1px 3px rgba(0,0,0,0.06); }
section h2 { font-size: 16px; margin: 0 0 14px; border-left: 4px solid #23ade5; padding-left: 10px; }
table { border-collapse: collapse; width: 100%; font-size: 13px; }
th, td { border-bottom: 1px solid #e8eaed; padding: 7px 10px; text-align: left; }
th { background: #f8f9fb; color: #5a6b7c; font-weight: 600; }
tr:hover td { background: #fbfcfe; }
.chart { margin-top: 12px; }
.chart-title { font-size: 13px; color: #5a6b7c; margin: 14px 0 6px; }
.note { font-size: 12px; color: #98a2ad; margin-top: 10px; }
.table-wrap { overflow-x: auto; }
.llm-report { line-height: 1.85; font-size: 14px; }
.llm-report h1, .llm-report h2, .llm-report h3 { margin: 18px 0 8px; }
.llm-report pre { background: #f5f6f8; padding: 10px; border-radius: 6px; overflow-x: auto; }
footer { text-align: center; color: #98a2ad; font-size: 12px; margin-top: 28px; }
"""


def _esc(value) -> str:
    return html.escape(str(value))


def _zone_label(zone_type: str) -> str:
    return _ZONE_LABELS.get(zone_type, zone_type or "整体")


def _group_key(item: AggregatedData) -> str:
    return f"{item.tname} · {_zone_label(item.zone_type)}"


def _fmt(value, digits: int = 4) -> str:
    if value is None:
        return "--"
    if isinstance(value, float):
        return f"{value:.{digits}f}".rstrip('0').rstrip('.') or "0"
    return _esc(value)


def _svg_bar_chart(pairs: List[tuple], color: str = "#23ade5") -> str:
    """(标签, 占比) 列表 → 内联 SVG 横向条形图；占比按 0-1 渲染"""
    if not pairs:
        return '<p class="note">无分布数据</p>'
    width, row_h, label_w, bar_max = 620, 24, 110, 440
    height = row_h * len(pairs)
    bars = []
    for i, (label, ratio) in enumerate(pairs):
        try:
            value = max(0.0, min(1.0, float(ratio)))
        except (TypeError, ValueError):
            continue
        y = i * row_h
        bar_w = int(round(bar_max * value))
        bars.append(
            f'<text x="{label_w - 8}" y="{y + 16}" text-anchor="end" font-size="12" fill="#5a6b7c">{_esc(label)}</text>'
            f'<rect x="{label_w}" y="{y + 4}" width="{bar_w}" height="14" rx="3" fill="{color}" opacity="0.85"/>'
            f'<text x="{label_w + bar_w + 6}" y="{y + 16}" font-size="11" fill="#7f8c9b">{value:.1%}</text>'
        )
    return f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">{"".join(bars)}</svg>'


def _table(headers: List[str], rows: List[List]) -> str:
    head = "".join(f"<th>{_esc(h)}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{_fmt(c)}</td>" for c in row) + "</tr>" for row in rows)
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def _markdown_to_html(md: str) -> str:
    """极简 markdown 转 HTML：标题/列表/粗体/行内代码/代码块/段落，先整体转义防注入"""
    text = html.escape(md)
    lines, out, in_code, in_list = text.splitlines(), [], False, False
    for line in lines:
        if line.strip().startswith("```"):
            if in_code:
                out.append("</pre>")
                in_code = False
            else:
                if in_list:
                    out.append("</ul>")
                    in_list = False
                out.append("<pre>")
                in_code = True
            continue
        if in_code:
            out.append(line)
            continue
        stripped = line.strip()
        heading = re.match(r"^(#{1,4})\s+(.*)$", stripped)
        if heading:
            if in_list:
                out.append("</ul>")
                in_list = False
            level = len(heading.group(1))
            out.append(f"<h{level}>{heading.group(2)}</h{level}>")
            continue
        if stripped.startswith(("- ", "* ")):
            if not in_list:
                out.append("<ul>")
                in_list = True
            out.append(f"<li>{stripped[2:]}</li>")
            continue
        if in_list:
            out.append("</ul>")
            in_list = False
        if stripped:
            rendered = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", stripped)
            rendered = re.sub(r"`(.+?)`", r"<code>\1</code>", rendered)
            out.append(f"<p>{rendered}</p>")
    if in_list:
        out.append("</ul>")
    if in_code:
        out.append("</pre>")
    return "\n".join(out)


class HtmlReportGenerator:

    def __init__(self, output_dir: str):
        self.output_dir = output_dir

    def write(self, aggregated: List[AggregatedData], metadata: Dict,
              llm_report_md: Optional[str] = None) -> str:
        os.makedirs(self.output_dir, exist_ok=True)
        filepath = os.path.join(self.output_dir, HTML_REPORT_FILENAME)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(self.render(aggregated, metadata, llm_report_md))
        logger.info(f"HTML 可视化报告已保存: {filepath}")
        return filepath

    def write_corpus(self, summary_df, tests_df, metadata: Dict) -> str:
        os.makedirs(self.output_dir, exist_ok=True)
        filepath = os.path.join(self.output_dir, CORPUS_REPORT_FILENAME)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(self.render_corpus(summary_df, tests_df, metadata))
        logger.info(f"语料库 HTML 报告已保存: {filepath}")
        return filepath

    def render_corpus(self, summary_df, tests_df, metadata: Dict) -> str:
        tnames = metadata.get("tnames") or []
        mode_label = "合并分析" if len(tnames) <= 1 else "比对分析"
        hero = (
            "<header class='hero'>"
            f"<h1>语料库分析报告（{mode_label}）</h1>"
            f"<div class='meta'>分区 {_esc('、'.join(tnames) or '—')} · "
            f"视频 {metadata.get('video_count', 0)} 个 · 弹幕总量 {metadata.get('total_danmaku', 0):,} 条</div>"
            "</header>"
        )
        cards = [
            (metadata.get("video_count", 0), "视频数"),
            (f"{metadata.get('total_danmaku', 0):,}", "弹幕总量"),
            (len(tnames), "分区数"),
            (len(summary_df), "聚合组数"),
        ]
        cards_html = '<div class="cards">' + "".join(
            f'<div class="card"><div class="num">{_esc(value)}</div><div class="lbl">{label}</div></div>'
            for value, label in cards
        ) + "</div>"

        if tests_df is not None and len(tests_df):
            tests_html = (
                "<section><h2>推断统计（未校正 p 值）</h2>"
                + self._render_df_table(tests_df)
                + '<p class="note">配对/组间检验均为探索性分析，p 值未做多重比较校正。</p></section>'
            )
        else:
            tests_html = "<section><h2>推断统计</h2><p class='note'>未执行推断检验（无可用比较轴或未启用语料库级统计）。</p></section>"

        parts = [
            "<!DOCTYPE html>",
            '<html lang="zh-CN"><head><meta charset="utf-8">',
            "<title>语料库分析报告</title>",
            f"<style>{_CSS}</style></head><body><div class='container'>",
            hero,
            cards_html,
            "<section><h2>描写层聚合</h2>" + self._render_df_table(summary_df)
            + '<p class="note">*_mean/*_std 为跨视频均值/标准差，分布列为弹幕数加权占比。</p></section>',
            tests_html,
            "<footer>DanmakuScope 语料库级产出</footer>",
            "</div></body></html>",
        ]
        return "\n".join(parts)

    @staticmethod
    def _fmt_cell(value) -> str:
        if value is None:
            return "--"
        if isinstance(value, float) and math.isnan(value):
            return "--"
        if isinstance(value, float):
            return f"{value:.4f}".rstrip('0').rstrip('.') or "0"
        return _esc(value)

    def _render_df_table(self, df) -> str:
        head = "".join(f"<th>{_esc(str(c))}</th>" for c in df.columns)
        body = "".join(
            "<tr>" + "".join(f"<td>{self._fmt_cell(v)}</td>" for v in row) + "</tr>"
            for row in df.itertuples(index=False, name=None)
        )
        return f'<div class="table-wrap"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'

    def render(self, aggregated: List[AggregatedData], metadata: Dict,
               llm_report_md: Optional[str] = None) -> str:
        parts = [
            "<!DOCTYPE html>",
            '<html lang="zh-CN"><head><meta charset="utf-8">',
            f"<title>弹幕语料分析报告 - {_esc(metadata.get('bvid', ''))}</title>",
            f"<style>{_CSS}</style></head><body><div class='container'>",
            self._render_hero(metadata),
            self._render_cards(aggregated, metadata),
            self._render_overview_table(aggregated),
            self._render_distributions(aggregated),
            self._render_consensus(aggregated),
        ]
        if llm_report_md:
            parts.append(
                "<section><h2>社会语言学分析报告</h2>"
                f'<div class="llm-report">{_markdown_to_html(llm_report_md)}</div></section>'
            )
        parts.append(
            f"<footer>DanmakuScope {_esc(metadata.get('pipeline_version', ''))} · "
            f"Prompt {_esc(metadata.get('prompt_version', ''))} · "
            f"生成于 {_esc((metadata.get('generated_at') or '')[:19])}</footer>"
        )
        parts.append("</div></body></html>")
        return "\n".join(parts)

    def _render_hero(self, metadata: Dict) -> str:
        tags = "、".join(metadata.get("tags", [])[:5]) or "—"
        source = "XML 兜底（存在数量上限）" if metadata.get("danmaku_source") == "xml" else "protobuf"
        return (
            "<header class='hero'>"
            f"<h1>{_esc(metadata.get('title') or metadata.get('bvid', ''))}</h1>"
            f"<div class='meta'>BV号 {_esc(metadata.get('bvid', ''))} · 分区 {_esc(metadata.get('tname', ''))} · "
            f"发布 {_esc(str(metadata.get('pubdate', ''))[:10])} · 播放 {_esc(metadata.get('view_count', 0))} · "
            f"标签 {tags}<br>弹幕来源 {source}</div>"
            "</header>"
        )

    def _render_cards(self, aggregated: List[AggregatedData], metadata: Dict) -> str:
        cards = [
            (metadata.get("danmaku_count", sum(i.danmaku_count for i in aggregated)), "弹幕总数"),
            (sum(i.segment_count for i in aggregated), "时序分段数"),
            (len({i.tname for i in aggregated}), "官方分区数"),
            (metadata.get("view_count", 0), "视频播放量"),
        ]
        return '<div class="cards">' + "".join(
            f'<div class="card"><div class="num">{_esc(value)}</div><div class="lbl">{label}</div></div>'
            for value, label in cards
        ) + "</div>"

    def _render_overview_table(self, aggregated: List[AggregatedData]) -> str:
        rows = [
            [_group_key(i), i.danmaku_count, i.avg_word_length, i.content_word_density,
             i.punctuation_emoji_rate, i.cooperative_principle_violation_rate]
            for i in aggregated
        ]
        return (
            "<section><h2>描写层概览</h2>"
            + _table(["分区 · 冷热区", "弹幕数", "平均词长", "实词密度", "标点与表情率", "合作原则违背率"], rows)
            + '<p class="note">实词密度 = 名词/动词/形容词占比；合作原则违背率为 LLM 软标签加权结果。</p></section>'
        )

    def _render_distributions(self, aggregated: List[AggregatedData]) -> str:
        sections = []
        for item in aggregated:
            charts = [
                ("情感分布", [( _EMOTION_LABELS.get(k, k), v) for k, v in item.emotion_distribution.items()], "#fb7299"),
                ("句类分布", [(_SENTENCE_LABELS.get(k, k), v) for k, v in item.sentence_function_distribution.items()], "#23ade5"),
                ("互动类型分布", [(_INTERACTION_LABELS.get(k, k), v) for k, v in item.interaction_type_distribution.items()], "#52c41a"),
                ("正字法状态分布", [(_ORTHOGRAPHY_LABELS.get(k, k), v) for k, v in item.orthography_status_distribution.items()], "#faad14"),
            ]
            rendered = "".join(
                f'<div class="chart-title">{title}</div><div class="chart">{_svg_bar_chart(pairs, color)}</div>'
                for title, pairs, color in charts if pairs
            )
            sections.append(f"<section><h2>语用层分布 · {_esc(_group_key(item))}</h2>{rendered}</section>")
        return "\n".join(sections)

    def _render_consensus(self, aggregated: List[AggregatedData]) -> str:
        rows = []
        for item in aggregated:
            ci = item.consensus_ci or {}
            status = ci.get("status", "ok")
            ci_text = (
                f"[{_fmt(ci.get('lower'))}, {_fmt(ci.get('upper'))}]"
                if status == "ok" else "样本不足"
            )
            rows.append([_group_key(item), item.high_consensus_rate, item.medium_consensus_rate,
                         item.low_consensus_rate, item.avg_weight_multiplier, ci_text])
        return (
            "<section><h2>双路共识统计</h2>"
            + _table(["分区 · 冷热区", "高共识率", "中共识率", "低共识率", "平均权重系数", "高共识率 95% CI"], rows)
            + '<p class="note">共识基于双温度路径输出的归一化 JSD；低共识样本按权重 0.2 保留（零丢弃）。</p></section>'
        )
