"""
多格式导出模块 - LaTeX 表格片段与 APA 格式统计文本
消费报告 ZIP 或聚合 CSV，面向论文写作场景产出可直接引用的片段
"""

import io
import os
import re
import zipfile
from typing import Optional

import pandas as pd

from .utils.logger import get_logger

logger = get_logger(__name__)

# 单视频报告表 → 展示名；语料库快照内的 corpus_summary.csv 优先于单视频表
SINGLE_VIDEO_TABLES = {
    "table_lexical_by_partition.csv": "词类统计",
    "table_orthography.csv": "正字法统计",
    "table_sentence_function.csv": "句类分布",
    "table_emotion.csv": "情感分布",
    "table_interaction_type.csv": "互动类型分布",
    "table_consensus_stats.csv": "共识统计",
}
CORPUS_SUMMARY_FILENAME = "corpus_summary.csv"
STATS_TESTS_FILENAME = "statistical_tests.csv"

_LATEX_SPECIAL = re.compile(r'([&%$#_{}~^\\])')
_EXPORT_DIMENSIONS = {
    "tname": "分区",
    "zone_type": "冷热区",
    "time_period": "时段",
}


class Exporter:

    def export(self, source_path: str, fmt: str, output_dir: Optional[str] = None) -> str:
        """入口：ZIP/CSV → 指定格式文件路径；未知格式抛 ValueError"""
        fmt = (fmt or "").lower()
        if fmt not in ("latex", "apa"):
            raise ValueError(f"不支持的导出格式: {fmt}（可选 latex / apa）")
        if source_path.lower().endswith(".zip"):
            content = self.from_zip(source_path, fmt)
        elif source_path.lower().endswith(".csv"):
            content = self.from_csv(source_path, fmt)
        else:
            raise ValueError(f"不支持的输入类型（仅支持 .zip / .csv）: {source_path}")

        out_dir = output_dir or os.getcwd()
        os.makedirs(out_dir, exist_ok=True)
        ext = "tex" if fmt == "latex" else "txt"
        out_path = os.path.join(out_dir, f"{os.path.splitext(os.path.basename(source_path))[0]}_{fmt}.{ext}")
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(content)
        logger.info(f"{fmt.upper()} 导出已保存: {out_path}")
        return out_path

    def from_zip(self, zip_path: str, fmt: str) -> str:
        with zipfile.ZipFile(zip_path, 'r') as zipf:
            names = set(zipf.namelist())
            if CORPUS_SUMMARY_FILENAME in names:
                df = self._read_csv_bytes(zipf.read(CORPUS_SUMMARY_FILENAME))
                if fmt == "latex":
                    parts = [self.to_latex_table(df, "语料库级比较表", "tab:corpus_summary")]
                    if STATS_TESTS_FILENAME in names:
                        parts.append(self.stats_to_latex(self._read_csv_bytes(zipf.read(STATS_TESTS_FILENAME))))
                else:
                    parts = [self.to_apa_text(df)]
                    if STATS_TESTS_FILENAME in names:
                        parts.append(self.stats_to_apa(self._read_csv_bytes(zipf.read(STATS_TESTS_FILENAME))))
                return "\n\n".join(parts)

            if fmt == "latex":
                parts = [
                    self.to_latex_table(self._read_csv_bytes(zipf.read(name)), f"{title}表", f"tab:{name[6:-4]}")
                    for name, title in SINGLE_VIDEO_TABLES.items() if name in names
                ]
                if not parts:
                    raise ValueError(f"ZIP 内未找到可导出的聚合表: {zip_path}")
                return "\n\n".join(parts)

            parts = []
            for name, title in SINGLE_VIDEO_TABLES.items():
                if name in names:
                    parts.append(f"### {title}\n\n{self.to_apa_text(self._read_csv_bytes(zipf.read(name)))}")
            if not parts:
                raise ValueError(f"ZIP 内未找到可导出的聚合表: {zip_path}")
            return "\n\n".join(parts)

    def from_csv(self, csv_path: str, fmt: str) -> str:
        df = pd.read_csv(csv_path, encoding='utf-8-sig')
        return self.to_latex_table(df) if fmt == "latex" else self.to_apa_text(df)

    @staticmethod
    def _read_csv_bytes(data: bytes) -> pd.DataFrame:
        return pd.read_csv(io.BytesIO(data), encoding='utf-8-sig')

    @staticmethod
    def _escape_latex(value) -> str:
        return _LATEX_SPECIAL.sub(r'\\\1', str(value))

    @staticmethod
    def _format_cell(value) -> str:
        if pd.isna(value):
            return "--"
        if isinstance(value, float):
            return f"{value:.4f}".rstrip('0').rstrip('.') or "0"
        return Exporter._escape_latex(value)

    def to_latex_table(self, df: pd.DataFrame, caption: str = "", label: str = "") -> str:
        """booktabs 风格 tabular 片段（可直接 \\input 进论文主文件）"""
        columns = [str(c) for c in df.columns]
        header = " & ".join(self._escape_latex(c) for c in columns) + " \\\\"
        body = [
            " & ".join(self._format_cell(v) for v in row) + " \\\\"
            for row in df.itertuples(index=False, name=None)
        ]
        lines = ["\\begin{table}[htbp]", "\\centering"]
        if caption:
            lines.append(f"\\caption{{{self._escape_latex(caption)}}}")
        if label:
            lines.append(f"\\label{{{label}}}")
        lines.append(f"\\begin{{tabular}}{{{'l' * len(columns)}}}")
        lines += ["\\toprule", header, "\\midrule", *body, "\\bottomrule", "\\end{tabular}", "\\end{table}"]
        return "\n".join(lines)

    def stats_to_latex(self, stats_df: pd.DataFrame) -> str:
        """statistical_tests.csv → booktabs 检验结果表（混合 dtype 空值归一为 --，与 APA 文本同源不重复计算）"""
        df = stats_df.copy()
        for col in df.columns:
            df[col] = df[col].map(lambda v: "" if pd.isna(v) else v)
        return self.to_latex_table(df, "语料库级推断统计检验结果（未校正 p 值）", "tab:statistical_tests")

    def to_apa_text(self, df: pd.DataFrame) -> str:
        """APA 风格描述统计文本：*_mean/*_std 配对优先，其余数值列按 M (SD) 报告"""
        paragraphs = []
        for _, row in df.iterrows():
            dims = "、".join(
                f"{label} {row[col]}"
                for col, label in _EXPORT_DIMENSIONS.items()
                if col in df.columns and pd.notna(row.get(col)) and str(row.get(col)) != ""
            ) or "整体"
            n_videos = int(row["video_count"]) if "video_count" in df.columns and pd.notna(row.get("video_count")) else None
            n_danmaku = int(row["total_danmaku"]) if "total_danmaku" in df.columns and pd.notna(row.get("total_danmaku")) else None

            pairs = []
            consumed = set()
            for col in df.columns:
                if col.endswith("_mean"):
                    base = col[:-5]
                    std_col = f"{base}_std"
                    mean_val = row.get(col)
                    if pd.isna(mean_val):
                        continue
                    std_val = row.get(std_col) if std_col in df.columns else None
                    consumed.update({col, std_col})
                    if pd.isna(std_val):
                        pairs.append(f"{base} M = {self._apa_num(mean_val)}")
                    else:
                        pairs.append(f"{base} M = {self._apa_num(mean_val)}, SD = {self._apa_num(std_val)}")
            for col in df.columns:
                value = row.get(col)
                if col in consumed or col in _EXPORT_DIMENSIONS or not isinstance(value, (int, float)) or pd.isna(value):
                    continue
                pairs.append(f"{col} = {self._apa_num(value)}")

            counts = []
            if n_videos is not None:
                counts.append(f"k = {n_videos} 个视频")
            if n_danmaku is not None:
                counts.append(f"N = {n_danmaku:,} 条弹幕")
            count_text = f"（{'，'.join(counts)}）" if counts else ""
            paragraphs.append(f"{dims}{count_text}：{'；'.join(pairs)}。" if pairs else f"{dims}{count_text}。")
        return "\n".join(paragraphs)

    @staticmethod
    def _apa_num(value) -> str:
        return f"{float(value):.3f}"

    def stats_to_apa(self, stats_df: pd.DataFrame) -> str:
        """statistical_tests.csv → APA 推断统计文本（KW 总检验 + 逐对 MWU + 冷热区配对 Wilcoxon + note 注记 + 样本不足说明，均标注未校正）"""
        lines = []
        kw_rows = stats_df[stats_df["test_type"] == "Kruskal-Wallis"]
        for _, row in kw_rows.iterrows():
            if pd.isna(row.get("p_value")):
                lines.append(f"指标 {row['metric']}：样本量不足，未执行 Kruskal-Wallis 检验（{row.get('note', '')}）。")
                continue
            axis_word = "时段" if "检验轴：时段" in str(row.get("note", "")) else "分区"
            kw_effect = row.get("effect_size")
            kw_effect_text = f"，ε² = {self._apa_num(kw_effect)}" if pd.notna(kw_effect) and str(kw_effect).strip() != "" else ""
            lines.append(
                f"指标 {row['metric']} 的{axis_word}间差异检验：Kruskal-Wallis H = {self._apa_num(row['statistic'])}，"
                f"p {self._apa_p(row['p_value'])}{kw_effect_text}（未校正 p 值）。"
            )

        status_rows = stats_df[
            (stats_df["test_type"] == "sample_status")
            & stats_df["note"].astype(str).str.contains("insufficient_sample", na=False)
        ]
        for _, row in status_rows.iterrows():
            lines.append(f"分区 {row['group1']} 样本量不足（n = {int(row['n1'])}），未纳入推断检验。")

        mwu = stats_df[stats_df["test_type"] == "Mann-Whitney U"]
        if not mwu.empty:
            lines.append("逐对比较（Mann-Whitney U，未校正 p 值）：")
            for _, row in mwu.iterrows():
                effect = f"，Cliff's δ = {self._apa_num(row['effect_size'])}（{row['effect_magnitude']}）" if pd.notna(row.get("effect_size")) else ""
                lines.append(
                    f"  {row['group1']} vs {row['group2']}：U = {self._apa_num(row['statistic'])}，"
                    f"p {self._apa_p(row['p_value'])}{effect}。"
                )

        wilcoxon = stats_df[stats_df["test_type"] == "Wilcoxon 符号秩（配对）"]
        if not wilcoxon.empty:
            lines.append("冷热区配对比较（Wilcoxon 符号秩检验，未校正 p 值）：")
            for _, row in wilcoxon.iterrows():
                effect = f"，Cliff's δ = {self._apa_num(row['effect_size'])}（{row['effect_magnitude']}）" if pd.notna(row.get("effect_size")) else ""
                lines.append(
                    f"  指标 {row['metric']}：{row['group1']} vs {row['group2']}（n = {int(float(row['n1']))} 对）："
                    f"W = {self._apa_num(row['statistic'])}，p {self._apa_p(row['p_value'])}{effect}。"
                )

        note_rows = stats_df[stats_df["test_type"] == "note"]
        for _, row in note_rows.iterrows():
            note = str(row.get("note", "")).strip()
            if note:
                lines.append(f"注：{note}。")
        return "\n".join(lines)

    @staticmethod
    def _apa_p(p) -> str:
        p = float(p)
        return "< .001" if p < 0.001 else f"= {p:.3f}".replace("0.", ".")
