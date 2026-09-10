"""
语料库可视化模块 - 双后端脚本模板生成（R/ggplot2 与 Python/matplotlib+seaborn，均零运行时依赖）
消费 corpus_videos.csv 视频级观测表，按 VISUALIZATION_BACKEND 配置产出可直接执行的可视化脚本：
组间比较箱线图（叠加 statistical_tests.csv 预计算的 Kruskal-Wallis p 值）、分布列堆叠条形图、
冷热区配对箱线图（叠加 Wilcoxon 符号秩 p 值，观测表含双区数据时生成）；
单分区+时间分桶场景自动改按时段绘图（与 corpus_compare 分组键分流一致）。
统计检验由 Python 侧 statistical_validator.corpus_compare 预计算，生成脚本不重复计算，
亦不实施任何多重比较校正（p 值均为未校正）。
"""

import os
from typing import List, Optional

import pandas as pd

from .config import get_settings
from .report_schema import SCALAR_FIELDS, STATS_TESTS_FILENAME, VIDEOS_CSV_FILENAME
from .utils.logger import get_logger

logger = get_logger(__name__)

R_SCRIPT_FILENAME = "corpus_plots.R"
PYTHON_SCRIPT_FILENAME = "corpus_plots.py"

# 模板占位符：{scalars}/{partitions} 为 R 字符向量字面量，{csv_filename}/{stats_filename} 为文件名
R_SCRIPT_TEMPLATE = '''\
# DanmakuScope 语料库可视化脚本（自动生成模板，可自由修改）
# 统计单位：视频级观测（corpus_videos.csv 每行一个视频）
# 推断检验结果由 Python 侧 statistical_tests.csv 预计算（未校正 p 值，无多重比较校正），本脚本仅读取叠加，不重复计算
# 用法：Rscript corpus_plots.R [corpus_videos.csv 路径] [statistical_tests.csv 路径]

args <- commandArgs(trailingOnly = TRUE)
csv_path <- if (length(args) >= 1) args[1] else "{csv_filename}"
stats_path <- if (length(args) >= 2) args[2] else "{stats_filename}"

for (pkg in c("ggplot2", "dplyr", "tidyr")) {{
  if (!requireNamespace(pkg, quietly = TRUE)) install.packages(pkg)
}}
library(ggplot2)
library(dplyr)
library(tidyr)

videos <- read.csv(csv_path, fileEncoding = "UTF-8-BOM", stringsAsFactors = FALSE)

identity_cols <- c("bvid", "tname", "pubdate", "prompt_version", "zone_type", "danmaku_count", "time_period")
scalar_metrics <- c({scalars})
partitions <- c({partitions})

videos <- videos %>% filter(!is.na(tname), tname != "")
if (length(partitions) > 0) {{
  videos$tname <- factor(as.character(videos$tname), levels = partitions)
  videos <- videos %>% filter(!is.na(tname))
}}

# ---- 0. 比较轴自适应：单分区+时间分桶时改按时段绘图（与 corpus_compare 分组键分流一致） ----
axis_label <- "分区 (tname)"
if ("time_period" %in% names(videos)) {{
  tp_ok <- !is.na(videos$time_period) & trimws(as.character(videos$time_period)) != ""
  if (length(unique(trimws(as.character(videos$time_period[tp_ok])))) >= 2 &&
      length(unique(trimws(as.character(videos$tname)))) <= 1) {{
    videos <- videos[tp_ok, ]
    videos$tname <- trimws(as.character(videos$time_period))
    videos <- videos[order(videos$tname), ]
    axis_label <- "时段 (time_period)"
  }}
}}

# ---- 1. 读取 Python 侧预计算推断检验结果（缺失时降级为纯箱线图） ----
kw_labels <- NULL
wil_labels <- NULL
if (file.exists(stats_path)) {{
  stats_df <- read.csv(stats_path, fileEncoding = "UTF-8-BOM", stringsAsFactors = FALSE)
  print(stats_df)
  kw <- stats_df[stats_df$test_type == "Kruskal-Wallis" & !is.na(stats_df$p_value), ]
  if (nrow(kw) > 0) {{
    kw_zone <- ifelse(grepl("冷热区分层：", kw$note), sub(".*冷热区分层：([^；]+).*", "\\\\1", kw$note), "")
    kw_key <- paste(kw$metric, kw_zone, sep = " / ")
    kw_labels <- setNames(sprintf("%s\\nKW p=%.4g", kw_key, kw$p_value), kw_key)
  }}
  wil <- stats_df[stats_df$test_type == "Wilcoxon 符号秩（配对）" & !is.na(stats_df$p_value), ]
  if (nrow(wil) > 0) {{
    wil_labels <- setNames(sprintf("%.4g", wil$p_value), wil$metric)
  }}
}} else {{
  message("未找到 ", stats_path, "，箱线图不叠加检验结果")
}}

# ---- 2. 核心指标分区间箱线图（每视频一个点，分面标签叠加 KW p 值） ----
videos$plot_zone <- if ("zone_type" %in% names(videos)) ifelse(is.na(videos$zone_type), "", videos$zone_type) else ""
if ("bvid" %in% names(videos)) videos <- videos %>% distinct(bvid, plot_zone, .keep_all = TRUE)
metric_long <- videos %>%
  select(tname, plot_zone, any_of(scalar_metrics)) %>%
  pivot_longer(-c(tname, plot_zone), names_to = "metric", values_to = "value")

metric_long$facet_label <- paste(metric_long$metric, metric_long$plot_zone, sep = " / ")
if (!is.null(kw_labels)) {{
  idx <- match(metric_long$facet_label, names(kw_labels))
  hit <- !is.na(idx)
  metric_long$facet_label[hit] <- unname(kw_labels[idx[hit]])
}}

p_box <- ggplot(metric_long, aes(x = tname, y = value, fill = tname)) +
  geom_boxplot(outlier.shape = NA, alpha = 0.6) +
  geom_jitter(width = 0.15, size = 1.5, alpha = 0.7) +
  facet_wrap(~ facet_label, scales = "free_y") +
  theme_bw() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1), legend.position = "none") +
  labs(x = axis_label, y = "指标值")
ggsave("corpus_boxplots.pdf", p_box, width = 10, height = 8)
ggsave("corpus_boxplots.png", p_box, width = 10, height = 8, dpi = 300)

# ---- 3. 分布图消费同包汇总表，避免重新以弹幕数代替实际分母 ----
summary_path <- file.path(dirname(csv_path), "corpus_summary.csv")
if (file.exists(summary_path)) {{
  summary <- read.csv(summary_path, fileEncoding = "UTF-8-BOM", stringsAsFactors = FALSE, check.names = FALSE)
  summary$plot_group <- as.character(summary$tname)
  for (dimension in intersect(c("time_period", "zone_type"), names(summary))) {{
    summary$plot_group <- paste(summary$plot_group, summary[[dimension]], sep = " / ")
  }}
  dist_cols <- names(summary)[grepl("_mean$", names(summary)) & !sub("_mean$", "", names(summary)) %in% scalar_metrics & !grepl("^hard_", names(summary))]
  if (length(dist_cols) > 0) {{
    dist_long <- summary %>%
      select(plot_group, all_of(dist_cols)) %>%
      pivot_longer(-plot_group, names_to = "category", values_to = "value") %>%
      mutate(category = sub("_mean$", "", category), group = sub("_.*$", "", category)) %>%
      filter(!is.na(value))
    dist_long$group[dist_long$category %in% c("positive", "neutral", "negative")] <- "emotion"
    dist_long$group[dist_long$category %in% c("assertion", "question", "exclamation", "directive", "fragment")] <- "sentence_function"
    dist_long$group[dist_long$category %in% c("check_in", "identity_claim", "mocking", "info_request", "expression", "other")] <- "interaction_type"
    if (nrow(dist_long) > 0) {{
      p_dist <- ggplot(dist_long, aes(x = plot_group, y = value, fill = category)) +
        geom_col() + facet_wrap(~ group, scales = "free") + theme_bw() +
        theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
        labs(x = axis_label, y = "有效标注或词项加权占比", fill = "类别")
      ggsave("corpus_distributions.pdf", p_dist, width = 12, height = 8)
      ggsave("corpus_distributions.png", p_dist, width = 12, height = 8, dpi = 300)
    }}
  }}
}} else {{
  message("缺少 corpus_summary.csv，跳过分布图，避免按错误分母重新聚合")
}}

# ---- 4. 冷热区配对比较箱线图（叠加 Wilcoxon 符号秩 p 值；观测表无双区数据时跳过） ----
if ("zone_type" %in% names(videos)) {{
  paired <- videos %>% filter(zone_type %in% c("hot_zone", "cold_zone"))
  paired_metrics <- intersect(scalar_metrics, names(paired))
  if (nrow(paired) > 0 && length(unique(paired$zone_type)) >= 2 && length(paired_metrics) > 0) {{
    zone_long <- paired %>%
      select(zone_type, all_of(paired_metrics)) %>%
      pivot_longer(-zone_type, names_to = "metric", values_to = "value")
    zone_long$zone_type <- factor(zone_long$zone_type, levels = c("hot_zone", "cold_zone"))
    zone_long$facet_label <- zone_long$metric
    if (!is.null(wil_labels)) {{
      idx <- match(zone_long$metric, names(wil_labels))
      hit <- !is.na(idx)
      zone_long$facet_label[hit] <- paste0(zone_long$metric[hit], "\\nWilcoxon p=", unname(wil_labels[idx[hit]]))
    }}
    p_zone <- ggplot(zone_long, aes(x = zone_type, y = value, fill = zone_type)) +
      geom_boxplot(outlier.shape = NA, alpha = 0.6) +
      geom_jitter(width = 0.15, size = 1.5, alpha = 0.7) +
      facet_wrap(~ facet_label, scales = "free_y") +
      theme_bw() +
      theme(axis.text.x = element_text(angle = 45, hjust = 1), legend.position = "none") +
      labs(x = "冷热区 (zone_type)", y = "指标值")
    ggsave("corpus_zone_paired.pdf", p_zone, width = 10, height = 8)
    ggsave("corpus_zone_paired.png", p_zone, width = 10, height = 8, dpi = 300)
  }}
}}

cat("完成：corpus_boxplots.* / corpus_distributions.* / corpus_zone_paired.*（如适用）\\n")
'''

# Python 后端模板：占位符同 R 模板；生成代码刻意不使用花括号（dict/f-string），避免 .format 转义噪声
PYTHON_SCRIPT_TEMPLATE = '''\
# DanmakuScope 语料库可视化脚本（Python 后端自动生成模板，可自由修改）
# 统计单位：视频级观测（corpus_videos.csv 每行一个视频）
# 推断检验结果由 Python 侧 statistical_tests.csv 预计算（未校正 p 值，无多重比较校正），本脚本仅读取叠加，不重复计算
# 用法：python corpus_plots.py [corpus_videos.csv 路径] [statistical_tests.csv 路径]
# 依赖：pip install "danmaku-analyzer[viz]" 或 pip install "matplotlib>=3.7" "seaborn>=0.12" pandas

import os
import sys

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

csv_path = sys.argv[1] if len(sys.argv) > 1 else "{csv_filename}"
stats_path = sys.argv[2] if len(sys.argv) > 2 else "{stats_filename}"

# 中文分区名如显示为方框，取消下行注释并换成本机已安装的中文字体
# plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

videos = pd.read_csv(csv_path, encoding="utf-8-sig")
scalar_metrics = [{scalars}]
partitions = [{partitions}]

videos = videos[videos["tname"].notna() & (videos["tname"] != "")]
if partitions:
    videos["tname"] = pd.Categorical(videos["tname"], categories=partitions)
    videos = videos[videos["tname"].notna()]

# ---- 0. 比较轴自适应：单分区+时间分桶时改按时段绘图（与 corpus_compare 分组键分流一致） ----
axis_label = "分区 (tname)"
if "time_period" in videos.columns:
    tp = videos["time_period"].dropna().astype(str).str.strip()
    tp = tp[tp != ""]
    if tp.nunique() >= 2 and videos["tname"].astype(str).str.strip().nunique() <= 1:
        videos = videos[videos["time_period"].notna() & (videos["time_period"].astype(str).str.strip() != "")]
        videos["tname"] = videos["time_period"].astype(str)
        videos = videos.sort_values("tname")
        axis_label = "时段 (time_period)"

# ---- 1. 读取 Python 侧预计算推断检验结果（缺失时降级为纯箱线图） ----
kw_labels = dict()
wilcoxon_labels = dict()
if os.path.exists(stats_path):
    stats_df = pd.read_csv(stats_path, encoding="utf-8-sig")
    print(stats_df)
    kw = stats_df[(stats_df["test_type"] == "Kruskal-Wallis") & stats_df["p_value"].notna()]
    for _, row in kw.iterrows():
        note = str(row.get("note", ""))
        zone = note.split("冷热区分层：", 1)[1].split("；", 1)[0] if "冷热区分层：" in note else ""
        kw_labels[(row["metric"], zone)] = "%.4g" % row["p_value"]
    wil = stats_df[(stats_df["test_type"] == "Wilcoxon 符号秩（配对）") & stats_df["p_value"].notna()]
    for _, row in wil.iterrows():
        wilcoxon_labels[row["metric"]] = "%.4g" % row["p_value"]
else:
    print("未找到 " + stats_path + "，箱线图不叠加检验结果")

# ---- 2. 核心指标组间比较箱线图（每视频一个点，子图标题叠加 KW p 值） ----
videos["_plot_zone"] = videos["zone_type"].fillna("").astype(str) if "zone_type" in videos.columns else ""
if "bvid" in videos.columns:
    videos = videos.drop_duplicates(["bvid", "_plot_zone"])
metric_long = videos.melt(
    id_vars=["tname", "_plot_zone"],
    value_vars=[m for m in scalar_metrics if m in videos.columns],
    var_name="metric", value_name="value",
)
metrics_present = sorted(set(zip(metric_long["metric"], metric_long["_plot_zone"])))
if metrics_present:
    cols_n = min(3, len(metrics_present))
    rows_n = (len(metrics_present) + cols_n - 1) // cols_n
    fig, axes = plt.subplots(rows_n, cols_n, figsize=(5 * cols_n, 4 * rows_n), squeeze=False)
    for i, (metric, zone) in enumerate(metrics_present):
        ax = axes[i // cols_n][i % cols_n]
        sub = metric_long[(metric_long["metric"] == metric) & (metric_long["_plot_zone"] == zone)]
        sns.boxplot(data=sub, x="tname", y="value", ax=ax)
        sns.stripplot(data=sub, x="tname", y="value", color="0.3", size=3, ax=ax)
        title = metric + (" / " + zone if zone else "")
        if (metric, zone) in kw_labels:
            title += "\\nKW p=" + kw_labels[(metric, zone)]
        ax.set_title(title)
        ax.set_xlabel(axis_label)
        ax.tick_params(axis="x", rotation=45)
    for j in range(len(metrics_present), rows_n * cols_n):
        axes[j // cols_n][j % cols_n].axis("off")
    fig.tight_layout()
    fig.savefig("corpus_boxplots.png", dpi=300)
    fig.savefig("corpus_boxplots.pdf")

# ---- 3. 分布图直接消费同包汇总表，保持各指标实际分母 ----
summary_path = os.path.join(os.path.dirname(os.path.abspath(csv_path)), "corpus_summary.csv")
if os.path.exists(summary_path):
    summary = pd.read_csv(summary_path, encoding="utf-8-sig")
    summary["plot_group"] = summary["tname"].astype(str)
    for dimension in ["time_period", "zone_type"]:
        if dimension in summary.columns:
            summary["plot_group"] += " / " + summary[dimension].fillna("").astype(str)
    dist_cols = [c for c in summary.columns if c.endswith("_mean") and c[:-5] not in scalar_metrics and not c.startswith("hard_")]
    if dist_cols:
        agg = summary.melt(id_vars="plot_group", value_vars=dist_cols, var_name="category", value_name="value")
        agg["category"] = agg["category"].str.removesuffix("_mean")
        agg["group"] = agg["category"].str.split("_").str[0]
        agg.loc[agg["category"].isin(["positive", "neutral", "negative"]), "group"] = "emotion"
        agg.loc[agg["category"].isin(["assertion", "question", "exclamation", "directive", "fragment"]), "group"] = "sentence_function"
        agg.loc[agg["category"].isin(["check_in", "identity_claim", "mocking", "info_request", "expression", "other"]), "group"] = "interaction_type"
        agg = agg.dropna(subset=["value"])
        groups = sorted(agg["group"].unique())
        if groups:
            fig2, axes2 = plt.subplots(1, len(groups), figsize=(6 * len(groups), 5), squeeze=False)
            for gi, group in enumerate(groups):
                ax = axes2[0][gi]
                pivot = agg[agg["group"] == group].pivot(index="plot_group", columns="category", values="value")
                pivot.plot(kind="bar", stacked=True, ax=ax)
                ax.set_title(group)
                ax.set_ylabel("有效标注或词项加权占比")
                ax.tick_params(axis="x", rotation=45)
            fig2.tight_layout()
            fig2.savefig("corpus_distributions.png", dpi=300)
            fig2.savefig("corpus_distributions.pdf")
else:
    print("缺少 corpus_summary.csv，跳过分布图，避免按错误分母重新聚合")

# ---- 4. 冷热区配对比较箱线图（叠加 Wilcoxon 符号秩 p 值；观测表无双区数据时跳过） ----
if "zone_type" in videos.columns:
    paired = videos[videos["zone_type"].astype(str).isin(["hot_zone", "cold_zone"])]
    zone_metrics = [m for m in scalar_metrics if m in paired.columns]
    if paired["zone_type"].astype(str).nunique() >= 2 and zone_metrics:
        zone_long = paired.melt(
            id_vars="zone_type", value_vars=zone_metrics,
            var_name="metric", value_name="value",
        )
        zone_order = ["hot_zone", "cold_zone"]
        cols_n = min(3, len(zone_metrics))
        rows_n = (len(zone_metrics) + cols_n - 1) // cols_n
        fig3, axes3 = plt.subplots(rows_n, cols_n, figsize=(5 * cols_n, 4 * rows_n), squeeze=False)
        for i, metric in enumerate(zone_metrics):
            ax = axes3[i // cols_n][i % cols_n]
            sub = zone_long[zone_long["metric"] == metric]
            sns.boxplot(data=sub, x="zone_type", y="value", order=zone_order, ax=ax)
            sns.stripplot(data=sub, x="zone_type", y="value", order=zone_order, color="0.3", size=3, ax=ax)
            title = metric
            if metric in wilcoxon_labels:
                title += "\\nWilcoxon p=" + wilcoxon_labels[metric]
            ax.set_title(title)
            ax.set_xlabel("冷热区 (zone_type)")
            ax.tick_params(axis="x", rotation=45)
        for j in range(len(zone_metrics), rows_n * cols_n):
            axes3[j // cols_n][j % cols_n].axis("off")
        fig3.tight_layout()
        fig3.savefig("corpus_zone_paired.png", dpi=300)
        fig3.savefig("corpus_zone_paired.pdf")

print("完成：corpus_boxplots.* / corpus_distributions.* / corpus_zone_paired.*（如适用）")
'''


class CorpusVisualizer:

    @staticmethod
    def _escape_r_string(value: str) -> str:
        """R 字符串字面量转义：防分区名含反斜杠/双引号时破坏生成脚本语法"""
        return value.replace("\\", "\\\\").replace('"', '\\"')

    def render_r_script(
        self,
        csv_filename: str = VIDEOS_CSV_FILENAME,
        partitions: Optional[List[str]] = None,
        stats_filename: str = STATS_TESTS_FILENAME,
    ) -> str:
        scalars = ", ".join(f'"{name}"' for name in SCALAR_FIELDS)
        partition_literal = ", ".join(f'"{self._escape_r_string(name)}"' for name in (partitions or []))
        return R_SCRIPT_TEMPLATE.format(
            scalars=scalars,
            partitions=partition_literal,
            csv_filename=csv_filename,
            stats_filename=stats_filename,
        )

    def write_r_script(self, out_dir: str, csv_filename: str = VIDEOS_CSV_FILENAME) -> str:
        os.makedirs(out_dir, exist_ok=True)
        filepath = os.path.join(out_dir, R_SCRIPT_FILENAME)
        partitions = self._read_partitions(out_dir, csv_filename)
        with open(filepath, 'w', encoding='utf-8', newline='\n') as f:
            f.write(self.render_r_script(csv_filename, partitions))
        logger.info(f"R 可视化脚本已保存: {filepath}（注入分区 {len(partitions)} 个）")
        return filepath

    def render_python_script(
        self,
        csv_filename: str = VIDEOS_CSV_FILENAME,
        partitions: Optional[List[str]] = None,
        stats_filename: str = STATS_TESTS_FILENAME,
    ) -> str:
        scalars = ", ".join(f'"{name}"' for name in SCALAR_FIELDS)
        partition_literal = ", ".join(f'"{name}"' for name in (partitions or []))
        return PYTHON_SCRIPT_TEMPLATE.format(
            scalars=scalars,
            partitions=partition_literal,
            csv_filename=csv_filename,
            stats_filename=stats_filename,
        )

    def write_python_script(self, out_dir: str, csv_filename: str = VIDEOS_CSV_FILENAME) -> str:
        os.makedirs(out_dir, exist_ok=True)
        filepath = os.path.join(out_dir, PYTHON_SCRIPT_FILENAME)
        partitions = self._read_partitions(out_dir, csv_filename)
        with open(filepath, 'w', encoding='utf-8', newline='\n') as f:
            f.write(self.render_python_script(csv_filename, partitions))
        logger.info(f"Python 可视化脚本已保存: {filepath}（注入分区 {len(partitions)} 个）")
        return filepath

    def write_script(self, out_dir: str, csv_filename: str = VIDEOS_CSV_FILENAME) -> str:
        """按 VISUALIZATION_BACKEND 配置分发后端（r / python）"""
        if get_settings().VISUALIZATION_BACKEND == "r":
            return self.write_r_script(out_dir, csv_filename)
        return self.write_python_script(out_dir, csv_filename)

    @staticmethod
    def _read_partitions(out_dir: str, csv_filename: str) -> List[str]:
        """自观测表提取分区清单注入模板（固定 x 轴因子水平）；表缺失或不可读返回空清单"""
        path = os.path.join(out_dir, csv_filename)
        if not os.path.exists(path):
            return []
        try:
            df = pd.read_csv(path, encoding='utf-8-sig', usecols=["tname"])
            return sorted(t for t in df["tname"].dropna().astype(str).unique() if t)
        except Exception as e:
            logger.warning(f"读取分区清单失败，R 脚本不注入分区水平: {e}")
            return []
