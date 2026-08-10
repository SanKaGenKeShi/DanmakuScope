"""
语料库可视化模块 - 生成 R/ggplot2 脚本模板（零新增 Python 依赖）
消费 corpus_videos.csv 视频级观测表，输出可直接 Rscript 出图的 corpus_plots.R：
分区间箱线图（叠加 statistical_tests.csv 预计算的 Kruskal-Wallis p 值）、分布列堆叠条形图。
统计检验由 Python 侧 statistical_validator.corpus_compare 预计算，R 脚本不重复计算，
亦不实施任何多重比较校正（p 值均为未校正）。
"""

import os
from typing import List, Optional

import pandas as pd

from .corpus_builder import SCALAR_FIELDS
from .utils.logger import get_logger

logger = get_logger(__name__)

R_SCRIPT_FILENAME = "corpus_plots.R"
VIDEOS_CSV_FILENAME = "corpus_videos.csv"
STATS_CSV_FILENAME = "statistical_tests.csv"

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

identity_cols <- c("bvid", "tname", "pubdate", "prompt_version", "zone_type", "danmaku_count")
scalar_metrics <- c({scalars})
partitions <- c({partitions})

videos <- videos %>% filter(!is.na(tname), tname != "")
if (length(partitions) > 0) {{
  videos$tname <- factor(as.character(videos$tname), levels = partitions)
  videos <- videos %>% filter(!is.na(tname))
}}

# ---- 1. 读取 Python 侧预计算推断检验结果（缺失时降级为纯箱线图） ----
kw_labels <- NULL
if (file.exists(stats_path)) {{
  stats_df <- read.csv(stats_path, fileEncoding = "UTF-8-BOM", stringsAsFactors = FALSE)
  print(stats_df)
  kw <- stats_df[stats_df$test_type == "Kruskal-Wallis" & !is.na(stats_df$p_value), ]
  if (nrow(kw) > 0) {{
    kw_labels <- setNames(sprintf("%s\\nKW p=%.4g", kw$metric, kw$p_value), kw$metric)
  }}
}} else {{
  message("未找到 ", stats_path, "，箱线图不叠加检验结果")
}}

# ---- 2. 核心指标分区间箱线图（每视频一个点，分面标签叠加 KW p 值） ----
metric_long <- videos %>%
  select(tname, all_of(scalar_metrics)) %>%
  pivot_longer(-tname, names_to = "metric", values_to = "value")

metric_long$facet_label <- metric_long$metric
if (!is.null(kw_labels)) {{
  idx <- match(metric_long$metric, names(kw_labels))
  hit <- !is.na(idx)
  metric_long$facet_label[hit] <- unname(kw_labels[idx[hit]])
}}

p_box <- ggplot(metric_long, aes(x = tname, y = value, fill = tname)) +
  geom_boxplot(outlier.shape = NA, alpha = 0.6) +
  geom_jitter(width = 0.15, size = 1.5, alpha = 0.7) +
  facet_wrap(~ facet_label, scales = "free_y") +
  theme_bw() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1), legend.position = "none") +
  labs(x = "分区 (tname)", y = "指标值")
ggsave("corpus_boxplots.pdf", p_box, width = 10, height = 8)
ggsave("corpus_boxplots.png", p_box, width = 10, height = 8, dpi = 300)

# ---- 3. 分布列堆叠条形图（按弹幕数加权均值，前缀分组） ----
dist_cols <- setdiff(names(videos), c(identity_cols, scalar_metrics))
if (length(dist_cols) > 0) {{
  dist_long <- videos %>%
    select(tname, danmaku_count, all_of(dist_cols)) %>%
    pivot_longer(-c(tname, danmaku_count), names_to = "category", values_to = "value") %>%
    mutate(group = ifelse(grepl("_", category), sub("_.*$", "", category), "other")) %>%
    group_by(group, tname, category) %>%
    summarise(value = weighted.mean(value, danmaku_count, na.rm = TRUE), .groups = "drop")

  p_dist <- ggplot(dist_long, aes(x = tname, y = value, fill = category)) +
    geom_col() +
    facet_wrap(~ group, scales = "free") +
    theme_bw() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
    labs(x = "分区 (tname)", y = "加权占比", fill = "类别")
  ggsave("corpus_distributions.pdf", p_dist, width = 12, height = 8)
  ggsave("corpus_distributions.png", p_dist, width = 12, height = 8, dpi = 300)
}}

cat("完成：corpus_boxplots.* / corpus_distributions.*\\n")
'''


class CorpusVisualizer:

    def render_r_script(
        self,
        csv_filename: str = VIDEOS_CSV_FILENAME,
        partitions: Optional[List[str]] = None,
        stats_filename: str = STATS_CSV_FILENAME,
    ) -> str:
        scalars = ", ".join(f'"{name}"' for name in SCALAR_FIELDS)
        partition_literal = ", ".join(f'"{name}"' for name in (partitions or []))
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
