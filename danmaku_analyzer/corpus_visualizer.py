"""
语料库可视化模块 - 生成 R/ggplot2 脚本模板（零新增 Python 依赖）
消费 corpus_videos.csv 视频级观测表，输出可直接运行的 corpus_plots.R：
分区间箱线图、Kruskal-Wallis + 事后检验表、分布列堆叠条形图
"""

import os

from .corpus_builder import SCALAR_FIELDS
from .utils.logger import get_logger

logger = get_logger(__name__)

R_SCRIPT_FILENAME = "corpus_plots.R"
VIDEOS_CSV_FILENAME = "corpus_videos.csv"

# 模板占位符：{scalars} 为 R 字符向量字面量，{csv_filename} 为观测表文件名
R_SCRIPT_TEMPLATE = '''\
# DanmakuScope 语料库可视化脚本（自动生成模板，可自由修改）
# 统计单位：视频级观测（corpus_videos.csv 每行一个视频）
# 用法：Rscript corpus_plots.R [corpus_videos.csv 路径]

args <- commandArgs(trailingOnly = TRUE)
csv_path <- if (length(args) >= 1) args[1] else "{csv_filename}"

for (pkg in c("ggplot2", "dplyr", "tidyr")) {{
  if (!requireNamespace(pkg, quietly = TRUE)) install.packages(pkg)
}}
library(ggplot2)
library(dplyr)
library(tidyr)

videos <- read.csv(csv_path, fileEncoding = "UTF-8-BOM", stringsAsFactors = FALSE)

identity_cols <- c("bvid", "tname", "pubdate", "prompt_version", "zone_type", "danmaku_count")
scalar_metrics <- c({scalars})

videos <- videos %>% filter(!is.na(tname), tname != "")

# ---- 1. 核心指标分区间箱线图（每视频一个点） ----
metric_long <- videos %>%
  select(tname, all_of(scalar_metrics)) %>%
  pivot_longer(-tname, names_to = "metric", values_to = "value")

p_box <- ggplot(metric_long, aes(x = tname, y = value, fill = tname)) +
  geom_boxplot(outlier.shape = NA, alpha = 0.6) +
  geom_jitter(width = 0.15, size = 1.5, alpha = 0.7) +
  facet_wrap(~ metric, scales = "free_y") +
  theme_bw() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1), legend.position = "none") +
  labs(x = "分区 (tname)", y = "指标值")
ggsave("corpus_boxplots.pdf", p_box, width = 10, height = 8)
ggsave("corpus_boxplots.png", p_box, width = 10, height = 8, dpi = 300)

# ---- 2. Kruskal-Wallis + 事后两两比较（BH-FDR 校正） ----
test_rows <- list()
for (m in scalar_metrics) {{
  vals <- videos[[m]]
  groups <- videos$tname
  keep <- !is.na(vals) & groups != ""
  if (length(unique(groups[keep])) < 2) next
  kw <- kruskal.test(vals[keep] ~ groups[keep])
  pw <- pairwise.wilcox.test(vals[keep], groups[keep], p.adjust.method = "BH", exact = FALSE)
  pmat <- as.data.frame(as.table(pw$p.value))
  pmat <- pmat[!is.na(pmat$Freq), ]
  test_rows[[m]] <- data.frame(
    metric = m, comparison = paste(pmat$Var2, pmat$Var1, sep = " vs "),
    kw_H = round(kw$statistic, 4), kw_p = kw$p.value,
    p_adjusted = pmat$Freq, stringsAsFactors = FALSE
  )
}}
if (length(test_rows) > 0) {{
  stats_df <- bind_rows(test_rows)
  write.csv(stats_df, "corpus_stats_R.csv", row.names = FALSE, fileEncoding = "UTF-8")
  print(stats_df)
}}

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

cat("完成：corpus_boxplots.* / corpus_distributions.* / corpus_stats_R.csv\\n")
'''


class CorpusVisualizer:

    def render_r_script(self, csv_filename: str = VIDEOS_CSV_FILENAME) -> str:
        scalars = ", ".join(f'"{name}"' for name in SCALAR_FIELDS)
        return R_SCRIPT_TEMPLATE.format(scalars=scalars, csv_filename=csv_filename)

    def write_r_script(self, out_dir: str, csv_filename: str = VIDEOS_CSV_FILENAME) -> str:
        os.makedirs(out_dir, exist_ok=True)
        filepath = os.path.join(out_dir, R_SCRIPT_FILENAME)
        with open(filepath, 'w', encoding='utf-8', newline='\n') as f:
            f.write(self.render_r_script(csv_filename))
        logger.info(f"R 可视化脚本已保存: {filepath}")
        return filepath
