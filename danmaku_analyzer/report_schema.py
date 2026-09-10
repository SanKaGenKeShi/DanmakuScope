"""报告表名与指标列的单一数据源：产出文件名、聚合表规格、列组与统计检验列。

所有报告/语料库产出的契约文件名与列组常量集中于此，reporter/corpus_builder/
exporter/report_archive/statistical_validator/corpus_visualizer 均引用本模块，
消除同一文件名在多处各写一份导致的漂移。本模块为纯常量叶子模块，不依赖包内其他模块。
"""

from typing import Dict, List

# ========== 单视频聚合表（英文契约名，供程序回读） ==========
TABLE_LEXICAL = "table_lexical_by_partition.csv"
TABLE_ORTHOGRAPHY = "table_orthography.csv"
TABLE_SENTENCE_FUNCTION = "table_sentence_function.csv"
TABLE_EMOTION = "table_emotion.csv"
TABLE_INTERACTION_TYPE = "table_interaction_type.csv"
TABLE_CONSENSUS_STATS = "table_consensus_stats.csv"

# ========== 单视频其他核心产出 ==========
METADATA_FILENAME = "metadata.json"
HEATMAP_FILENAME = "heatmap_data.json"
KAPPA_READY_FILENAME = "kappa_ready.csv"
RAW_DANMAKU_FILENAME = "danmaku_raw.csv"

# ========== 语料库级产出 ==========
CORPUS_SUMMARY_FILENAME = "corpus_summary.csv"
VIDEOS_CSV_FILENAME = "corpus_videos.csv"
STATS_TESTS_FILENAME = "statistical_tests.csv"
MERGED_RAW_FILENAME = "danmaku_corpus.csv"
CORPUS_METADATA_FILENAME = "corpus_metadata.json"

# ZIP 内需回读的表：文件名 → 除 danmaku_count 外的标量列（其余列视为分布占比）
TABLE_SPECS: Dict[str, List[str]] = {
    TABLE_LEXICAL: ["avg_word_length", "content_word_density", "punctuation_emoji_rate"],
    TABLE_CONSENSUS_STATS: ["high_consensus_rate", "medium_consensus_rate", "low_consensus_rate", "avg_weight_multiplier"],
    TABLE_EMOTION: ["cooperative_principle_violation_rate"],
    TABLE_SENTENCE_FUNCTION: [],
    TABLE_INTERACTION_TYPE: [],
    TABLE_ORTHOGRAPHY: [],
}

# 单视频报告表 → 展示名（exporter 消费；语料库快照内 corpus_summary.csv 优先于单视频表）
SINGLE_VIDEO_TABLES: Dict[str, str] = {
    TABLE_LEXICAL: "词类统计",
    TABLE_ORTHOGRAPHY: "正字法统计",
    TABLE_SENTENCE_FUNCTION: "句类分布",
    TABLE_EMOTION: "情感分布",
    TABLE_INTERACTION_TYPE: "互动类型分布",
    TABLE_CONSENSUS_STATS: "共识统计",
}

# 单视频 ZIP 必备核心文件（ReportArchive 完整性校验）
CORE_FILENAMES = frozenset({
    METADATA_FILENAME,
    TABLE_LEXICAL,
    TABLE_ORTHOGRAPHY,
    TABLE_SENTENCE_FUNCTION,
    TABLE_EMOTION,
    TABLE_INTERACTION_TYPE,
    TABLE_CONSENSUS_STATS,
    HEATMAP_FILENAME,
    RAW_DANMAKU_FILENAME,
})

# 聚合表中不视为分布类别的列（分母、记录数、失败数与共识 CI）
NON_DIST_COLUMNS = {
    "tname", "zone_type", "danmaku_count", "_source_table",
    "total_word_count", "total_char_count", "label_weight_sum", "valid_label_count",
    "llm_record_count", "failed_record_count", "degraded_record_count",
    "high_consensus_ci_lower", "high_consensus_ci_upper", "high_consensus_ci_status",
}

# 视频级标量指标（组级取均值/标准差，diff 与检验按此清单消费）
SCALAR_FIELDS = [
    "avg_word_length", "content_word_density", "punctuation_emoji_rate",
    "high_consensus_rate", "medium_consensus_rate", "low_consensus_rate",
    "avg_weight_multiplier", "cooperative_principle_violation_rate",
]

# diff 数值差异判定容差（浮点回读噪声）
DIFF_NUMERIC_TOLERANCE = 1e-9
# 参与变更比对的字段：身份/版本/规模 + 全部标量指标
DIFF_FIELDS = ("tname", "prompt_version", "danmaku_count") + tuple(SCALAR_FIELDS)

# statistical_tests.csv 列顺序契约
STATISTICAL_TESTS_COLUMNS = [
    "metric", "test_type", "group1", "group2", "n1", "n2",
    "statistic", "p_value", "effect_size", "effect_magnitude", "note",
]

UNCORRECTED_P_NOTE = "未校正 p 值（未实施多重比较校正）"
