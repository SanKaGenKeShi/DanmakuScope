"""
报告生成器模块 - 导出交叉表 + 热力图数据 + kappa_ready.csv + 原始弹幕
聚合表同步写出中文版（文件名与列头翻译），英文版保持契约名供程序回读
"""

import os
import json
import csv
import shutil
from typing import List, Dict, Any, Optional
from datetime import datetime

import pandas as pd

from .config import get_settings
from .llm_config import get_llm_settings
from .aggregator import AggregatedData
from .utils.logger import get_logger

logger = get_logger(__name__)

# kappa_ready 中 LLM 输出字段解包映射：(CSV 列名, 维度键, 字段名, 默认值)
_KAPPA_LLM_FIELDS = [
    ("emotion_label", "emotion", "label", ""),
    ("emotion_confidence", "emotion", "confidence", 0),
    ("cooperative_principle_violated", "cooperative_principle", "violated", False),
    ("cooperative_principle_maxim", "cooperative_principle", "maxim", ""),
    ("interaction_type_label", "interaction_type", "label", ""),
    ("interaction_type_confidence", "interaction_type", "confidence", 0),
    ("sentence_function_label", "sentence_function", "label", ""),
    ("sentence_function_confidence", "sentence_function", "confidence", 0),
    ("orthography_status", "orthography", "status", ""),
    ("orthography_confidence", "orthography", "confidence", 0),
]

# 中文版产出映射：英文契约文件名 → 中文文件名（未登记的表不写中文版）
_ZH_TABLE_FILENAMES = {
    "table_lexical_by_partition.csv": "词类统计表.csv",
    "table_orthography.csv": "正字法统计表.csv",
    "table_sentence_function.csv": "句类分布表.csv",
    "table_emotion.csv": "情感分布表.csv",
    "table_interaction_type.csv": "互动类型分布表.csv",
    "table_consensus_stats.csv": "共识统计表.csv",
    "danmaku_raw.csv": "原始弹幕.csv",
}

_ZH_COLUMN_LABELS = {
    "tname": "分区",
    "zone_type": "冷热区",
    "danmaku_count": "弹幕数",
    "avg_word_length": "平均词长",
    "content_word_density": "实词密度",
    "punctuation_emoji_rate": "标点与表情率",
    "cooperative_principle_violation_rate": "合作原则违背率",
    "high_consensus_rate": "高共识率",
    "medium_consensus_rate": "中共识率",
    "low_consensus_rate": "低共识率",
    "avg_weight_multiplier": "平均权重系数",
    "high_consensus_ci_lower": "高共识率置信下限",
    "high_consensus_ci_upper": "高共识率置信上限",
    "high_consensus_ci_status": "置信区间状态",
    "uid_hash": "用户哈希",
    "content": "弹幕内容",
    "time_sec": "时间点(秒)",
    "identity_type": "身份类型",
}

_ZH_LABEL_VALUES = {
    "positive": "正面", "neutral": "中性", "negative": "负面",
    "check_in": "打卡报到", "identity_claim": "身份声明", "mocking": "调侃嘲讽",
    "info_request": "信息询问", "expression": "情感表达", "other": "其他",
    "assertion": "陈述", "question": "疑问", "exclamation": "感叹",
    "directive": "祈使", "fragment": "碎片",
    "standard": "规范书写", "community_variant": "社区变体", "non_standard_typo": "非规范错字",
    "hot_zone": "热区", "cold_zone": "冷区",
}

_ZH_HARD_METRICS = {
    "uppercase_abbr_per_1000": "每千字大写缩写数",
    "number_symbol_per_1000": "每千字数字表意串数",
    "emoticon_per_1000": "每千字颜文字数",
}

_ZH_POS_FLAGS = {
    "n": "名词", "v": "动词", "a": "形容词", "d": "副词", "r": "代词",
    "m": "数词", "q": "量词", "p": "介词", "c": "连词", "u": "助词",
    "f": "方位词", "s": "处所词", "t": "时间词", "b": "区别词", "z": "状态词",
    "i": "成语", "j": "简称", "l": "习用语", "e": "叹词", "y": "语气词",
    "o": "拟声词", "h": "前接成分", "k": "后接成分", "x": "非语素字",
    "w": "标点", "eng": "外语", "un": "未知词",
}


class Reporter:
    
    def __init__(self, output_dir: Optional[str] = None):
        self.settings = get_settings()
        self.output_dir = self.settings.resolve_data_path(output_dir or self.settings.OUTPUT_DIR)
        os.makedirs(self.output_dir, exist_ok=True)
        self.zh_reports: Dict[str, str] = {}
    
    def generate_reports(
        self, 
        aggregated_data: List[AggregatedData],
        kappa_records: Optional[List[Dict]] = None,
        metadata: Optional[Dict] = None
    ) -> Dict[str, str]:
        logger.info(f"开始生成报告，共 {len(aggregated_data)} 个聚合组")
        
        reports = {}
        
        reports["lexical_by_partition"] = self._generate_lexical_table(aggregated_data)
        reports["orthography"] = self._generate_orthography_table(aggregated_data)
        reports["sentence_function"] = self._generate_sentence_function_table(aggregated_data)
        reports["emotion"] = self._generate_emotion_table(aggregated_data)
        reports["interaction_type"] = self._generate_interaction_type_table(aggregated_data)
        reports["consensus_stats"] = self._generate_consensus_table(aggregated_data)
        reports["heatmap_data"] = self._generate_heatmap_data(aggregated_data)
        
        if kappa_records:
            reports["kappa_ready"] = self._generate_kappa_ready(kappa_records)
        
        reports["metadata"] = self._generate_metadata(aggregated_data, metadata)
        reports.update(self.zh_reports)
        
        logger.info(f"报告生成完成，共 {len(reports)} 个文件")
        return reports
    
    async def _run_llm_report(self, generate_call, filename: str, label: str) -> Optional[str]:
        """LLM 报告公共流程：启用检查 → 生成 → 落盘；失败/未启用返回 None"""
        if not self.settings.ENABLE_LLM_ANALYSIS_REPORT:
            logger.info("LLM分析报告生成未启用")
            return None
        
        try:
            report_content = await generate_call()
            if not report_content:
                logger.error(f"{label}生成失败: 未获得有效报告内容")
                return None
            
            filepath = os.path.join(self.output_dir, filename)
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(report_content)
            
            logger.info(f"{label}已保存: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"{label}生成失败: {e}")
            return None
    
    async def generate_llm_analysis_report(
        self,
        aggregated_data: List[AggregatedData],
        metadata: Optional[Dict] = None
    ) -> Optional[str]:
        from .report_generator import AnalysisReportGenerator
        
        report_gen = AnalysisReportGenerator()
        report_metadata = {
            "bvid": metadata.get("bvid", "") if metadata else "",
            "title": metadata.get("title", "") if metadata else "",
            "tname": metadata.get("tname", "") if metadata else "",
            "tags": metadata.get("tags", []) if metadata else [],
        }
        aggregated_dicts = [data.to_flat_dict() for data in aggregated_data]
        
        return await self._run_llm_report(
            lambda: report_gen.generate(aggregated_data=aggregated_dicts, metadata=report_metadata),
            "sociolinguistic_analysis_report.md", "LLM分析报告"
        )
    
    async def generate_corpus_analysis_report(
        self,
        summary_csv_path: str,
        videos_csv_path: str,
        corpus_metadata: Dict,
    ) -> Optional[str]:
        """语料库级 LLM 比较分析报告（与单视频报告共用同一生成入口）"""
        from .report_generator import AnalysisReportGenerator
    
        report_gen = AnalysisReportGenerator()
        return await self._run_llm_report(
            lambda: report_gen.generate_corpus_report(summary_csv_path, videos_csv_path, corpus_metadata),
            "corpus_analysis_report.md", "语料库LLM分析报告"
        )
    
    def generate_corpus_html_report(self, build_result, comparison=None) -> str:
        """语料库级 HTML 可视化报告（corpus_report.html），渲染委派 HtmlReportGenerator"""
        from .html_report import HtmlReportGenerator
    
        summary_df = pd.read_csv(build_result.csv_path, encoding='utf-8-sig')
        videos_df = pd.read_csv(build_result.videos_csv_path, encoding='utf-8-sig')
        tests_df = comparison.to_dataframe() if comparison is not None and comparison.rows else None
        metadata = {
            "tnames": list(build_result.tnames),
            "video_count": int(videos_df["bvid"].nunique()) if "bvid" in videos_df.columns else 0,
            "total_danmaku": int(videos_df["danmaku_count"].sum()) if "danmaku_count" in videos_df.columns else 0,
        }
        return HtmlReportGenerator(self.output_dir).write_corpus(summary_df, tests_df, metadata)
    
    def generate_methodology(self, metadata: Dict, sampling: Optional[Dict] = None) -> str:
        """方法论描述（methodology.md）随报告入包，渲染委派 MethodologyGenerator"""
        from .methodology import MethodologyGenerator

        return MethodologyGenerator(self.output_dir).write(metadata, sampling)

    def generate_html_report(
        self,
        aggregated_data: List[AggregatedData],
        metadata: Optional[Dict] = None,
        llm_report_md: Optional[str] = None,
    ) -> str:
        """HTML 可视化报告（离线单文件，内联 CSS/SVG），渲染委派 HtmlReportGenerator；同名中文副本仅供浏览"""
        from .html_report import HTML_REPORT_FILENAME, HtmlReportGenerator

        enriched = dict(metadata or {})
        enriched.setdefault("prompt_version", get_llm_settings().PROMPT_VERSION)
        enriched.setdefault("generated_at", datetime.now().isoformat(timespec='seconds'))
        path = HtmlReportGenerator(self.output_dir).write(aggregated_data, enriched, llm_report_md)
        try:
            zh_path = os.path.join(self.output_dir, "分析报告.html")
            shutil.copyfile(path, zh_path)
            self.zh_reports[f"{os.path.splitext(HTML_REPORT_FILENAME)[0]}_zh"] = zh_path
        except OSError as e:
            logger.warning(f"HTML 报告中文副本写出失败（主报告不受影响）: {e}")
        return path

    def export_formatted(self, source_path: str, fmt: str) -> str:
        """多格式导出（latex/apa），格式化逻辑委派 Exporter"""
        from .exporter import Exporter

        return Exporter().export(source_path, fmt, self.output_dir)

    def _write_dataframe(self, rows: List[Dict], filename: str, description: str) -> str:
        df = pd.DataFrame(rows)
        filepath = os.path.join(self.output_dir, filename)
        df.to_csv(filepath, index=False, encoding='utf-8-sig')
        logger.info(f"{description}已保存: {filepath}")
        self._write_zh_twin(df, filename, description)
        return filepath

    def _write_zh_twin(self, df: pd.DataFrame, filename: str, description: str) -> None:
        """中文版双写：英文契约版保程序回读，中文版翻译列头供人工阅读；失败仅告警不影响主产出"""
        zh_filename = _ZH_TABLE_FILENAMES.get(filename)
        if not zh_filename:
            return
        zh_path = os.path.join(self.output_dir, zh_filename)
        try:
            zh_df = df.rename(columns=_translate_column)
            zh_df.to_csv(zh_path, index=False, encoding='utf-8-sig')
        except OSError as e:
            logger.warning(f"{description}中文版写出失败（英文版不受影响）: {e}")
            return
        self.zh_reports[f"{os.path.splitext(filename)[0]}_zh"] = zh_path

    def generate_raw_danmaku(self, danmaku_list: list) -> str:
        """全量原始弹幕（未清洗）入包，支撑论文语料附录与复核；XML 兜底样本的截断性由 metadata 的 danmaku_source 标注"""
        rows = [
            {
                "uid_hash": item.uid_hash,
                "content": item.content,
                "time_sec": item.time_sec,
                "identity_type": item.identity_type,
            }
            for item in danmaku_list
        ]
        df = pd.DataFrame(rows, columns=["uid_hash", "content", "time_sec", "identity_type"])
        filepath = os.path.join(self.output_dir, "danmaku_raw.csv")
        df.to_csv(filepath, index=False, encoding='utf-8-sig')
        logger.info(f"原始弹幕表已保存: {filepath}（{len(df)} 条）")
        self._write_zh_twin(df, "danmaku_raw.csv", "原始弹幕表")
        return filepath
    
    def _generate_lexical_table(self, data: List[AggregatedData]) -> str:
        rows = []
        for item in data:
            row = {
                "tname": item.tname,
                "zone_type": item.zone_type,
                "danmaku_count": item.danmaku_count,
                "avg_word_length": item.avg_word_length,
                "content_word_density": item.content_word_density,
                "punctuation_emoji_rate": item.punctuation_emoji_rate,
            }
            for pos, ratio in item.pos_distribution.items():
                row[f"pos_{pos}"] = ratio
            for syllable_type, ratio in item.syllable_distribution.items():
                row[f"syllable_{syllable_type}"] = ratio
            
            rows.append(row)
        
        return self._write_dataframe(rows, "table_lexical_by_partition.csv", "词类统计表")
    
    def _generate_orthography_table(self, data: List[AggregatedData]) -> str:
        rows = []
        for item in data:
            row = {
                "tname": item.tname,
                "zone_type": item.zone_type,
                "danmaku_count": item.danmaku_count,
            }
            for metric, value in item.orthography_hard_metrics.items():
                row[f"hard_{metric}"] = value
            for status, ratio in item.orthography_status_distribution.items():
                row[f"soft_{status}"] = ratio
            
            rows.append(row)
        
        return self._write_dataframe(rows, "table_orthography.csv", "正字法统计表")
    
    def _generate_sentence_function_table(self, data: List[AggregatedData]) -> str:
        rows = []
        for item in data:
            row = {
                "tname": item.tname,
                "zone_type": item.zone_type,
                "danmaku_count": item.danmaku_count,
            }
            for sf, ratio in item.sentence_function_distribution.items():
                row[sf] = ratio
            
            rows.append(row)
        
        return self._write_dataframe(rows, "table_sentence_function.csv", "句类分布表")
    
    def _generate_emotion_table(self, data: List[AggregatedData]) -> str:
        rows = []
        for item in data:
            row = {
                "tname": item.tname,
                "zone_type": item.zone_type,
                "danmaku_count": item.danmaku_count,
                "cooperative_principle_violation_rate": item.cooperative_principle_violation_rate,
            }
            for emotion, ratio in item.emotion_distribution.items():
                row[emotion] = ratio
            
            rows.append(row)
        
        return self._write_dataframe(rows, "table_emotion.csv", "情感分布表")
    
    def _generate_interaction_type_table(self, data: List[AggregatedData]) -> str:
        rows = []
        for item in data:
            row = {
                "tname": item.tname,
                "zone_type": item.zone_type,
                "danmaku_count": item.danmaku_count,
            }
            for it, ratio in item.interaction_type_distribution.items():
                row[it] = ratio
            
            rows.append(row)
        
        return self._write_dataframe(rows, "table_interaction_type.csv", "互动类型分布表")
    
    def _generate_consensus_table(self, data: List[AggregatedData]) -> str:
        rows = []
        for item in data:
            ci = item.consensus_ci or {}
            row = {
                "tname": item.tname,
                "zone_type": item.zone_type,
                "danmaku_count": item.danmaku_count,
                "high_consensus_rate": item.high_consensus_rate,
                "medium_consensus_rate": item.medium_consensus_rate,
                "low_consensus_rate": item.low_consensus_rate,
                "avg_weight_multiplier": item.avg_weight_multiplier,
                "high_consensus_ci_lower": ci.get("lower"),
                "high_consensus_ci_upper": ci.get("upper"),
                "high_consensus_ci_status": ci.get("status", "ok"),
            }
            rows.append(row)
        
        return self._write_dataframe(rows, "table_consensus_stats.csv", "共识统计表")
    
    def _generate_heatmap_data(self, data: List[AggregatedData]) -> str:
        heatmap_data = {
            "emotion_heatmap": {},
            "sentence_function_heatmap": {},
            "orthography_heatmap": {},
        }
        
        for item in data:
            key = f"{item.tname}_{item.zone_type}"
            
            heatmap_data["emotion_heatmap"][key] = item.emotion_distribution
            heatmap_data["sentence_function_heatmap"][key] = item.sentence_function_distribution
            heatmap_data["orthography_heatmap"][key] = item.orthography_status_distribution
        
        filepath = os.path.join(self.output_dir, "heatmap_data.json")
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(heatmap_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"热力图数据已保存: {filepath}")
        return filepath
    
    def _generate_kappa_ready(self, records: List[Dict]) -> str:
        if not records:
            return ""
        
        fieldnames = [
            "uid_hash", "time_segment", "raw_text", 
            "tname", "zone_type", "consensus_level", "weight_multiplier"
        ]
        
        llm_fields = [
            "emotion_label", "emotion_confidence",
            "cooperative_principle_violated", "cooperative_principle_maxim",
            "interaction_type_label", "interaction_type_confidence",
            "sentence_function_label", "sentence_function_confidence",
            "orthography_status", "orthography_confidence",
        ]
        fieldnames.extend(llm_fields)
        
        filepath = os.path.join(self.output_dir, "kappa_ready.csv")
        
        with open(filepath, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for record in records:
                row = {
                    "uid_hash": record.get("uid_hash", ""),
                    "time_segment": record.get("time_segment", ""),
                    "raw_text": record.get("raw_text", ""),
                    "tname": record.get("tname", ""),
                    "zone_type": record.get("zone_type", ""),
                    "consensus_level": record.get("consensus_level", ""),
                    "weight_multiplier": record.get("weight_multiplier", 1.0),
                }
                
                llm_output = record.get("llm_output", {})
                for csv_field, dim_key, field_name, default in _KAPPA_LLM_FIELDS:
                    row[csv_field] = llm_output.get(dim_key, {}).get(field_name, default)
                
                writer.writerow(row)
        
        logger.info(f"kappa_ready.csv 已保存: {filepath}")
        return filepath
    
    def _generate_metadata(
        self, 
        data: List[AggregatedData],
        extra_metadata: Optional[Dict] = None
    ) -> str:
        llm_cfg = get_llm_settings()
        metadata = {
            "generated_at": datetime.now().isoformat(),
            "prompt_version": llm_cfg.PROMPT_VERSION,
            "total_videos": sum(item.video_count for item in data),
            "total_danmaku": sum(item.danmaku_count for item in data),
            "total_segments": sum(item.segment_count for item in data),
            "partitions": list(set(item.tname for item in data)),
            "settings": {
                "moe": self.settings.MOE,
                "confidence_level": self.settings.CONFIDENCE_LEVEL,
                "enable_significance_testing": self.settings.ENABLE_SIGNIFICANCE_TESTING,
                "segmentation_mode": self.settings.SEGMENTATION_MODE,
                "min_segment_samples": self.settings.MIN_SEGMENT_SAMPLES,
                "enable_dual_path": llm_cfg.ENABLE_DUAL_PATH,
                "jsd_threshold_low": llm_cfg.JSD_THRESHOLD_LOW,
                "jsd_threshold_medium": llm_cfg.JSD_THRESHOLD_MEDIUM,
                "low_consensus_weight": llm_cfg.LOW_CONSENSUS_WEIGHT,
                "context_time_window": self.settings.CONTEXT_TIME_WINDOW,
                "max_context_tokens": self.settings.MAX_CONTEXT_TOKENS,
            },
        }
        
        if extra_metadata:
            metadata.update(extra_metadata)
        
        filepath = os.path.join(self.output_dir, "metadata.json")
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        logger.info(f"元数据已保存: {filepath}")
        return filepath


def _translate_column(column: str) -> str:
    """列头翻译：固定列直查，动态列按前缀展开（pos_/hard_/soft_），未命中保留原名不致断链"""
    if column in _ZH_COLUMN_LABELS:
        return _ZH_COLUMN_LABELS[column]
    if column.startswith("pos_"):
        flag = column[4:]
        # 精确匹配：前缀匹配会把未知旗标误并入已知词性（zzz → 状态词zz）
        return f"词性_{_ZH_POS_FLAGS.get(flag, flag)}"
    if column.startswith("syllable_"):
        return f"音节_{_ZH_LABEL_VALUES.get(column[9:], column[9:])}"
    if column.startswith("hard_"):
        return _ZH_HARD_METRICS.get(column[5:], column)
    if column.startswith("soft_"):
        return f"LLM判定_{_ZH_LABEL_VALUES.get(column[5:], column[5:])}"
    return _ZH_LABEL_VALUES.get(column, column)

