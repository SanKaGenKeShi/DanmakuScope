"""
报告生成器模块 - 导出交叉表 + 热力图数据 + kappa_ready.csv
"""

import os
import json
import csv
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


class Reporter:
    
    def __init__(self, output_dir: Optional[str] = None):
        self.settings = get_settings()
        self.output_dir = self.settings.resolve_data_path(output_dir or self.settings.OUTPUT_DIR)
        os.makedirs(self.output_dir, exist_ok=True)
    
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
    
    def generate_methodology(self, metadata: Dict, sampling: Optional[Dict] = None) -> str:
        """方法论描述（methodology.md）随报告入包，渲染委派 MethodologyGenerator"""
        from .methodology import MethodologyGenerator

        return MethodologyGenerator(self.output_dir).write(metadata, sampling)

    def export_formatted(self, source_path: str, fmt: str) -> str:
        """多格式导出（latex/apa），格式化逻辑委派 Exporter"""
        from .exporter import Exporter

        return Exporter().export(source_path, fmt, self.output_dir)

    def _write_dataframe(self, rows: List[Dict], filename: str, description: str) -> str:
        df = pd.DataFrame(rows)
        filepath = os.path.join(self.output_dir, filename)
        df.to_csv(filepath, index=False, encoding='utf-8-sig')
        logger.info(f"{description}已保存: {filepath}")
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

