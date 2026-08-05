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
        logger.info("开始生成报告")
        
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
    
    async def generate_llm_analysis_report(
        self,
        aggregated_data: List[AggregatedData],
        metadata: Optional[Dict] = None
    ) -> Optional[str]:
        if not self.settings.ENABLE_LLM_ANALYSIS_REPORT:
            logger.info("LLM分析报告生成未启用")
            return None
        
        try:
            from .report_generator import AnalysisReportGenerator
            
            logger.info("开始生成LLM分析报告")
            
            report_gen = AnalysisReportGenerator()
            
            report_metadata = {
                "bvid": metadata.get("bvid", "") if metadata else "",
                "title": metadata.get("title", "") if metadata else "",
                "tname": metadata.get("tname", "") if metadata else "",
                "tags": metadata.get("tags", []) if metadata else [],
            }
            
            aggregated_dicts = []
            for data in aggregated_data:
                aggregated_dicts.append({
                    "tname": data.tname,
                    "zone_type": data.zone_type,
                    "danmaku_count": data.danmaku_count,
                    "video_count": data.video_count,
                    "segment_count": data.segment_count,
                    "emotion_distribution": data.emotion_distribution,
                    "sentence_function_distribution": data.sentence_function_distribution,
                    "interaction_type_distribution": data.interaction_type_distribution,
                    "orthography_status_distribution": data.orthography_status_distribution,
                    "high_consensus_rate": data.high_consensus_rate,
                    "medium_consensus_rate": data.medium_consensus_rate,
                    "low_consensus_rate": data.low_consensus_rate,
                    "avg_weight_multiplier": data.avg_weight_multiplier,
                    "avg_word_length": data.avg_word_length,
                    "content_word_density": data.content_word_density,
                    "punctuation_emoji_rate": data.punctuation_emoji_rate,
                    "pos_distribution": data.pos_distribution,
                    "syllable_distribution": data.syllable_distribution,
                    "orthography_hard_metrics": data.orthography_hard_metrics,
                })
            
            report_content = await report_gen.generate(
                aggregated_data=aggregated_dicts,
                metadata=report_metadata,
                prompt_version=get_llm_settings().PROMPT_VERSION
            )
            
            if not report_content:
                logger.error("LLM分析报告生成失败: 未获得有效报告内容")
                return None
            
            filepath = os.path.join(self.output_dir, "sociolinguistic_analysis_report.md")
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(report_content)
            
            logger.info(f"LLM分析报告已保存: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"LLM分析报告生成失败: {e}")
            return None
    
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
        
        df = pd.DataFrame(rows)
        filepath = os.path.join(self.output_dir, "table_lexical_by_partition.csv")
        df.to_csv(filepath, index=False, encoding='utf-8-sig')
        
        logger.info(f"词类统计表已保存: {filepath}")
        return filepath
    
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
        
        df = pd.DataFrame(rows)
        filepath = os.path.join(self.output_dir, "table_orthography.csv")
        df.to_csv(filepath, index=False, encoding='utf-8-sig')
        
        logger.info(f"正字法统计表已保存: {filepath}")
        return filepath
    
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
        
        df = pd.DataFrame(rows)
        filepath = os.path.join(self.output_dir, "table_sentence_function.csv")
        df.to_csv(filepath, index=False, encoding='utf-8-sig')
        
        logger.info(f"句类分布表已保存: {filepath}")
        return filepath
    
    def _generate_emotion_table(self, data: List[AggregatedData]) -> str:
        rows = []
        for item in data:
            row = {
                "tname": item.tname,
                "zone_type": item.zone_type,
                "danmaku_count": item.danmaku_count,
            }
            for emotion, ratio in item.emotion_distribution.items():
                row[emotion] = ratio
            
            rows.append(row)
        
        df = pd.DataFrame(rows)
        filepath = os.path.join(self.output_dir, "table_emotion.csv")
        df.to_csv(filepath, index=False, encoding='utf-8-sig')
        
        logger.info(f"情感分布表已保存: {filepath}")
        return filepath
    
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
        
        df = pd.DataFrame(rows)
        filepath = os.path.join(self.output_dir, "table_interaction_type.csv")
        df.to_csv(filepath, index=False, encoding='utf-8-sig')
        
        logger.info(f"互动类型分布表已保存: {filepath}")
        return filepath
    
    def _generate_consensus_table(self, data: List[AggregatedData]) -> str:
        rows = []
        for item in data:
            row = {
                "tname": item.tname,
                "zone_type": item.zone_type,
                "danmaku_count": item.danmaku_count,
                "high_consensus_rate": item.high_consensus_rate,
                "medium_consensus_rate": item.medium_consensus_rate,
                "low_consensus_rate": item.low_consensus_rate,
                "avg_weight_multiplier": item.avg_weight_multiplier,
            }
            rows.append(row)
        
        df = pd.DataFrame(rows)
        filepath = os.path.join(self.output_dir, "table_consensus_stats.csv")
        df.to_csv(filepath, index=False, encoding='utf-8-sig')
        
        logger.info(f"共识统计表已保存: {filepath}")
        return filepath
    
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
                row["emotion_label"] = llm_output.get("emotion", {}).get("label", "")
                row["emotion_confidence"] = llm_output.get("emotion", {}).get("confidence", 0)
                row["cooperative_principle_violated"] = llm_output.get("cooperative_principle", {}).get("violated", False)
                row["cooperative_principle_maxim"] = llm_output.get("cooperative_principle", {}).get("maxim", "")
                row["interaction_type_label"] = llm_output.get("interaction_type", {}).get("label", "")
                row["interaction_type_confidence"] = llm_output.get("interaction_type", {}).get("confidence", 0)
                row["sentence_function_label"] = llm_output.get("sentence_function", {}).get("label", "")
                row["sentence_function_confidence"] = llm_output.get("sentence_function", {}).get("confidence", 0)
                row["orthography_status"] = llm_output.get("orthography", {}).get("status", "")
                row["orthography_confidence"] = llm_output.get("orthography", {}).get("confidence", 0)
                
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

