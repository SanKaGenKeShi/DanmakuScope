"""
语料库级聚合模块 - 从单视频 ZIP 报告回读并跨视频聚合
输出语料库级比较表（CSV），统计单位为视频（供分区间检验消费）
"""

import io
import json
import os
import zipfile
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pandas as pd

from . import __version__
from .config import get_settings
from .corpus_store import CorpusStore
from .utils.logger import get_logger

logger = get_logger(__name__)

METADATA_FILENAME = "metadata.json"

# ZIP 内需回读的表：文件名 → 除 danmaku_count 外的标量列（其余列视为分布占比）
TABLE_SPECS = {
    "table_lexical_by_partition.csv": ["avg_word_length", "content_word_density", "punctuation_emoji_rate"],
    "table_consensus_stats.csv": ["high_consensus_rate", "medium_consensus_rate", "low_consensus_rate", "avg_weight_multiplier"],
    "table_emotion.csv": ["cooperative_principle_violation_rate"],
    "table_sentence_function.csv": [],
    "table_interaction_type.csv": [],
}

NON_DIST_COLUMNS = {"tname", "zone_type", "danmaku_count"}

SCALAR_FIELDS = [
    "avg_word_length", "content_word_density", "punctuation_emoji_rate",
    "high_consensus_rate", "medium_consensus_rate", "low_consensus_rate",
    "avg_weight_multiplier", "cooperative_principle_violation_rate",
]


def validate_zip_archive(zip_path: str, expected_count: int) -> bool:
    """ZIP 完整性校验：存在非空 + 条目数一致 + 首个条目可读"""
    if not os.path.exists(zip_path) or os.path.getsize(zip_path) == 0:
        return False
    try:
        with zipfile.ZipFile(zip_path, 'r') as zipf:
            names = zipf.namelist()
            if len(names) != expected_count:
                return False
            if names:
                zipf.read(names[0])
            return True
    except Exception:
        return False


@dataclass
class VideoSummary:
    """单视频级摘要：语料库比较的观测单位"""
    bvid: str
    tname: str
    pubdate: str
    prompt_version: str
    zone_type: Optional[str]  # zone_policy=all 时保留分区信息，其余为 None
    danmaku_count: int = 0
    scalars: Dict[str, float] = field(default_factory=dict)
    distributions: Dict[str, float] = field(default_factory=dict)  # 列名（含前缀）→ 占比


@dataclass
class CorpusBuildResult:
    """语料库聚合产物：打包前的散落文件清单与来源 ZIP"""
    csv_path: str
    videos_csv_path: str
    source_zip_paths: List[str] = field(default_factory=list)
    output_dir: str = ""
    warnings: List[str] = field(default_factory=list)
    zip_path: Optional[str] = None
    zip_valid: bool = False


class CorpusBuilder:

    def build_from_zips(self, zip_paths: List[str], output_dir: Optional[str] = None) -> CorpusBuildResult:
        """回读多个 ZIP → 登记索引 → 聚合输出语料库级 CSV，返回构建结果"""
        store = CorpusStore()
        summaries: List[VideoSummary] = []
        readable_zips: List[str] = []
        for path in zip_paths:
            try:
                metadata, tables = self.read_zip(path)
            except (OSError, zipfile.BadZipFile, KeyError, ValueError) as e:
                logger.warning(f"跳过无法回读的 ZIP: {path} - {e}")
                continue
            readable_zips.append(path)
            try:
                store.register_video(self._index_entry(metadata, path))
            except ValueError as e:
                logger.warning(f"索引登记失败（不阻断聚合）: {path} - {e}")
            try:
                summaries.extend(self.summarize_video(metadata, tables))
            except ValueError as e:
                logger.warning(f"跳过无法摘要的 ZIP: {path} - {e}")

        return self._aggregate_and_write(summaries, output_dir, readable_zips)

    def build_from_index(self, output_dir: Optional[str] = None) -> CorpusBuildResult:
        """从语料库索引登记的全部视频聚合"""
        store = CorpusStore()
        summaries: List[VideoSummary] = []
        readable_zips: List[str] = []
        for video in store.get_videos():
            zip_path = store.resolve_zip_path(video.get("zip_path", ""))
            if not os.path.exists(zip_path):
                logger.warning(f"索引中 ZIP 不存在，跳过: {video.get('bvid')} - {zip_path}")
                continue
            try:
                metadata, tables = self.read_zip(zip_path)
            except (OSError, zipfile.BadZipFile, KeyError, ValueError) as e:
                logger.warning(f"跳过无法回读的 ZIP: {zip_path} - {e}")
                continue
            readable_zips.append(zip_path)
            try:
                summaries.extend(self.summarize_video(metadata, tables))
            except ValueError as e:
                logger.warning(f"跳过无法摘要的 ZIP: {zip_path} - {e}")

        return self._aggregate_and_write(summaries, output_dir, readable_zips)

    def read_zip(self, zip_path: str) -> Tuple[Dict, Dict[str, pd.DataFrame]]:
        """回读 ZIP：metadata.json + 各聚合表 CSV；缺失必需文件时抛 KeyError"""
        with zipfile.ZipFile(zip_path, 'r') as zipf:
            names = set(zipf.namelist())
            if METADATA_FILENAME not in names:
                raise KeyError(f"缺少 {METADATA_FILENAME}: {zip_path}")
            metadata = json.loads(zipf.read(METADATA_FILENAME).decode('utf-8'))
            tables = {}
            for filename in TABLE_SPECS:
                if filename in names:
                    tables[filename] = pd.read_csv(io.BytesIO(zipf.read(filename)), encoding='utf-8-sig')
                else:
                    logger.warning(f"ZIP 缺少表 {filename}: {zip_path}")
        return metadata, tables

    def summarize_video(self, metadata: Dict, tables: Dict[str, pd.DataFrame]) -> List[VideoSummary]:
        """按冷热区策略把单视频各表折叠为 VideoSummary（policy=all 时每个区一条）"""
        policy = get_settings().CORPUS_ZONE_POLICY
        bvid = metadata.get("bvid", "")
        # 旧版 ZIP 的 metadata.json 无 tname 透传字段，partitions 为可靠回退源
        tname = metadata.get("tname") or (metadata.get("partitions") or [""])[0]
        if not tname:
            raise ValueError(f"无法确定分区（metadata 缺 tname 与 partitions）: {bvid or '未知视频'}")

        per_zone_rows = self._collect_zone_rows(tables)
        zone_groups = self._apply_zone_policy(per_zone_rows, policy, bvid)

        summaries = []
        for zone_label, rows in zone_groups:
            merged = self._merge_rows(rows)
            summaries.append(VideoSummary(
                bvid=bvid,
                tname=tname,
                pubdate=metadata.get("pubdate", ""),
                prompt_version=metadata.get("prompt_version", ""),
                zone_type=zone_label,
                danmaku_count=int(merged.pop("danmaku_count", 0)),
                scalars={k: merged.pop(k, 0.0) for k in SCALAR_FIELDS},
                distributions=merged,
            ))
        return summaries

    def _collect_zone_rows(self, tables: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, Dict]]:
        """把各表行按 zone_type 归拢：zone → {表名: 行dict}"""
        per_zone = defaultdict(dict)
        for filename, df in tables.items():
            if "zone_type" not in df.columns:
                continue
            for _, row in df.iterrows():
                per_zone[row["zone_type"]][filename] = row.to_dict()
        return per_zone

    def _apply_zone_policy(self, per_zone: Dict, policy: str, bvid: str) -> List[Tuple[Optional[str], List[Dict]]]:
        """返回 [(zone_label, 行列表)]；hot_only 无热区时跳过该视频并告警"""
        if policy == "hot_only":
            if "hot_zone" not in per_zone:
                logger.warning(f"视频 {bvid} 无 hot_zone 行，按 hot_only 策略跳过")
                return []
            return [(None, list(per_zone["hot_zone"].values()))]
        if policy == "weighted":
            rows = [row for zone_rows in per_zone.values() for row in zone_rows.values()]
            return [(None, rows)]
        # all：两区各保留，语料库行带 zone_type 维度
        return [(zone, list(zone_rows.values())) for zone, zone_rows in sorted(per_zone.items())]

    def _merge_rows(self, rows: List[Dict]) -> Dict[str, float]:
        """按 danmaku_count 加权合并同视频多行（表间按列名合并，权重为各自行弹幕数）"""
        actual_total = sum(max(r.get("danmaku_count", 0), 0) for r in rows)
        merged = {}
        if actual_total <= 0:
            logger.warning("合并行弹幕数为 0，退化为等权平均")
            weights = [1.0] * len(rows)
        else:
            weights = [max(r.get("danmaku_count", 0), 0) for r in rows]

        all_columns = set()
        for r in rows:
            all_columns.update(r.keys())

        # 同一 zone 在多张表中重复携带 danmaku_count，按 zone 去重后求和才是视频真实弹幕数
        zone_counts = {}
        for r in rows:
            zone = r.get("zone_type", "")
            zone_counts[zone] = max(zone_counts.get(zone, 0), max(r.get("danmaku_count", 0), 0))
        merged["danmaku_count"] = sum(zone_counts.values())
        for col in all_columns:
            if col in NON_DIST_COLUMNS:
                continue
            values = [(r[col], w) for r, w in zip(rows, weights) if pd.notna(r.get(col))]
            if not values:
                continue
            merged[col] = sum(float(v) * w for v, w in values) / sum(w for _, w in values)
        return merged

    def _aggregate_and_write(
        self,
        summaries: List[VideoSummary],
        output_dir: Optional[str],
        source_zip_paths: Optional[List[str]] = None,
    ) -> CorpusBuildResult:
        if not summaries:
            raise ValueError("无可聚合的视频摘要（请检查 ZIP 是否包含 metadata.json 与聚合表）")

        self._check_prompt_versions(summaries)

        rows, warnings = self._aggregate_groups(summaries)

        settings = get_settings()
        out_dir = settings.resolve_data_path(output_dir or settings.OUTPUT_DIR)
        os.makedirs(out_dir, exist_ok=True)
        filepath = self._write_summary_table(rows, summaries, out_dir)

        # 视频级观测表：KW/Dunn 等检验的原始观测来源（组级 mean/std 无法还原个体值）
        videos_path = os.path.join(out_dir, "corpus_videos.csv")
        self._write_video_observations(summaries, videos_path)
        return CorpusBuildResult(
            csv_path=filepath,
            videos_csv_path=videos_path,
            source_zip_paths=list(source_zip_paths or []),
            output_dir=out_dir,
            warnings=warnings,
        )

    def _aggregate_groups(self, summaries: List[VideoSummary]) -> Tuple[List[Dict], List[str]]:
        """按分区/时段/冷热区分组聚合，返回（组级行, 样本量告警）"""
        settings = get_settings()
        temporal = settings.ENABLE_TEMPORAL_GROUPING
        granularity = settings.TEMPORAL_GRANULARITY
        min_videos = settings.CORPUS_MIN_VIDEOS_PER_PARTITION

        groups: Dict[Tuple, List[VideoSummary]] = defaultdict(list)
        for s in summaries:
            time_period = self._bucket_pubdate(s.pubdate, granularity) if temporal else ""
            key = (s.tname, time_period, s.zone_type or "")
            groups[key].append(s)

        warnings: List[str] = []
        rows = []
        for (tname, time_period, zone_type), items in sorted(groups.items()):
            if len(items) < min_videos:
                msg = f"分区 {tname}{f'({time_period})' if time_period else ''} 视频数 {len(items)} < {min_videos}，结果仅供参考"
                logger.warning(msg)
                warnings.append(msg)
            rows.append(self._aggregate_group(tname, time_period, zone_type, items))
        return rows, warnings

    def _write_summary_table(self, rows: List[Dict], summaries: List[VideoSummary], out_dir: str) -> str:
        filepath = os.path.join(out_dir, "corpus_summary.csv")
        pd.DataFrame(rows).to_csv(filepath, index=False, encoding='utf-8-sig')
        logger.info(f"语料库聚合表已保存: {filepath}（{len(rows)} 组，{len(summaries)} 个视频观测）")
        return filepath

    def package_snapshot(
        self,
        result: CorpusBuildResult,
        extra_files: Optional[List[str]] = None,
    ) -> str:
        """把语料库产物打包为自包含快照 ZIP（时间戳命名）

        包内：corpus_metadata.json + 聚合 CSV + 附加文件（R 脚本/LLM 报告）+ videos/ 下的源视频 ZIP。
        源视频 ZIP 仅复制入包，原文件一律保留；校验通过才删除包内已收录的散落源文件。
        """
        out_dir = result.output_dir
        meta = self.build_snapshot_metadata(result)
        meta_path = os.path.join(out_dir, "corpus_metadata.json")
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        zip_filename = f"[corpus]_{meta['video_count']}videos_{timestamp}.zip"
        zip_path = os.path.join(out_dir, zip_filename)

        loose_files = [result.csv_path, result.videos_csv_path, meta_path] + list(extra_files or [])
        # 源 ZIP 按基名去重（同名冲突保留先出现的），全部置于 videos/ 前缀下
        source_entries: List[Tuple[str, str]] = []
        seen_names = set()
        for path in result.source_zip_paths:
            base = os.path.basename(path)
            if base in seen_names:
                logger.warning(f"源 ZIP 基名冲突，跳过重复收录: {path}")
                continue
            if not os.path.exists(path):
                logger.warning(f"源 ZIP 已不存在，无法收录: {path}")
                continue
            seen_names.add(base)
            source_entries.append((path, f"videos/{base}"))

        try:
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for path in loose_files:
                    if os.path.exists(path):
                        zipf.write(path, os.path.basename(path))
                for path, arcname in source_entries:
                    zipf.write(path, arcname)
        except OSError as e:
            logger.error(f"语料库快照打包失败: {zip_path} - {e}")
            return zip_path

        expected_count = sum(1 for p in loose_files if os.path.exists(p)) + len(source_entries)
        if validate_zip_archive(zip_path, expected_count):
            result.zip_path = zip_path
            result.zip_valid = True
            deleted = 0
            for path in loose_files:
                if os.path.exists(path):
                    try:
                        os.remove(path)
                        deleted += 1
                    except OSError as e:
                        logger.warning(f"删除失败: {os.path.basename(path)} - {e}")
            logger.info(f"语料库快照已打包: {zip_filename}（源视频 ZIP 收录 {len(source_entries)} 个，原文件保留；已删除 {deleted} 个散落源文件）")
        else:
            logger.warning(f"语料库快照校验失败，保留散落文件: {zip_path}")
        return zip_path

    def build_snapshot_metadata(self, result: CorpusBuildResult) -> Dict:
        """构建快照元数据（自描述证据链：纳入视频/版本/策略/告警），供打包与 LLM 报告共用"""
        settings = get_settings()
        bvids = self._read_bvids(result.videos_csv_path)
        return {
            "generated_at": datetime.now().isoformat(timespec='seconds'),
            "pipeline_version": __version__,
            "zone_policy": settings.CORPUS_ZONE_POLICY,
            "temporal_grouping": settings.ENABLE_TEMPORAL_GROUPING,
            "temporal_granularity": settings.TEMPORAL_GRANULARITY,
            "video_count": len(bvids),
            "bvids": bvids,
            "prompt_versions": self._read_prompt_versions(result.videos_csv_path),
            "warnings": result.warnings,
        }

    @staticmethod
    def _read_bvids(videos_csv_path: str) -> List[str]:
        try:
            df = pd.read_csv(videos_csv_path, encoding='utf-8-sig', usecols=["bvid"])
            return sorted(df["bvid"].astype(str).unique().tolist())
        except Exception as e:
            logger.warning(f"回读 bvid 清单失败: {e}")
            return []

    @staticmethod
    def _read_prompt_versions(videos_csv_path: str) -> List[str]:
        try:
            df = pd.read_csv(videos_csv_path, encoding='utf-8-sig', usecols=["prompt_version"])
            return sorted(str(v) for v in df["prompt_version"].dropna().unique() if str(v))
        except Exception as e:
            logger.warning(f"回读 prompt_version 清单失败: {e}")
            return []

    def _write_video_observations(self, summaries: List[VideoSummary], filepath: str):
        records = []
        for s in summaries:
            record = {
                "bvid": s.bvid, "tname": s.tname, "pubdate": s.pubdate,
                "prompt_version": s.prompt_version,
                "zone_type": s.zone_type or "", "danmaku_count": s.danmaku_count,
            }
            record.update(s.scalars)
            record.update(s.distributions)
            records.append(record)
        pd.DataFrame(records).to_csv(filepath, index=False, encoding='utf-8-sig')
        logger.info(f"视频级观测表已保存: {filepath}（{len(records)} 行）")

    def _aggregate_group(self, tname: str, time_period: str, zone_type: str, items: List[VideoSummary]) -> Dict:
        """组级聚合：标量取视频级均值/标准差，分布按弹幕数加权均值"""
        row: Dict = {"tname": tname}
        if time_period:
            row["time_period"] = time_period
        if zone_type:
            row["zone_type"] = zone_type
        row["video_count"] = len(items)
        row["total_danmaku"] = sum(s.danmaku_count for s in items)

        for name in SCALAR_FIELDS:
            values = [s.scalars.get(name, 0.0) for s in items]
            row[f"{name}_mean"] = float(pd.Series(values).mean())
            row[f"{name}_std"] = float(pd.Series(values).std(ddof=1)) if len(values) > 1 else 0.0

        total_weight = sum(s.danmaku_count for s in items) or 1
        dist_columns = set()
        for s in items:
            dist_columns.update(s.distributions.keys())
        for col in sorted(dist_columns):
            weighted = sum(s.distributions.get(col, 0.0) * s.danmaku_count for s in items)
            row[f"{col}_mean"] = weighted / total_weight
        return row

    def _check_prompt_versions(self, summaries: List[VideoSummary]):
        versions = {s.prompt_version for s in summaries if s.prompt_version}
        if len(versions) > 1:
            logger.warning(f"语料库混入多个 prompt_version: {sorted(versions)}，软标签跨版本可比性存疑")

    @staticmethod
    def _bucket_pubdate(pubdate: str, granularity: str) -> str:
        if not pubdate:
            return "unknown"
        try:
            dt = datetime.fromisoformat(pubdate)
        except ValueError:
            return "unknown"
        if granularity == "year":
            return str(dt.year)
        if granularity == "quarter":
            return f"{dt.year}-Q{(dt.month - 1) // 3 + 1}"
        return f"{dt.year}-{dt.month:02d}"

    @staticmethod
    def _index_entry(metadata: Dict, zip_path: str) -> Dict:
        return {
            "bvid": metadata.get("bvid", ""),
            "title": metadata.get("title", ""),
            "tname": metadata.get("tname") or (metadata.get("partitions") or [""])[0],
            "pubdate": metadata.get("pubdate", ""),
            "danmaku_count": metadata.get("danmaku_count", 0),
            "pipeline_version": metadata.get("pipeline_version", ""),
            "prompt_version": metadata.get("prompt_version", ""),
            "zip_path": zip_path,
        }
