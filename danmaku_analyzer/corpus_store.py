"""
语料库索引模块 - DATA_ROOT/corpus_index.json 持久化
记录已分析视频的身份与结果位置，供语料库级聚合回读
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Optional

from .config import get_settings
from .utils.logger import get_logger

logger = get_logger(__name__)

SCHEMA_VERSION = "1.0"
DEFAULT_INDEX_FILENAME = "corpus_index.json"


class CorpusStore:
    """语料库索引读写：JSON 结构，bvid 为唯一键，重复登记时覆盖旧记录"""

    def __init__(self, index_path: Optional[str] = None):
        settings = get_settings()
        self.index_path = settings.resolve_data_path(index_path or DEFAULT_INDEX_FILENAME)

    def load(self) -> Dict:
        """读取索引；文件不存在或损坏时返回空索引并告警"""
        if not os.path.exists(self.index_path):
            return self._empty_index()
        try:
            with open(self.index_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if "videos" not in data:
                logger.warning(f"语料库索引缺少 videos 字段，视为空索引: {self.index_path}")
                return self._empty_index()
            return data
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"语料库索引读取失败，视为空索引: {self.index_path} - {e}")
            return self._empty_index()

    def save(self, index: Dict):
        with open(self.index_path, 'w', encoding='utf-8') as f:
            json.dump(index, f, ensure_ascii=False, indent=2)
        logger.info(f"语料库索引已保存: {self.index_path}（{len(index['videos'])} 个视频）")

    def register_video(self, entry: Dict) -> Dict:
        """登记一个已分析视频；bvid 已存在时覆盖并提示"""
        bvid = entry.get("bvid", "")
        if not bvid:
            raise ValueError("索引条目缺少 bvid 字段")
        entry.setdefault("analyzed_at", datetime.now().isoformat())
        if "zip_path" in entry:
            entry["zip_path"] = self._relativize(entry["zip_path"])

        index = self.load()
        existing = [v for v in index["videos"] if v.get("bvid") == bvid]
        if existing:
            logger.info(f"视频 {bvid} 已在索引中，覆盖旧记录")
            index["videos"] = [v for v in index["videos"] if v.get("bvid") != bvid]
        index["videos"].append(entry)
        self.save(index)
        return entry

    def get_videos(self) -> List[Dict]:
        return self.load()["videos"]

    def resolve_zip_path(self, stored_path: str) -> str:
        """回读索引中的 zip_path：绝对路径直接返回，相对路径基于 DATA_ROOT 解析"""
        return get_settings().resolve_data_path(stored_path)

    def _relativize(self, path: str) -> str:
        """尽量存相对 DATA_ROOT 的路径；跨驱动器或越层过深时回退绝对路径"""
        abs_path = os.path.abspath(path)
        data_root = get_settings().DATA_ROOT
        try:
            rel = os.path.relpath(abs_path, data_root)
        except ValueError:
            return abs_path
        if rel.startswith(".." + os.sep):
            depth = rel.count(".." + os.sep)
            if depth > data_root.rstrip(os.sep).count(os.sep):
                return abs_path
        return rel

    @staticmethod
    def _empty_index() -> Dict:
        return {"schema_version": SCHEMA_VERSION, "videos": []}
