"""
语料库补足建议模块 - 缺口分析（离线） + 候选视频推荐（在线）
离线：基于 corpus_index.json 统计各分区已有视频数与 CORPUS_MIN_VIDEOS_PER_PARTITION 的差距
在线：经 bilibili-api search 接口按分区 + 弹幕数排序拉取候选（BV号/标题/播放量/弹幕数/发布日期），
自动剔除已在语料库索引中的视频
"""

import html
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Set

from .config import get_settings
from .corpus_store import CorpusStore
from .partitions import PARTITION_TO_TID
from .utils.logger import get_logger

logger = get_logger(__name__)

# 搜索接口标题中的关键词高亮标签，展示前需剔除
_KEYWORD_TAG_RE = re.compile(r"</?em[^>]*>")


def _clean_title(raw: str) -> str:
    """去除高亮标签与 HTML 实体转义（如 <em class=\"keyword\">、&amp;）"""
    return html.unescape(_KEYWORD_TAG_RE.sub("", raw)).strip()


@dataclass
class PartitionGap:
    """单分区语料库缺口"""
    tname: str
    have: int
    min_required: int
    missing: int

    @property
    def is_sufficient(self) -> bool:
        return self.missing <= 0

    def to_dict(self) -> dict:
        return {
            "tname": self.tname,
            "have": self.have,
            "min_required": self.min_required,
            "missing": self.missing,
            "is_sufficient": self.is_sufficient,
        }


@dataclass
class CandidateVideo:
    """在线候选视频"""
    bvid: str
    title: str
    play: int
    danmaku_count: int
    pubdate: str

    def to_dict(self) -> dict:
        return {
            "bvid": self.bvid,
            "title": self.title,
            "play": self.play,
            "danmaku_count": self.danmaku_count,
            "pubdate": self.pubdate,
        }


@dataclass
class SuggestResult:
    """suggest 命令完整结果：缺口列表 + 每分区候选列表"""
    gaps: List[PartitionGap] = field(default_factory=list)
    candidates: Dict[str, List[CandidateVideo]] = field(default_factory=dict)


class CorpusSuggester:

    def __init__(self, store: Optional[CorpusStore] = None):
        self.settings = get_settings()
        self.store = store or CorpusStore()

    def analyze_gaps(self) -> List[PartitionGap]:
        """统计各分区已有视频数与最小要求数的差距（未知分区单独列出但不计缺口）"""
        min_required = self.settings.CORPUS_MIN_VIDEOS_PER_PARTITION
        counts: Dict[str, int] = {}
        for v in self.store.get_videos():
            tname = (v.get("tname") or "").strip() or "未知"
            counts[tname] = counts.get(tname, 0) + 1

        gaps = []
        for tname in sorted(counts.keys(), key=lambda t: (-counts[t], t)):
            have = counts[tname]
            missing = max(0, min_required - have) if tname != "未知" else 0
            gaps.append(PartitionGap(tname=tname, have=have, min_required=min_required, missing=missing))
        return gaps

    def insufficient_partitions(self) -> List[str]:
        """有缺口的分区名列表（供在线候选的默认目标）"""
        return [g.tname for g in self.analyze_gaps() if not g.is_sufficient]

    async def fetch_candidates(
        self,
        partitions: List[str],
        per_partition: int = 10,
    ) -> Dict[str, List[CandidateVideo]]:
        """按分区拉取候选视频：分区过滤 + 弹幕数排序，剔除索引已有 bvid"""
        from bilibili_api import search

        existing: Set[str] = {v.get("bvid", "") for v in self.store.get_videos()}
        result: Dict[str, List[CandidateVideo]] = {}
        for tname in partitions:
            tid = PARTITION_TO_TID.get(tname)
            if tid is None:
                logger.warning(f"分区 '{tname}' 无 tid 映射，仅按关键词搜索（可能混入其他分区）")
            try:
                raw = await search.search_by_type(
                    keyword=tname,
                    search_type=search.SearchObjectType.VIDEO,
                    order_type=search.OrderVideo.DM,
                    video_zone_type=tid,
                    page=1,
                    page_size=max(per_partition * 2, 20),
                )
            except Exception as e:
                logger.warning(f"分区 '{tname}' 候选搜索失败: {e}")
                result[tname] = []
                continue
            result[tname] = self._parse_results(raw, existing, per_partition)
            logger.info(f"分区 '{tname}' 获取候选 {len(result[tname])} 个")
        return result

    @staticmethod
    def _parse_results(raw: dict, existing: Set[str], limit: int) -> List[CandidateVideo]:
        """搜索结果 → CandidateVideo：剔除已收录、按弹幕数降序、取前 limit 个"""
        candidates = []
        for item in raw.get("result") or []:
            bvid = item.get("bvid") or ""
            if not bvid or bvid in existing:
                continue
            pubdate_ts = item.get("pubdate") or 0
            candidates.append(CandidateVideo(
                bvid=bvid,
                title=_clean_title(item.get("title") or ""),
                play=int(item.get("play") or 0),
                danmaku_count=int(item.get("video_review") or 0),
                pubdate=datetime.fromtimestamp(pubdate_ts).strftime("%Y-%m-%d") if pubdate_ts else "",
            ))
        candidates.sort(key=lambda c: c.danmaku_count, reverse=True)
        return candidates[:limit]

    async def suggest(self, partitions: Optional[List[str]] = None, per_partition: int = 10) -> SuggestResult:
        """完整建议：缺口分析 + 候选获取（partitions 为空时取所有有缺口的分区）"""
        gaps = self.analyze_gaps()
        targets = partitions if partitions is not None else self.insufficient_partitions()
        candidates = await self.fetch_candidates(targets, per_partition) if targets else {}
        return SuggestResult(gaps=gaps, candidates=candidates)
