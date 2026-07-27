"""
用户级去重模块 - 仅对 real_user 去重
unknown_device 标记为 "unknown_device"，排除在去重之外，但计入总量
"""

from typing import List, Dict, Tuple
from dataclasses import dataclass
from collections import defaultdict

from .crawler import DanmakuItem
from .utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class DeduplicationResult:
    """去重结果"""
    deduplicated_danmaku: List[DanmakuItem]  # 去重后的弹幕列表
    total_count: int  # 总弹幕数（去重前）
    unique_real_user_count: int  # 去重后的唯一真实用户数
    unknown_device_count: int  # unknown_device 数量
    duplicate_count: int  # 重复弹幕数量
    
    @property
    def deduplication_rate(self) -> float:
        """去重率"""
        if self.total_count == 0:
            return 0.0
        return self.duplicate_count / self.total_count


class UserDeduplicator:

    def deduplicate(self, danmaku_list: List[DanmakuItem]) -> DeduplicationResult:
        if not danmaku_list:
            return DeduplicationResult(
                deduplicated_danmaku=[],
                total_count=0,
                unique_real_user_count=0,
                unknown_device_count=0,
                duplicate_count=0,
            )
        
        logger.info(f"开始用户去重，共 {len(danmaku_list)} 条弹幕")
        
        # 分离 real_user 和 unknown_device
        real_user_danmaku = []
        unknown_device_danmaku = []
        
        for danmaku in danmaku_list:
            if danmaku.identity_type == "unknown_device":
                unknown_device_danmaku.append(danmaku)
            else:
                real_user_danmaku.append(danmaku)
        
        # 对 real_user 进行去重（基于 uid_hash）
        user_danmaku_map: Dict[str, List[DanmakuItem]] = defaultdict(list)
        
        for danmaku in real_user_danmaku:
            user_danmaku_map[danmaku.uid_hash].append(danmaku)
        
        # 每个用户只保留一条弹幕（保留第一条）
        deduplicated_real_user = []
        duplicate_count = 0
        
        for uid_hash, user_danmaku_list in user_danmaku_map.items():
            if len(user_danmaku_list) > 1:
                # 保留第一条，其余视为重复
                deduplicated_real_user.append(user_danmaku_list[0])
                duplicate_count += len(user_danmaku_list) - 1
            else:
                deduplicated_real_user.append(user_danmaku_list[0])
        
        # 合并去重后的 real_user 和 unknown_device
        deduplicated_danmaku = deduplicated_real_user + unknown_device_danmaku
        
        # 按时间排序
        deduplicated_danmaku.sort(key=lambda x: x.time_sec)
        
        result = DeduplicationResult(
            deduplicated_danmaku=deduplicated_danmaku,
            total_count=len(danmaku_list),
            unique_real_user_count=len(deduplicated_real_user),
            unknown_device_count=len(unknown_device_danmaku),
            duplicate_count=duplicate_count,
        )
        
        logger.info(
            f"用户去重完成：总弹幕 {result.total_count}，"
            f"唯一真实用户 {result.unique_real_user_count}，"
            f"unknown_device {result.unknown_device_count}，"
            f"重复 {result.duplicate_count}，"
            f"去重率 {result.deduplication_rate:.2%}"
        )
        
        return result
    
    def deduplicate_batch(self, danmaku_lists: List[List[DanmakuItem]]) -> List[DeduplicationResult]:
        return [self.deduplicate(dl) for dl in danmaku_lists]
