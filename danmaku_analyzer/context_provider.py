"""
微语境构建模块 - 提取前后时间窗口内的弹幕作为上下文
严禁跨段提取，最大 token 数限制
"""

from typing import List, Optional, Tuple
from dataclasses import dataclass

from .config import get_settings
from .crawler import DanmakuItem
from .timeline_segmenter import TimeSegment
from .utils.token_counter import count_tokens


@dataclass
class ContextWindow:
    """微语境窗口"""
    target_danmaku: DanmakuItem  # 目标弹幕
    before_context: List[DanmakuItem]  # 前文弹幕
    after_context: List[DanmakuItem]  # 后文弹幕
    total_tokens: int  # 总 token 数
    
    def to_prompt_text(self) -> str:
        """转换为提示文本"""
        lines = []
        
        # 前文
        if self.before_context:
            lines.append("【前文弹幕】")
            for d in self.before_context:
                lines.append(f"- {d.content}")
        
        # 目标弹幕
        lines.append("【当前弹幕】")
        lines.append(f">>> {self.target_danmaku.content}")
        
        # 后文
        if self.after_context:
            lines.append("【后文弹幕】")
            for d in self.after_context:
                lines.append(f"- {d.content}")
        
        return "\n".join(lines)


class ContextProvider:
    
    def __init__(self):
        self.settings = get_settings()
        self.time_window = self.settings.CONTEXT_TIME_WINDOW
        self.max_tokens = self.settings.MAX_CONTEXT_TOKENS
    
    def get_context(
        self, 
        target_danmaku: DanmakuItem, 
        segment: TimeSegment,
        segment_danmaku_list: List[DanmakuItem]
    ) -> ContextWindow:
        target_time = target_danmaku.time_sec
        
        # 计算时间窗口范围
        window_start = max(segment.start_time, target_time - self.time_window)
        window_end = min(segment.end_time, target_time + self.time_window)
        
        # 获取窗口内的弹幕（严格限制在段内）
        before_context = []
        after_context = []
        
        for danmaku in segment_danmaku_list:
            # 跳过目标弹幕本身
            if danmaku.time_sec == target_time and danmaku.content == target_danmaku.content:
                continue
            
            # 严格限制在时间窗口内
            if window_start <= danmaku.time_sec < target_time:
                before_context.append(danmaku)
            elif target_time < danmaku.time_sec <= window_end:
                after_context.append(danmaku)
        
        # 按时间排序
        before_context.sort(key=lambda x: x.time_sec)
        after_context.sort(key=lambda x: x.time_sec)
        
        # 应用 token 限制
        before_context, after_context = self._apply_token_limit(
            before_context, after_context, target_danmaku
        )
        
        # 计算总 token 数
        total_tokens = self._count_total_tokens(before_context, after_context, target_danmaku)
        
        context = ContextWindow(
            target_danmaku=target_danmaku,
            before_context=before_context,
            after_context=after_context,
            total_tokens=total_tokens,
        )
        
        return context
    
    def get_context_batch(
        self, 
        target_danmaku_list: List[DanmakuItem], 
        segment: TimeSegment,
        segment_danmaku_list: List[DanmakuItem]
    ) -> List[ContextWindow]:
        return [self.get_context(t, segment, segment_danmaku_list) for t in target_danmaku_list]
    
    def _apply_token_limit(
        self, 
        before_context: List[DanmakuItem], 
        after_context: List[DanmakuItem],
        target_danmaku: DanmakuItem
    ) -> Tuple[List[DanmakuItem], List[DanmakuItem]]:
        # 计算目标弹幕的 token 数
        target_tokens = count_tokens(target_danmaku.content)
        remaining_tokens = self.max_tokens - target_tokens
        
        if remaining_tokens <= 0:
            return [], []
        
        # 分配 token 给前后文（前文优先）
        before_tokens_budget = int(remaining_tokens * 0.6)
        after_tokens_budget = remaining_tokens - before_tokens_budget
        
        # 过滤前文
        filtered_before = []
        used_tokens = 0
        # 从最近的开始添加
        for danmaku in reversed(before_context):
            danmaku_tokens = count_tokens(danmaku.content)
            if used_tokens + danmaku_tokens <= before_tokens_budget:
                filtered_before.insert(0, danmaku)
                used_tokens += danmaku_tokens
            else:
                break
        
        # 过滤后文
        filtered_after = []
        used_tokens = 0
        for danmaku in after_context:
            danmaku_tokens = count_tokens(danmaku.content)
            if used_tokens + danmaku_tokens <= after_tokens_budget:
                filtered_after.append(danmaku)
                used_tokens += danmaku_tokens
            else:
                break
        
        return filtered_before, filtered_after
    
    def _count_total_tokens(
        self, 
        before_context: List[DanmakuItem], 
        after_context: List[DanmakuItem],
        target_danmaku: DanmakuItem
    ) -> int:
        total = count_tokens(target_danmaku.content)
        
        for danmaku in before_context:
            total += count_tokens(danmaku.content)
        
        for danmaku in after_context:
            total += count_tokens(danmaku.content)
        
        return total

