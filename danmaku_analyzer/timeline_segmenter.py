"""
时序动态切分模块 - 基于弹幕时间戳密度的突变点检测
使用 ruptures PELT 算法进行动态切分
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass

import ruptures

from .config import get_settings
from .crawler import DanmakuItem
from .utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class TimeSegment:
    """时间段"""
    start_time: float  # 开始时间（秒）
    end_time: float  # 结束时间（秒）
    danmaku_indices: List[int]  # 弹幕索引列表
    density: float  # 弹幕密度（条/秒）
    zone_type: str  # "hot_zone" 或 "cold_zone"
    
    @property
    def duration(self) -> float:
        """持续时间（秒）"""
        return self.end_time - self.start_time
    
    @property
    def danmaku_count(self) -> int:
        """弹幕数量"""
        return len(self.danmaku_indices)


class TimelineSegmenter:
    
    def __init__(self):
        self.settings = get_settings()
        self.min_segment_samples = self.settings.MIN_SEGMENT_SAMPLES
        self.segmentation_mode = self.settings.SEGMENTATION_MODE
    
    def segment(self, danmaku_list: List[DanmakuItem]) -> List[TimeSegment]:
        if not danmaku_list:
            return []
        
        logger.info(f"开始时序切分，共 {len(danmaku_list)} 条弹幕，模式: {self.segmentation_mode}")
        
        # 提取时间戳
        timestamps = np.array([d.time_sec for d in danmaku_list])
        
        if self.segmentation_mode == "fixed":
            # 固定模式：按 MIN_SEGMENT_SAMPLES 等分
            segments = self._fixed_segment(danmaku_list, timestamps)
        else:
            # 动态模式：基于密度突变点检测
            density_signal = self._compute_density_signal(timestamps)
            breakpoints = self._detect_changepoints(density_signal)
            segments = self._split_by_breakpoints(danmaku_list, timestamps, breakpoints)
            # 应用保底逻辑：合并过小的段
            segments = self._merge_small_segments(segments)
        
        # 标记 hot/cold zone
        segments = self._label_zones(segments)
        
        logger.info(f"时序切分完成，共 {len(segments)} 个时间段")
        return segments
    
    def _fixed_segment(self, danmaku_list: List[DanmakuItem], timestamps: np.ndarray) -> List[TimeSegment]:
        segments = []
        n = len(danmaku_list)
        step = self.min_segment_samples
        
        for start_idx in range(0, n, step):
            end_idx = min(start_idx + step, n)
            start_time = float(timestamps[start_idx])
            end_time = float(timestamps[end_idx - 1])
            duration = end_time - start_time
            density = (end_idx - start_idx) / duration if duration > 0 else 0.0
            
            segment = TimeSegment(
                start_time=start_time,
                end_time=end_time,
                danmaku_indices=list(range(start_idx, end_idx)),
                density=density,
                zone_type="",
            )
            segments.append(segment)
        
        logger.info(f"固定模式切分：每段 {step} 条，共 {len(segments)} 段")
        return segments
    
    def _compute_density_signal(self, timestamps: np.ndarray, window_size: float = 1.0) -> np.ndarray:
        if len(timestamps) < 2:
            return np.array([len(timestamps)])
        
        # 确定时间范围
        min_time = timestamps.min()
        max_time = timestamps.max()
        
        # 创建时间窗口
        num_windows = int((max_time - min_time) / window_size) + 1
        density = np.zeros(num_windows)
        
        # 计算每个窗口的弹幕数
        for t in timestamps:
            window_idx = int((t - min_time) / window_size)
            if 0 <= window_idx < num_windows:
                density[window_idx] += 1
        
        return density
    
    def _detect_changepoints(self, signal: np.ndarray) -> List[int]:
        """PELT 突变点检测，失败则均匀切分"""
        if len(signal) < 3:
            return []
        
        try:
            # 使用 PELT 算法
            algo = ruptures.Pelt(model="l2", min_size=1, jump=1)
            algo.fit(signal.reshape(-1, 1))
            
            # 检测突变点
            breakpoints = algo.predict(pen=1)
            
            # 移除最后一个点（信号末尾）
            if breakpoints and breakpoints[-1] == len(signal):
                breakpoints = breakpoints[:-1]
            
            logger.debug(f"检测到 {len(breakpoints)} 个突变点")
            return breakpoints
            
        except Exception as e:
            logger.warning(f"突变点检测失败: {e}，使用均匀切分")
            # 保底：均匀切分
            return list(range(1, len(signal), max(1, len(signal) // 5)))
    
    def _split_by_breakpoints(
        self, 
        danmaku_list: List[DanmakuItem], 
        timestamps: np.ndarray, 
        breakpoints: List[int],
        window_size: float = 1.0
    ) -> List[TimeSegment]:
        if not danmaku_list:
            return []
        
        min_time = timestamps.min()
        segments = []
        prev_idx = 0
        
        # 添加突变点对应的弹幕索引
        breakpoint_indices = []
        for bp in breakpoints:
            # 找到该时间窗口对应的弹幕索引
            bp_time = min_time + bp * window_size
            idx = np.searchsorted(timestamps, bp_time)
            breakpoint_indices.append(idx)
        
        # 切分弹幕
        all_indices = [0] + breakpoint_indices + [len(danmaku_list)]
        
        for i in range(len(all_indices) - 1):
            start_idx = all_indices[i]
            end_idx = all_indices[i + 1]
            
            if start_idx >= end_idx:
                continue
            
            # 计算时间范围
            start_time = timestamps[start_idx]
            end_time = timestamps[end_idx - 1] if end_idx > start_idx else start_time
            
            # 计算密度
            duration = end_time - start_time
            density = (end_idx - start_idx) / duration if duration > 0 else 0.0
            
            segment = TimeSegment(
                start_time=start_time,
                end_time=end_time,
                danmaku_indices=list(range(start_idx, end_idx)),
                density=density,
                zone_type="",  # 后续标记
            )
            segments.append(segment)
        
        return segments
    
    def _merge_small_segments(self, segments: List[TimeSegment]) -> List[TimeSegment]:
        """合并小于 MIN_SEGMENT_SAMPLES 的段"""
        if not segments:
            return []
        
        merged = []
        i = 0
        
        while i < len(segments):
            current = segments[i]
            
            # 如果当前段太小，尝试与前一段合并
            if current.danmaku_count < self.min_segment_samples and merged:
                prev = merged[-1]
                # 合并到前一段
                merged_segment = TimeSegment(
                    start_time=prev.start_time,
                    end_time=current.end_time,
                    danmaku_indices=prev.danmaku_indices + current.danmaku_indices,
                    density=(prev.danmaku_count + current.danmaku_count) / 
                            (current.end_time - prev.start_time) if (current.end_time - prev.start_time) > 0 else 0.0,
                    zone_type="",  # 后续重新标记
                )
                merged[-1] = merged_segment
            else:
                merged.append(current)
            
            i += 1
        
        # 如果第一段太小，与第二段合并
        if len(merged) > 1 and merged[0].danmaku_count < self.min_segment_samples:
            first = merged[0]
            second = merged[1]
            merged_segment = TimeSegment(
                start_time=first.start_time,
                end_time=second.end_time,
                danmaku_indices=first.danmaku_indices + second.danmaku_indices,
                density=(first.danmaku_count + second.danmaku_count) / 
                        (second.end_time - first.start_time) if (second.end_time - first.start_time) > 0 else 0.0,
                zone_type="",
            )
            merged = [merged_segment] + merged[2:]
        
        return merged
    
    def _label_zones(self, segments: List[TimeSegment]) -> List[TimeSegment]:
        """基于密度均值+1.5MAD 标记 hot/cold zone"""
        if not segments:
            return []
        
        # 计算密度统计
        densities = np.array([s.density for s in segments])
        
        if len(densities) < 2:
            # 只有一个段，默认标记为 cold_zone
            segments[0].zone_type = "cold_zone"
            return segments
        
        # 计算均值和 MAD（中位数绝对偏差）
        mean_density = np.mean(densities)
        median_density = np.median(densities)
        mad = np.median(np.abs(densities - median_density))
        
        # 阈值：均值 + 1.5 MAD
        threshold = mean_density + 1.5 * mad
        
        # 标记 zone
        for segment in segments:
            if segment.density > threshold:
                segment.zone_type = "hot_zone"
            else:
                segment.zone_type = "cold_zone"
        
        hot_count = sum(1 for s in segments if s.zone_type == "hot_zone")
        cold_count = sum(1 for s in segments if s.zone_type == "cold_zone")
        logger.info(f"Zone 标记完成：{hot_count} 个热区，{cold_count} 个冷区")
        
        return segments

