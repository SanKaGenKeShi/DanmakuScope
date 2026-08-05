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
    start_time: float  # 秒
    end_time: float  # 秒
    danmaku_indices: List[int]
    density: float  # 条/秒
    zone_type: str  # "hot_zone" 或 "cold_zone"
    
    @property
    def duration(self) -> float:
        return self.end_time - self.start_time
    
    @property
    def danmaku_count(self) -> int:
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
        
        timestamps = np.array([d.time_sec for d in danmaku_list])
        
        if self.segmentation_mode == "fixed":
            segments = self._fixed_segment(danmaku_list, timestamps)
        else:
            density_signal = self._compute_density_signal(timestamps)
            breakpoints = self._detect_changepoints(density_signal)
            segments = self._split_by_breakpoints(danmaku_list, timestamps, breakpoints)
            segments = self._merge_small_segments(segments)
        
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
        
        min_time = timestamps.min()
        max_time = timestamps.max()
        num_windows = int((max_time - min_time) / window_size) + 1
        
        # 向量化直方统计（等价于逐条按窗口累加，弹幕量大时显著更快）
        density, _ = np.histogram(
            timestamps,
            bins=num_windows,
            range=(min_time, min_time + num_windows * window_size),
        )
        return density.astype(float)
    
    def _detect_changepoints(self, signal: np.ndarray) -> List[int]:
        """PELT 突变点检测，失败则均匀切分"""
        if len(signal) < 3:
            return []
        
        try:
            algo = ruptures.Pelt(model="l2", min_size=1, jump=1)
            algo.fit(signal.reshape(-1, 1))
            breakpoints = algo.predict(pen=1)
            
            # predict 会在信号末尾追加 len(signal)，需移除
            if breakpoints and breakpoints[-1] == len(signal):
                breakpoints = breakpoints[:-1]
            
            logger.debug(f"检测到 {len(breakpoints)} 个突变点")
            return breakpoints
            
        except Exception as e:
            logger.warning(f"突变点检测失败: {e}，使用均匀切分")
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
        
        breakpoint_indices = []
        for bp in breakpoints:
            bp_time = min_time + bp * window_size
            idx = np.searchsorted(timestamps, bp_time)
            breakpoint_indices.append(idx)
        
        all_indices = [0] + breakpoint_indices + [len(danmaku_list)]
        
        for i in range(len(all_indices) - 1):
            start_idx = all_indices[i]
            end_idx = all_indices[i + 1]
            
            if start_idx >= end_idx:
                continue
            
            start_time = timestamps[start_idx]
            end_time = timestamps[end_idx - 1] if end_idx > start_idx else start_time
            
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
    
    @staticmethod
    def _combine(prev: TimeSegment, current: TimeSegment) -> TimeSegment:
        """相邻两段合并（时间跨度、弹幕索引、密度重算），zone_type 留待后续重新标记"""
        duration = current.end_time - prev.start_time
        return TimeSegment(
            start_time=prev.start_time,
            end_time=current.end_time,
            danmaku_indices=prev.danmaku_indices + current.danmaku_indices,
            density=(prev.danmaku_count + current.danmaku_count) / duration if duration > 0 else 0.0,
            zone_type="",
        )
    
    def _merge_small_segments(self, segments: List[TimeSegment]) -> List[TimeSegment]:
        """合并小于 MIN_SEGMENT_SAMPLES 的段"""
        if not segments:
            return []
        
        merged = []
        i = 0
        
        while i < len(segments):
            current = segments[i]
            
            if current.danmaku_count < self.min_segment_samples and merged:
                merged[-1] = self._combine(merged[-1], current)
            else:
                merged.append(current)
            
            i += 1
        
        # 首段无前驱可并，向后合并直至满足阈值（或仅剩一段）
        while len(merged) > 1 and merged[0].danmaku_count < self.min_segment_samples:
            merged = [self._combine(merged[0], merged[1])] + merged[2:]
        
        return merged
    
    def _label_zones(self, segments: List[TimeSegment]) -> List[TimeSegment]:
        """基于密度均值+1.5MAD 标记 hot/cold zone"""
        if not segments:
            return []
        
        densities = np.array([s.density for s in segments])
        
        if len(densities) < 2:
            segments[0].zone_type = "cold_zone"
            return segments
        
        mean_density = np.mean(densities)
        median_density = np.median(densities)
        mad = np.median(np.abs(densities - median_density))
        
        threshold = mean_density + 1.5 * mad
        
        for segment in segments:
            if segment.density > threshold:
                segment.zone_type = "hot_zone"
            else:
                segment.zone_type = "cold_zone"
        
        hot_count = sum(1 for s in segments if s.zone_type == "hot_zone")
        cold_count = sum(1 for s in segments if s.zone_type == "cold_zone")
        logger.info(f"Zone 标记完成：{hot_count} 个热区，{cold_count} 个冷区")
        
        return segments

