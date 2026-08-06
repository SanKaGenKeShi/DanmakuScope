"""
日志工具模块 - 统一日志配置
"""

import os
import sys
from typing import Optional
from loguru import logger

try:
    from loguru import Logger
except ImportError:  # 旧版 loguru 未导出 Logger 类
    Logger = type(logger)

from ..config import get_settings


def setup_logger(
    log_file: Optional[str] = None,
    level: str = "INFO",
    rotation: str = "10 MB",
    retention: str = "7 days"
) -> None:
    settings = get_settings()
    
    logger.remove()
    
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=level,
        colorize=True,
    )
    
    if log_file is None:
        raw_log_dir = settings.LOG_DIR
        # 相对路径基于 DATA_ROOT 解析（用户可写目录）
        log_dir = settings.resolve_data_path(raw_log_dir)
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, "danmaku_analyzer.log")
    
    logger.add(
        log_file,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level=level,
        rotation=rotation,
        retention=retention,
        encoding="utf-8",
    )


def get_logger(name: str) -> Logger:
    """获取绑定模块名的日志记录器"""
    return logger.bind(name=name)
