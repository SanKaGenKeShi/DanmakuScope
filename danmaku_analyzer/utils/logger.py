"""
日志工具模块 - 统一日志配置
"""

import os
import sys
from typing import Optional
from loguru import logger
from ..config import get_settings


def setup_logger(
    log_file: Optional[str] = None,
    level: str = "INFO",
    rotation: str = "10 MB",
    retention: str = "7 days"
) -> None:
    """
    设置日志配置
    
    Args:
        log_file: 日志文件路径（可选）
        level: 日志级别
        rotation: 日志轮转大小
        retention: 日志保留时间
    """
    settings = get_settings()
    
    # 移除默认处理器
    logger.remove()
    
    # 添加控制台处理器
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=level,
        colorize=True,
    )
    
    # 添加文件处理器
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


def get_logger(name: str) -> logger.__class__:
    """
    获取日志记录器
    
    Args:
        name: 日志记录器名称
        
    Returns:
        logger: 日志记录器
    """
    return logger.bind(name=name)


# 初始化日志
setup_logger()
