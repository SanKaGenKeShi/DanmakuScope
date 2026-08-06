"""
Token 计数器模块 - 计算文本的 token 数量
LLM 配置可能是任意厂商模型（tiktoken 无对应编码器），统一用 cl100k_base 作通用估算
"""

import tiktoken
from functools import lru_cache

from .logger import get_logger

logger = get_logger(__name__)


@lru_cache(maxsize=1)
def _get_encoder() -> tiktoken.Encoding:
    return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    if not text:
        return 0
    
    try:
        return len(_get_encoder().encode(text))
    except Exception as e:
        logger.warning(f"Token 计数失败: {e}，使用字符数估算")
        # 保底：中文字符约 0.5 token/字，其余字符约 0.3 token/字符
        chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        other_chars = len(text) - chinese_chars
        return max(1, int(chinese_chars * 0.5 + other_chars * 0.3))
