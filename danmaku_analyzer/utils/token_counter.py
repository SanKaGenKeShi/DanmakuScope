"""
Token 计数器模块 - 计算文本的 token 数量
"""

import tiktoken
from typing import Optional
from functools import lru_cache

from .logger import get_logger

logger = get_logger(__name__)


@lru_cache(maxsize=128)
def get_encoder(model: str = "gpt-4o-mini") -> tiktoken.Encoding:
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        logger.warning(f"未找到模型 {model} 的编码器，使用 cl100k_base")
        return tiktoken.get_encoding("cl100k_base")


def count_tokens(
    text: str, 
    model: str = "gpt-4o-mini"
) -> int:
    if not text:
        return 0
    
    try:
        encoder = get_encoder(model)
        tokens = encoder.encode(text)
        return len(tokens)
    except Exception as e:
        logger.warning(f"Token 计数失败: {e}，使用字符数估算")
        # 保底：中文字符约 0.5 token/字，其余字符约 0.3 token/字符
        chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        other_chars = len(text) - chinese_chars
        return max(1, int(chinese_chars * 0.5 + other_chars * 0.3))


def count_tokens_batch(
    texts: list[str], 
    model: str = "gpt-4o-mini"
) -> list[int]:
    return [count_tokens(text, model) for text in texts]


def truncate_to_tokens(
    text: str, 
    max_tokens: int, 
    model: str = "gpt-4o-mini"
) -> str:
    if not text:
        return ""
    
    try:
        encoder = get_encoder(model)
        tokens = encoder.encode(text)
        
        if len(tokens) <= max_tokens:
            return text
        
        truncated_tokens = tokens[:max_tokens]
        truncated_text = encoder.decode(truncated_tokens)
        
        logger.info(f"文本截断：{len(tokens)} -> {max_tokens} tokens")
        return truncated_text
        
    except Exception as e:
        logger.warning(f"Token 截断失败: {e}，使用字符截断")
        # 保底：按字符截断（中文1字符约2token，英文1单词约1token）
        estimated_tokens = 0
        truncated_chars = []
        
        for char in text:
            if '\u4e00' <= char <= '\u9fff':
                estimated_tokens += 2
            else:
                estimated_tokens += 1
            
            if estimated_tokens > max_tokens:
                break
            
            truncated_chars.append(char)
        
        return "".join(truncated_chars)
