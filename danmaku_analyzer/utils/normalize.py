"""文本归一化：分析前统一 Unicode 规范形式，保障分词与硬统计的可复现性。"""

import re
import unicodedata

_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def normalize_text(raw: str) -> str:
    """Unicode NFC 归一 + 换行统一（\r\n/\r → \n）+ 控制字符去除（保留 \n）+ 首尾空白去除。"""
    text = unicodedata.normalize("NFC", raw)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return _CONTROL_CHARS.sub("", text).strip()
