"""
智能输入适配器 - 支持 BV号、完整 B 站链接、AV号（自动转换）
核心正则提取逻辑，AVBV 转换使用 bilibili-api
"""

import re
from typing import Optional, Literal
from dataclasses import dataclass
from enum import Enum

from .logger import get_logger

logger = get_logger(__name__)


class InputType(Enum):
    """输入类型"""
    BV = "bv"
    AV = "av"
    URL = "url"
    UNKNOWN = "unknown"


@dataclass
class ParsedInput:
    """解析结果"""
    input_type: InputType
    bvid: Optional[str] = None
    avid: Optional[int] = None
    original_input: str = ""
    
    def to_dict(self) -> dict:
        return {
            "input_type": self.input_type.value,
            "bvid": self.bvid,
            "avid": self.avid,
            "original_input": self.original_input,
        }


class InputParser:
    """智能输入解析器"""
    
    # BV 号正则
    BV_PATTERN = re.compile(r'^BV[a-zA-Z0-9]{10}$', re.IGNORECASE)
    
    # AV 号正则
    AV_PATTERN = re.compile(r'^av(\d+)$', re.IGNORECASE)
    
    # B站链接正则
    URL_PATTERNS = [
        # 完整链接
        re.compile(r'https?://www\.bilibili\.com/video/(BV[a-zA-Z0-9]{10})', re.IGNORECASE),
        re.compile(r'https?://www\.bilibili\.com/video/(av\d+)', re.IGNORECASE),
        # 短链接
        re.compile(r'https?://b23\.tv/([a-zA-Z0-9]+)', re.IGNORECASE),
        # 嵌入链接
        re.compile(r'https?://player\.bilibili\.com/player\.html\?.*?bvid=(BV[a-zA-Z0-9]{10})', re.IGNORECASE),
        re.compile(r'https?://player\.bilibili\.com/player\.html\?.*?aid=(\d+)', re.IGNORECASE),
    ]
    
    def __init__(self):
        """初始化解析器"""
        logger.info("输入解析器初始化完成")
    
    def parse(self, input_str: str) -> ParsedInput:
        """
        解析输入
        
        Args:
            input_str: 输入字符串（BV号、AV号或URL）
            
        Returns:
            ParsedInput: 解析结果
        """
        input_str = input_str.strip()
        
        if not input_str:
            return ParsedInput(
                input_type=InputType.UNKNOWN,
                original_input=input_str,
            )
        
        # 尝试解析 BV 号
        bv_result = self._parse_bv(input_str)
        if bv_result:
            return bv_result
        
        # 尝试解析 AV 号
        av_result = self._parse_av(input_str)
        if av_result:
            return av_result
        
        # 尝试解析 URL
        url_result = self._parse_url(input_str)
        if url_result:
            return url_result
        
        # 未知格式
        logger.warning(f"无法解析输入: {input_str}")
        return ParsedInput(
            input_type=InputType.UNKNOWN,
            original_input=input_str,
        )
    
    def _parse_bv(self, input_str: str) -> Optional[ParsedInput]:
        """
        解析 BV 号
        
        Args:
            input_str: 输入字符串
            
        Returns:
            Optional[ParsedInput]: 解析结果
        """
        match = self.BV_PATTERN.match(input_str)
        if match:
            # 前缀标准化为 BV，ID 部分保留原始大小写（B站BV号大小写敏感）
            bvid = 'BV' + input_str[2:]
            
            logger.info(f"解析到 BV 号: {bvid}")
            return ParsedInput(
                input_type=InputType.BV,
                bvid=bvid,
                original_input=input_str,
            )
        return None
    
    def _parse_av(self, input_str: str) -> Optional[ParsedInput]:
        """
        解析 AV 号
        
        Args:
            input_str: 输入字符串
            
        Returns:
            Optional[ParsedInput]: 解析结果
        """
        match = self.AV_PATTERN.match(input_str)
        if match:
            avid = int(match.group(1))
            logger.info(f"解析到 AV 号: av{avid}")
            return ParsedInput(
                input_type=InputType.AV,
                avid=avid,
                original_input=input_str,
            )
        return None
    
    def _parse_url(self, input_str: str) -> Optional[ParsedInput]:
        """
        解析 URL
        
        Args:
            input_str: 输入字符串
            
        Returns:
            Optional[ParsedInput]: 解析结果
        """
        for pattern in self.URL_PATTERNS:
            match = pattern.search(input_str)
            if match:
                captured = match.group(1)
                
                # 判断是 BV 还是 AV
                if captured.upper().startswith('BV'):
                    # 前缀标准化为 BV，ID 部分保留原始大小写
                    bvid = 'BV' + captured[2:]
                    
                    logger.info(f"从 URL 解析到 BV 号: {bvid}")
                    return ParsedInput(
                        input_type=InputType.URL,
                        bvid=bvid,
                        original_input=input_str,
                    )
                elif captured.isdigit() or (captured.lower().startswith('av') and captured[2:].isdigit()):
                    # 处理 av12345 格式
                    if captured.lower().startswith('av'):
                        avid = int(captured[2:])
                    else:
                        avid = int(captured)
                    logger.info(f"从 URL 解析到 AV 号: av{avid}")
                    return ParsedInput(
                        input_type=InputType.URL,
                        avid=avid,
                        original_input=input_str,
                    )
        
        return None
    
    async def resolve_to_bvid(self, parsed_input: ParsedInput) -> str:
        """
        将解析结果转换为 BV 号
        
        Args:
            parsed_input: 解析结果
            
        Returns:
            str: BV 号
        """
        if parsed_input.bvid:
            return parsed_input.bvid
        
        if parsed_input.avid:
            # 使用 bilibili-api 转换
            try:
                from bilibili_api import video
                v = video.Video(aid=parsed_input.avid)
                info = await v.get_info()
                bvid = info.get("bvid", "")
                
                if bvid:
                    logger.info(f"AV 号转换成功: av{parsed_input.avid} -> {bvid}")
                    return bvid
                else:
                    raise ValueError(f"无法获取 BV 号: av{parsed_input.avid}")
                    
            except Exception as e:
                logger.error(f"AV 号转换失败: {e}")
                raise
        
        raise ValueError(f"无法转换为 BV 号: {parsed_input.original_input}")


# 便捷函数
def parse_input(input_str: str) -> ParsedInput:
    """解析输入的便捷函数"""
    parser = InputParser()
    return parser.parse(input_str)


async def resolve_to_bvid(input_str: str) -> str:
    """解析并转换为 BV 号的便捷函数"""
    parser = InputParser()
    parsed = parser.parse(input_str)
    return await parser.resolve_to_bvid(parsed)
