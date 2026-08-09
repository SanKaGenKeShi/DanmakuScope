"""
输入解析器单元测试
"""

import asyncio
from danmaku_analyzer.utils.input_parser import InputParser, InputType, ParsedInput


class TestInputParser:
    """输入解析器测试类"""
    
    def setup_method(self):
        """测试前准备"""
        self.parser = InputParser()
    
    def test_parse_bv(self):
        """测试 BV 号解析"""
        # 标准 BV 号
        result = self.parser.parse("BV1xx411c7mD")
        assert result.input_type == InputType.BV
        assert result.bvid == "BV1xx411c7mD"
        
        # 小写 BV 号
        result = self.parser.parse("bv1xx411c7mD")
        assert result.input_type == InputType.BV
        assert result.bvid == "BV1xx411c7mD"
        
        # 无前缀
        result = self.parser.parse("1xx411c7mD")
        assert result.input_type == InputType.UNKNOWN
    
    def test_parse_av(self):
        """测试 AV 号解析"""
        # 标准 AV 号
        result = self.parser.parse("av12345")
        assert result.input_type == InputType.AV
        assert result.avid == 12345
        
        # 大写 AV 号
        result = self.parser.parse("AV12345")
        assert result.input_type == InputType.AV
        assert result.avid == 12345
        
        # 无前缀数字
        result = self.parser.parse("12345")
        assert result.input_type == InputType.UNKNOWN
    
    def test_parse_url(self):
        """测试 URL 解析"""
        # 标准视频链接
        result = self.parser.parse("https://www.bilibili.com/video/BV1xx411c7mD")
        assert result.input_type == InputType.URL
        assert result.bvid == "BV1xx411c7mD"
        
        # AV 号链接
        result = self.parser.parse("https://www.bilibili.com/video/av12345")
        assert result.input_type == InputType.URL
        assert result.avid == 12345
        
        # 嵌入链接
        result = self.parser.parse("https://player.bilibili.com/player.html?bvid=BV1xx411c7mD")
        assert result.input_type == InputType.URL
        assert result.bvid == "BV1xx411c7mD"
    
    def test_parse_empty(self):
        """测试空输入"""
        result = self.parser.parse("")
        assert result.input_type == InputType.UNKNOWN
        
        result = self.parser.parse("   ")
        assert result.input_type == InputType.UNKNOWN
    
    def test_parse_invalid(self):
        """测试无效输入"""
        result = self.parser.parse("invalid_input")
        assert result.input_type == InputType.UNKNOWN
        
        result = self.parser.parse("https://www.example.com")
        assert result.input_type == InputType.UNKNOWN
    
    def test_parsed_input_to_dict(self):
        """测试 ParsedInput 转字典"""
        parsed = ParsedInput(
            input_type=InputType.BV,
            bvid="BV1XX411C7MD",
            avid=None,
            original_input="BV1xx411c7mD"
        )
        
        result = parsed.to_dict()
        assert result["input_type"] == "bv"
        assert result["bvid"] == "BV1XX411C7MD"
        assert result["avid"] is None
        assert result["original_input"] == "BV1xx411c7mD"


class TestInputParserEdgeCases:
    """边界情况测试"""
    
    def setup_method(self):
        """测试前准备"""
        self.parser = InputParser()
    
    def test_bv_with_extra_spaces(self):
        """测试带空格的 BV 号"""
        result = self.parser.parse("  BV1xx411c7mD  ")
        assert result.input_type == InputType.BV
        assert result.bvid == "BV1xx411c7mD"
    
    def test_url_with_query_params(self):
        """测试带查询参数的 URL"""
        result = self.parser.parse(
            "https://www.bilibili.com/video/BV1xx411c7mD?spm_id_from=333.999.0.0"
        )
        assert result.input_type == InputType.URL
        assert result.bvid == "BV1xx411c7mD"
    
    def test_av_with_large_number(self):
        """测试大数字 AV 号"""
        result = self.parser.parse("av999999999")
        assert result.input_type == InputType.AV
        assert result.avid == 999999999
