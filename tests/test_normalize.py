"""文本归一化单测：NFC 规范组合、换行统一、控制字符去除与首尾空白。"""

from danmaku_analyzer.utils.normalize import normalize_text


class TestNormalizeText:

    def test_nfc_composes_combining_sequence(self):
        # e + 组合尖音符 → NFC 规范组合为单码位 é
        assert normalize_text("e\u0301") == "\u00e9"

    def test_idempotent_on_normalized_text(self):
        text = "前方高能 yyds 233"
        once = normalize_text(text)
        assert once == text
        assert normalize_text(once) == once

    def test_full_width_preserved_by_nfc(self):
        # NFC 只做规范组合，不做兼容折叠（全角属 NFKC 范畴），故保持不变
        assert normalize_text("ａｂｃ") == "ａｂｃ"

    def test_crlf_and_cr_unified_to_lf(self):
        assert normalize_text("a\r\nb\rc") == "a\nb\nc"

    def test_control_chars_removed_newline_kept(self):
        assert normalize_text("a\x00b\x1fc") == "abc"
        assert normalize_text("a\tb\nc") == "a\tb\nc"

    def test_surrounding_whitespace_stripped(self):
        assert normalize_text("  生草  ") == "生草"

    def test_control_only_input_becomes_empty(self):
        # 归一后为空 → 触发 crawler 的 `if not content: continue` 跳过守卫
        assert normalize_text("\x00\x01\x02") == ""
        assert normalize_text("   \r\n  ") == ""
