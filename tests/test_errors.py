"""异常层次契约：统一基类 + 边界错误类型 + 输入解析中止。"""

import asyncio

import pytest

from danmaku_analyzer.account import QrLoginError
from danmaku_analyzer.errors import DanmakuScopeError, InputParseError
from danmaku_analyzer.pipeline import _stage_resolve_input


def _noop_progress(stage, message):
    pass


class TestErrorHierarchy:

    def test_input_parse_error_is_base_and_value_error(self):
        assert issubclass(InputParseError, DanmakuScopeError)
        assert issubclass(InputParseError, ValueError)

    def test_qr_login_error_unified_under_base(self):
        assert issubclass(QrLoginError, DanmakuScopeError)

    def test_base_catches_input_parse_error(self):
        with pytest.raises(DanmakuScopeError):
            raise InputParseError("无法解析输入: xxx")


class TestInputParseAbort:

    def test_stage_resolve_input_raises_typed_error(self):
        with pytest.raises(InputParseError):
            asyncio.run(_stage_resolve_input("无效输入@@@", _noop_progress))

    def test_typed_error_still_matches_value_error(self):
        # 双继承保证既有 pytest.raises(ValueError) 契约不破
        with pytest.raises(ValueError):
            asyncio.run(_stage_resolve_input("无效输入@@@", _noop_progress))
