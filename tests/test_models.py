"""规范模型不变量：冻结不可变 + 字段校验（VideoMeta/DanmakuItem/SocialVariables）。"""

import dataclasses
from datetime import datetime

import pytest
from pydantic import ValidationError

from danmaku_analyzer.crawler import DanmakuItem, VideoMeta
from danmaku_analyzer.social_variables import SocialVariables


def _meta(**overrides):
    base = dict(bvid="BV1xx", title="t", tname="游戏", pubdate=datetime(2025, 1, 1))
    base.update(overrides)
    return VideoMeta(**base)


def _item(**overrides):
    base = dict(uid_hash="u1", content="前方高能", time_sec=1.0, identity_type="real_user")
    base.update(overrides)
    return DanmakuItem(**base)


class TestFrozenModels:

    def test_social_variables_immutable(self):
        with pytest.raises(dataclasses.FrozenInstanceError):
            SocialVariables(tname="游戏", tags=["a"]).tname = "音乐"

    def test_video_meta_immutable(self):
        with pytest.raises(ValidationError):
            _meta().title = "改标题"

    def test_danmaku_item_immutable(self):
        with pytest.raises(ValidationError):
            _item().content = "改写"


class TestInvariants:

    def test_video_meta_requires_non_empty_bvid(self):
        with pytest.raises(ValidationError):
            _meta(bvid="")

    def test_danmaku_item_rejects_negative_time(self):
        with pytest.raises(ValidationError):
            _item(time_sec=-1.0)

    def test_danmaku_item_accepts_zero_time(self):
        assert _item(time_sec=0.0).time_sec == 0.0
