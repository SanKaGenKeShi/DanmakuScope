"""
account.py 单元测试 - 凭证转换与 Cookie 提取（纯函数，无网络依赖）
"""

import asyncio
import json

import httpx
import pytest

from danmaku_analyzer.account import (
    _to_credential,
    _extract_cookies,
    logout,
    remote_logout,
    QrLoginError,
    COOKIE_KEYS,
)


def make_response(cookies: dict = None) -> httpx.Response:
    # httpx 解析 cookies 需要关联 request 实例
    resp = httpx.Response(
        200,
        headers=[("set-cookie", f"{name}={value}") for name, value in (cookies or {}).items()],
        request=httpx.Request("GET", "https://passport.bilibili.com/"),
    )
    return resp


class TestToCredential:

    def test_完整映射(self):
        credential = _to_credential({
            "SESSDATA": "sess_123",
            "bili_jct": "jct_456",
            "buvid3": "buvid_789",
            "DedeUserID": "10001",
        })
        assert credential == {
            "sessdata": "sess_123",
            "bili_jct": "jct_456",
            "buvid3": "buvid_789",
            "dedeuserid": "10001",
        }

    def test_缺失字段回退空串(self):
        credential = _to_credential({"SESSDATA": "sess_only"})
        assert credential["sessdata"] == "sess_only"
        assert credential["bili_jct"] == ""
        assert credential["buvid3"] == ""
        assert credential["dedeuserid"] == ""

    def test_空字典(self):
        credential = _to_credential({})
        assert all(v == "" for v in credential.values())


class TestExtractCookies:

    def test_从响应Cookie提取(self):
        resp = make_response({"SESSDATA": "s1", "bili_jct": "j1", "buvid3": "b1"})
        items = _extract_cookies(resp, {})
        assert items["SESSDATA"] == "s1"
        assert items["bili_jct"] == "j1"
        assert items["buvid3"] == "b1"

    def test_过滤COOKIE_KEYS之外的Cookie(self):
        resp = make_response({"SESSDATA": "s1", "unrelated_cookie": "x"})
        items = _extract_cookies(resp, {})
        assert "SESSDATA" in items
        assert "unrelated_cookie" not in items

    def test_空值Cookie被忽略(self):
        resp = make_response({"SESSDATA": ""})
        items = _extract_cookies(resp, {"url": ""})
        assert items == {}

    def test_回退到跳转URL查询参数(self):
        resp = make_response()
        data = {"url": "https://www.bilibili.com/?SESSDATA=s2&bili_jct=j2&DedeUserID=99"}
        items = _extract_cookies(resp, data)
        assert items == {"SESSDATA": "s2", "bili_jct": "j2", "DedeUserID": "99"}

    def test_响应Cookie优先于URL参数(self):
        resp = make_response({"SESSDATA": "from_cookie"})
        data = {"url": "https://www.bilibili.com/?SESSDATA=from_url"}
        items = _extract_cookies(resp, data)
        assert items["SESSDATA"] == "from_cookie"

    def test_COOKIE_KEYS包含buvid3(self):
        assert "buvid3" in COOKIE_KEYS


class TestLogout:

    def test_删除已保存凭证(self, tmp_path):
        path = tmp_path / "credential.json"
        path.write_text(json.dumps({"sessdata": "s1"}), encoding="utf-8")
        assert logout(str(path)) is True
        assert not path.exists()

    def test_无凭证文件返回False(self, tmp_path):
        assert logout(str(tmp_path / "missing.json")) is False


class _FakeLogoutClient:
    def __init__(self, resp):
        self._resp = resp

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def post(self, url, data=None):
        self.posted = (url, data)
        return self._resp


def _logout_resp(content_type: str, body: str, status_code: int = 200) -> httpx.Response:
    return httpx.Response(
        status_code,
        headers={"content-type": content_type},
        content=body.encode("utf-8"),
        request=httpx.Request("POST", "https://passport.bilibili.com/login/exit/v2"),
    )


class TestRemoteLogout:

    def test_JSON成功响应(self, monkeypatch):
        fake = _FakeLogoutClient(_logout_resp("application/json", '{"code": 0, "status": true}'))
        monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: fake)
        assert asyncio.run(remote_logout({"sessdata": "s1", "bili_jct": "j1", "buvid3": "b1"})) is True
        assert fake.posted[1]["biliCSRF"] == "j1"

    def test_重定向HTML响应视为成功(self, monkeypatch):
        fake = _FakeLogoutClient(_logout_resp("text/html", "<html></html>", status_code=302))
        monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: fake)
        assert asyncio.run(remote_logout({"sessdata": "s1", "bili_jct": "j1"})) is True

    def test_缺少bili_jct直接视为已退出(self):
        assert asyncio.run(remote_logout({"sessdata": "s1"})) is True

    def test_接口业务报错抛出(self, monkeypatch):
        fake = _FakeLogoutClient(_logout_resp("application/json", '{"code": -111, "message": "csrf 校验失败"}'))
        monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: fake)
        with pytest.raises(QrLoginError):
            asyncio.run(remote_logout({"sessdata": "s1", "bili_jct": "j1"}))
