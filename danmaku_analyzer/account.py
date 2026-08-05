"""
账号模块 - B站二维码登录与凭证管理
"""

import asyncio
import json
import os
import time
from typing import Callable, Optional, TYPE_CHECKING
from urllib.parse import parse_qsl, urlparse

import httpx

from .config import get_settings
from .utils.logger import get_logger

if TYPE_CHECKING:
    from bilibili_api import Credential

logger = get_logger(__name__)

QR_GENERATE_API = "https://passport.bilibili.com/x/passport-login/web/qrcode/generate"
QR_POLL_API = "https://passport.bilibili.com/x/passport-login/web/qrcode/poll"
LOGIN_INFO_API = "https://api.bilibili.com/x/web-interface/nav"
BUVID_SPI_API = "https://api.bilibili.com/x/frontend/finger/spi"

QR_POLL_INTERVAL = 2.0
QR_LOGIN_TIMEOUT = 180.0
QR_UNSCANNED = 86101
QR_SCANNED = 86090
QR_EXPIRED = 86038

COOKIE_KEYS = ("SESSDATA", "bili_jct", "buvid3", "DedeUserID", "DedeUserID__ckMd5", "sid")

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Referer": "https://www.bilibili.com/",
}


class QrLoginError(Exception):
    """二维码登录失败"""


def default_credential_path() -> str:
    return os.path.join(get_settings().DATA_ROOT, "credential.json")


def credential_from_file(path: str) -> Optional["Credential"]:
    """JSON 凭证文件 → Credential（字段：sessdata/bili_jct/buvid3），失败返回 None"""
    from bilibili_api import Credential
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        sessdata = data.get("sessdata") or data.get("SESSDATA", "")
        bili_jct = data.get("bili_jct") or data.get("BILIBILI_JCT", "")
        buvid3 = data.get("buvid3") or data.get("BILIBILI_BUVID3", "")
        if not sessdata:
            logger.warning(f"凭证文件缺少 sessdata 字段: {path}")
            return None
        return Credential(sessdata=sessdata, bili_jct=bili_jct, buvid3=buvid3)
    except Exception as e:
        logger.error(f"凭证文件加载失败: {path} - {e}")
        return None


def resolve_credential(credential_file: Optional[str] = None) -> tuple[Optional["Credential"], str]:
    """凭证解析唯一入口，三级回退：指定文件 → 登录保存的默认文件 → Settings 环境变量

    返回 (credential, source)，source ∈ {'file', 'login', 'settings', ''}
    """
    from bilibili_api import Credential

    if credential_file:
        credential = credential_from_file(credential_file)
        if credential:
            return credential, "file"

    default_path = default_credential_path()
    if os.path.exists(default_path):
        credential = credential_from_file(default_path)
        if credential:
            return credential, "login"

    settings = get_settings()
    logger.info(f"检查凭证: SESSDATA={'有' if settings.BILIBILI_SESSDATA else '无'}")
    if settings.BILIBILI_SESSDATA:
        return Credential(
            sessdata=settings.BILIBILI_SESSDATA,
            bili_jct=settings.BILIBILI_JCT,
            buvid3=settings.BILIBILI_BUVID3,
        ), "settings"
    return None, ""


async def qr_login(
    status_callback: Optional[Callable[[str, str], None]] = None,
    timeout: float = QR_LOGIN_TIMEOUT,
) -> dict:
    """二维码登录：生成 → 轮询扫码 → 提取 Cookie，返回凭证字典"""
    async with httpx.AsyncClient(headers=DEFAULT_HEADERS, timeout=10) as client:
        resp = await client.get(QR_GENERATE_API)
        resp.raise_for_status()
        payload = resp.json()
        if payload.get("code") != 0:
            raise QrLoginError(payload.get("message") or "获取二维码失败")

        data = payload.get("data") or {}
        login_url = str(data.get("url") or "").strip()
        qr_key = str(data.get("qrcode_key") or "").strip()
        if not login_url or not qr_key:
            raise QrLoginError("二维码接口返回数据不完整")

        if status_callback:
            status_callback("qr_ready", login_url)

        deadline = time.monotonic() + timeout
        scanned_notified = False
        while True:
            await asyncio.sleep(QR_POLL_INTERVAL)
            if time.monotonic() > deadline:
                raise QrLoginError("登录超时，请重新执行 login")

            resp = await client.get(QR_POLL_API, params={"qrcode_key": qr_key})
            resp.raise_for_status()
            data = resp.json().get("data") or {}
            code = int(data.get("code", -1))

            if code == QR_UNSCANNED:
                continue
            if code == QR_SCANNED:
                if not scanned_notified and status_callback:
                    status_callback("scanned", "已扫码，请在手机上确认登录")
                    scanned_notified = True
                continue
            if code == QR_EXPIRED:
                raise QrLoginError("二维码已过期，请重新执行 login")
            if code != 0:
                continue

            cookies = _extract_cookies(resp, data)
            if not cookies.get("SESSDATA"):
                raise QrLoginError("登录成功，但未能提取到 SESSDATA")
            if not cookies.get("buvid3"):
                # QR 登录响应通常不下发 buvid3 Cookie，通过指纹接口补全以保证凭证完整
                cookies["buvid3"] = await _fetch_buvid3(client)
            logger.info(f"二维码登录成功，DedeUserID: {cookies.get('DedeUserID', '未知')}")
            return _to_credential(cookies)


async def _fetch_buvid3(client: httpx.AsyncClient) -> str:
    """通过指纹接口获取 buvid3，失败时返回空串（不阻断登录）"""
    try:
        resp = await client.get(BUVID_SPI_API)
        resp.raise_for_status()
        buvid3 = str((resp.json().get("data") or {}).get("b_3") or "")
        if buvid3:
            return buvid3
        logger.warning("buvid3 指纹接口未返回 b_3 字段，凭证中 buvid3 为空")
    except Exception as e:
        logger.warning(f"buvid3 获取失败，凭证中 buvid3 为空: {e}")
    return ""


def _extract_cookies(resp: httpx.Response, data: dict) -> dict:
    """优先从响应 Cookie 提取，回退到跳转 URL 的查询参数"""
    items = {}
    for name, value in resp.cookies.items():
        if name in COOKIE_KEYS and value:
            items[name] = value
    if not items:
        success_url = str(data.get("url") or "")
        for name, value in parse_qsl(urlparse(success_url).query):
            if name in COOKIE_KEYS and value:
                items[name] = value
    return items


def _to_credential(cookies: dict) -> dict:
    """Cookie 字段 → 凭证文件格式（与 credential_from_file 兼容）"""
    return {
        "sessdata": cookies.get("SESSDATA", ""),
        "bili_jct": cookies.get("bili_jct", ""),
        "buvid3": cookies.get("buvid3", ""),
        "dedeuserid": cookies.get("DedeUserID", ""),
    }


def save_credential(credential: dict, path: Optional[str] = None) -> str:
    path = path or default_credential_path()
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(credential, f, ensure_ascii=False, indent=2)
    return path


def load_credential(path: Optional[str] = None) -> Optional[dict]:
    path = path or default_credential_path()
    if not os.path.exists(path):
        return None
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if data.get("sessdata"):
            return data
        logger.warning(f"凭证文件缺少 sessdata 字段: {path}")
    except Exception as e:
        logger.warning(f"凭证文件读取失败: {path} - {e}")
    return None


async def fetch_account_info(sessdata: str) -> dict:
    """通过 nav 接口校验凭证有效性并返回账号信息"""
    headers = dict(DEFAULT_HEADERS)
    headers["Cookie"] = f"SESSDATA={sessdata}"
    async with httpx.AsyncClient(headers=headers, timeout=10) as client:
        resp = await client.get(LOGIN_INFO_API)
        resp.raise_for_status()
        payload = resp.json()
    data = payload.get("data") or {}
    if payload.get("code") == -101 or not data.get("isLogin"):
        return {"is_login": False, "uname": "", "mid": ""}
    return {
        "is_login": True,
        "uname": str(data.get("uname") or "").strip(),
        "mid": str(data.get("mid") or ""),
    }
