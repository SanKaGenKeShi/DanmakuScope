"""
爬虫模块 - 抓取B站弹幕及视频元数据
支持 BV号、AV号、完整链接解析
"""

import asyncio
import hashlib
import re
from datetime import datetime
from typing import List, Optional, Literal
from pydantic import BaseModel, Field, ConfigDict, field_serializer

from bilibili_api import video, Credential, HEADERS
import httpx

from .config import get_settings
from .utils.logger import get_logger

logger = get_logger(__name__)

# B站 tid → 一级分区名称映射（当 API 返回 tname 为空时兜底）
# 包含所有一级分区及其子分区 ID
TID_TO_TNAME: dict[int, str] = {
    # 动画
    1: "动画", 24: "动画", 25: "动画", 47: "动画", 210: "动画", 86: "动画", 253: "动画",
    # 番剧
    13: "番剧", 33: "番剧", 32: "番剧", 51: "番剧",
    # 国创
    167: "国创", 153: "国创", 168: "国创", 169: "国创", 195: "国创", 170: "国创",
    # 音乐
    3: "音乐", 31: "音乐", 30: "音乐", 59: "音乐", 54: "音乐", 28: "音乐",
    198: "音乐", 29: "音乐", 193: "音乐", 243: "音乐", 244: "音乐",
    # 舞蹈
    129: "舞蹈", 20: "舞蹈", 199: "舞蹈", 200: "舞蹈", 154: "舞蹈", 156: "舞蹈",
    # 游戏
    4: "游戏", 17: "游戏", 171: "游戏", 172: "游戏", 65: "游戏", 173: "游戏",
    121: "游戏", 136: "游戏", 19: "游戏",
    # 知识
    36: "知识", 201: "知识", 124: "知识", 228: "知识", 207: "知识", 208: "知识",
    209: "知识", 229: "知识",
    # 科技
    188: "科技", 95: "科技", 230: "科技", 231: "科技", 232: "科技", 233: "科技",
    # 运动
    234: "运动", 235: "运动", 249: "运动", 164: "运动", 236: "运动", 237: "运动", 238: "运动",
    # 汽车
    223: "汽车", 245: "汽车", 246: "汽车", 247: "汽车", 248: "汽车", 176: "汽车", 224: "汽车",
    # 生活
    160: "生活", 138: "生活", 239: "生活", 161: "生活", 162: "生活", 21: "生活",
    # 美食
    211: "美食", 76: "美食", 212: "美食", 213: "美食", 214: "美食", 215: "美食",
    # 动物圈
    217: "动物圈", 218: "动物圈", 219: "动物圈", 222: "动物圈", 221: "动物圈", 220: "动物圈", 75: "动物圈",
    # 鬼畜
    119: "鬼畜", 22: "鬼畜", 26: "鬼畜", 126: "鬼畜", 216: "鬼畜", 127: "鬼畜",
    # 时尚
    155: "时尚", 157: "时尚", 252: "时尚", 158: "时尚", 159: "时尚",
    # 娱乐
    5: "娱乐", 71: "娱乐", 241: "娱乐", 242: "娱乐", 137: "娱乐",
    # 影视
    181: "影视", 182: "影视", 183: "影视", 85: "影视", 184: "影视",
    # 纪录片
    177: "纪录片", 37: "纪录片", 178: "纪录片", 179: "纪录片",
    # 电影
    23: "电影", 147: "电影", 145: "电影", 146: "电影", 83: "电影",
    # 电视剧
    11: "电视剧", 185: "电视剧", 187: "电视剧",
}


class VideoMeta(BaseModel):
    """视频元数据模型"""
    model_config = ConfigDict(strict=False)
    
    bvid: str = Field(description="BV号")
    title: str = Field(description="视频标题")
    tname: str = Field(description="官方一级分区 - 唯一硬分组变量")
    tags: List[str] = Field(default_factory=list, description="用户自定义标签列表 - 仅用作 LLM 提示上下文，不聚类")
    pubdate: datetime = Field(description="发布时间")
    view_count: int = Field(default=0, description="播放量")
    like_count: int = Field(default=0, description="点赞数")
    
    @field_serializer('pubdate')
    def serialize_pubdate(self, v: datetime) -> str:
        return v.isoformat()


class DanmakuItem(BaseModel):
    """弹幕数据模型"""
    model_config = ConfigDict(strict=False)
    
    uid_hash: str = Field(description="用户UID哈希，若为 0 则标记为 unknown_device")
    content: str = Field(description="弹幕内容")
    time_sec: float = Field(description="弹幕出现时间（秒）")
    identity_type: Literal["real_user", "unknown_device"] = Field(description="身份类型")
    
    @field_serializer('time_sec')
    def serialize_time_sec(self, v: float) -> float:
        return round(v, 3)


class BilibiliCrawler:
    
    def __init__(self, credential: Optional[Credential] = None):
        self.settings = get_settings()
        self.credential = credential
    
    async def fetch_video_metadata(self, bvid: str) -> VideoMeta:
        """获取视频元数据（分区名优先用 API tname，为空则 tid 映射兖底）"""
        logger.info(f"开始获取视频元数据: {bvid}")
        
        try:
            v = video.Video(bvid=bvid, credential=self.credential)
            info = await v.get_info()
            
            # 提取标签
            tags = []
            try:
                tag_info = await v.get_tags()
                tags = [tag["tag_name"] for tag in tag_info]
            except Exception as e:
                logger.warning(f"获取标签失败: {e}")
            
            # 提取分区名称：优先用 API 返回的 tname，为空则通过 tid 映射兜底
            tname = info.get("tname", "") or ""
            if not tname:
                tid = info.get("tid", 0)
                tname = TID_TO_TNAME.get(tid, "")
                if tname:
                    logger.info(f"tname 为空，通过 tid={tid} 映射得到分区: {tname}")
                else:
                    logger.warning(f"tname 为空且 tid={tid} 无映射，分区未知")
            
            # 构建元数据
            meta = VideoMeta(
                bvid=bvid,
                title=info.get("title", ""),
                tname=tname,
                tags=tags,
                pubdate=datetime.fromtimestamp(info.get("pubdate", 0)),
                view_count=info.get("stat", {}).get("view", 0),
                like_count=info.get("stat", {}).get("like", 0),
            )
            
            logger.info(f"视频元数据获取成功: {meta.title} (分区: {meta.tname})")
            return meta
            
        except Exception as e:
            logger.error(f"获取视频元数据失败: {e}")
            raise
    
    async def fetch_danmaku(self, bvid: str, cid: Optional[int] = None) -> List[DanmakuItem]:
        """protobuf 分段接口拉取全量弹幕，失败回退 XML"""
        logger.info(f"开始获取弹幕（protobuf 分段接口），BVID: {bvid}")
        
        try:
            v = video.Video(bvid=bvid, credential=self.credential)
            
            # 获取弹幕列表（自动按 6 分钟分段拉取全部）
            raw_danmakus = await v.get_danmakus(cid=cid)
            
            # 转换为内部数据模型
            danmaku_list = self._convert_danmakus(raw_danmakus)
            
            logger.info(f"弹幕获取成功，共 {len(danmaku_list)} 条")
            return danmaku_list
            
        except Exception as e:
            logger.warning(f"protobuf 接口获取弹幕失败: {e}，回退到 XML 接口")
            # 回退：使用旧 XML 接口（有数量上限，但作为兜底）
            return await self._fetch_danmaku_xml_fallback(bvid, cid)
    
    def _convert_danmakus(self, raw_danmakus) -> List[DanmakuItem]:
        """bilibili-api Danmaku 对象 → 内部 DanmakuItem"""
        danmaku_list = []
        
        for dm in raw_danmakus:
            # crc32_id 是发送者 UID 的 CRC32 哈希
            crc32_id = getattr(dm, 'crc32_id', '') or ''
            uid = getattr(dm, 'uid', -1)
            
            if crc32_id and crc32_id != '0' and crc32_id != '':
                uid_hash = crc32_id
                identity_type = "real_user"
            elif uid and uid > 0:
                uid_hash = hashlib.md5(str(uid).encode()).hexdigest()[:16]
                identity_type = "real_user"
            else:
                uid_hash = "unknown_device"
                identity_type = "unknown_device"
            
            content = (getattr(dm, 'text', '') or '').strip()
            if not content:
                continue
            
            time_sec = getattr(dm, 'dm_time', 0.0) or 0.0
            
            danmaku = DanmakuItem(
                uid_hash=uid_hash,
                content=content,
                time_sec=time_sec,
                identity_type=identity_type,
            )
            danmaku_list.append(danmaku)
        
        return danmaku_list
    
    async def _fetch_danmaku_xml_fallback(self, bvid: str, cid: Optional[int] = None) -> List[DanmakuItem]:
        """XML 接口兖底（上限约 1000 条）"""
        if cid is None:
            cid = await self.fetch_video_cid(bvid)
        
        logger.info(f"使用 XML 兜底接口获取弹幕，CID: {cid}")
        url = f"https://comment.bilibili.com/{cid}.xml"
        
        async with httpx.AsyncClient() as client:
            response = await client.get(url, headers=HEADERS)
            response.raise_for_status()
        
        danmaku_list = self._parse_danmaku_xml(response.text)
        logger.info(f"XML 兜底获取完成，共 {len(danmaku_list)} 条（注意：此接口有数量上限）")
        return danmaku_list
    
    def _parse_danmaku_xml(self, xml_content: str) -> List[DanmakuItem]:
        import xml.etree.ElementTree as ET
        
        danmaku_list = []
        
        try:
            root = ET.fromstring(xml_content)
            
            for d_elem in root.findall(".//d"):
                # 解析属性
                attrs = d_elem.get("p", "").split(",")
                if len(attrs) < 8:
                    continue
                
                time_sec = float(attrs[0])
                uid_str = attrs[6]  # 用户ID哈希
                
                # 处理UID
                if uid_str == "0" or uid_str == "":
                    uid_hash = "unknown_device"
                    identity_type = "unknown_device"
                else:
                    uid_hash = hashlib.md5(uid_str.encode()).hexdigest()[:16]
                    identity_type = "real_user"
                
                content = d_elem.text or ""
                
                danmaku = DanmakuItem(
                    uid_hash=uid_hash,
                    content=content.strip(),
                    time_sec=time_sec,
                    identity_type=identity_type,
                )
                danmaku_list.append(danmaku)
            
        except Exception as e:
            logger.error(f"解析弹幕XML失败: {e}")
            raise
        
        return danmaku_list
    
    async def fetch_video_cid(self, bvid: str) -> int:
        try:
            v = video.Video(bvid=bvid, credential=self.credential)
            info = await v.get_info()
            cid = info.get("cid", 0)
            
            if cid == 0:
                # 尝试从分P信息获取
                pages = info.get("pages", [])
                if pages:
                    cid = pages[0].get("cid", 0)
            
            return cid
            
        except Exception as e:
            logger.error(f"获取视频CID失败: {e}")
            raise
    
    async def fetch_all(self, bvid: str) -> tuple[VideoMeta, List[DanmakuItem]]:
        meta = await self.fetch_video_metadata(bvid)
        danmaku_list = await self.fetch_danmaku(bvid)
        return meta, danmaku_list

