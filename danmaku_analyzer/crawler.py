"""
爬虫模块 - 抓取B站弹幕及视频元数据
支持 BV号、AV号、完整链接解析
"""

import hashlib
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import List, Optional, Literal
from pydantic import BaseModel, Field, ConfigDict, field_serializer

from bilibili_api import video, Credential, HEADERS
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from .partitions import TID_TO_TNAME
from .utils.logger import get_logger

logger = get_logger(__name__)


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
    cid: int = Field(default=0, description="主分P cid（内部透传，避免弹幕兜底路径重复请求 get_info）")
    
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
        self.credential = credential
    
    async def fetch_video_metadata(self, bvid: str) -> VideoMeta:
        """获取视频元数据（分区名优先用 API tname，为空则 tid 映射兜底）"""
        logger.info(f"开始获取视频元数据: {bvid}")
        
        try:
            v = video.Video(bvid=bvid, credential=self.credential)
            info = await v.get_info()
            
            tags = []
            try:
                tag_info = await v.get_tags()
                tags = [tag["tag_name"] for tag in tag_info]
            except Exception as e:
                logger.warning(f"获取标签失败: {e}")
            
            tname = info.get("tname", "") or ""
            if not tname:
                tid = info.get("tid", 0)
                tname = TID_TO_TNAME.get(tid, "")
                if tname:
                    logger.info(f"tname 为空，通过 tid={tid} 映射得到分区: {tname}")
                else:
                    logger.warning(f"tname 为空且 tid={tid} 无映射，分区未知")

            cid = info.get("cid", 0)
            if not cid:
                pages = info.get("pages", [])
                if pages:
                    cid = pages[0].get("cid", 0)

            meta = VideoMeta(
                bvid=bvid,
                title=info.get("title", ""),
                tname=tname,
                tags=tags,
                pubdate=datetime.fromtimestamp(info.get("pubdate", 0)),
                view_count=info.get("stat", {}).get("view", 0),
                like_count=info.get("stat", {}).get("like", 0),
                cid=cid,
            )
            
            logger.info(f"视频元数据获取成功: {meta.title} (分区: {meta.tname})")
            return meta
            
        except Exception as e:
            logger.error(f"获取视频元数据失败: {e}")
            raise
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def fetch_danmaku(self, bvid: str, cid: Optional[int] = None) -> List[DanmakuItem]:
        """protobuf 分段接口拉取全量弹幕，失败回退 XML；两路均失败时 tenacity 指数退避重试"""
        logger.info(f"开始获取弹幕（protobuf 分段接口），BVID: {bvid}")
        
        try:
            v = video.Video(bvid=bvid, credential=self.credential)
            
            raw_danmakus = await v.get_danmakus(cid=cid)
            
            danmaku_list = self._convert_danmakus(raw_danmakus)
            
            logger.info(f"弹幕获取成功，共 {len(danmaku_list)} 条")
            return danmaku_list
            
        except Exception as e:
            logger.warning(f"protobuf 接口获取弹幕失败: {e}，回退到 XML 接口")
            return await self._fetch_danmaku_xml_fallback(bvid, cid)
    
    def _convert_danmakus(self, raw_danmakus) -> List[DanmakuItem]:
        """bilibili-api Danmaku 对象 → 内部 DanmakuItem"""
        danmaku_list = []
        
        for dm in raw_danmakus:
            # crc32_id 是发送者 UID 的 CRC32 哈希
            crc32_id = getattr(dm, 'crc32_id', '') or ''
            uid = getattr(dm, 'uid', -1)
            
            if crc32_id and crc32_id != '0':
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
        """XML 接口兜底（上限约 1000 条）"""
        if cid is None:
            cid = await self._fetch_video_cid(bvid)
        
        logger.info(f"使用 XML 兜底接口获取弹幕，CID: {cid}")
        url = f"https://comment.bilibili.com/{cid}.xml"
        
        async with httpx.AsyncClient() as client:
            response = await client.get(url, headers=HEADERS)
            response.raise_for_status()
        
        danmaku_list = self._parse_danmaku_xml(response.text)
        logger.info(f"XML 兜底获取完成，共 {len(danmaku_list)} 条（注意：此接口有数量上限）")
        return danmaku_list
    
    def _parse_danmaku_xml(self, xml_content: str) -> List[DanmakuItem]:
        danmaku_list = []
        skipped_count = 0
        
        try:
            root = ET.fromstring(xml_content)
            
            for d_elem in root.findall(".//d"):
                attrs = d_elem.get("p", "").split(",")
                if len(attrs) < 8:
                    skipped_count += 1
                    continue
                
                time_sec = float(attrs[0])
                uid_str = attrs[6]  # 用户ID哈希
                
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
        
        if skipped_count:
            logger.warning(f"XML 弹幕解析跳过 {skipped_count} 条属性不完整的记录（p 属性字段 < 8）")
        
        return danmaku_list
    
    async def _fetch_video_cid(self, bvid: str) -> int:
        try:
            v = video.Video(bvid=bvid, credential=self.credential)
            info = await v.get_info()
            cid = info.get("cid", 0)
            
            if cid == 0:
                pages = info.get("pages", [])
                if pages:
                    cid = pages[0].get("cid", 0)
            
            return cid
            
        except Exception as e:
            logger.error(f"获取视频CID失败: {e}")
            raise
    
    async def fetch_all(self, bvid: str) -> tuple[VideoMeta, List[DanmakuItem]]:
        meta = await self.fetch_video_metadata(bvid)
        # 透传元数据阶段已取得的 cid，XML 兜底路径无需再次 get_info
        danmaku_list = await self.fetch_danmaku(bvid, cid=meta.cid or None)
        return meta, danmaku_list

