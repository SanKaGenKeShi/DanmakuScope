"""
社会变量锚定 - 仅提取官方分区 tname 作为硬分组键
tags 不聚类，原样传给 Prompt 构建器作上下文
"""

from typing import List
from dataclasses import dataclass

from .crawler import VideoMeta


@dataclass
class SocialVariables:
    tname: str   # 官方一级分区，唯一硬分组变量
    tags: List[str]  # 用户标签，仅做 LLM 上下文

    def to_dict(self) -> dict:
        return {"tname": self.tname, "tags": self.tags}


class SocialVariableExtractor:

    def extract(self, video_meta: VideoMeta) -> SocialVariables:
        return SocialVariables(tname=video_meta.tname, tags=video_meta.tags)

    def extract_batch(self, video_metas: List[VideoMeta]) -> List[SocialVariables]:
        return [self.extract(m) for m in video_metas]
