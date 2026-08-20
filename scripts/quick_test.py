import asyncio
import sys
import os

# 添加项目根目录到 sys.path（支持包导入）
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from danmaku_analyzer.crawler import BilibiliCrawler
from danmaku_analyzer.utils.logger import setup_logger

async def test():
    setup_logger(level='WARNING')
    crawler = BilibiliCrawler()
    try:
        meta, danmaku_list, source = await crawler.fetch_all('BV1uu4y1s7TB')
        print(f'标题: {meta.title}')
        print(f'分区: {meta.tname}')
        print(f'标签: {meta.tags[:5]}')
        print(f'弹幕数: {len(danmaku_list)}（来源: {source}）')
        print(f'播放量: {meta.view_count}')
        print(f'点赞: {meta.like_count}')
        
        print('\n前5条弹幕:')
        for i, d in enumerate(danmaku_list[:5]):
            print(f'  {i+1}. [{d.time_sec:.1f}s] {d.content}')
    except Exception as e:
        print(f'错误: {e}')

asyncio.run(test())
