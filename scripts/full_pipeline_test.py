"""
全流程测试 - 复用 pipeline.py 核心逻辑
"""

import asyncio
import sys
import os

# 添加项目根目录到 sys.path（支持包导入）
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from danmaku_analyzer.pipeline import analyze_video
from danmaku_analyzer.utils.logger import setup_logger
from danmaku_analyzer.config import get_settings

import logging

# 设置日志级别为WARNING以减少输出
logging.basicConfig(level=logging.WARNING)


def print_progress(stage: str, message: str):
    print(f"  [{stage}] {message}")


async def full_pipeline_test():
    settings = get_settings()
    
    print("="*60)
    print("DanmakuScope - 全流程测试")
    print("="*60)
    print(f"LLM配置:")
    print(f"  复杂任务: {settings.llm.COMPLEX_LLM_MODEL} @ {settings.llm.COMPLEX_LLM_BASE_URL}")
    print(f"  简单任务: {settings.llm.SIMPLE_LLM_MODEL} @ {settings.llm.SIMPLE_LLM_BASE_URL}")
    print(f"分析策略: 频次最高的前 {settings.TOP_N} 条弹幕")
    if settings.ENABLE_LLM_ANALYSIS_REPORT:
        print(f"LLM分析报告: 已启用 (模型: {settings.llm.effective_analysis_report_model})")
    else:
        print(f"LLM分析报告: 未启用")
    print("="*60)
    
    bvid = "BV1Ha41187qw"
    
    try:
        result = await analyze_video(
            input_str=bvid,
            freq_based=True,
            progress_callback=print_progress
        )
        
        print("\n" + "="*60)
        print("✅ 全流程测试完成！")
        print("="*60)
        print(f"  BV号: {result.bvid}")
        print(f"  标题: {result.title}")
        print(f"  分区: {result.tname}")
        print(f"  时间段: {result.segments_count}")
        print(f"  聚合组: {result.aggregated_count}")
        
        if result.zip_path:
            print(f"  报告: {result.zip_path}")
        
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(full_pipeline_test())
