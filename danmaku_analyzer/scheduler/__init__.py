"""
任务调度子包 - 单进程异步任务队列 + JSON Lines 持久化
替代 compare_videos 的线性循环：批量中断后按任务状态无损恢复，不引入外部队列依赖
"""

from .task_queue import ScheduledTask, TaskScheduler, SchedulerResult, TASK_STATE_RELPATH, TERMINAL_STATUSES

__all__ = ["ScheduledTask", "TaskScheduler", "SchedulerResult", "TASK_STATE_RELPATH", "TERMINAL_STATUSES"]
