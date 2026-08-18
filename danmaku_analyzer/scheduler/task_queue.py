"""
任务队列 - asyncio.Queue 执行 + JSON Lines 状态持久化（DATA_ROOT/scheduler/tasks.jsonl）
状态文件与 progress.jsonl 同格式约定（追加记录，按键取最新），中断后 submit 自动恢复
"""

import asyncio
import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Awaitable, Callable, Dict, List, Optional

from ..config import get_settings
from ..utils.logger import get_logger

logger = get_logger(__name__)

TASK_STATE_RELPATH = os.path.join("scheduler", "tasks.jsonl")

# done/reused 为终态（恢复时跳过执行）；running 视同 pending（中断时未落终态）
TERMINAL_STATUSES = ("done", "reused")
TaskHandler = Callable[["ScheduledTask"], Awaitable[None]]


@dataclass
class ScheduledTask:
    """单个调度任务：input 为唯一键（批次内重复输入合并为同一任务）"""
    input: str
    status: str = "pending"
    bvid: str = ""
    zip_path: str = ""
    error: str = ""
    updated_at: str = ""

    def to_record(self) -> Dict:
        return {
            "input": self.input, "status": self.status, "bvid": self.bvid,
            "zip_path": self.zip_path, "error": self.error, "updated_at": self.updated_at,
        }

    @classmethod
    def from_record(cls, record: Dict) -> "ScheduledTask":
        return cls(
            input=record.get("input", ""), status=record.get("status", "pending"),
            bvid=record.get("bvid", ""), zip_path=record.get("zip_path", ""),
            error=record.get("error", ""), updated_at=record.get("updated_at", ""),
        )


@dataclass
class SchedulerResult:
    """一轮调度执行后的汇总"""
    done: int = 0
    reused: int = 0
    failed: int = 0
    tasks: List[ScheduledTask] = field(default_factory=list)


class TaskScheduler:
    """单进程异步任务队列：asyncio.Queue 并发执行 + JSON Lines 状态持久化，中断后按状态无损恢复"""

    def __init__(self, state_path: Optional[str] = None, workers: Optional[int] = None):
        settings = get_settings()
        self.state_path = state_path or settings.resolve_data_path(TASK_STATE_RELPATH)
        self.workers = max(1, workers or settings.SCHEDULER_WORKERS)
        self.tasks: List[ScheduledTask] = []
        # 多 worker 并发落盘时保护 读取→重写 临界区，避免后写覆盖先写丢失终态
        self._persist_lock = asyncio.Lock()

    def load_state(self) -> Dict[str, ScheduledTask]:
        """读取状态文件，按 input 索引（后记录覆盖先记录）；缺失/坏行跳过"""
        state: Dict[str, ScheduledTask] = {}
        if not os.path.exists(self.state_path):
            return state
        with open(self.state_path, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning(f"任务状态文件行损坏，跳过: {line[:80]}")
                    continue
                if record.get("input"):
                    state[record["input"]] = ScheduledTask.from_record(record)
        return state

    def submit(self, inputs: List[str], recover: bool = True) -> List[ScheduledTask]:
        """登记本批任务；recover 且历史终态（done/reused）存在时直接恢复跳过，其余一律重入队；
        recover=False（如 --no-reuse 全量重分析）无视历史状态全部重新执行"""
        state = self.load_state() if recover else {}
        tasks: Dict[str, ScheduledTask] = {}
        for raw in inputs:
            if raw in tasks:
                continue
            prior = state.get(raw)
            tasks[raw] = prior if prior and prior.status in TERMINAL_STATUSES else ScheduledTask(input=raw)
        self.tasks = list(tasks.values())
        skipped = sum(1 for t in self.tasks if t.status in TERMINAL_STATUSES)
        if skipped:
            logger.info(f"调度器恢复历史状态: {skipped} 个任务已完成，跳过执行")
        return self.tasks

    def _persist_sync(self, task: ScheduledTask) -> None:
        """状态变更即时落盘：按键取最新后全量原子重写（任务量小，换取中断零丢失）"""
        task.updated_at = datetime.now().isoformat(timespec='seconds')
        state = self.load_state()
        state[task.input] = task
        os.makedirs(os.path.dirname(self.state_path), exist_ok=True)
        tmp_path = self.state_path + ".tmp"
        with open(tmp_path, 'w', encoding='utf-8') as f:
            for record in state.values():
                f.write(json.dumps(record.to_record(), ensure_ascii=False) + "\n")
        os.replace(tmp_path, self.state_path)

    async def _persist_locked(self, task: ScheduledTask) -> None:
        async with self._persist_lock:
            self._persist_sync(task)

    async def run(self, handler: TaskHandler) -> SchedulerResult:
        """并发执行非终态任务；handler 通过修改 task 字段回报结果（未置终态视为 done），
        handler 抛错则该任务标 failed 且不中断其余任务"""
        pending = [t for t in self.tasks if t.status not in TERMINAL_STATUSES]
        if pending:
            logger.info(f"调度器启动: {len(pending)} 个任务入队，并发 {min(self.workers, len(pending))}")
        queue: asyncio.Queue = asyncio.Queue()
        for task in pending:
            queue.put_nowait(task)

        async def worker():
            while True:
                try:
                    task = queue.get_nowait()
                except asyncio.QueueEmpty:
                    return
                task.status = "running"
                task.error = ""
                await self._persist_locked(task)
                try:
                    await handler(task)
                    if task.status not in TERMINAL_STATUSES and task.status != "failed":
                        task.status = "done"
                except Exception as e:
                    task.status = "failed"
                    task.error = str(e)
                    logger.error(f"任务失败: {task.input} - {e}")
                await self._persist_locked(task)

        await asyncio.gather(*[worker() for _ in range(min(self.workers, len(pending)) or 1)])
        return SchedulerResult(
            done=sum(1 for t in self.tasks if t.status == "done"),
            reused=sum(1 for t in self.tasks if t.status == "reused"),
            failed=sum(1 for t in self.tasks if t.status == "failed"),
            tasks=self.tasks,
        )
