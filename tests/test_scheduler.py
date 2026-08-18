"""
任务调度器单元测试 - 状态持久化 / 终态恢复 / 失败隔离 / 重复输入合并
"""

import asyncio
import json

import pytest

from danmaku_analyzer.scheduler import TaskScheduler


@pytest.fixture
def state_file(tmp_path):
    return str(tmp_path / "tasks.jsonl")


class TestTaskScheduler:

    def test_submit_and_run_marks_done(self, state_file):
        scheduler = TaskScheduler(state_path=state_file, workers=2)
        scheduler.submit(["BV1a", "BV1b"])

        async def handler(task):
            task.bvid = task.input
            task.zip_path = f"/tmp/{task.input}.zip"

        result = asyncio.run(scheduler.run(handler))
        assert result.done == 2 and result.failed == 0
        assert all(t.status == "done" for t in result.tasks)
        assert all(t.updated_at for t in result.tasks)

    def test_interrupted_batch_recovers_losslessly(self, state_file):
        """中断后重新提交同批任务：done 跳过，failed 重跑"""
        s1 = TaskScheduler(state_path=state_file)
        s1.submit(["BV1a", "BV1b"])

        async def handler(task):
            if task.input == "BV1b":
                raise RuntimeError("crash")
            task.zip_path = "/tmp/a.zip"

        asyncio.run(s1.run(handler))

        s2 = TaskScheduler(state_path=state_file)
        tasks = s2.submit(["BV1a", "BV1b"])
        assert tasks[0].status == "done"
        assert tasks[0].zip_path == "/tmp/a.zip"

        executed = []

        async def handler2(task):
            executed.append(task.input)

        result = asyncio.run(s2.run(handler2))
        assert executed == ["BV1b"]
        # BV1a 以历史终态计入 done，BV1b 重跑成功，无失败
        assert result.done == 2 and result.failed == 0

    def test_running_state_treated_as_pending_on_recovery(self, state_file):
        """模拟中断时 running中的任务未落终态：恢复后重入队"""
        with open(state_file, "w", encoding="utf-8") as f:
            f.write(json.dumps({"input": "BV1x", "status": "running"}) + "\n")
        scheduler = TaskScheduler(state_path=state_file)
        tasks = scheduler.submit(["BV1x"])
        assert tasks[0].status == "pending"

    def test_duplicate_inputs_merged(self, state_file):
        scheduler = TaskScheduler(state_path=state_file)
        tasks = scheduler.submit(["BV1a", "BV1a", "BV1b"])
        assert len(tasks) == 2

    def test_handler_failure_isolated(self, state_file):
        scheduler = TaskScheduler(state_path=state_file, workers=1)
        scheduler.submit(["BV1bad", "BV1ok"])

        async def handler(task):
            if task.input == "BV1bad":
                raise RuntimeError("boom")

        result = asyncio.run(scheduler.run(handler))
        assert result.failed == 1 and result.done == 1
        failed = next(t for t in result.tasks if t.status == "failed")
        assert "boom" in failed.error

    def test_reused_terminal_status_recovered(self, state_file):
        scheduler = TaskScheduler(state_path=state_file)
        scheduler.submit(["BV1a"])

        async def handler(task):
            task.status = "reused"
            task.zip_path = "/tmp/a.zip"

        asyncio.run(scheduler.run(handler))

        s2 = TaskScheduler(state_path=state_file)
        tasks = s2.submit(["BV1a"])
        assert tasks[0].status == "reused"

    def test_corrupt_state_line_skipped(self, state_file):
        with open(state_file, "w", encoding="utf-8") as f:
            f.write("not-json\n")
            f.write(json.dumps({"input": "BV1ok", "status": "done"}) + "\n")
        scheduler = TaskScheduler(state_path=state_file)
        state = scheduler.load_state()
        assert set(state) == {"BV1ok"}

    def test_missing_state_file_yields_fresh_tasks(self, state_file):
        scheduler = TaskScheduler(state_path=state_file)
        tasks = scheduler.submit(["BV1new"])
        assert tasks[0].status == "pending"

    def test_submit_without_recover_ignores_terminal_state(self, state_file):
        """--no-reuse 全量重分析：历史终态不得静默跳过任务（语义回归保护）"""
        s1 = TaskScheduler(state_path=state_file)
        s1.submit(["BV1a"])

        async def handler(task):
            task.zip_path = "/tmp/a.zip"

        asyncio.run(s1.run(handler))

        s2 = TaskScheduler(state_path=state_file)
        tasks = s2.submit(["BV1a"], recover=False)
        assert tasks[0].status == "pending"

    def test_concurrent_persist_keeps_all_terminal_states(self, state_file):
        """多 worker 并发落盘不得丢失终态（持久化锁回归保护）"""
        scheduler = TaskScheduler(state_path=state_file, workers=4)
        inputs = [f"BV1{i:02d}" for i in range(12)]
        scheduler.submit(inputs)

        async def handler(task):
            task.zip_path = f"/tmp/{task.input}.zip"

        result = asyncio.run(scheduler.run(handler))
        assert result.done == len(inputs) and result.failed == 0
        state = TaskScheduler(state_path=state_file).load_state()
        assert all(state[key].status == "done" for key in inputs)
        assert all(state[key].zip_path.endswith(".zip") for key in inputs)
