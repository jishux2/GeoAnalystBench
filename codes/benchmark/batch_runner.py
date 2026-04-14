# codes/benchmark/batch_runner.py
"""
通用任务批处理框架

抽象出跨场景共享的调度骨架：从任务池中筛选待处理条目，
以信号量控制并发度，逐任务记录终态并支持断点续跑。

参考代码执行器和智能体编排器各自继承此框架，
仅需实现 _execute_single 方法来定义单任务的执行逻辑。
"""

from __future__ import annotations

import asyncio
import json
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional

from .task_index import TaskIndex


class TaskStatus:
    """单个任务的状态记录。"""

    SUCCESS = "success"
    FAILED = "failed"
    PENDING = "pending"
    NO_CODE = "no_code"     # 参考代码不存在

    def __init__(self, task_id: str, source: str):
        self.task_id = task_id
        self.source = source
        self.status: str = self.PENDING
        self.result: Optional[Dict[str, Any]] = None
        self.started_at: Optional[float] = None
        self.completed_at: Optional[float] = None
        self.error: Optional[str] = None


class StatusRegistry:
    """
    任务状态的持久化注册表

    每个任务的状态以独立的 JSON 文件存储在其任务目录下，
    避免全局文件在并发写入时的冲突风险。
    """

    STATUS_FILENAME = "run_status.json"

    def __init__(self, root_dir: Path):
        """
        Args:
            root_dir: 状态文件所在的目录树根节点
                      （ground_truth_root 或 workspace_root）
        """
        self._root = root_dir

    def _status_path(self, source: str, task_id: str) -> Path:
        return self._root / source / task_id / self.STATUS_FILENAME

    def load(self, source: str, task_id: str) -> Optional[Dict[str, Any]]:
        """读取指定任务的状态记录。不存在则返回 None。"""
        path = self._status_path(source, task_id)
        if not path.exists():
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def save(self, source: str, task_id: str, record: Dict[str, Any]):
        """写入任务状态。"""
        path = self._status_path(source, task_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2, ensure_ascii=False, default=str)

    def get_status(self, source: str, task_id: str) -> str:
        """获取任务状态字符串。"""
        record = self.load(source, task_id)
        if record is None:
            return TaskStatus.PENDING
        return record.get("status", TaskStatus.PENDING)

    def is_completed(self, source: str, task_id: str) -> bool:
        status = self.get_status(source, task_id)
        return status in (TaskStatus.SUCCESS, TaskStatus.NO_CODE)

    def is_failed(self, source: str, task_id: str) -> bool:
        return self.get_status(source, task_id) == TaskStatus.FAILED


class BatchRunner(ABC):
    """
    任务批处理的抽象基类

    子类实现 _execute_single 定义单任务逻辑，
    框架负责筛选、并发、状态管理和断点续跑。
    """

    def __init__(
        self,
        task_index: TaskIndex,
        status_registry: StatusRegistry,
        max_concurrent: int = 4,
    ):
        self.index = task_index
        self.registry = status_registry
        self.max_concurrent = max_concurrent
        self._semaphore: Optional[asyncio.Semaphore] = None
        self._results: Dict[str, Dict[str, Any]] = {}

    async def run(
        self,
        task_ids: Optional[List[str]] = None,
        sources: Optional[List[str]] = None,
        limit: Optional[int] = None,
        skip_success: bool = True,
        skip_failed: bool = False,
    ):
        """
        执行一批任务。

        Args:
            task_ids: 明确指定的任务 ID，为 None 时取全集
            sources: 限定来源范围
            limit: 数量限制
            skip_success: 跳过已成功的任务
            skip_failed: 跳过已失败的任务（用于避免重复执行
                         已知失败且未修复的任务）
        """
        # 筛选候选任务
        candidates = self.index.filter(
            sources=sources,
            task_ids=task_ids,
            limit=None,   # 先不限数量，等状态过滤后再截取
        )

        # 状态过滤
        filtered = []
        for tid in candidates:
            entry = self.index.get(tid)
            source = entry["source"]
            if skip_success and self.registry.is_completed(source, tid):
                continue
            if skip_failed and self.registry.is_failed(source, tid):
                continue
            filtered.append(tid)

        # 数量限制在最后施加
        if limit is not None and limit > 0:
            filtered = filtered[:limit]

        if not filtered:
            self._report_summary([], candidates)
            return

        self._semaphore = asyncio.Semaphore(self.max_concurrent)

        tasks = [
            self._run_with_semaphore(tid)
            for tid in filtered
        ]
        await asyncio.gather(*tasks, return_exceptions=True)

        self._report_summary(filtered, candidates)

    async def _run_with_semaphore(self, task_id: str):
        """信号量控制下的单任务执行包装。"""
        async with self._semaphore:
            entry = self.index.get(task_id)
            source = entry["source"]

            status = TaskStatus(task_id, source)
            status.started_at = time.time()

            try:
                result = await self._execute_single(task_id, entry)

                # 优先采纳执行方显式指定的状态，否则按 success 字段推断
                if "status" in result:
                    status.status = result["status"]
                else:
                    status.status = (
                        TaskStatus.SUCCESS if result.get("success")
                        else TaskStatus.FAILED
                    )

                status.result = result
                status.error = result.get("error")
            except Exception as e:
                status.status = TaskStatus.FAILED
                status.error = f"{type(e).__name__}: {e}"
                status.result = {"success": False, "error": status.error}

            status.completed_at = time.time()

            self.registry.save(source, task_id, {
                "status": status.status,
                "started_at": status.started_at,
                "completed_at": status.completed_at,
                "elapsed": status.completed_at - status.started_at,
                "error": status.error,
                "result": status.result,
            })

            self._results[task_id] = status.result

    @abstractmethod
    async def _execute_single(
        self, task_id: str, entry: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        子类实现：执行单个任务的具体逻辑。

        Args:
            task_id: 任务标识
            entry: 任务索引中的完整记录

        Returns:
            包含 success 布尔值的结果字典
        """
        ...

    def _report_summary(
        self, executed: List[str], candidates: List[str]
    ):
        """输出执行摘要。"""
        total = len(candidates)
        ran = len(executed)
        skipped = total - ran
        succeeded = sum(
            1 for tid in executed
            if self._results.get(tid, {}).get("success", False)
        )
        failed = ran - succeeded

        print(f"\n{'=' * 50}")
        print(f"Batch execution summary")
        print(f"{'=' * 50}")
        print(f"  Candidates:  {total}")
        print(f"  Skipped:     {skipped}")
        print(f"  Executed:    {ran}")
        print(f"  Succeeded:   {succeeded}")
        print(f"  Failed:      {failed}")
        print(f"{'=' * 50}")