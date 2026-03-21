# codes/evaluator/dialogue_manager.py
"""
任务状态注册表

在新架构下，详细的对话历史由各子智能体的ContextJournal
独立维护。此模块仅负责任务级终态的记录与查询，
支持断点续跑时跳过已完成的任务。
"""

import json
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime


class TaskStatusRegistry:
    """
    任务状态注册表

    以任务为粒度维护简要的终态记录。
    每个任务对应outputs/下的coordinator_journal.json，
    注册表通过扫描这些文件还原已知状态。
    """

    def __init__(self, workspace_root: str = "evaluation_workspace"):
        self.workspace_root = Path(workspace_root)

    def is_completed(self, task_id: int) -> bool:
        journal_path = self._journal_path(task_id)
        if not journal_path.exists():
            return False
        with open(journal_path, "r", encoding="utf-8") as f:
            journal = json.load(f)
        return "outcome" in journal and journal["outcome"] is not None

    def get_result(self, task_id: int) -> Optional[Dict]:
        journal_path = self._journal_path(task_id)
        if not journal_path.exists():
            return None
        with open(journal_path, "r", encoding="utf-8") as f:
            journal = json.load(f)
        return journal.get("outcome")

    def get_status(self, task_id: int) -> str:
        """
        获取任务状态。

        Returns:
            'success' / 'failed' / 'pending'
        """
        result = self.get_result(task_id)
        if result is None:
            return "pending"
        return "success" if result.get("success") else "failed"

    def filter_pending(self, task_ids: List[int]) -> List[int]:
        """从候选列表中筛选出尚未完成的任务。"""
        return [tid for tid in task_ids if not self.is_completed(tid)]

    def get_tasks_by_status(
        self, status: str, scope: Optional[List[int]] = None
    ) -> List[int]:
        """
        按状态筛选任务。

        Args:
            status: 'success' / 'failed' / 'pending'
            scope: 限定扫描范围，None时扫描全部

        Returns:
            符合条件的任务ID列表
        """
        if scope is not None:
            candidates = scope
        else:
            candidates = self._scan_all_task_ids()

        return sorted(
            tid for tid in candidates if self.get_status(tid) == status
        )

    def _journal_path(self, task_id: int) -> Path:
        return self.workspace_root / str(task_id) / "outputs" / "coordinator_journal.json"

    def _scan_all_task_ids(self) -> List[int]:
        """扫描工作空间下所有任务目录。"""
        if not self.workspace_root.exists():
            return []
        ids = []
        for d in self.workspace_root.iterdir():
            if d.is_dir() and d.name.isdigit():
                ids.append(int(d.name))
        return ids