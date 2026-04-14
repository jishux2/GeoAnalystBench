# codes/benchmark/task_index.py
"""
基准测试集的任务索引与信息组装

从 datasets.json 中解析全部任务条目，构建支持多维度
筛选的内存索引，并将每条任务的元数据组装为可递交
智能体团队的结构化文本描述。

取代原有的 TaskInfoExtractor 和 WorkspaceManager，
将解析、筛选、文本生成三项职能收归单一模块。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set


class TaskIndex:
    """
    基准测试集的任务索引

    启动时一次性加载 datasets.json，解析每条任务记录
    并构建按来源分组的内存索引。对外提供多条件组合筛选
    与任务描述文本的组装能力。
    """

    def __init__(self, dataset_path: str = "benchmark/datasets.json"):
        """
        Args:
            dataset_path: datasets.json 的路径，相对于项目根目录
        """
        self._path = Path(dataset_path)
        self._tasks: Dict[str, Dict[str, Any]] = {}   # task_ID -> 任务记录
        self._by_source: Dict[str, List[str]] = {}     # source -> [task_ID, ...]
        self._metadata: Dict[str, Any] = {}

        self._load()

    def _load(self):
        """解析 JSON 文件，构建内存索引。"""
        with open(self._path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        self._metadata = raw.get("metadata", {})

        for entry in raw.get("tasks", []):
            task_id = entry["task_ID"]
            source = entry.get("source", "unknown")

            self._tasks[task_id] = entry
            self._by_source.setdefault(source, []).append(task_id)

    # ================================================================
    # 查询接口
    # ================================================================

    @property
    def all_task_ids(self) -> List[str]:
        """全部任务 ID，按在 JSON 中的出现顺序。"""
        return list(self._tasks.keys())

    @property
    def sources(self) -> List[str]:
        """全部来源标识，按首次出现顺序。"""
        return list(self._by_source.keys())

    @property
    def metadata(self) -> Dict[str, Any]:
        return dict(self._metadata)

    def get(self, task_id: str) -> Optional[Dict[str, Any]]:
        """按 ID 获取单条任务记录的完整字典。"""
        return self._tasks.get(task_id)

    def task_ids_by_source(self, source: str) -> List[str]:
        """获取指定来源下的全部任务 ID。"""
        return list(self._by_source.get(source, []))

    def filter(
        self,
        sources: Optional[List[str]] = None,
        task_ids: Optional[List[str]] = None,
        solvable: Optional[str] = None,
        has_reference: Optional[bool] = None,
        limit: Optional[int] = None,
    ) -> List[str]:
        """
        多条件组合筛选，返回符合全部条件的任务 ID 列表。

        各条件之间取交集，筛选完成后再施加数量限制。

        Args:
            sources: 限定来源范围
            task_ids: 限定候选 ID 范围
            solvable: 限定 solvable 字段值（如 "ACCEPT"）
            has_reference: True 仅保留有参考代码的任务，
                           False 仅保留无参考代码的任务
            limit: 从筛选结果中取前 N 条
        """
        candidates = list(self._tasks.keys())

        if sources is not None:
            allowed = set()
            for s in sources:
                allowed.update(self._by_source.get(s, []))
            candidates = [tid for tid in candidates if tid in allowed]

        if task_ids is not None:
            id_set = set(task_ids)
            candidates = [tid for tid in candidates if tid in id_set]

        if solvable is not None:
            candidates = [
                tid for tid in candidates
                if self._tasks[tid].get("solvable") == solvable
            ]

        if has_reference is not None:
            candidates = [
                tid for tid in candidates
                if self._has_reference_code(tid) == has_reference
            ]

        if limit is not None and limit > 0:
            candidates = candidates[:limit]

        return candidates

    # ================================================================
    # 任务信息提取
    # ================================================================

    def extract_task_info(self, task_id: str) -> Dict[str, Any]:
        """
        提取指定任务的结构化信息，供下游消费。

        返回字典包含原始字段以及组装好的完整任务描述文本。
        """
        entry = self._tasks.get(task_id)
        if entry is None:
            raise KeyError(f"Unknown task ID: {task_id}")

        source = entry.get("source", "unknown")
        task_text = entry.get("task_text", "")
        used_dataset = entry.get("used_dataset", [])
        solvable = entry.get("solvable", "")

        # 将嵌套的 used_dataset 展平为去重的文件名集合
        flat_files = self._flatten_dataset_refs(used_dataset)

        info = {
            "task_id": task_id,
            "source": source,
            "task_text": task_text,
            "used_dataset": used_dataset,
            "dataset_files": flat_files,
            "solvable": solvable,
            "has_reference": self._has_reference_code(task_id),
        }

        info["full_text"] = self._compose_description(info)

        return info

    def get_reference_code(self, task_id: str) -> Optional[str]:
        """
        获取任务的参考实现代码。

        Returns:
            代码字符串，无参考代码时返回 None
        """
        entry = self._tasks.get(task_id)
        if entry is None:
            return None
        solutions = entry.get("reference_solutions", [])
        if not solutions:
            return None
        return solutions[0]

    # ================================================================
    # 内部方法
    # ================================================================

    def _has_reference_code(self, task_id: str) -> bool:
        entry = self._tasks.get(task_id)
        if entry is None:
            return False
        solutions = entry.get("reference_solutions", [])
        return len(solutions) > 0 and bool(solutions[0].strip())

    @staticmethod
    def _flatten_dataset_refs(used_dataset: Any) -> List[str]:
        """
        将 used_dataset 的各种形态统一为去重的文件名列表。

        处理三种输入：
        - 平铺列表 ["a.shp", "b.tif"]
        - 嵌套列表 [["a.shp", "b.tif"], ["a.shp", "c.csv"]]
        - 空列表 []
        """
        if not used_dataset:
            return []

        files: Set[str] = set()
        for item in used_dataset:
            if isinstance(item, list):
                files.update(item)
            elif isinstance(item, str):
                files.add(item)

        return sorted(files)

    def _compose_description(self, info: Dict[str, Any]) -> str:
        """
        将任务元数据组装为递交智能体团队的文本描述。

        文本包含任务目标、数据源清单及其存储位置的定位指引，
        不附加任何编码规范或输出格式约束。
        """
        parts = [f"[Task Objective]\n{info['task_text']}"]

        files = info["dataset_files"]
        source = info["source"]

        if files:
            file_listing = "\n".join(f"  - {f}" for f in files)
            parts.append(
                f"[Input Data]\n"
                f"The following source files are available for this task, "
                f"located under the '{source}' subdirectory of the shared "
                f"dataset repository:\n{file_listing}"
            )
        else:
            parts.append(
                "[Input Data]\n"
                "This task does not rely on locally stored dataset files. "
                "Required data should be obtained programmatically through "
                "the appropriate library APIs as indicated in the task description."
            )

        parts.append(f"[Source Collection]\n{source}")

        return "\n\n".join(parts)

    # ================================================================
    # 持久化回写
    # ================================================================

    def save(self, output_path: Optional[str] = None):
        """
        将当前索引状态回写为 JSON 文件。

        用于排序或编辑操作后的持久化。
        """
        target = Path(output_path) if output_path else self._path

        # 按 source 分组排序后重建 tasks 列表
        ordered_tasks = []
        for source in sorted(self._by_source.keys()):
            for tid in self._by_source[source]:
                ordered_tasks.append(self._tasks[tid])

        payload = {
            "metadata": self._metadata,
            "tasks": ordered_tasks,
        }

        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    def reorder_by_source(self):
        """
        按 source 字段重新排列任务顺序。

        将同一来源的任务聚拢在一起，解决原 JSON 中
        偶发的错位问题（如落单的 GeoBenchX 记录混入
        其他来源段落）。
        """
        reordered: Dict[str, Dict[str, Any]] = {}
        for source in sorted(self._by_source.keys()):
            for tid in self._by_source[source]:
                reordered[tid] = self._tasks[tid]
        self._tasks = reordered