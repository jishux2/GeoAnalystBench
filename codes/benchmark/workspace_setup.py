# codes/benchmark/workspace_setup.py
"""
评测目录结构的构建与路径解析

根据 TaskIndex 中的任务清单，在磁盘上建立三套平行的
目录体系（数据集、参考基线、智能体工作空间），将参考
代码部署至对应位置，并为运行时的路径拼接提供统一入口。
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from .task_index import TaskIndex


class BenchmarkLayout:
    """
    评测目录结构的定义与路径解析

    三个顶层区域：
    - benchmark_datasets/      数据集，按source分子目录
    - benchmark_ground_truth/  参考代码与其执行产物
    - benchmark_workspace/     智能体协作空间

    每个任务在后两个区域中占据 {source}/{task_ID}/ 的位置，
    其下的 pred_results/ 子目录收纳脚本的交付产物。
    """

    DATASET_ROOT = "benchmark_datasets"
    GROUND_TRUTH_ROOT = "benchmark_ground_truth"
    WORKSPACE_ROOT = "benchmark_workspace"

    REFERENCE_SCRIPT_NAME = "reference_solution.py"
    REFERENCE_OUTPUT_SUBDIR = "output"
    WORKSPACE_OUTPUT_SUBDIR = "pred_results"

    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root).resolve()

    # ================================================================
    # 顶层目录
    # ================================================================

    @property
    def dataset_root(self) -> Path:
        return self.project_root / self.DATASET_ROOT

    @property
    def ground_truth_root(self) -> Path:
        return self.project_root / self.GROUND_TRUTH_ROOT

    @property
    def workspace_root(self) -> Path:
        return self.project_root / self.WORKSPACE_ROOT

    # ================================================================
    # 任务级路径解析
    # ================================================================

    def dataset_dir(self, source: str) -> Path:
        """数据集目录下指定来源的子目录。"""
        return self.dataset_root / source

    def ground_truth_task_dir(self, source: str, task_id: str) -> Path:
        """参考基线侧的任务根目录。"""
        return self.ground_truth_root / source / task_id

    def ground_truth_script(self, source: str, task_id: str) -> Path:
        """参考脚本的完整路径。"""
        return self.ground_truth_task_dir(source, task_id) / self.REFERENCE_SCRIPT_NAME

    def ground_truth_output(self, source: str, task_id: str) -> Path:
        """参考代码执行产物的收纳目录。"""
        return self.ground_truth_task_dir(source, task_id) / self.REFERENCE_OUTPUT_SUBDIR

    def workspace_task_dir(self, source: str, task_id: str) -> Path:
        """智能体工作空间侧的任务根目录。"""
        return self.workspace_root / source / task_id

    def workspace_output(self, source: str, task_id: str) -> Path:
        """智能体生成代码执行产物的收纳目录。"""
        return self.workspace_task_dir(source, task_id) / self.WORKSPACE_OUTPUT_SUBDIR

    # ================================================================
    # 环境变量值
    # ================================================================

    def env_input_root(self, source: str) -> str:
        """TASK_INPUT_ROOT 环境变量应设定的值。"""
        return str(self.dataset_dir(source).resolve())

    def env_output_root_ground_truth(self, source: str, task_id: str) -> str:
        """参考代码执行时 TASK_OUTPUT_ROOT 的值。"""
        return str(self.ground_truth_output(source, task_id).resolve())

    def env_output_root_workspace(self, source: str, task_id: str) -> str:
        """智能体脚本执行时 TASK_OUTPUT_ROOT 的值。"""
        return str(self.workspace_output(source, task_id).resolve())


class WorkspaceBuilder:
    """
    目录结构的物理构建器

    从 TaskIndex 读取全部任务条目，在磁盘上创建完整的
    目录树并部署参考脚本。支持增量模式——已存在的目录
    和文件不会被覆盖，仅补充缺失的部分。
    """

    def __init__(
        self,
        layout: BenchmarkLayout,
        task_index: TaskIndex,
    ):
        self.layout = layout
        self.index = task_index

    def build_all(self, deploy_reference: bool = True):
        """
        构建完整的目录树。

        Args:
            deploy_reference: 是否将参考代码写入 ground_truth 目录
        """
        all_ids = self.index.all_task_ids

        sources_seen = set()
        for task_id in all_ids:
            entry = self.index.get(task_id)
            source = entry["source"]
            sources_seen.add(source)

            # 参考基线侧
            gt_task = self.layout.ground_truth_task_dir(source, task_id)
            gt_task.mkdir(parents=True, exist_ok=True)
            self.layout.ground_truth_output(source, task_id).mkdir(
                parents=True, exist_ok=True
            )

            # 智能体工作空间侧
            ws_task = self.layout.workspace_task_dir(source, task_id)
            ws_task.mkdir(parents=True, exist_ok=True)
            self.layout.workspace_output(source, task_id).mkdir(
                parents=True, exist_ok=True
            )

            # 部署参考脚本
            if deploy_reference:
                self._deploy_reference_script(source, task_id)

        # 确保数据集目录下每个来源的子目录存在
        for source in sources_seen:
            self.layout.dataset_dir(source).mkdir(parents=True, exist_ok=True)

    def build_single(self, task_id: str, deploy_reference: bool = True):
        """为单个任务构建目录结构。"""
        entry = self.index.get(task_id)
        if entry is None:
            raise KeyError(f"Unknown task ID: {task_id}")

        source = entry["source"]

        self.layout.ground_truth_task_dir(source, task_id).mkdir(
            parents=True, exist_ok=True
        )
        self.layout.ground_truth_output(source, task_id).mkdir(
            parents=True, exist_ok=True
        )
        self.layout.workspace_task_dir(source, task_id).mkdir(
            parents=True, exist_ok=True
        )
        self.layout.workspace_output(source, task_id).mkdir(
            parents=True, exist_ok=True
        )
        self.layout.dataset_dir(source).mkdir(parents=True, exist_ok=True)

        if deploy_reference:
            self._deploy_reference_script(source, task_id)

    def _deploy_reference_script(self, source: str, task_id: str):
        """将参考代码写入 ground_truth 目录。"""
        code = self.index.get_reference_code(task_id)
        if not code:
            return

        script_path = self.layout.ground_truth_script(source, task_id)
        if script_path.exists():
            existing = script_path.read_text(encoding="utf-8")
            if existing == code:
                return

        script_path.write_text(code, encoding="utf-8")