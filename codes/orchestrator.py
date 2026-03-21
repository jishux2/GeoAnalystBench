# codes/orchestrator.py
"""
迭代修复流程编排器

在新架构下，编排器的职责收敛为任务级并发调度：
遍历任务列表，为每个任务启动一个Coordinator实例，
通过信号量控制同时活跃的任务数量。任务内部的
多智能体协作、轮次推进、终态判定全部由Coordinator自治。
"""

from __future__ import annotations

import asyncio
import json
import subprocess
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional

from prompt_builder import TaskInfoExtractor
from evaluator.workspace_manager import WorkspaceManager


class IterativeRepairOrchestrator:
    """迭代修复编排器"""

    def __init__(
        self,
        api_key: str,
        max_concurrent: int = 4,
        temperature: float = 0.7,
        workspace_root: str = "evaluation_workspace",
        skill_root: str = "skills",
        check_interval: float = 60.0,
        task_timeout: float = 900.0,
    ):
        """
        Args:
            api_key: DeepSeek API密钥
            max_concurrent: 最大并发任务数
            temperature: 采样温度
            workspace_root: 工作空间根目录
            skill_root: 技能目录根路径
            check_interval: 主控节点检查间隔（秒）
            task_timeout: 单任务全局超时（秒）
        """
        self.api_key = api_key
        self.max_concurrent = max_concurrent
        self.temperature = temperature
        self.workspace_root = Path(workspace_root)
        self.skill_root = Path(skill_root)
        self.check_interval = check_interval
        self.task_timeout = task_timeout

        self.task_extractor = TaskInfoExtractor()
        self.workspace_mgr = WorkspaceManager(workspace_root)

        self._load_interpreter_config()

        # 信号量控制任务级并发
        self._semaphore: Optional[asyncio.Semaphore] = None

        # 结果收集
        self._results: Dict[int, Dict[str, Any]] = {}

    async def run(self, task_ids: List[int]):
        """执行完整的修复流程。"""
        from evaluator.dialogue_manager import TaskStatusRegistry

        registry = TaskStatusRegistry(str(self.workspace_root))

        # 跳过已完成的任务
        pending_ids = registry.filter_pending(task_ids)
        skipped = len(task_ids) - len(pending_ids)

        # 加载已完成任务的结果
        for tid in task_ids:
            if tid not in pending_ids:
                result = registry.get_result(tid)
                if result:
                    self._results[tid] = result

        print("=" * 60)
        print("GeoAnalystBench - 多智能体协作修复系统")
        print("=" * 60)
        print(f"总任务数：{len(task_ids)}")
        if skipped > 0:
            print(f"已完成（跳过）：{skipped}")
        print(f"待处理：{len(pending_ids)}")
        print(f"并发上限：{self.max_concurrent}")
        print(f"单任务超时：{self.task_timeout}s")
        print("=" * 60)

        if not pending_ids:
            print("所有任务均已完成")
            self._print_summary(task_ids)
            return

        self._semaphore = asyncio.Semaphore(self.max_concurrent)

        with ProcessPoolExecutor(max_workers=self.max_concurrent) as executor:
            tasks = [
                self._run_single_task(task_id, executor)
                for task_id in pending_ids
            ]
            await asyncio.gather(*tasks, return_exceptions=True)

        self._print_summary(task_ids)

    async def _run_single_task(
        self,
        task_id: int,
        executor: ProcessPoolExecutor,
    ):
        """
        单个任务的完整生命周期。

        通过信号量获取执行令牌后启动Coordinator，
        完成后释放令牌并记录结果。
        """
        async with self._semaphore:
            print(f"\n[Task {task_id}] 启动")

            try:
                result = await self._execute_task(task_id, executor)
                self._results[task_id] = result

                status = "成功" if result.get("success") else "失败"
                print(f"[Task {task_id}] {status}")

                # 触发评估
                await self._evaluate_task(task_id, result)

            except Exception as e:
                print(f"[Task {task_id}] 异常：{e}")
                self._results[task_id] = {
                    "success": False,
                    "error": str(e),
                }
                import traceback
                traceback.print_exc()

    async def _execute_task(
        self,
        task_id: int,
        executor: ProcessPoolExecutor,
    ) -> Dict[str, Any]:
        """
        构造并运行单个任务的Coordinator。

        Args:
            task_id: 任务编号
            executor: 进程池，传递给Coordinator供脚本执行使用

        Returns:
            Coordinator的终态结果
        """
        from agents.coordinator import Coordinator

        # 提取任务信息
        task_info = self.task_extractor.extract(task_id)
        task_meta = self.workspace_mgr.task_configs.get(task_id, {})

        task_config = {
            "prompt": task_info,
            "categories": task_meta.get("categories", []),
            "is_opensource": task_meta.get("is_opensource", True),
        }

        # 准备目录
        working_dir = self.workspace_root / str(task_id)
        working_dir.mkdir(parents=True, exist_ok=True)
        output_dir = working_dir / "outputs"
        output_dir.mkdir(parents=True, exist_ok=True)

        # 选择解释器
        interpreter = self._get_interpreter(task_config)

        # 创建并运行Coordinator
        coordinator = Coordinator(
            task_id=task_id,
            api_key=self.api_key,
            task_config=task_config,
            working_dir=working_dir,
            output_dir=output_dir,
            interpreter=interpreter,
            skill_root=self.skill_root,
            check_interval=self.check_interval,
            timeout=self.task_timeout,
            process_executor=executor,
        )

        return await coordinator.run()

    async def _evaluate_task(self, task_id: int, result: Dict[str, Any]):
        """
        任务终态确定后触发模型辅助评估。

        评估素材从各子智能体的日志文件中汇集。
        """
        try:
            from evaluator.task_evaluator import TaskEvaluator
            from deepseek.deepseek_client import DeepSeekClient

            evaluator = TaskEvaluator(str(self.workspace_root))

            async with DeepSeekClient(self.api_key) as eval_client:
                await evaluator.evaluate(
                    task_id=task_id,
                    result=result,
                    api_client=eval_client,
                    temperature=0.3,
                )
            print(f"[Task {task_id}] 评估完成")

        except Exception as e:
            print(f"[Task {task_id}] 评估失败：{e}")

    def _get_interpreter(self, task_config: Dict) -> str:
        """根据技术栈属性选择解释器。"""
        if task_config.get("is_opensource", True):
            return self.opensource_interpreter
        if not self.arcgis_interpreter:
            raise RuntimeError("任务需要ArcGIS环境，但未配置解释器")
        return self.arcgis_interpreter

    def _load_interpreter_config(self):
        """从配置文件加载解释器路径。"""
        config_path = Path("codes/evaluator_config.json")
        if not config_path.exists():
            raise FileNotFoundError(
                "解释器配置文件不存在！请先运行: python codes/setup_evaluation_env.py"
            )

        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        self.opensource_interpreter = config["opensource_interpreter"]
        self.arcgis_interpreter = config.get("arcgis_interpreter")

        self._verify_interpreter(self.opensource_interpreter)

    def _verify_interpreter(self, interpreter_path: str):
        """验证Python解释器是否可用。"""
        try:
            result = subprocess.run(
                [interpreter_path, "--version"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode != 0:
                raise RuntimeError(f"解释器不可用：{interpreter_path}")
            print(f"解释器验证通过：{result.stdout.strip()}")
        except Exception as e:
            raise RuntimeError(f"解释器验证失败 {interpreter_path}: {e}")

    def _print_summary(self, task_ids: List[int]):
        """输出最终统计。"""
        total = len(task_ids)
        success = sum(
            1 for tid in task_ids
            if self._results.get(tid, {}).get("success", False)
        )
        failed = total - success

        print("\n" + "=" * 60)
        print("执行完成")
        print("=" * 60)
        print(f"总任务数：{total}")
        print(f"成功：{success}")
        print(f"失败：{failed}")
        if total > 0:
            print(f"成功率：{success / total * 100:.1f}%")
        print("=" * 60)