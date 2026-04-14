# codes/orchestrator.py
"""
智能体协作修复的批量编排

继承通用批处理框架，为每个任务实例化一个 Coordinator
驱动多智能体协作团队。筛选、并发控制、状态持久化与
断点续跑能力从 BatchRunner 继承。
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional

from benchmark.task_index import TaskIndex
from benchmark.workspace_setup import BenchmarkLayout
from benchmark.batch_runner import BatchRunner, StatusRegistry


class AgentOrchestrator(BatchRunner):

    def __init__(
        self,
        task_index: TaskIndex,
        layout: BenchmarkLayout,
        api_key: str,
        interpreter: str,
        skill_root: str = "skills",
        max_concurrent: int = 4,
        temperature: float = 0.7,
        check_interval: float = 2400.0,
        task_timeout: float = 2400.0,
        process_executor: Optional[ProcessPoolExecutor] = None,
        enable_evaluation: bool = False,
    ):
        registry = StatusRegistry(layout.workspace_root)
        super().__init__(task_index, registry, max_concurrent)

        self.layout = layout
        self.api_key = api_key
        self.interpreter = interpreter
        self.skill_root = Path(skill_root)
        self.temperature = temperature
        self.check_interval = check_interval
        self.task_timeout = task_timeout
        self._process_executor = process_executor
        self.enable_evaluation = enable_evaluation

    async def _execute_single(
        self, task_id: str, entry: Dict[str, Any]
    ) -> Dict[str, Any]:
        from agents.coordinator import Coordinator

        source = entry["source"]
        task_info = self.index.extract_task_info(task_id)

        working_dir = self.layout.workspace_task_dir(source, task_id)
        output_dir = working_dir / "outputs"
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"  [{source}/{task_id}] launching agent team...")

        coordinator = Coordinator(
            task_id=task_id,
            api_key=self.api_key,
            task_info=task_info,
            working_dir=working_dir,
            output_dir=output_dir,
            interpreter=self.interpreter,
            skill_root=self.skill_root,
            check_interval=self.check_interval,
            timeout=self.task_timeout,
            process_executor=self._process_executor,
            layout=self.layout,
        )

        result = await coordinator.run()

        # 智能体完成后触发评估
        if self.enable_evaluation:
            try:
                await self._evaluate_task(task_id)
                print(f"  [{source}/{task_id}] evaluation complete")
            except Exception as e:
                print(f"  [{source}/{task_id}] evaluation failed: {e}")

        return result

    async def _evaluate_task(self, task_id: str):
        from benchmark.artifact_processor import ArtifactProcessor
        from benchmark.task_evaluator import TaskEvaluator
        from benchmark.eval_client import EvalClient
        from benchmark.batch_runner import StatusRegistry

        entry = self.index.get(task_id)
        source = entry["source"]

        processor = ArtifactProcessor(
            interpreter=self.interpreter,
            skills_dir=self.skill_root,
        )
        evaluator = TaskEvaluator(
            task_index=self.index,
            layout=self.layout,
            artifact_processor=processor,
        )

        async with EvalClient() as client:
            result = await evaluator.evaluate(
                task_id=task_id,
                eval_client=client,
            )

        # 同步写入评估状态文件
        eval_registry = StatusRegistry(self.layout.workspace_root)
        eval_registry.STATUS_FILENAME = "eval_status.json"
        success = not result.get("parse_error", False)
        eval_registry.save(source, task_id, {
            "status": "success" if success else "failed",
            "result": {
                "success": success,
                "overall_score": result.get("overall_score"),
            },
        })