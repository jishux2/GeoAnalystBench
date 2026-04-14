# codes/benchmark/run_evaluation.py
"""
独立评估入口

对已完成智能体运行的任务单独触发评估，
无需重新运行智能体。

运行方式：
    python codes/benchmark/run_evaluation.py [选项]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmark.task_index import TaskIndex
from benchmark.workspace_setup import BenchmarkLayout
from benchmark.batch_runner import BatchRunner, StatusRegistry
from benchmark.artifact_processor import ArtifactProcessor
from benchmark.task_evaluator import TaskEvaluator
from benchmark.eval_client import EvalClient


def load_interpreter() -> str:
    config_path = Path("codes/evaluator_config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    return config["opensource_interpreter"]


class EvalStatusRegistry(StatusRegistry):
    """评估专用的状态注册表，使用独立的状态文件名。"""
    STATUS_FILENAME = "eval_status.json"


class EvaluationRunner(BatchRunner):

    def __init__(
        self,
        task_index: TaskIndex,
        layout: BenchmarkLayout,
        interpreter: str,
        skill_root: Path,
        max_concurrent: int = 4,
    ):
        registry = EvalStatusRegistry(layout.workspace_root)
        super().__init__(task_index, registry, max_concurrent)

        self.layout = layout
        self.interpreter = interpreter
        self.skill_root = skill_root

    async def _execute_single(
        self, task_id: str, entry: dict
    ) -> dict:
        source = entry["source"]

        # 检查是否有生成代码
        script_path = (
            self.layout.workspace_task_dir(source, task_id) / "current_script.py"
        )
        if not script_path.exists():
            return {
                "success": False,
                "error": "No generated script found for evaluation",
            }

        processor = ArtifactProcessor(
            interpreter=self.interpreter,
            skills_dir=self.skill_root,
        )
        evaluator = TaskEvaluator(
            task_index=self.index,
            layout=self.layout,
            artifact_processor=processor,
        )

        print(f"  [{source}/{task_id}] evaluating...")

        async with EvalClient() as client:
            result = await evaluator.evaluate(
                task_id=task_id,
                eval_client=client,
            )

        success = not result.get("parse_error", False)
        return {
            "success": success,
            "overall_score": result.get("overall_score"),
            "error": result.get("summary") if not success else None,
        }


def main():
    parser = argparse.ArgumentParser(description="独立评估入口")
    parser.add_argument("--sources", type=str, default=None)
    parser.add_argument("--tasks", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--concurrent", type=int, default=4)
    parser.add_argument("--rerun-failed", action="store_true")
    args = parser.parse_args()

    sources = args.sources.split(",") if args.sources else None
    task_ids = args.tasks.split(",") if args.tasks else None
    interpreter = load_interpreter()

    index = TaskIndex("benchmark/datasets.json")
    layout = BenchmarkLayout(".")

    print("=" * 60)
    print("独立评估")
    print("=" * 60)

    runner = EvaluationRunner(
        task_index=index,
        layout=layout,
        interpreter=interpreter,
        skill_root=Path("skills"),
        max_concurrent=args.concurrent,
    )

    asyncio.run(runner.run(
        task_ids=task_ids,
        sources=sources,
        limit=args.limit,
        skip_success=True,
        skip_failed=not args.rerun_failed,
    ))


if __name__ == "__main__":
    main()