# codes/run_iterative_repair.py
"""
多智能体协作修复系统主入口
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from benchmark.task_index import TaskIndex
from benchmark.workspace_setup import BenchmarkLayout
from orchestrator import AgentOrchestrator


def load_interpreter() -> str:
    config_path = Path("codes/evaluator_config.json")
    if not config_path.exists():
        print("错误：未找到 evaluator_config.json")
        print("请先运行：python codes/setup_evaluation_env.py")
        sys.exit(1)
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    return config["opensource_interpreter"]


def main():
    parser = argparse.ArgumentParser(description="多智能体协作修复")
    parser.add_argument("--sources", type=str, default=None,
                        help="限定来源（逗号分隔）")
    parser.add_argument("--tasks", type=str, default=None,
                        help="限定任务ID（逗号分隔）")
    parser.add_argument("--limit", type=int, default=None,
                        help="最多执行数量")
    parser.add_argument("--concurrent", type=int, default=4,
                        help="并发数")
    parser.add_argument("--rerun-failed", action="store_true",
                        help="重新执行已失败的任务")
    parser.add_argument("--interpreter", type=str, default=None,
                        help="Python解释器路径")
    parser.add_argument("--timeout", type=float, default=3600.0,
                        help="单任务超时（秒）")
    parser.add_argument("--evaluate", action="store_true",
                        help="智能体完成后自动触发评估")
    args = parser.parse_args()

    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        print("错误：未设置DEEPSEEK_API_KEY环境变量")
        return

    sources = args.sources.split(",") if args.sources else None
    task_ids = args.tasks.split(",") if args.tasks else None
    interpreter = args.interpreter or load_interpreter()

    index = TaskIndex("benchmark/datasets.json")
    layout = BenchmarkLayout(".")

    print("=" * 60)
    print("多智能体协作修复系统")
    print("=" * 60)
    print(f"  任务总数：{len(index.all_task_ids)}")
    print(f"  解释器：  {interpreter}")
    print(f"  并发数：  {args.concurrent}")
    print(f"  超时：    {args.timeout}s")
    print("=" * 60)

    with ProcessPoolExecutor(max_workers=args.concurrent) as executor:
        orchestrator = AgentOrchestrator(
            task_index=index,
            layout=layout,
            api_key=api_key,
            interpreter=interpreter,
            max_concurrent=args.concurrent,
            temperature=0.7,
            check_interval=args.timeout,
            task_timeout=args.timeout,
            process_executor=executor,
            enable_evaluation=args.evaluate,
        )

        try:
            asyncio.run(orchestrator.run(
                task_ids=task_ids,
                sources=sources,
                limit=args.limit,
                skip_success=True,
                skip_failed=not args.rerun_failed,
            ))
        except KeyboardInterrupt:
            print("\n流程被用户中断，已完成的结果已保存")


if __name__ == "__main__":
    main()