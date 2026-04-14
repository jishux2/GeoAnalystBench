# codes/benchmark/run_reference.py
"""
参考代码批量执行入口

从 datasets.json 加载任务清单，筛选待执行的任务，
并发运行参考脚本并记录状态。

运行方式：
    python codes/benchmark/run_reference.py [选项]

选项：
    --sources SRC1,SRC2    限定来源范围（逗号分隔）
    --tasks ID1,ID2        限定任务 ID（逗号分隔）
    --limit N              最多执行 N 个任务
    --concurrent N         并发数（默认 4）
    --timeout N            单脚本超时秒数（默认 300）
    --rerun-failed         重新执行已失败的任务
    --interpreter PATH     Python 解释器路径
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmark.task_index import TaskIndex
from benchmark.workspace_setup import BenchmarkLayout
from benchmark.reference_runner import ReferenceRunner


def load_interpreter() -> str:
    """从配置文件加载解释器路径。"""
    config_path = Path("codes/evaluator_config.json")
    if not config_path.exists():
        print("错误：未找到 evaluator_config.json")
        print("请先运行：python codes/setup_evaluation_env.py")
        sys.exit(1)
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    return config["opensource_interpreter"]


def main():
    parser = argparse.ArgumentParser(description="参考代码批量执行")
    parser.add_argument("--sources", type=str, default=None,
                        help="限定来源（逗号分隔）")
    parser.add_argument("--tasks", type=str, default=None,
                        help="限定任务 ID（逗号分隔）")
    parser.add_argument("--limit", type=int, default=None,
                        help="最多执行数量")
    parser.add_argument("--concurrent", type=int, default=4,
                        help="并发数")
    parser.add_argument("--timeout", type=int, default=300,
                        help="单脚本超时（秒）")
    parser.add_argument("--rerun-failed", action="store_true",
                        help="重新执行已失败的任务")
    parser.add_argument("--interpreter", type=str, default=None,
                        help="Python 解释器路径")
    args = parser.parse_args()

    # 解析参数
    sources = args.sources.split(",") if args.sources else None
    task_ids = args.tasks.split(",") if args.tasks else None
    interpreter = args.interpreter or load_interpreter()

    # 加载索引
    index = TaskIndex("benchmark/datasets.json")
    layout = BenchmarkLayout(".")

    print("=" * 60)
    print("参考代码批量执行")
    print("=" * 60)
    print(f"  任务总数：{len(index.all_task_ids)}")
    print(f"  解释器：  {interpreter}")
    print(f"  并发数：  {args.concurrent}")
    print(f"  超时：    {args.timeout}s")
    if sources:
        print(f"  来源筛选：{sources}")
    if task_ids:
        print(f"  ID 筛选： {task_ids}")
    if args.limit:
        print(f"  数量限制：{args.limit}")
    print("=" * 60)

    with ProcessPoolExecutor(max_workers=args.concurrent) as executor:
        runner = ReferenceRunner(
            task_index=index,
            layout=layout,
            interpreter=interpreter,
            max_concurrent=args.concurrent,
            script_timeout=args.timeout,
            executor=executor,
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