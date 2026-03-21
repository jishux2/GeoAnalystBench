# codes/run_iterative_repair.py
"""
多智能体协作修复系统主入口
"""

import sys
import os
import asyncio
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from orchestrator import IterativeRepairOrchestrator
from evaluator.workspace_manager import WorkspaceManager


def main():
    api_key = os.getenv("DEEPSEEK_API_KEY")

    if not api_key:
        print("错误：未设置DEEPSEEK_API_KEY环境变量")
        print("  Linux/macOS: export DEEPSEEK_API_KEY='your_key'")
        print("  Windows CMD: set DEEPSEEK_API_KEY=your_key")
        print("  PowerShell:  $env:DEEPSEEK_API_KEY='your_key'")
        return

    workspace_mgr = WorkspaceManager()

    task_ids = workspace_mgr.filter_tasks(
        opensource_only=True,
        categories=["DR", "F"],
    )

    print(f"筛选结果：{len(task_ids)}个任务")
    print(f"任务ID：{task_ids}")

    if not task_ids:
        print("未找到符合条件的任务")
        return

    orchestrator = IterativeRepairOrchestrator(
        api_key=api_key,
        max_concurrent=4,
        temperature=0.7,
        workspace_root="evaluation_workspace",
        skill_root="skills",
        check_interval=60.0,
        task_timeout=900.0,
    )

    try:
        asyncio.run(orchestrator.run(task_ids))

        # from evaluator.summary_reporter import SummaryReporter
        # reporter = SummaryReporter()
        # reporter.generate()
        # print("汇总报告已生成")

    except KeyboardInterrupt:
        print("\n流程被用户中断，已完成的结果已保存")

    except Exception as e:
        print(f"执行过程中发生错误：{e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()