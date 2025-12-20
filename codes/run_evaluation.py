# codes/run_evaluation.py
"""
GeoCode Validator - 代码执行与评估系统
自动化验证模型生成代码的可执行性
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from evaluator.workspace_manager import WorkspaceManager
from evaluator.code_executor import CodeExecutor
from evaluator.evaluation_reporter import EvaluationReporter


def main():
    """主执行流程"""
    print("="*60)
    print("GeoCode Validator - 代码执行评估系统")
    print("="*60)
    
    # 初始化管理器
    workspace_mgr = WorkspaceManager()
    
    # 第一步：筛选任务
    # 示例：仅执行开源任务中DR和F类别的任务
    print("\n正在筛选任务...")
    task_ids = workspace_mgr.filter_tasks(
        opensource_only=True,
        categories=['DR', 'F']
    )
    
    print(f"筛选结果：共{len(task_ids)}个任务")
    print(f"任务ID：{task_ids}\n")
    
    if not task_ids:
        print("未找到符合条件的任务，退出")
        return
    
    # 第二步：初始化工作空间
    workspace_mgr.setup_workspace(
        task_ids=task_ids,
        prompt_type="domain_and_dataset",
        force_overwrite=False
    )
    
    # 第三步：执行代码
    executor = CodeExecutor(
        timeout=300,  # 5分钟超时
        max_workers=4  # 并发数
    )
    
    results = executor.execute_batch(
        task_ids=task_ids,
        prompt_type="domain_and_dataset",
        use_concurrent=True  # 启用并发
    )
    
    # 第四步：生成报告
    reporter = EvaluationReporter()
    report = reporter.generate_summary_report()
    reporter.print_summary(report)
    
    print("\n评估流程完成！")


if __name__ == "__main__":
    main()