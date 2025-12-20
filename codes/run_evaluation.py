# codes/run_evaluation.py
"""
GeoCode Validator - 代码执行与评估系统

自动化验证模型生成的地理空间分析代码的可执行性。
核心流程包括：
1. 从推理结果中提取Python脚本
2. 注入必要的运行时辅助逻辑（目录创建、后端配置等）
3. 在隔离环境中并发执行代码
4. 收集执行日志与错误信息
5. 生成多维度统计报告

典型用法：
    python codes/run_evaluation.py

配置项：
    - 任务筛选条件：修改filter_tasks()的参数
    - 并发度与超时：调整CodeExecutor()的初始化参数
    - 提示词类型：修改prompt_type（默认domain_and_dataset）
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
    
    # 初始化工作空间管理器
    # 该组件负责从CSV结果文件提取代码、构建任务索引、管理目录结构
    workspace_mgr = WorkspaceManager()
    
    # 第一步：按条件筛选待评测任务
    # 支持通过开源属性、方法论类别、任务ID等维度组合过滤
    # 当前配置聚焦于DR和F类别的开源任务，对应论文中模型表现较弱的维度
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
    # 从responses.csv提取模型生成的代码，注入必要的辅助逻辑
    # force_overwrite=True确保应用最新的代码增强策略（如matplotlib后端设置）
    workspace_mgr.setup_workspace(
        task_ids=task_ids,
        prompt_type="domain_and_dataset",
        force_overwrite=True
    )
    
    # 第三步：配置代码执行引擎
    # timeout控制单任务最长运行时间，防止死循环或资源耗尽
    # max_workers决定并发度，需在执行效率与系统稳定性间权衡
    # 地理空间处理为CPU密集型操作，过高并发可能导致内存竞争
    executor = CodeExecutor(
        timeout=300,
        max_workers=4
    )
    
    # 批量执行已提取的代码
    # use_concurrent=True启用进程池并发，显著缩短总耗时
    # 若遇到matplotlib内存分配错误等并发问题，可临时改为False串行执行
    results = executor.execute_batch(
        task_ids=task_ids,
        prompt_type="domain_and_dataset",
        use_concurrent=True
    )
    
    # 第四步：生成评估报告
    # 报告采集工作空间内所有已执行任务的状态，支持增量式累积统计
    # 输出包括JSON格式的详细数据和终端打印的可读摘要
    reporter = EvaluationReporter()
    report = reporter.generate_summary_report()
    reporter.print_summary(report)
    
    print("\n评估流程完成！")


if __name__ == "__main__":
    main()