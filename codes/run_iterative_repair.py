# codes/run_iterative_repair.py
"""
迭代修复系统主入口
支持增量执行、断点续跑和灵活的任务筛选
"""

import sys
import os
import asyncio
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from orchestrator import IterativeRepairOrchestrator
from evaluator.workspace_manager import WorkspaceManager


def main():
    """主执行函数"""
    # 从环境变量获取API密钥
    api_key = os.getenv("DEEPSEEK_API_KEY")
    
    if not api_key:
        print("错误：未设置DEEPSEEK_API_KEY环境变量")
        print("\n请根据操作系统选择对应命令：")
        print("  Linux/macOS: export DEEPSEEK_API_KEY='your_api_key_here'")
        print("  Windows CMD: set DEEPSEEK_API_KEY=your_api_key_here")
        print("  PowerShell:  $env:DEEPSEEK_API_KEY='your_api_key_here'")
        return
    
    print("="*60)
    print("GeoAnalystBench - 迭代修复系统")
    print("="*60)
    print(f"API密钥：{'*' * (len(api_key) - 8)}{api_key[-8:]}")
    print("="*60)
    
    # 初始化工作空间管理器用于任务筛选
    workspace_mgr = WorkspaceManager()
    
    # 筛选任务：开源任务中的DR和F类别
    task_ids = workspace_mgr.filter_tasks(
        opensource_only=True,
        categories=['DR', 'F']
    )
    
    print(f"\n筛选结果：{len(task_ids)}个任务")
    print(f"任务ID：{task_ids}\n")
    
    if not task_ids:
        print("未找到符合条件的任务，退出")
        return
    
    # 创建编排器
    orchestrator = IterativeRepairOrchestrator(
        api_key=api_key,
        max_rounds=3,
        max_concurrent=4,  # 并发数可根据机器性能调整
        temperature=0.7,
        enable_thinking=True,
        workspace_root="evaluation_workspace"
    )
    
    # 执行迭代修复
    try:
        asyncio.run(orchestrator.run(task_ids))
        print("\n迭代修复完成！")
    
    except KeyboardInterrupt:
        print("\n\n修复流程被用户中断")
        print("已完成的结果已保存，下次运行将自动跳过成功的任务")
    
    except Exception as e:
        print(f"\n执行过程中发生错误：{e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()