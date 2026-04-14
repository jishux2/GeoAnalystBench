# codes/benchmark/setup_dirs.py
"""
目录结构初始化脚本

从 datasets.json 读取任务清单，构建 benchmark_ground_truth
和 benchmark_workspace 的完整目录树，并将参考代码部署到位。

运行方式：
    python codes/benchmark/setup_dirs.py
"""

import sys
from pathlib import Path

# 确保 codes/ 在导入路径中
sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmark.task_index import TaskIndex
from benchmark.workspace_setup import BenchmarkLayout, WorkspaceBuilder


def main():
    print("加载任务索引...")
    index = TaskIndex("benchmark/datasets.json")
    print(f"  共 {len(index.all_task_ids)} 个任务，{len(index.sources)} 个来源")

    layout = BenchmarkLayout(".")
    builder = WorkspaceBuilder(layout, index)

    # 顺便执行排序，修复 JSON 中的错位问题
    print("\n按来源重排任务顺序...")
    index.reorder_by_source()
    index.save()
    print("  datasets.json 已更新")

    print("\n构建目录树...")
    builder.build_all(deploy_reference=True)

    # 统计部署情况
    deployed = 0
    no_code = 0
    for tid in index.all_task_ids:
        entry = index.get(tid)
        script = layout.ground_truth_script(entry["source"], tid)
        if script.exists():
            deployed += 1
        else:
            no_code += 1

    print(f"\n完成：")
    print(f"  参考脚本已部署：{deployed}")
    print(f"  无参考代码：    {no_code}")
    print(f"  ground_truth：  {layout.ground_truth_root}")
    print(f"  workspace：     {layout.workspace_root}")


if __name__ == "__main__":
    main()