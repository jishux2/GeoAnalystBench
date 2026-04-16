# codes/benchmark/export_dashboard_data.py
"""
将任务索引和各维度状态汇总为前端面板可消费的JSON文件。

扫描 datasets.json、ground_truth 和 workspace 下的状态文件，
输出一份 dashboard_data.json 供 HTML 面板加载。

运行方式：
    python codes/benchmark/export_dashboard_data.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmark.task_index import TaskIndex
from benchmark.workspace_setup import BenchmarkLayout


EVAL_DIMENSIONS = [
    "task_alignment", "execution_validity", "output_completeness",
    "result_fidelity", "visualization_quality", "code_craftsmanship",
]


def export(output_path: str = "dashboard_data.json"):
    index = TaskIndex("benchmark/datasets.json")
    layout = BenchmarkLayout(".")

    tasks = []
    for task_id in index.all_task_ids:
        entry = index.get(task_id)
        source = entry["source"]

        # 基础信息
        record = {
            "task_ID": task_id,
            "source": source,
            "task_text": entry.get("task_text", ""),
            "solvable": entry.get("solvable", ""),
            "has_reference": index._has_reference_code(task_id),
        }

        # 参考代码执行状态
        ref_status_path = (
            layout.ground_truth_task_dir(source, task_id) / "run_status.json"
        )
        if ref_status_path.exists():
            with open(ref_status_path, "r", encoding="utf-8") as f:
                ref_data = json.load(f)
            record["ref_status"] = ref_data.get("status", "pending")
        elif not record["has_reference"]:
            record["ref_status"] = "no_code"
        else:
            record["ref_status"] = "pending"

        # 智能体运行状态
        agent_status_path = (
            layout.workspace_task_dir(source, task_id) / "run_status.json"
        )
        if agent_status_path.exists():
            with open(agent_status_path, "r", encoding="utf-8") as f:
                agent_data = json.load(f)
            record["agent_status"] = agent_data.get("status", "pending")
        else:
            record["agent_status"] = "pending"

        # 评估状态
        eval_status_path = (
            layout.workspace_task_dir(source, task_id) / "eval_status.json"
        )
        eval_result_path = (
            layout.workspace_task_dir(source, task_id) / "eval_result.json"
        )
        if eval_status_path.exists():
            with open(eval_status_path, "r", encoding="utf-8") as f:
                eval_data = json.load(f)
            record["eval_status"] = eval_data.get("status", "pending")
        else:
            record["eval_status"] = "pending"

        # 评估分数、维度详情、评语与时间戳
        record["eval_score"] = None
        record["eval_dimensions"] = {}
        record["eval_rationales"] = {}
        record["eval_completed_at"] = None

        if eval_result_path.exists():
            try:
                with open(eval_result_path, "r", encoding="utf-8") as f:
                    eval_result = json.load(f)
                if not eval_result.get("parse_error"):
                    record["eval_score"] = eval_result.get("overall_score")
                    record["eval_status"] = "completed"
                    record["eval_summary"] = eval_result.get("summary", "")
                    for dim in EVAL_DIMENSIONS:
                        if dim in eval_result and isinstance(eval_result[dim], dict):
                            record["eval_dimensions"][dim] = eval_result[dim].get("score")
                            record["eval_rationales"][dim] = eval_result[dim].get("rationale", "")
                    # 文件修改时间作为评估完成的时序依据
                    record["eval_completed_at"] = os.path.getmtime(str(eval_result_path))
            except (json.JSONDecodeError, KeyError):
                pass

        tasks.append(record)

    # 统计元数据
    source_counts = {}
    for t in tasks:
        source_counts[t["source"]] = source_counts.get(t["source"], 0) + 1

    output = {
        "generated_at": __import__("datetime").datetime.now().isoformat(),
        "total": len(tasks),
        "source_counts": source_counts,
        "tasks": tasks,
    }

    out_path = Path(output_path)
    out_path.write_text(
        json.dumps(output, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Exported {len(tasks)} tasks to {out_path.resolve()}")


if __name__ == "__main__":
    output = sys.argv[1] if len(sys.argv) > 1 else "dashboard_data.json"
    export(output)