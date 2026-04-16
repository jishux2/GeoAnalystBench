# codes/benchmark/check_datasets.py
"""
数据文件可达性预检

遍历任务索引中的全部条目，逐一验证 used_dataset 字段
声明的文件是否存在于对应来源的数据集目录下。对 Shapefile
额外检查 .dbf/.shx/.prj 伴生文件的齐备性。

输出缺失清单供操作者人工审查与补全。

运行方式：
    python codes/benchmark/check_datasets.py [--sources SRC1,SRC2] [--tasks ID1,ID2]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Set

sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmark.task_index import TaskIndex
from benchmark.workspace_setup import BenchmarkLayout

# Shapefile 的必要伴生文件后缀
SHP_COMPANIONS = [".dbf", ".shx"]
# 可选但推荐的伴生文件
SHP_OPTIONAL = [".prj", ".cpg"]


def check_single_task(
    task_id: str,
    entry: Dict,
    layout: BenchmarkLayout,
) -> List[Dict]:
    """
    检查单个任务的数据文件可达性。

    Returns:
        缺失项列表，每项含 file（文件名）、kind（缺失类别）、detail（说明）
    """
    source = entry["source"]
    dataset_dir = layout.dataset_dir(source)
    missing = []

    files = TaskIndex._flatten_dataset_refs(entry.get("used_dataset", []))
    if not files:
        return missing

    for filename in files:
        filepath = dataset_dir / filename
        if not filepath.exists():
            missing.append({
                "file": filename,
                "kind": "primary",
                "detail": f"Not found in {dataset_dir}",
            })
            continue

        # Shapefile 伴生文件检查
        if filepath.suffix.lower() == ".shp":
            stem = filepath.stem
            for companion_suffix in SHP_COMPANIONS:
                companion = dataset_dir / f"{stem}{companion_suffix}"
                if not companion.exists():
                    missing.append({
                        "file": f"{stem}{companion_suffix}",
                        "kind": "companion_required",
                        "detail": f"Required Shapefile component missing for {filename}",
                    })
            for optional_suffix in SHP_OPTIONAL:
                companion = dataset_dir / f"{stem}{optional_suffix}"
                if not companion.exists():
                    missing.append({
                        "file": f"{stem}{optional_suffix}",
                        "kind": "companion_optional",
                        "detail": f"Optional Shapefile component missing for {filename}",
                    })

    return missing


def main():
    parser = argparse.ArgumentParser(description="数据文件可达性预检")
    parser.add_argument("--sources", type=str, default=None,
                        help="限定来源（逗号分隔）")
    parser.add_argument("--tasks", type=str, default=None,
                        help="限定任务ID（逗号分隔）")
    args = parser.parse_args()

    sources = args.sources.split(",") if args.sources else None
    task_ids = args.tasks.split(",") if args.tasks else None

    index = TaskIndex("benchmark/datasets.json")
    layout = BenchmarkLayout(".")

    candidates = index.filter(sources=sources, task_ids=task_ids)

    # 按严重程度分桶收集
    blocking_tasks = []     # primary 或 companion_required
    optional_tasks = []     # 仅有 companion_optional

    total_primary = 0
    total_required = 0
    total_optional = 0

    for tid in candidates:
        entry = index.get(tid)
        problems = check_single_task(tid, entry, layout)
        if not problems:
            continue

        blocking = [p for p in problems if p["kind"] in ("primary", "companion_required")]
        optional = [p for p in problems if p["kind"] == "companion_optional"]

        for p in problems:
            if p["kind"] == "primary":
                total_primary += 1
            elif p["kind"] == "companion_required":
                total_required += 1
            else:
                total_optional += 1

        if blocking:
            blocking_tasks.append((tid, entry["source"], blocking, optional))
        elif optional:
            optional_tasks.append((tid, entry["source"], optional))

    # 输出报告
    print("=" * 60)
    print("数据文件可达性预检报告")
    print("=" * 60)
    print(f"  扫描任务数：          {len(candidates)}")
    print(f"  阻断性缺失的任务：    {len(blocking_tasks)}")
    print(f"  仅可选缺失的任务：    {len(optional_tasks)}")
    print(f"  无缺失的任务：        {len(candidates) - len(blocking_tasks) - len(optional_tasks)}")
    print("-" * 60)
    print(f"  主文件缺失：          {total_primary}")
    print(f"  必要伴生文件缺失：    {total_required}")
    print(f"  可选伴生文件缺失：    {total_optional}")
    print("=" * 60)

    if not blocking_tasks and not optional_tasks:
        print("\n全部数据文件就位，无需补全。")
        return

    # 第一段：阻断性缺失，逐条列举
    if blocking_tasks:
        print(f"\n{'!' * 60}")
        print(f"  阻断性缺失（{len(blocking_tasks)} 个任务无法执行）")
        print(f"{'!' * 60}")

        for tid, source, blocking, optional in blocking_tasks:
            print(f"\n  [{source}/{tid}]")
            for p in blocking:
                label = "主文件" if p["kind"] == "primary" else "必要组件"
                print(f"    ✗ {p['file']}  ({label})")
            if optional:
                print(f"    + {len(optional)} 个可选伴生文件也缺失")

    # 第二段：可选缺失，按来源汇总而非逐条列举
    if optional_tasks:
        print(f"\n{'-' * 60}")
        print(f"  可选伴生文件缺失（{len(optional_tasks)} 个任务，不阻断执行）")
        print(f"{'-' * 60}")

        # 按来源聚合统计
        by_source: Dict[str, Dict[str, int]] = {}
        for tid, source, optional in optional_tasks:
            if source not in by_source:
                by_source[source] = {".prj": 0, ".cpg": 0}
            for p in optional:
                suffix = Path(p["file"]).suffix.lower()
                if suffix in by_source[source]:
                    by_source[source][suffix] += 1

        for source in sorted(by_source):
            counts = by_source[source]
            parts = []
            if counts[".prj"] > 0:
                parts.append(f".prj ×{counts['.prj']}")
            if counts[".cpg"] > 0:
                parts.append(f".cpg ×{counts['.cpg']}")
            print(f"  {source}: {', '.join(parts)}")

    # 涉及的数据集目录
    affected_sources: Set[str] = set()
    for tid, source, *_ in blocking_tasks:
        affected_sources.add(source)
    for tid, source, _ in optional_tasks:
        affected_sources.add(source)

    print(f"\n涉及的数据集目录：")
    for source in sorted(affected_sources):
        print(f"  {layout.dataset_dir(source)}")


if __name__ == "__main__":
    main()