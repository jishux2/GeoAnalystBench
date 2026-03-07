# codes/evaluator/summary_reporter.py
"""
全局汇总报告生成器
聚合所有已完成任务的评估结果，输出结构化统计报告
"""

import json
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
from collections import Counter


class SummaryReporter:
    """全局汇总报告生成器"""

    def __init__(self, workspace_root: str = "evaluation_workspace"):
        self.workspace_root = Path(workspace_root)

    def generate(self, output_path: str = "results/evaluation_report.md") -> str:
        """
        生成汇总报告

        Args:
            output_path: 报告输出路径

        Returns:
            报告文本内容
        """
        tasks = self._collect_all_tasks()

        if not tasks:
            return "No completed tasks found."

        report = self._render_report(tasks)

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(report, encoding='utf-8')

        return report

    def _collect_all_tasks(self) -> List[Dict]:
        """扫描工作空间收集所有任务数据"""
        tasks = []

        if not self.workspace_root.exists():
            return tasks

        for task_dir in sorted(self.workspace_root.iterdir(), key=lambda p: int(p.name) if p.name.isdigit() else 0):
            if not task_dir.is_dir() or not task_dir.name.isdigit():
                continue

            task_id = int(task_dir.name)

            # 读取对话历史
            history_path = task_dir / "dialogue_history.json"
            if not history_path.exists():
                continue

            with open(history_path, 'r', encoding='utf-8') as f:
                history = json.load(f)

            status = history.get('status', 'unknown')
            if status not in ('success', 'failed'):
                continue

            # 读取评估结果
            eval_path = task_dir / "outputs" / "evaluation.json"
            evaluation = None
            if eval_path.exists():
                with open(eval_path, 'r', encoding='utf-8') as f:
                    evaluation = json.load(f)

            # 统计轮次信息
            rounds = history.get('rounds', [])
            rounds_used = len(rounds)

            # 统计工具调用
            tool_usage = Counter()
            total_turns = 0
            for round_dir in sorted((task_dir / "outputs").iterdir()) if (task_dir / "outputs").exists() else []:
                if not round_dir.is_dir():
                    continue
                trace_path = round_dir / "debug_trace.json"
                if trace_path.exists():
                    with open(trace_path, 'r', encoding='utf-8') as f:
                        trace = json.load(f)
                    for turn in trace.get('turns', []):
                        total_turns += 1
                        for tc in turn.get('tool_calls', []):
                            name = tc.get('function', {}).get('name', 'unknown')
                            tool_usage[name] += 1

            tasks.append({
                'task_id': task_id,
                'status': status,
                'rounds_used': rounds_used,
                'max_rounds': history.get('max_rounds', 3),
                'total_turns': total_turns,
                'tool_usage': dict(tool_usage),
                'categories': history.get('metadata', {}).get('categories', []),
                'failure_reason': history.get('failure_reason'),
                'final_diagnosis': history.get('diagnosis'),
                'evaluation': evaluation
            })

        return tasks

    def _render_report(self, tasks: List[Dict]) -> str:
        """渲染完整报告"""
        sections = []

        sections.append("# Iterative Repair Evaluation Report\n")
        sections.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        # 总览
        sections.append(self._render_overview(tasks))

        # 按类别统计
        sections.append(self._render_category_breakdown(tasks))

        # 工具使用分布
        sections.append(self._render_tool_distribution(tasks))

        # 逐任务详情
        sections.append(self._render_task_details(tasks))

        return '\n'.join(sections)

    def _render_overview(self, tasks: List[Dict]) -> str:
        """渲染总览统计"""
        total = len(tasks)
        success = sum(1 for t in tasks if t['status'] == 'success')
        failed = total - success

        success_rounds = [t['rounds_used'] for t in tasks if t['status'] == 'success']
        failed_rounds = [t['rounds_used'] for t in tasks if t['status'] == 'failed']

        avg_success_rounds = sum(success_rounds) / len(success_rounds) if success_rounds else 0
        avg_failed_rounds = sum(failed_rounds) / len(failed_rounds) if failed_rounds else 0

        total_turns = sum(t['total_turns'] for t in tasks)
        avg_turns = total_turns / total if total else 0

        lines = []
        lines.append("## Overview\n")
        lines.append(f"| Metric | Value |")
        lines.append(f"|--------|-------|")
        lines.append(f"| Total tasks | {total} |")
        lines.append(f"| Successful | {success} ({success/total*100:.1f}%) |")
        lines.append(f"| Failed | {failed} ({failed/total*100:.1f}%) |")
        lines.append(f"| Avg rounds (success) | {avg_success_rounds:.1f} |")
        lines.append(f"| Avg rounds (failed) | {avg_failed_rounds:.1f} |")
        lines.append(f"| Total agent turns | {total_turns} |")
        lines.append(f"| Avg turns per task | {avg_turns:.1f} |")
        lines.append("")

        return '\n'.join(lines)

    def _render_category_breakdown(self, tasks: List[Dict]) -> str:
        """按方法论类别渲染统计"""
        category_stats = {}

        for task in tasks:
            for cat in task['categories']:
                if cat not in category_stats:
                    category_stats[cat] = {'total': 0, 'success': 0}
                category_stats[cat]['total'] += 1
                if task['status'] == 'success':
                    category_stats[cat]['success'] += 1

        if not category_stats:
            return ""

        lines = []
        lines.append("## Performance by Category\n")
        lines.append("| Category | Total | Success | Rate |")
        lines.append("|----------|-------|---------|------|")

        for cat in sorted(category_stats.keys()):
            stats = category_stats[cat]
            rate = stats['success'] / stats['total'] * 100 if stats['total'] else 0
            lines.append(f"| {cat} | {stats['total']} | {stats['success']} | {rate:.1f}% |")

        lines.append("")
        return '\n'.join(lines)

    def _render_tool_distribution(self, tasks: List[Dict]) -> str:
        """渲染工具使用分布"""
        global_usage = Counter()
        for task in tasks:
            for name, count in task['tool_usage'].items():
                global_usage[name] += count

        if not global_usage:
            return ""

        lines = []
        lines.append("## Tool Usage Distribution\n")
        lines.append("| Tool | Invocations |")
        lines.append("|------|-------------|")

        for name, count in global_usage.most_common():
            lines.append(f"| {name} | {count} |")

        lines.append("")
        return '\n'.join(lines)

    def _render_task_details(self, tasks: List[Dict]) -> str:
        """渲染逐任务详情"""
        lines = []
        lines.append("## Task Details\n")

        for task in tasks:
            status_icon = "✓" if task['status'] == 'success' else "✗"
            lines.append(f"### Task {task['task_id']} {status_icon}\n")

            lines.append(f"- **Status**: {task['status']}")
            lines.append(f"- **Rounds**: {task['rounds_used']} / {task['max_rounds']}")
            lines.append(f"- **Agent turns**: {task['total_turns']}")

            if task['categories']:
                lines.append(f"- **Categories**: {', '.join(task['categories'])}")

            if task['status'] == 'failed' and task.get('failure_reason'):
                lines.append(f"- **Failure reason**: {task['failure_reason']}")

            # 最终诊断摘要
            diag = task.get('final_diagnosis')
            if diag and diag.get('root_cause'):
                root_cause = diag['root_cause']
                if len(root_cause) > 200:
                    root_cause = root_cause[:197] + "..."
                lines.append(f"\n**Final diagnosis**: {root_cause}")

            # AI评估摘要
            evaluation = task.get('evaluation')
            if evaluation and evaluation.get('analysis'):
                analysis = evaluation['analysis']
                if len(analysis) > 500:
                    analysis = analysis[:497] + "..."
                lines.append(f"\n**AI evaluation**: {analysis}")

            lines.append("")

        return '\n'.join(lines)