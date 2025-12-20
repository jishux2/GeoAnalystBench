# codes/evaluator/evaluation_reporter.py
"""
评估报告生成器
汇总执行结果并生成分析报告
"""

import json
import pandas as pd
from pathlib import Path
from typing import List, Dict
from datetime import datetime
from collections import defaultdict


class EvaluationReporter:
    """评估报告生成器"""
    
    def __init__(self, workspace_root: str = "evaluation_workspace"):
        self.workspace_root = Path(workspace_root)
    
    def generate_summary_report(
        self,
        output_path: str = "results/evaluation_summary.json"
    ) -> Dict:
        """
        生成评估摘要报告
        
        Returns:
            包含统计信息的字典
        """
        # 收集所有任务的执行状态
        task_results = []
        
        for task_dir in sorted(self.workspace_root.iterdir()):
            if not task_dir.is_dir():
                continue
            
            config_path = task_dir / "evaluation.json"
            if not config_path.exists():
                continue
            
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                task_results.append(config)
            except Exception as e:
                print(f"警告：读取{config_path}失败 - {e}")
        
        if not task_results:
            return {'error': '未找到任何评估结果'}
        
        # 统计分析
        total = len(task_results)
        success = sum(1 for r in task_results if r.get('execution_status') == 'success')
        failed = sum(1 for r in task_results if r.get('execution_status') == 'failed')
        pending = sum(1 for r in task_results if r.get('execution_status') == 'pending')
        
        # 按类别统计
        category_stats = defaultdict(lambda: {'total': 0, 'success': 0, 'failed': 0})
        for result in task_results:
            for cat in result.get('categories', []):
                category_stats[cat]['total'] += 1
                if result.get('execution_status') == 'success':
                    category_stats[cat]['success'] += 1
                elif result.get('execution_status') == 'failed':
                    category_stats[cat]['failed'] += 1
        
        # 错误类型统计
        error_types = defaultdict(int)
        for result in task_results:
            if result.get('error_type'):
                error_types[result['error_type']] += 1
        
        # 构建报告
        report = {
            'generated_at': datetime.now().isoformat(),
            'overall': {
                'total_tasks': total,
                'success': success,
                'failed': failed,
                'pending': pending,
                'success_rate': f"{success/total*100:.2f}%" if total > 0 else "0%"
            },
            'by_category': {
                cat: {
                    'total': stats['total'],
                    'success': stats['success'],
                    'failed': stats['failed'],
                    'success_rate': f"{stats['success']/stats['total']*100:.2f}%" if stats['total'] > 0 else "0%"
                }
                for cat, stats in category_stats.items()
            },
            'error_distribution': dict(error_types),
            'opensource_vs_closed': self._analyze_opensource_split(task_results)
        }
        
        # 保存报告
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"\n评估报告已保存至：{output_path}")
        
        return report
    
    def _analyze_opensource_split(self, task_results: List[Dict]) -> Dict:
        """分析开源/闭源任务的执行差异"""
        opensource = {'total': 0, 'success': 0}
        closed = {'total': 0, 'success': 0}
        
        for result in task_results:
            if result.get('is_opensource'):
                opensource['total'] += 1
                if result.get('execution_status') == 'success':
                    opensource['success'] += 1
            else:
                closed['total'] += 1
                if result.get('execution_status') == 'success':
                    closed['success'] += 1
        
        return {
            'opensource': {
                **opensource,
                'success_rate': f"{opensource['success']/opensource['total']*100:.2f}%" if opensource['total'] > 0 else "0%"
            },
            'closed_source': {
                **closed,
                'success_rate': f"{closed['success']/closed['total']*100:.2f}%" if closed['total'] > 0 else "0%"
            }
        }
    
    def print_summary(self, report: Dict):
        """在终端打印摘要信息"""
        print("\n" + "="*60)
        print("评估结果摘要")
        print("="*60)
        
        overall = report['overall']
        print(f"\n整体统计：")
        print(f"  总任务数：{overall['total_tasks']}")
        print(f"  成功：{overall['success']} ({overall['success_rate']})")
        print(f"  失败：{overall['failed']}")
        print(f"  待执行：{overall['pending']}")
        
        print(f"\n按类别统计：")
        for cat, stats in report['by_category'].items():
            cat_name = {
                'DP': '模式检测',
                'DR': '位置关联',
                'F': '路径优化',
                'M': '形态测量',
                'S': '空间插值',
                'U': '位置理解'
            }.get(cat, cat)
            
            print(f"  {cat_name}（{cat}）：{stats['success']}/{stats['total']} ({stats['success_rate']})")
        
        if report.get('error_distribution'):
            print(f"\n常见错误类型：")
            sorted_errors = sorted(
                report['error_distribution'].items(),
                key=lambda x: x[1],
                reverse=True
            )
            for error_type, count in sorted_errors[:5]:
                print(f"  {error_type}：{count}次")