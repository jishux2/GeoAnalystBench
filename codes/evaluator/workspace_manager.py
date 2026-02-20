# codes/evaluator/workspace_manager.py
"""
评测工作空间管理器
负责目录结构初始化和模型生成代码的物理化
"""

import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Set


class WorkspaceManager:
    """评测工作空间管理器"""
    
    # 空间分析方法论分类的标准描述
    TASK_CATEGORIES = {
        'DP': 'detecting and quantifying patterns',
        'DR': 'determining how places are related',
        'F': 'finding the best locations and paths',
        'M': 'measuring size, shape, and distribution',
        'S': 'spatial interpolation and predictive modeling',
        'U': 'understanding where'
    }
    
    # 数据集中采用完整描述，通过反向查找建立缩写索引
    # 首字母大写是为了匹配CSV中"Making predictions"这类格式
    CATEGORY_REVERSE_MAP = {
        desc.capitalize(): abbr 
        for abbr, desc in TASK_CATEGORIES.items()
    }
    
    def __init__(
        self,
        workspace_root: str = "evaluation_workspace",
        dataset_path: str = "dataset/GeoAnalystBench.csv",
        responses_path: str = "results/deepseek/code_responses.csv"
    ):
        """
        初始化管理器
        
        Args:
            workspace_root: 评测工作空间根目录
            dataset_path: 基准测试集路径
            responses_path: 模型响应结果路径
        """
        self.workspace_root = Path(workspace_root)
        self.dataset_path = Path(dataset_path)
        self.responses_path = Path(responses_path)
        
        # 加载任务元数据
        self.tasks_df = pd.read_csv(dataset_path)
        self.responses_df = None
        
        # 任务配置索引
        self.task_configs: Dict[int, Dict] = {}
        self._build_task_index()
    
    def _build_task_index(self):
        """构建任务配置索引"""
        for idx, row in self.tasks_df.iterrows():
            task_id = idx + 1  # 任务ID从1开始
            
            # 从三个类别列中提取方法论标签
            # CSV存储的是完整英文描述，需转换为缩写形式以便筛选
            categories = []
            for cat_col in ['Task Categories1', 'Task Categories2', 'Task Categories3']:
                if pd.notna(row.get(cat_col)):
                    cat_full = row[cat_col].strip()
                    cat_abbr = self.CATEGORY_REVERSE_MAP.get(cat_full)
                    if cat_abbr:
                        categories.append(cat_abbr)
            
            self.task_configs[task_id] = {
                'id': task_id,
                'title': row['Task'],
                'is_opensource': row['Open Source'] == 'T',
                'categories': categories,
                'reference_code': row['CodeString'],
                'workflow_length': row['Task Length']
            }
    
    def filter_tasks(
        self,
        opensource_only: bool = False,
        categories: Optional[List[str]] = None,
        task_ids: Optional[List[int]] = None
    ) -> List[int]:
        """
        按条件筛选任务
        
        Args:
            opensource_only: 仅包含开源任务
            categories: 方法论类别列表（如['DR', 'F']）
            task_ids: 明确指定的任务ID列表
        
        Returns:
            符合条件的任务ID列表
        """
        selected = set(self.task_configs.keys())
        
        # 应用开源过滤
        if opensource_only:
            selected = {
                tid for tid in selected
                if self.task_configs[tid]['is_opensource']
            }
        
        # 应用类别过滤
        if categories:
            selected = {
                tid for tid in selected
                if any(cat in self.task_configs[tid]['categories'] for cat in categories)
            }
        
        # 应用ID过滤
        if task_ids:
            selected = selected & set(task_ids)
        
        return sorted(selected)