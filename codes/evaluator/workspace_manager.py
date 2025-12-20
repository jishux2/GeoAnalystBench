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
    
    # 六种空间分析方法论分类
    CATEGORY_REVERSE_MAP = {
        'Detecting and quantifying patterns': 'DP',
        'Determining how places are related': 'DR',
        'Finding the best locations and paths': 'F',
        'Making predictions': 'S',
        'Measuring size, shape, and distribution': 'M',
        'Understanding where': 'U'
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
            
            # 解析方法论类别
            categories = []
            for cat_col in ['Task Categories1', 'Task Categories2', 'Task Categories3']:
                if pd.notna(row.get(cat_col)):
                    cat_full = row[cat_col].strip()
                    # 转换为缩写
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
    
    def extract_code_from_response(self, response_text: str) -> str:
        """
        从模型响应中提取Python代码
        
        处理多种可能的格式：
        1. 标准markdown代码块：```python\n...\n```
        2. 无语言标记的代码块：```\n...\n```
        3. 纯文本代码（无围栏标记）
        
        Args:
            response_text: 模型原始响应
        
        Returns:
            提取的纯代码文本
        """
        import re
        
        # 尝试提取markdown代码块
        pattern = r'```(?:python)?\s*(.*?)```'
        matches = re.findall(pattern, response_text, re.DOTALL)
        
        if matches:
            # 取最长的代码块（避免误提取示例片段）
            code = max(matches, key=len).strip()
        else:
            # 未找到代码块标记，假定全文为代码
            code = response_text.strip()
        
        # 规范化换行符
        code = code.replace('\r\n', '\n').replace('\r', '\n')
        
        # 自动添加pred_results目录创建逻辑
        if 'pred_results/' in code:
            if 'makedirs' not in code and 'mkdir' not in code.lower():
                lines = code.split('\n')
                insert_pos = 0
                
                for i, line in enumerate(lines):
                    if line.strip().startswith(('import ', 'from ')):
                        insert_pos = i + 1
                
                dir_code = [
                    '',
                    '# 确保输出目录存在',
                    "import os",
                    "os.makedirs('pred_results', exist_ok=True)",
                    ''
                ]
                
                lines = lines[:insert_pos] + dir_code + lines[insert_pos:]
                code = '\n'.join(lines)
        
        # 添加matplotlib后端设置（解决并发内存问题）
        if 'matplotlib.pyplot' in code and 'matplotlib.use' not in code:
            # 找到matplotlib.pyplot的导入位置
            lines = code.split('\n')
            
            for i, line in enumerate(lines):
                # 匹配 "import matplotlib.pyplot as plt" 或 "import matplotlib.pyplot"
                if 'import matplotlib.pyplot' in line:
                    # 在这一行之前插入后端设置
                    backend_code = [
                        'import matplotlib',
                        "matplotlib.use('Agg')  # 设置后端避免并发内存问题"
                    ]
                    lines = lines[:i] + backend_code + lines[i:]
                    code = '\n'.join(lines)
                    break
        
        return code
    
    def setup_workspace(
        self,
        task_ids: Optional[List[int]] = None,
        prompt_type: str = "domain_and_dataset",
        force_overwrite: bool = False
    ):
        """
        初始化评测工作空间
        
        为指定任务创建目录结构并提取生成代码
        
        Args:
            task_ids: 需要设置的任务ID列表，None表示全部任务
            prompt_type: 使用的提示词类型
            force_overwrite: 是否覆盖已存在的代码文件
        """
        # 加载模型响应数据
        if self.responses_df is None:
            if not self.responses_path.exists():
                raise FileNotFoundError(f"响应文件不存在：{self.responses_path}")
            self.responses_df = pd.read_csv(self.responses_path)
        
        # 确定待处理任务
        if task_ids is None:
            task_ids = list(self.task_configs.keys())
        
        print(f"\n开始初始化工作空间...")
        print(f"任务数量：{len(task_ids)}")
        print(f"提示词类型：{prompt_type}")
        print(f"覆盖模式：{'是' if force_overwrite else '否'}\n")
        
        success_count = 0
        skip_count = 0
        error_count = 0
        
        for task_id in task_ids:
            try:
                task_dir = self.workspace_root / str(task_id)
                task_dir.mkdir(parents=True, exist_ok=True)
                
                # 创建子目录
                (task_dir / "generated").mkdir(exist_ok=True)
                (task_dir / "outputs" / prompt_type / "results").mkdir(parents=True, exist_ok=True)
                
                # 提取并保存生成代码
                code_path = task_dir / "generated" / f"{prompt_type}.py"
                
                if code_path.exists() and not force_overwrite:
                    skip_count += 1
                    continue
                
                # 从响应数据中查找对应记录
                mask = (
                    (self.responses_df['task_id'] == task_id) &
                    (self.responses_df['prompt_type'] == prompt_type) &
                    (self.responses_df['error_info'].isna() | (self.responses_df['error_info'] == ''))
                )
                
                matching_responses = self.responses_df[mask]
                
                if matching_responses.empty:
                    print(f"警告：任务{task_id}未找到有效的{prompt_type}响应")
                    error_count += 1
                    continue
                
                # 取第一条有效响应（三次重复中的任意一次）
                response_text = matching_responses.iloc[0]['response_content']
                code = self.extract_code_from_response(response_text)
                
                # 写入代码文件
                with open(code_path, 'w', encoding='utf-8', newline='\n') as f:
                    f.write(code)
                
                # 创建任务级配置文件
                config = {
                    'task_id': task_id,
                    'title': self.task_configs[task_id]['title'],
                    'categories': self.task_configs[task_id]['categories'],
                    'is_opensource': self.task_configs[task_id]['is_opensource'],
                    'prompt_type': prompt_type,
                    'code_extracted': True,
                    'execution_status': 'pending'
                }
                
                config_path = task_dir / "evaluation.json"
                with open(config_path, 'w', encoding='utf-8') as f:
                    json.dump(config, f, indent=2, ensure_ascii=False)
                
                success_count += 1
            
            except Exception as e:
                print(f"错误：任务{task_id}设置失败 - {e}")
                error_count += 1
        
        print(f"\n工作空间初始化完成")
        print(f"成功：{success_count}")
        print(f"跳过：{skip_count}")
        print(f"失败：{error_count}")
    
    def get_task_info(self, task_id: int) -> Dict:
        """获取任务完整信息"""
        if task_id not in self.task_configs:
            raise ValueError(f"无效的任务ID：{task_id}")
        
        return self.task_configs[task_id].copy()
    
    def list_pending_tasks(self, prompt_type: str = "domain_and_dataset") -> List[int]:
        """列出待执行的任务"""
        pending = []
        
        for task_id in self.task_configs.keys():
            task_dir = self.workspace_root / str(task_id)
            config_path = task_dir / "evaluation.json"
            
            if not config_path.exists():
                continue
            
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            if config.get('execution_status') == 'pending':
                pending.append(task_id)
        
        return pending