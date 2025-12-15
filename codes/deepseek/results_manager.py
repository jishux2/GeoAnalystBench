# codes/deepseek/results_manager.py
"""
推理结果管理器
负责结果的持久化存储、增量更新和状态追踪
"""

import pandas as pd
import csv
import os
import re
from pathlib import Path
from typing import Dict, Set, Tuple, Optional
from threading import Lock
from datetime import datetime
import aiofiles
import asyncio


class ResultsManager:
    """推理结果管理器"""
    
    def __init__(self, output_dir: str = "results/deepseek"):
        """
        初始化管理器
        
        Args:
            output_dir: 结果输出目录
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.workflow_file = self.output_dir / "workflow_responses.csv"
        self.code_file = self.output_dir / "code_responses.csv"
        
        # 线程安全的写入锁
        self.async_lock = None  # 先设为None
        
        # 已完成任务的索引集合
        self.completed_tasks: Dict[str, Set[Tuple[int, str, int]]] = {
            'workflow': set(),
            'code': set()
        }
        
        # 初始化输出文件
        self._initialize_files()
        
        # 加载已有结果
        self._load_existing_results()
    
    def _initialize_files(self):
        """初始化CSV文件结构"""
        header = [
            'task_id', 'response_id', 'prompt_type', 'response_type',
            'Arcpy', 'llm_model', 'response_content', 'task_length',
            'error_info', 'timestamp'
        ]
        
        for file_path in [self.workflow_file, self.code_file]:
            if not file_path.exists():
                with open(file_path, 'w', newline='', encoding='utf-8') as f:
                    csv.writer(f).writerow(header)
    
    def _load_existing_results(self):
        """加载已有的推理结果，构建完成状态索引"""
        for task_type, file_path in [
            ('workflow', self.workflow_file),
            ('code', self.code_file)
        ]:
            if not file_path.exists() or file_path.stat().st_size == 0:
                continue
            
            try:
                df = pd.read_csv(file_path)
                for _, row in df.iterrows():
                    # 解析response_id获取重复序号
                    match = re.search(r'(\d+)(workflow|code)(\d+)$', str(row['response_id']))
                    if match:
                        task_id = int(match.group(1))
                        repeat_idx = int(match.group(3))
                        prompt_type = row['prompt_type']
                        
                        # 只有成功的结果才标记为已完成
                        if pd.isna(row.get('error_info')) or not row.get('error_info'):
                            self.completed_tasks[task_type].add(
                                (task_id, prompt_type, repeat_idx)
                            )
            
            except Exception as e:
                print(f"警告：加载{file_path}时出错：{e}")
    
    def is_completed(
        self,
        task_type: str,
        task_id: int,
        prompt_type: str,
        repeat_idx: int
    ) -> bool:
        """
        检查特定任务是否已完成
        
        Args:
            task_type: 'workflow' 或 'code'
            task_id: 任务编号
            prompt_type: 提示词类型
            repeat_idx: 重复序号（0-2）
        
        Returns:
            是否已完成
        """
        return (task_id, prompt_type, repeat_idx) in self.completed_tasks[task_type]
    
    def calculate_workflow_length(self, workflow_text: str) -> int:
        """从工作流文本中提取步骤数量"""
        import re
        
        # 匹配 tasks = [ ... ] 整个结构
        pattern = r'tasks\s*=\s*\[(.*?)\]'
        match = re.search(pattern, workflow_text, re.DOTALL)
        
        if not match:
            return 0
        
        # 提取列表内容
        tasks_content = match.group(1)
        
        # 统计列表中的引号字符串数量（匹配成对的引号）
        task_pattern = r'"[^"]*"'
        tasks = re.findall(task_pattern, tasks_content)
        
        return len(tasks)
    
    async def save_result(
        self,
        task_type: str,
        task_id: int,
        response_id: str,
        prompt_type: str,
        arcpy: bool,
        response_content: str,
        error_info: Optional[str] = None
    ):
        """
        保存单个推理结果
        
        Args:
            task_type: 'workflow' 或 'code'
            task_id: 任务编号
            response_id: 响应唯一标识
            prompt_type: 提示词类型
            arcpy: 是否使用ArcPy
            response_content: 模型响应内容
            error_info: 错误信息（如有）
        """
        print(f"[调试] save_result被调用: task_id={task_id}, response_id={response_id}")
        
        # 首次调用时初始化锁
        if self.async_lock is None:
            self.async_lock = asyncio.Lock()
            print(f"[调试] 异步锁已初始化")
        
        file_path = self.workflow_file if task_type == 'workflow' else self.code_file
        print(f"[调试] 准备写入文件: {file_path}")
        
        # 计算工作流长度
        task_length = 'none'
        if task_type == 'workflow' and not error_info:
            print(f"[调试] 准备计算工作流长度，文本长度={len(response_content)}")
            task_length = self.calculate_workflow_length(response_content)
            print(f"[调试] 工作流长度计算完成: {task_length}")
        
        row = [
            task_id,
            response_id,
            prompt_type,
            task_type,
            arcpy,
            'deepseek-chat',
            response_content if not error_info else '',
            task_length,
            error_info or '',
            datetime.now().isoformat()
        ]
        
        import csv
        import io
        output = io.StringIO()
        csv.writer(output).writerow(row)
        line = output.getvalue()
        print(f"[调试] CSV行已构建，长度={len(line)}")
            
        print(f"[调试] 准备获取锁...")
        async with self.async_lock:
            print(f"[调试] 已获取锁，开始写入...")
            async with aiofiles.open(file_path, 'a', encoding='utf-8') as f:
                await f.write(line)
            print(f"[调试] 写入完成")
            
        # 更新完成状态索引
        if not error_info:
            match = re.search(r'(\d+)(workflow|code)(\d+)$', response_id)
            if match:
                repeat_idx = int(match.group(3))
                self.completed_tasks[task_type].add(
                    (task_id, prompt_type, repeat_idx)
                )
    
    def get_statistics(self, task_type: str) -> Dict:
        """
        获取当前完成统计信息
        
        Args:
            task_type: 'workflow' 或 'code'
        
        Returns:
            包含统计数据的字典
        """
        total_tasks = 50 * 4 * 3  # 50任务 × 4配置 × 3重复
        completed = len(self.completed_tasks[task_type])
        
        return {
            'total': total_tasks,
            'completed': completed,
            'remaining': total_tasks - completed,
            'progress_pct': (completed / total_tasks * 100) if total_tasks > 0 else 0
        }