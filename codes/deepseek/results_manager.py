# codes/deepseek/results_manager.py
"""
推理结果管理器
负责结果的持久化存储、增量更新和状态追踪
"""

# 标准库
import csv
import os
import re
import asyncio
from pathlib import Path
from typing import Dict, Set, Tuple, Optional
from datetime import datetime

# 第三方库
import pandas as pd
import aiofiles


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
        
        # 异步锁必须在事件循环内创建，此处先初始化为None
        # 实际的Lock对象会在首次异步调用时由事件循环创建
        self.async_lock = None
        
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
            # 处理文件不存在、为空或缺失表头的情况
            if not file_path.exists() or file_path.stat().st_size == 0:
                with open(file_path, 'w', newline='', encoding='utf-8') as f:
                    csv.writer(f).writerow(header)
            else:
                # 检查已存在文件的首行是否为有效表头
                with open(file_path, 'r', encoding='utf-8') as f:
                    first_line = f.readline().strip()
                    # 若首行不是表头（可能因手动删除部分数据导致），则补充表头
                    if first_line and not first_line.startswith('task_id'):
                        f.seek(0)
                        content = f.read()
                        with open(file_path, 'w', newline='', encoding='utf-8') as fw:
                            csv.writer(fw).writerow(header)
                            fw.write(content)
    
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
                    # 从response_id中解析重复序号，格式为{task_id}{type}{repeat_idx}
                    match = re.search(r'(\d+)(workflow|code)(\d+)$', str(row['response_id']))
                    if match:
                        task_id = int(match.group(1))
                        repeat_idx = int(match.group(3))
                        prompt_type = row['prompt_type']
                        
                        # 仅将成功的结果标记为已完成，错误结果需要重新执行
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
        """
        从工作流文本中提取步骤数量
        
        通过正则表达式精确定位tasks列表的范围，避免误统计代码中其他位置的字符串
        （如matplotlib参数、图表标题等）
        """
        import re
        
        # 匹配整个tasks = [...] 结构，re.DOTALL使.能匹配换行符
        pattern = r'tasks\s*=\s*\[(.*?)\]'
        match = re.search(pattern, workflow_text, re.DOTALL)
        
        if not match:
            return 0
        
        # 仅在列表内容中统计引号包裹的字符串数量
        tasks_content = match.group(1)
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
        保存单个推理结果（异步版本）
        
        Args:
            task_type: 'workflow' 或 'code'
            task_id: 任务编号
            response_id: 响应唯一标识
            prompt_type: 提示词类型
            arcpy: 是否使用ArcPy
            response_content: 模型响应内容
            error_info: 错误信息（如有）
        """
        file_path = self.workflow_file if task_type == 'workflow' else self.code_file
        
        # 计算工作流长度
        task_length = 'none'
        if task_type == 'workflow' and not error_info:
            task_length = self.calculate_workflow_length(response_content)
        
        row = [
            task_id, response_id, prompt_type, task_type, arcpy,
            'deepseek-chat', response_content if not error_info else '',
            task_length, error_info or '', datetime.now().isoformat()
        ]
        
        # 在内存中构建CSV格式的行，避免csv.writer直接操作异步文件对象
        import csv
        import io
        output = io.StringIO()
        csv.writer(output).writerow(row)
        line = output.getvalue()
        
        # 使用异步锁保护文件写入，防止并发写入导致数据交错
        # 虽然写入是串行的，但配合异步I/O，等待锁的协程不会阻塞事件循环
        async with self.async_lock:
            async with aiofiles.open(file_path, 'a', encoding='utf-8') as f:
                await f.write(line)
        
        # 更新完成状态索引（在锁外执行，因为操作的是内存数据结构）
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