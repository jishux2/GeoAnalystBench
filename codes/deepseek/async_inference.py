# codes/deepseek/async_inference.py
"""
异步推理引擎
实现并发调度、进度追踪和批量处理
"""

import asyncio
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple
from tqdm.asyncio import tqdm
from datetime import datetime

from .deepseek_client import DeepSeekClient, DeepSeekAPIError
from .results_manager import ResultsManager


class AsyncInferenceEngine:
    """异步推理引擎"""
    
    def __init__(
        self,
        api_key: str,
        max_concurrent: int = 30,
        temperature: float = 0.7
    ):
        """
        初始化引擎
        
        Args:
            api_key: DeepSeek API密钥
            max_concurrent: 最大并发请求数
            temperature: 采样温度
        """
        self.api_key = api_key
        self.max_concurrent = max_concurrent
        self.temperature = temperature
        
        self.client: DeepSeekClient = None
        self.results_manager = ResultsManager()
        
        # 并发控制信号量
        self.semaphore = asyncio.Semaphore(max_concurrent)
        
        # 统计信息
        self.stats = {
            'success': 0,
            'failed': 0,
            'retried': 0
        }
    
    def _load_prompts(self, task_type: str) -> pd.DataFrame:
        """加载提示词文件"""
        file_path = f"prompts/{task_type}_prompts.csv"
        if not Path(file_path).exists():
            raise FileNotFoundError(f"提示词文件不存在：{file_path}")
        return pd.read_csv(file_path)
    
    def _build_task_list(
        self,
        task_type: str,
        prompts_df: pd.DataFrame
    ) -> List[Dict]:
        """
        构建待执行任务列表（仅包含未完成的任务）
        
        Args:
            task_type: 'workflow' 或 'code'
            prompts_df: 提示词数据框
        
        Returns:
            任务配置列表
        """
        tasks = []
        
        for _, row in prompts_df.iterrows():
            task_id = row['task_id']
            prompt_type = self._get_prompt_type(row)
            
            # 每个配置需要3次重复
            for repeat_idx in range(3):
                # 检查是否已完成
                if self.results_manager.is_completed(
                    task_type, task_id, prompt_type, repeat_idx
                ):
                    continue
                
                tasks.append({
                    'task_type': task_type,
                    'task_id': task_id,
                    'prompt_type': prompt_type,
                    'prompt_content': row['prompt_content'],
                    'arcpy': row['Arcpy'],
                    'repeat_idx': repeat_idx,
                    'response_id': f"{task_id}{task_type}{repeat_idx}"
                })
        
        return tasks
    
    def _get_prompt_type(self, row: pd.Series) -> str:
        """根据配置确定提示词类型"""
        if row['domain_knowledge'] and row['dataset']:
            return 'domain_and_dataset'
        elif row['domain_knowledge']:
            return 'domain'
        elif row['dataset']:
            return 'dataset'
        else:
            return 'original'
    
    async def _execute_single_task(self, task_config: Dict, pbar: tqdm):
        """
        执行单个推理任务
        
        Args:
            task_config: 任务配置字典
            pbar: 进度条对象
        """
        async with self.semaphore:
            task_type = task_config['task_type']
            prompt = task_config['prompt_content']
            
            try:
                # 根据任务类型选择生成方法
                if task_type == 'workflow':
                    response = await self.client.generate_workflow(
                        prompt, self.temperature
                    )
                else:
                    response = await self.client.generate_code(
                        prompt, self.temperature
                    )
                
                # 保存成功结果
                self.results_manager.save_result(
                    task_type=task_type,
                    task_id=task_config['task_id'],
                    response_id=task_config['response_id'],
                    prompt_type=task_config['prompt_type'],
                    arcpy=task_config['arcpy'],
                    response_content=response
                )
                
                self.stats['success'] += 1
            
            except DeepSeekAPIError as e:
                # 保存错误信息
                self.results_manager.save_result(
                    task_type=task_type,
                    task_id=task_config['task_id'],
                    response_id=task_config['response_id'],
                    prompt_type=task_config['prompt_type'],
                    arcpy=task_config['arcpy'],
                    response_content='',
                    error_info=str(e)
                )
                
                self.stats['failed'] += 1
            
            except Exception as e:
                # 捕获未预期的错误
                error_msg = f"未知错误：{type(e).__name__} - {str(e)}"
                self.results_manager.save_result(
                    task_type=task_type,
                    task_id=task_config['task_id'],
                    response_id=task_config['response_id'],
                    prompt_type=task_config['prompt_type'],
                    arcpy=task_config['arcpy'],
                    response_content='',
                    error_info=error_msg
                )
                
                self.stats['failed'] += 1
            
            finally:
                pbar.update(1)
    
    async def run_inference(self, task_type: str):
        """
        执行特定类型的批量推理
        
        Args:
            task_type: 'workflow' 或 'code'
        """
        print(f"\n{'='*60}")
        print(f"开始执行{task_type}推理任务")
        print(f"{'='*60}\n")
        
        # 加载提示词
        prompts_df = self._load_prompts(task_type)
        
        # 构建待执行任务列表
        tasks = self._build_task_list(task_type, prompts_df)
        
        if not tasks:
            print(f"所有{task_type}任务已完成，无需执行！")
            return
        
        print(f"待执行任务数：{len(tasks)}")
        print(f"最大并发数：{self.max_concurrent}")
        print(f"开始时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # 重置统计信息
        self.stats = {'success': 0, 'failed': 0, 'retried': 0}
        
        # 初始化客户端
        async with DeepSeekClient(self.api_key) as client:
            self.client = client
            
            # 创建进度条
            with tqdm(total=len(tasks), desc=f"{task_type}推理进度") as pbar:
                # 并发执行所有任务
                await asyncio.gather(*[
                    self._execute_single_task(task, pbar)
                    for task in tasks
                ])
        
        # 输出统计信息
        print(f"\n{'='*60}")
        print(f"{task_type}推理完成")
        print(f"{'='*60}")
        print(f"成功：{self.stats['success']}")
        print(f"失败：{self.stats['failed']}")
        print(f"成功率：{self.stats['success']/(len(tasks))*100:.2f}%")
        print(f"结束时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # 显示整体进度
        stats = self.results_manager.get_statistics(task_type)
        print(f"整体进度：{stats['completed']}/{stats['total']} "
              f"({stats['progress_pct']:.2f}%)")
        print(f"剩余任务：{stats['remaining']}\n")
    
    async def run_all(self):
        """依次执行工作流和代码推理"""
        await self.run_inference('workflow')
        await self.run_inference('code')