# codes/evaluator/dialogue_manager.py
"""
对话历史管理器
负责迭代修复过程中对话状态的持久化与查询
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime


class DialogueManager:
    """
    对话历史管理器
    
    架构设计：
    采用任务级独立文件存储（每个任务一个dialogue_history.json），
    相比集中式单文件方案的优势在于：
    - 并发写入天然隔离，无需锁机制
    - 单任务故障不影响其他记录
    - 调试时可直接查看特定任务的完整演进轨迹
    """
    
    def __init__(self, workspace_root: str = "evaluation_workspace"):
        self.workspace_root = Path(workspace_root)
    
    def _get_history_path(self, task_id: int) -> Path:
        """获取对话历史文件路径"""
        return self.workspace_root / str(task_id) / "dialogue_history.json"
    
    def initialize(
        self,
        task_id: int,
        initial_prompt: Dict[str, str],
        max_rounds: int = 3
    ):
        """
        初始化对话历史
        
        Args:
            task_id: 任务编号
            initial_prompt: 首轮提示词，包含task/instruction/domain_knowledge/dataset等字段
            max_rounds: 最大迭代轮次
        """
        history_path = self._get_history_path(task_id)
        
        # 确保任务目录存在
        history_path.parent.mkdir(parents=True, exist_ok=True)
        
        history = {
            "task_id": task_id,
            "max_rounds": max_rounds,
            "current_round": 1,
            "status": "pending",  # pending/success/failed/max_rounds_reached
            "created_at": datetime.now().isoformat(),
            "rounds": [
                {
                    "round": 1,
                    "prompt": initial_prompt,
                    "generated_code": None,
                    "execution": {
                        "status": "pending",
                        "error_trace": None,
                        "call_details": None
                    },
                    "diagnosis": None
                }
            ]
        }
        
        with open(history_path, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2, ensure_ascii=False)
    
    def get_history(self, task_id: int) -> Optional[Dict]:
        """
        读取完整对话历史
        
        Args:
            task_id: 任务编号
        
        Returns:
            对话历史字典，文件不存在时返回None
        """
        history_path = self._get_history_path(task_id)
        
        if not history_path.exists():
            return None
        
        with open(history_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def update_round(
        self,
        task_id: int,
        round_num: int,
        **updates
    ):
        """
        更新指定轮次的字段
        
        Args:
            task_id: 任务编号
            round_num: 轮次编号
            **updates: 要更新的字段，如generated_code='...', execution={'status': 'failed'}
        """
        history = self.get_history(task_id)
        
        if not history:
            raise ValueError(f"任务{task_id}的对话历史不存在，请先调用initialize")
        
        # 查找对应轮次
        round_data = None
        for r in history['rounds']:
            if r['round'] == round_num:
                round_data = r
                break
        
        if not round_data:
            raise ValueError(f"任务{task_id}的第{round_num}轮记录不存在")
        
        # 更新字段
        # 对于嵌套字典（如execution），采用合并策略而非整体覆盖，
        # 允许仅更新部分子字段而保留其他已有信息
        for key, value in updates.items():
            if key in round_data:
                if isinstance(round_data[key], dict) and isinstance(value, dict):
                    round_data[key].update(value)
                else:
                    round_data[key] = value
            else:
                round_data[key] = value
        
        # 写回文件
        history['updated_at'] = datetime.now().isoformat()
        self._save_history(task_id, history)
    
    def add_round(
        self,
        task_id: int,
        round_num: int,
        retrieved_docs: Optional[List[Dict]] = None,
        retrieved_examples: Optional[List[Dict]] = None
    ):
        """
        添加新轮次记录
        
        Args:
            task_id: 任务编号
            round_num: 新轮次编号
            retrieved_docs: 检索到的API文档
            retrieved_examples: 检索到的代码示例
        """
        history = self.get_history(task_id)
        
        if not history:
            raise ValueError(f"任务{task_id}的对话历史不存在")
        
        # 检查轮次是否已存在
        existing_rounds = [r['round'] for r in history['rounds']]
        if round_num in existing_rounds:
            raise ValueError(f"第{round_num}轮记录已存在，请使用update_round更新")
        
        new_round = {
            "round": round_num,
            "retrieved_docs": retrieved_docs or [],
            "retrieved_examples": retrieved_examples or [],
            "generated_patch": None,
            "execution": {
                "status": "pending",
                "error_trace": None,
                "call_details": None
            },
            "diagnosis": None
        }
        
        history['rounds'].append(new_round)
        history['current_round'] = round_num
        history['updated_at'] = datetime.now().isoformat()
        
        self._save_history(task_id, history)
    
    def mark_success(self, task_id: int):
        """标记任务执行成功"""
        history = self.get_history(task_id)
        if history:
            history['status'] = 'success'
            history['completed_at'] = datetime.now().isoformat()
            
            # 清除失败相关字段（若存在）
            # 避免状态转换后遗留前序失败标记导致的语义混淆
            history.pop('failure_reason', None)
            
            self._save_history(task_id, history)
    
    def mark_failed(self, task_id: int, reason: str = "max_rounds_reached"):
        """
        标记任务最终失败
        
        Args:
            task_id: 任务编号
            reason: 失败原因（max_rounds_reached/other）
        """
        history = self.get_history(task_id)
        if history:
            history['status'] = 'failed'
            history['failure_reason'] = reason
            history['completed_at'] = datetime.now().isoformat()
            self._save_history(task_id, history)
    
    def get_current_round(self, task_id: int) -> int:
        """获取当前轮次编号"""
        history = self.get_history(task_id)
        return history['current_round'] if history else 1
    
    def get_round_data(self, task_id: int, round_num: int) -> Optional[Dict]:
        """获取指定轮次的数据"""
        history = self.get_history(task_id)
        if not history:
            return None
        
        for r in history['rounds']:
            if r['round'] == round_num:
                return r
        return None
    
    def apply_patch(self, task_id: int, round_num: int) -> str:
        """
        基于字符串匹配应用补丁生成完整代码
        
        采用精确字符串匹配而非行号定位的设计考量：
        模型生成的行号容易因计数偏差或空行理解差异导致定位失准，
        而要求模型复制原始代码片段可直接复用其文本理解能力，
        通过str.replace实现的替换操作同时验证了匹配的唯一性
        
        Args:
            task_id: 任务编号
            round_num: 当前轮次
        
        Returns:
            应用补丁后的完整代码
        """
        history = self.get_history(task_id)
        
        if not history:
            raise ValueError(f"任务{task_id}的对话历史不存在")
        
        # 获取原始代码（第1轮生成的）
        round_1 = self.get_round_data(task_id, 1)
        if not round_1 or not round_1.get('generated_code'):
            raise ValueError(f"任务{task_id}缺少首轮生成代码")
        
        original_code = round_1['generated_code']
        
        if round_num == 1:
            return original_code
        
        # 获取当前轮次的补丁
        current_round = self.get_round_data(task_id, round_num)
        if not current_round or not current_round.get('generated_patch'):
            raise ValueError(f"任务{task_id}第{round_num}轮缺少补丁信息")
        
        patch = current_round['generated_patch']
        
        # 执行字符串精确匹配替换
        # 所有补丁始终针对首轮原始代码，避免累积定位偏移
        target = patch['target_code']
        replacement = patch['replacement_code']
        
        if target not in original_code:
            raise ValueError(
                f"补丁匹配失败：在原始代码中找不到目标片段\n"
                f"目标代码（前100字符）：{target[:100]}"
            )
        
        # 检查匹配的唯一性，防止歧义替换
        count = original_code.count(target)
        if count > 1:
            raise ValueError(
                f"补丁匹配歧义：目标代码在原始脚本中出现{count}次，无法唯一定位"
            )
        
        patched_code = original_code.replace(target, replacement, 1)
        
        return patched_code
    
    def _save_history(self, task_id: int, history: Dict):
        """内部方法：保存对话历史"""
        history_path = self._get_history_path(task_id)
        with open(history_path, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2, ensure_ascii=False)
    
    def list_pending_tasks(self) -> List[int]:
        """列出所有待处理的任务ID"""
        pending = []
        
        if not self.workspace_root.exists():
            return pending
        
        for task_dir in self.workspace_root.iterdir():
            if not task_dir.is_dir():
                continue
            
            try:
                task_id = int(task_dir.name)
                history = self.get_history(task_id)
                
                if history and history['status'] == 'pending':
                    pending.append(task_id)
            except (ValueError, KeyError):
                continue
        
        return sorted(pending)
    
    def get_tasks_by_status(self, status: str) -> List[int]:
        """
        按状态筛选任务
        
        Args:
            status: pending/success/failed
        
        Returns:
            任务ID列表
        """
        tasks = []
        
        if not self.workspace_root.exists():
            return tasks
        
        for task_dir in self.workspace_root.iterdir():
            if not task_dir.is_dir():
                continue
            
            try:
                task_id = int(task_dir.name)
                history = self.get_history(task_id)
                
                if history and history['status'] == status:
                    tasks.append(task_id)
            except (ValueError, KeyError):
                continue
        
        return sorted(tasks)