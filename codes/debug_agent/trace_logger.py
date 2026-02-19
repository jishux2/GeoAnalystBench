"""
Debug Agent执行轨迹记录器
"""

import json
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime


class TraceLogger:
    """记录Agent的完整执行轨迹"""
    
    def __init__(self, output_path: Path):
        """
        初始化记录器
        
        Args:
            output_path: 输出文件路径
        """
        self.output_path = output_path
        self.trace = {
            "start_time": datetime.now().isoformat(),
            "turns": []
        }
    
    def log_turn(
        self,
        turn_num: int,
        reasoning: Optional[str],
        tool_calls: Optional[List[Dict]],
        tool_results: Optional[List[str]],
        response_text: Optional[str]
    ):
        """
        记录一轮交互
        
        Args:
            turn_num: 轮次编号
            reasoning: 思考内容
            tool_calls: 工具调用列表
            tool_results: 工具执行结果
            response_text: 模型响应文本
        """
        turn_record = {
            "turn": turn_num,
            "reasoning": reasoning,
            "tool_calls": tool_calls or [],
            "tool_results": tool_results or [],
            "response_text": response_text
        }
        
        self.trace["turns"].append(turn_record)
        self._write_to_file()  # ← 每轮立即写入
    
    def finalize(self, final_diagnosis: Dict):
        """
        记录最终诊断并保存
        
        Args:
            final_diagnosis: 最终诊断结果
        """
        self.trace["end_time"] = datetime.now().isoformat()
        self.trace["final_diagnosis"] = final_diagnosis
        
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(self.output_path, 'w', encoding='utf-8') as f:
            json.dump(self.trace, f, indent=2, ensure_ascii=False)

    def _write_to_file(self):
        """统一的写入方法"""
        with open(self.output_path, 'w', encoding='utf-8') as f:
            json.dump(self.trace, f, indent=2, ensure_ascii=False)