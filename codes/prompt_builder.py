# codes/prompt_builder.py
"""
提示词构造器
支持首轮初始化和迭代修复场景的提示词生成
"""

import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional


class InitialPromptBuilder:
    """首轮代码生成提示词构造器"""
    
    def __init__(self, dataset_path: str = "dataset/GeoAnalystBench.csv"):
        """
        初始化构造器
        
        Args:
            dataset_path: 基准测试集路径
        """
        self.dataset_path = Path(dataset_path)
        self.tasks_df = pd.read_csv(dataset_path)
    
    @staticmethod
    def wrap_text(text, max_line_length=80):
        """将长文本按词边界折行，每行不超过指定字符数"""
        if not isinstance(text, str):
            return str(text)
        
        words = text.split()
        lines = []
        current_line = []
        current_length = 0
        
        for word in words:
            word_length = len(word) + 1  # 加上空格
            if current_length + word_length > max_line_length and current_line:
                lines.append(' '.join(current_line))
                current_line = [word]
                current_length = word_length
            else:
                current_line.append(word)
                current_length += word_length
        
        if current_line:
            lines.append(' '.join(current_line))
        
        return '\n'.join(lines)
    
    @staticmethod
    def format_multiline_text(text):
        """对多行文本的每一行进行折行处理"""
        if not isinstance(text, str):
            return str(text)
        
        formatted_lines = [InitialPromptBuilder.wrap_text(line) for line in text.split('\n')]
        return '\n'.join(formatted_lines)
    
    def build(self, task_id: int) -> Dict[str, str]:
        """
        构建首轮完整提示词（包含全部辅助信息）
        
        Args:
            task_id: 任务编号（1-50）
        
        Returns:
            包含各字段的提示词字典
        """
        row = self.tasks_df.iloc[task_id - 1]
        
        # 应用格式化处理
        task = self.wrap_text(row["Task"])
        instruction = self.wrap_text(row["Instruction"])
        domain_knowledge = self.wrap_text(row["Domain Knowledge"])
        dataset_desc = self.format_multiline_text(row["Dataset Description"])
        use_arcpy = row["Open Source"] != 'T'
        
        prompt_dict = {
            "task": task,
            "instruction": instruction,
            "domain_knowledge": domain_knowledge,
            "dataset_description": dataset_desc,
            "use_arcpy": use_arcpy
        }
        
        prompt_dict["full_text"] = self._format_prompt(prompt_dict)
        
        return prompt_dict
    
    def _format_prompt(self, prompt_dict: Dict) -> str:
        """
        将结构化字典格式化为完整提示词文本
        
        Args:
            prompt_dict: 包含各字段的字典
        
        Returns:
            格式化后的提示词文本
        """
        sections = []
        
        sections.append("As a Geospatial data scientist, generate a python file to solve the proposed task.\n")
        
        sections.append(f"[Task]:\n{prompt_dict['task']}\n")
        sections.append(f"[Instruction]:\n{prompt_dict['instruction']}\n")
        sections.append(f"[Domain Knowledge]:\n{prompt_dict['domain_knowledge']}\n")
        sections.append(f"[Dataset Description]:\n{prompt_dict['dataset_description']}\n")
        
        sections.append("[Key Notes]:")
        sections.append("1. Use **automatic reasoning** and clearly explain each subtask before performing it (ReAct approach).")
        sections.append("2. Using latest python packages for code generation.")
        sections.append("3. Put all code under main function, no helper functions.")
        sections.append("4. Limit your output to code, no extra information.")
        
        if prompt_dict['use_arcpy']:
            sections.append("5. Use latest **Arcpy** functions only.")
        else:
            sections.append("5. Use latest open source python packages only.")
        
        sections.append("6. Wrap critical third-party library functions (e.g., gpd.sjoin, gpd.overlay) using the monitor_call decorator. The decorator definition and error handling infrastructure will be automatically injected - you only need to add the wrapper calls.")
        
        sections.append("\n[Expected Sample Output Begin]")
        sections.append('''"""
import geopandas as gpd  # or other packages

# Wrap the specific geopandas functions you will call in your code
gpd.function1 = monitor_call('gpd.function1')(gpd.function1)
gpd.function2 = monitor_call('gpd.function2')(gpd.function2)
# Add more wrappers based on which functions you actually use

def main():
    path = "path"
    data = loaddata()
    # code for subtask1
    # code for subtask2
    # code for final task

if __name__ == "__main__":
    main()
"""''')
        sections.append("[Expected Sample Output End]")
        
        return "\n".join(sections)