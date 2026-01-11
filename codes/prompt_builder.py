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


class IterativePromptBuilder:
    """迭代修复提示词构造器"""
    
    def build_code_prompt(self, dialogue_history: Dict, round_num: int) -> str:
        """
        构建第N轮的代码补丁生成提示词
        
        策略：
        - 首轮完整代码：作为全局参照和补丁应用基准
        - 历史摘要（1到N-2轮）：包括诊断结果+应用的补丁+执行结果
        - 完整上轮信息（N-1轮）：包含补丁代码、检索文档、诊断结果
        - 当前轮检索资料（N轮）：API文档和代码示例
        
        Args:
            dialogue_history: 完整对话历史字典
            round_num: 当前轮次（2或3）
        
        Returns:
            格式化的提示词文本
        """
        if round_num < 2:
            raise ValueError("迭代提示词仅适用于第2轮及以后")
        
        sections = []
        
        # 开场白
        sections.append("Continue iterating on the geospatial analysis code based on previous attempts and diagnostic feedback.\n")
        
        # 第一部分：原始任务+首轮完整代码
        round_1 = dialogue_history['rounds'][0]
        initial_prompt = round_1['prompt']
        
        sections.append("[Original Task]:")
        sections.append(f"Task: {initial_prompt['task']}")
        sections.append(f"Instruction: {initial_prompt['instruction']}")
        sections.append(f"Domain Knowledge: {initial_prompt['domain_knowledge']}")
        sections.append(f"Dataset: {initial_prompt['dataset_description']}\n")
        
        sections.append("[Original Code (Round 1)]:")
        sections.append(f"```python\n{round_1['generated_code']}\n```\n")
        
        # 第二部分：历史摘要（第2轮到第N-2轮的补丁）
        if round_num > 2:
            sections.append("[Previous Iterations]:")
            
            # 遍历第2轮到第N-2轮的补丁
            for i in range(1, round_num - 2):  # 补丁所在轮次的索引
                patch_round = dialogue_history['rounds'][i]  # 第2到N-2轮
                prev_diagnosis_round = dialogue_history['rounds'][i - 1]  # 对应的诊断来自上一轮
                
                sections.append(f"\n--- Round {patch_round['round']} ---")
                
                # 上一轮的诊断（触发本轮补丁的原因）
                if prev_diagnosis_round.get('diagnosis'):
                    sections.append(f"Diagnosis from Round {prev_diagnosis_round['round']}: {prev_diagnosis_round['diagnosis'].get('root_cause', 'N/A')}")
                
                # 本轮应用的补丁
                if patch_round.get('generated_patch'):
                    patch = patch_round['generated_patch']
                    sections.append(f"Applied Patch:")
                    sections.append(f"```python\n{patch['replacement_code']}\n```")
                
                # 本轮执行结果
                exec_status = patch_round['execution'].get('status', 'unknown')
                sections.append(f"Result: {exec_status}")
            
            sections.append("")

        # 第三部分：上一轮详细信息（N-1轮）
        prev_round = dialogue_history['rounds'][round_num - 2]

        sections.append(f"[Round {round_num - 1} Details]:")

        # 如果N-1轮有补丁（即round_num >= 3），展示触发它的诊断和补丁
        if round_num >= 3:
            # 触发N-1轮补丁的诊断来自N-2轮
            diagnosis_round = dialogue_history['rounds'][round_num - 3]
            if diagnosis_round.get('diagnosis'):
                sections.append(f"Previous Diagnosis (Round {diagnosis_round['round']}): {diagnosis_round['diagnosis'].get('root_cause', 'N/A')}")
            
            # N-1轮应用的补丁
            if prev_round.get('generated_patch'):
                patch = prev_round['generated_patch']
                sections.append(f"\nApplied Patch:")
                sections.append(f"```python\n{patch['replacement_code']}\n```")

        # N-1轮的执行错误
        if prev_round['execution'].get('error_trace'):
            error_trace = prev_round['execution']['error_trace']
            sections.append(f"\nExecution Error:")
            sections.append(f"Type: {error_trace.get('error_type', 'Unknown')}")
            sections.append(f"Message: {error_trace.get('error_message', '')}")

        # N-1轮的诊断（用于本轮N的补丁生成）
        if prev_round.get('diagnosis'):
            sections.append(f"\nCurrent Diagnosis (Round {prev_round['round']}): {prev_round['diagnosis'].get('root_cause', 'N/A')}")

        sections.append("")
        
        # 第四部分：当前轮检索资料
        current_round = dialogue_history['rounds'][round_num - 1]
        
        if current_round.get('retrieved_docs'):
            sections.append("[Retrieved API Documentation]:")
            for doc in current_round['retrieved_docs']:
                sections.append(f"\n--- {doc.get('path', 'Unknown')} ---")
                sections.append(doc.get('content', '')[:1000])
            sections.append("")
        
        if current_round.get('retrieved_examples'):
            sections.append("[Retrieved Code Examples]:")
            for i, example in enumerate(current_round['retrieved_examples'][:3], 1):
                sections.append(f"\nExample {i} (relevance: {example.get('score', 0):.2f}):")
                sections.append(f"Source: {example.get('source', 'Unknown')}")
                sections.append(f"```python\n{example.get('code', '')}\n```")
            sections.append("")
        
        # 第五部分：输出要求
        sections.append("[Task for This Round]:")
        sections.append("Generate a code patch to fix the identified issue. Your response must be valid JSON:")
        sections.append('''{
  "target_code": "<exact code snippet from Round 1 that needs replacement>",
  "replacement_code": "<corrected code>"
}''')
        sections.append("\nCritical Requirements:")
        sections.append("- target_code: Copy the EXACT code snippet from Round 1 original code, including all whitespace and indentation")
        sections.append("- replacement_code: Provide the corrected version with IDENTICAL indentation level as the original")
        sections.append("- Indentation matters: Python code structure depends on precise spacing")
        sections.append("- The patch will be applied by string replacement - ensure target_code matches exactly")
        
        return "\n".join(sections)
    
    def build_diagnosis_prompt(
        self,
        dialogue_history: Dict,
        round_num: int,
        error_trace: Dict,
        call_details: List[Dict]
    ) -> str:
        """
        构建错误诊断提示词
        
        Args:
            dialogue_history: 完整对话历史
            round_num: 当前轮次
            error_trace: 全局异常捕获结果
            call_details: 装饰器监控记录
        
        Returns:
            格式化的提示词文本
        """
        sections = []
        
        sections.append("Analyze the runtime error from geospatial code execution and identify the root cause.\n")
        
        # 第一部分：原始任务背景+首轮完整代码
        round_1 = dialogue_history['rounds'][0]
        initial_prompt = round_1['prompt']
        
        sections.append("[Task Context]:")
        sections.append(f"Task: {initial_prompt['task']}")
        sections.append(f"Instruction: {initial_prompt['instruction']}\n")
        
        sections.append("[Original Code (Round 1)]:")
        sections.append(f"```python\n{round_1['generated_code']}\n```\n")
        
        # 第二部分：历史迭代（第2轮到第N轮的补丁+诊断）
        if round_num > 1:
            sections.append("[Previous Iterations]:")
            
            for i in range(1, round_num):  # 从第2轮到当前轮
                current = dialogue_history['rounds'][i]
                prev = dialogue_history['rounds'][i - 1]
                
                sections.append(f"\n--- Round {current['round']} ---")
                
                # 上一轮的诊断（触发本轮补丁）
                if prev.get('diagnosis'):
                    sections.append(f"Diagnosis from Round {prev['round']}: {prev['diagnosis'].get('root_cause', 'N/A')}")
                
                # 本轮应用的补丁
                if current.get('generated_patch'):
                    patch = current['generated_patch']
                    sections.append(f"Applied Patch:")
                    sections.append(f"```python\n{patch['replacement_code']}\n```")
                
                # 本轮执行结果
                exec_status = current['execution'].get('status', 'unknown')
                sections.append(f"Execution Result: {exec_status}")
            
            sections.append("")
        
        # 第三部分：当前轮次的完整错误信息
        sections.append(f"[Current Error (Round {round_num})]:")
        sections.append(f"Error Type: {error_trace.get('error_type', 'Unknown')}")
        sections.append(f"Error Message: {error_trace.get('error_message', '')}\n")
        
        # 完整的调用栈（不截断）
        if error_trace.get('stack_frames'):
            sections.append("Complete Stack Trace (most recent call last):")
            for frame in error_trace['stack_frames']:
                sections.append(f"\nFile: {frame.get('file', 'unknown')}, Line {frame.get('line', '?')}")
                sections.append(f"Function: {frame.get('function', 'unknown')}")
                sections.append(f"Code: {frame.get('code', '')}")
                
                # 局部变量（完整展示）
                if frame.get('locals'):
                    sections.append("Local Variables:")
                    # 使用JSON格式化，保持可读性
                    import json
                    sections.append(json.dumps(frame['locals'], indent=2, ensure_ascii=False))
        
        # 函数调用详情（完整展示）
        if call_details:
            sections.append("\n[Function Call Details]:")
            for detail in call_details:
                sections.append(f"\nFunction: {detail.get('function', 'unknown')}")
                sections.append(f"Error: {detail.get('error', '')}")
                sections.append(f"Arguments Summary:")
                
                import json
                sections.append(json.dumps({
                    'args': detail.get('args_summary', []),
                    'kwargs': detail.get('kwargs_summary', {})
                }, indent=2, ensure_ascii=False))
        
        # 第四部分：分析要求
        sections.append("\n[Analysis Task]:")
        sections.append("Provide a structured diagnosis in JSON format:")
        sections.append('''{
  "root_cause": "<natural language description of the fundamental issue>",
  "api_queries": [
    {"library": "<lib_name>", "version": "<version>", "api_path": "<module.class.method>"}
  ],
  "keywords": ["<keyword1>", "<keyword2>", ...],
  "example_query": "<semantic query for retrieving relevant code examples>"
}''')
        sections.append("\nGuidelines:")
        sections.append("- root_cause: Explain why the error occurred, not just what happened")
        sections.append("- api_queries: List APIs that need documentation lookup")
        sections.append("- keywords: Extract terms for BM25-based sparse retrieval")
        sections.append("- example_query: Natural language query for semantic code search")
        
        return "\n".join(sections)