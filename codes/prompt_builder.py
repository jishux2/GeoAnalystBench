# codes/prompt_builder.py
"""
任务信息提取器

从基准测试集的CSV记录中提取任务元数据，组装为
结构化的任务描述文本。不包含任何代码生成指导——
编码规范和工具使用说明由各智能体的技能文档承载。
"""

import pandas as pd
from pathlib import Path
from typing import Dict


class TaskInfoExtractor:
    """任务信息提取器"""

    def __init__(self, dataset_path: str = "dataset/GeoAnalystBench.csv"):
        self.dataset_path = Path(dataset_path)
        self.tasks_df = pd.read_csv(dataset_path)

    def extract(self, task_id: int) -> Dict[str, str]:
        """
        提取指定任务的结构化信息。

        Args:
            task_id: 任务编号（1-50）

        Returns:
            包含各字段的任务信息字典
        """
        row = self.tasks_df.iloc[task_id - 1]

        task = self._wrap_text(row["Task"])
        instruction = self._wrap_text(row["Instruction"])
        domain_knowledge = self._wrap_text(row["Domain Knowledge"])
        dataset_desc = self._format_multiline(row["Dataset Description"])
        use_arcpy = row["Open Source"] != "T"

        info = {
            "task": task,
            "instruction": instruction,
            "domain_knowledge": domain_knowledge,
            "dataset_description": dataset_desc,
            "use_arcpy": use_arcpy,
        }

        info["full_text"] = self._compose(info)

        return info

    def _compose(self, info: Dict) -> str:
        """
        将结构化字段组合为完整的任务描述文本。

        仅包含任务本身的语义信息，不附加任何
        代码生成指导或输出格式约束。
        """
        sections = [
            f"[Task]:\n{info['task']}",
            f"[Instruction]:\n{info['instruction']}",
            f"[Domain Knowledge]:\n{info['domain_knowledge']}",
            f"[Dataset Description]:\n{info['dataset_description']}",
        ]

        tech_note = (
            "This task requires ArcPy functions."
            if info["use_arcpy"]
            else "This task uses open-source Python packages."
        )
        sections.append(f"[Technology Stack]:\n{tech_note}")

        return "\n\n".join(sections)

    @staticmethod
    def _wrap_text(text, max_line_length=80) -> str:
        """将长文本按词边界折行。"""
        if not isinstance(text, str):
            return str(text)

        words = text.split()
        lines = []
        current_line = []
        current_length = 0

        for word in words:
            word_length = len(word) + 1
            if current_length + word_length > max_line_length and current_line:
                lines.append(" ".join(current_line))
                current_line = [word]
                current_length = word_length
            else:
                current_line.append(word)
                current_length += word_length

        if current_line:
            lines.append(" ".join(current_line))

        return "\n".join(lines)

    @staticmethod
    def _format_multiline(text) -> str:
        """对多行文本的每一行进行折行处理。"""
        if not isinstance(text, str):
            return str(text)

        return "\n".join(
            TaskInfoExtractor._wrap_text(line) for line in text.split("\n")
        )