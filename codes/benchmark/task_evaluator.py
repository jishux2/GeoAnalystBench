# codes/benchmark/task_evaluator.py
"""
单任务评估器

将产物收集、prompt构建和模型调用串联为完整的
评估管线。从磁盘上的执行记录和产物目录中汇集
材料，交由评估模型判定，持久化结构化结果。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from .task_index import TaskIndex
from .workspace_setup import BenchmarkLayout
from .artifact_processor import ArtifactProcessor
from .eval_client import EvalClient
from .eval_prompt import (
    build_system_prompt,
    build_reference_message,
    build_generated_message,
)


class TaskEvaluator:
    """
    单任务评估器

    汇集参考侧与生成侧的全部材料，构建评估prompt，
    调用评估模型获取判定结果，持久化至任务目录。
    """

    EVAL_RESULT_FILENAME = "eval_result.json"

    def __init__(
        self,
        task_index: TaskIndex,
        layout: BenchmarkLayout,
        artifact_processor: ArtifactProcessor,
    ):
        self.index = task_index
        self.layout = layout
        self.processor = artifact_processor

    async def evaluate(
        self,
        task_id: str,
        eval_client: EvalClient,
        temperature: float = 0.3,
    ) -> Dict[str, Any]:
        """
        对单个任务执行完整的评估流程。

        Returns:
            解析后的评估结果字典
        """
        entry = self.index.get(task_id)
        if entry is None:
            raise KeyError(f"Unknown task ID: {task_id}")

        source = entry["source"]
        task_text = entry.get("task_text", "")

        # 收集参考侧材料
        ref_code = self.index.get_reference_code(task_id)
        ref_execution = self._load_execution_info(
            self.layout.ground_truth_task_dir(source, task_id)
        )
        ref_artifacts = self.processor.process_directory(
            self.layout.ground_truth_output(source, task_id)
        )

        # 收集生成侧材料
        gen_code = self._load_generated_code(source, task_id)
        gen_execution = self._load_generated_execution(source, task_id)
        gen_artifacts = self.processor.process_directory(
            self.layout.workspace_output(source, task_id)
        )

        # 构建消息序列
        system_msg = {"role": "system", "content": build_system_prompt()}
        ref_msg = build_reference_message(
            task_text, ref_code, ref_execution, ref_artifacts
        )
        gen_msg = build_generated_message(
            gen_code, gen_execution, gen_artifacts
        )

        messages = [system_msg, ref_msg, gen_msg]

        # 调用评估模型
        raw_response = await eval_client.evaluate(
            messages=messages,
            temperature=temperature,
            max_tokens=4096,
        )

        # 解析JSON结果
        result = self._parse_eval_response(raw_response)

        # 持久化
        result_path = (
            self.layout.workspace_task_dir(source, task_id)
            / self.EVAL_RESULT_FILENAME
        )
        result_path.write_text(
            json.dumps(result, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        return result

    def _load_execution_info(self, task_dir: Path) -> Optional[Dict[str, Any]]:
        """从任务目录加载执行状态信息。"""
        stdout_path = task_dir / "stdout.txt"
        stderr_path = task_dir / "stderr.txt"
        status_path = task_dir / "run_status.json"

        if not status_path.exists():
            return None

        with open(status_path, "r", encoding="utf-8") as f:
            status = json.load(f)

        execution = {
            "returncode": status.get("result", {}).get("returncode"),
        }

        if stdout_path.exists():
            execution["stdout"] = stdout_path.read_text(encoding="utf-8")
        else:
            execution["stdout"] = ""

        if stderr_path.exists():
            execution["stderr"] = stderr_path.read_text(encoding="utf-8")
        else:
            execution["stderr"] = ""

        return execution

    def _load_generated_code(self, source: str, task_id: str) -> Optional[str]:
        """加载智能体生成的脚本。"""
        script_path = (
            self.layout.workspace_task_dir(source, task_id) / "current_script.py"
        )
        if not script_path.exists():
            return None
        return script_path.read_text(encoding="utf-8")

    def _load_generated_execution(
        self, source: str, task_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        加载生成代码的最终执行信息。

        从diagnostician的执行归档中取最后一次运行的记录。
        """
        diag_dir = (
            self.layout.workspace_task_dir(source, task_id)
            / "outputs" / "diagnostician"
        )
        if not diag_dir.exists():
            return None

        # 找到编号最大的run目录
        run_dirs = sorted(
            [d for d in diag_dir.iterdir() if d.is_dir() and d.name.startswith("run_")],
            key=lambda d: int(d.name.split("_")[1]) if d.name.split("_")[1].isdigit() else 0,
        )
        if not run_dirs:
            return None

        last_run = run_dirs[-1]

        execution = {"returncode": None, "stdout": "", "stderr": ""}

        stdout_path = last_run / "stdout.txt"
        stderr_path = last_run / "stderr.txt"
        error_trace = last_run / "error_trace.json"

        if stdout_path.exists():
            execution["stdout"] = stdout_path.read_text(encoding="utf-8")
        if stderr_path.exists():
            execution["stderr"] = stderr_path.read_text(encoding="utf-8")

        # error_trace.json的存在等价于执行失败
        if error_trace.exists():
            execution["returncode"] = 1
        else:
            execution["returncode"] = 0

        return execution

    def _parse_eval_response(self, raw: str) -> Dict[str, Any]:
        text = raw.strip()

        if text.startswith("```"):
            first_newline = text.index("\n")
            last_fence = text.rfind("```")
            if last_fence > first_newline:
                text = text[first_newline + 1:last_fence].strip()

        try:
            result = json.loads(text)
        except json.JSONDecodeError:
            return {
                "parse_error": True,
                "raw_response": raw,
                "summary": "Failed to parse evaluation response as JSON.",
            }

        # 规范化评分字段为整数
        if "overall_score" in result:
            try:
                result["overall_score"] = int(result["overall_score"])
            except (ValueError, TypeError):
                pass

        for dim in ["task_alignment", "execution_validity", "output_completeness",
                     "result_fidelity", "visualization_quality", "code_craftsmanship"]:
            if dim in result and isinstance(result[dim], dict) and "score" in result[dim]:
                try:
                    result[dim]["score"] = int(result[dim]["score"])
                except (ValueError, TypeError):
                    pass

        return result