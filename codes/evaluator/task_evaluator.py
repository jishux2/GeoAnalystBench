# codes/evaluator/task_evaluator.py
"""
单任务评估器
对已完成任务的调试过程与最终结果进行结构化评价
"""

import json
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime


class TaskEvaluator:
    """单任务调试过程评估器"""

    def __init__(self, workspace_root: str = "evaluation_workspace"):
        self.workspace_root = Path(workspace_root)

    async def evaluate(
        self,
        task_id: int,
        api_client,
        temperature: float = 0.7
    ) -> Dict:
        """
        对单个任务的调试历程进行评估

        Args:
            task_id: 任务编号
            api_client: DeepSeek API客户端实例
            temperature: 采样温度

        Returns:
            结构化的评估结果
        """
        task_dir = self.workspace_root / str(task_id)

        # 收集素材
        materials = self._gather_materials(task_id)

        # 构造评估提示词
        prompt = self._build_evaluation_prompt(materials)

        # 调用模型生成评估
        response = await api_client.chat_completion(
            messages=[
                {"role": "system", "content": self._system_prompt()},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=4096,
            thinking={"type": "enabled"}
        )

        # 解析并保存
        evaluation = {
            "task_id": task_id,
            "status": materials["status"],
            "rounds_used": materials["rounds_used"],
            "analysis": response,
            "evaluated_at": datetime.now().isoformat()
        }

        output_path = task_dir / "outputs" / "evaluation.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(evaluation, f, indent=2, ensure_ascii=False)

        return evaluation

    def _gather_materials(self, task_id: int) -> Dict:
        """收集任务的全部诊断素材"""
        task_dir = self.workspace_root / str(task_id)

        # 对话历史
        history_path = task_dir / "dialogue_history.json"
        history = {}
        if history_path.exists():
            with open(history_path, 'r', encoding='utf-8') as f:
                history = json.load(f)

        # 各轮次的调试轨迹
        traces = {}
        outputs_dir = task_dir / "outputs"
        if outputs_dir.exists():
            for round_dir in sorted(outputs_dir.iterdir()):
                if round_dir.is_dir() and round_dir.name.startswith("round_"):
                    trace_path = round_dir / "debug_trace.json"
                    if trace_path.exists():
                        with open(trace_path, 'r', encoding='utf-8') as f:
                            traces[round_dir.name] = json.load(f)

        return {
            "status": history.get("status", "unknown"),
            "rounds_used": len(history.get("rounds", [])),
            "max_rounds": history.get("max_rounds", 3),
            "history": history,
            "traces": traces,
            "final_diagnosis": history.get("diagnosis")
        }

    def _build_evaluation_prompt(self, materials: Dict) -> str:
        """构造评估提示词"""
        sections = []

        sections.append(f"## Task Outcome: {materials['status'].upper()}")
        sections.append(
            f"Completed {materials['rounds_used']} of "
            f"{materials['max_rounds']} available rounds.\n"
        )

        # 各轮次的调试轨迹摘要
        for round_name, trace in materials["traces"].items():
            sections.append(f"## {round_name.replace('_', ' ').title()} Trace\n")

            turns = trace.get("turns", [])
            sections.append(f"Total interactions: {len(turns)}\n")

            for turn in turns:
                turn_num = turn.get("turn", "?")
                tool_calls = turn.get("tool_calls", [])

                if tool_calls:
                    tool_names = [tc.get("function", {}).get("name", "?") for tc in tool_calls]
                    sections.append(f"**Turn {turn_num}**: {', '.join(tool_names)}")
                else:
                    sections.append(f"**Turn {turn_num}**: (no tool calls)")

                reasoning = turn.get("reasoning")
                if reasoning:
                    # 截取思考内容的关键部分
                    preview = reasoning[:500]
                    if len(reasoning) > 500:
                        preview += "..."
                    sections.append(f"Reasoning: {preview}\n")

            final = trace.get("final_diagnosis")
            if final:
                sections.append("**Final diagnosis from this round:**")
                if final.get("success"):
                    sections.append("Concluded with success.\n")
                else:
                    root_cause = final.get("root_cause", "")
                    patches = final.get("patches", [])
                    sections.append(f"Root cause: {root_cause}")
                    sections.append(f"Patches proposed: {len(patches)}\n")

        # 任务级最终诊断
        if materials["final_diagnosis"]:
            sections.append("## Final Task-Level Diagnosis\n")
            diag = materials["final_diagnosis"]
            sections.append(f"Root cause: {diag.get('root_cause', 'N/A')}")
            patches = diag.get("patches", [])
            if patches:
                sections.append(f"\nPatches ({len(patches)} total):")
                for i, p in enumerate(patches, 1):
                    sections.append(f"\nPatch {i}:")
                    sections.append(f"```\n{p.get('target_code', '')}\n```")
                    sections.append("→")
                    sections.append(f"```\n{p.get('replacement_code', '')}\n```")

        return "\n".join(sections)

    def _system_prompt(self) -> str:
        """评估专用的系统提示词"""
        return """You are an expert evaluator reviewing the debugging performance of an AI agent on geospatial Python code.

Analyze the provided debugging trace and produce a structured assessment covering:

1. **Strategy Evaluation**: Was the agent's overall approach well-reasoned? Did it follow an efficient path from symptom to root cause, or did it waste turns on unproductive exploration?

2. **Key Decision Points**: Identify the most consequential choices the agent made—both productive turns that advanced the diagnosis and missteps that led to dead ends.

3. **Diagnosis Quality**: Is the final root cause analysis accurate and complete? Does it capture the true underlying defect, or does it merely describe symptoms?

4. **Patch Assessment**: If patches were proposed, are they correct and sufficient? Do they address the root cause or just suppress the immediate error? Are there edge cases they might miss?

5. **Outcome Summary**: A concise verdict on the overall debugging session—what went well, what could improve, and whether the final result (success or failure) reflects the agent's capability or the task's difficulty.

Respond in well-structured prose. Be specific—reference particular turns, tool calls, and reasoning steps rather than offering generic commentary."""