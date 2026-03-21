# codes/evaluator/task_evaluator.py
"""
单任务评估器

从各子智能体的上下文日志、业务产出文件和主控节点记录中
汇集评估素材，构造结构化提示词并调用模型生成中文评估报告。
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime


class TaskEvaluator:

    def __init__(self, workspace_root: str = "evaluation_workspace"):
        self.workspace_root = Path(workspace_root)

    async def evaluate(
        self,
        task_id: int,
        result: Dict[str, Any],
        api_client,
        temperature: float = 0.3,
    ) -> Dict:
        task_dir = self.workspace_root / str(task_id)
        materials = self._gather_materials(task_id, result)
        prompt = self._build_evaluation_prompt(materials)

        response = await api_client.chat_completion(
            messages=[
                {"role": "system", "content": self._system_prompt()},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=8192,
            thinking={"type": "enabled"},
        )

        # 构建Markdown报告
        report = self._compose_report(task_id, materials, response)

        output_path = task_dir / "outputs" / "evaluation_report.md"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report, encoding="utf-8")

        return {"task_id": task_id, "report_path": str(output_path)}

    # ================================================================
    # 素材采集
    # ================================================================

    def _gather_materials(self, task_id: int, result: Dict) -> Dict:
        task_dir = self.workspace_root / str(task_id)
        outputs_dir = task_dir / "outputs"

        materials = {
            "task_id": task_id,
            "outcome": result,
        }

        # 初始脚本（coordinator备份）
        initial_code_path = outputs_dir / "initial_code.py"
        if initial_code_path.exists():
            materials["initial_code"] = initial_code_path.read_text(encoding="utf-8")

        # 最终脚本
        final_script_path = task_dir / "current_script.py"
        if final_script_path.exists():
            materials["final_code"] = final_script_path.read_text(encoding="utf-8")

        # 数据探查报告
        explorer_report = outputs_dir / "explorer" / "data_report.txt"
        if explorer_report.exists():
            materials["explorer_report"] = explorer_report.read_text(encoding="utf-8")

        # 各智能体的上下文日志
        for agent_name in ["explorer", "engineer", "diagnostician"]:
            journal_path = outputs_dir / agent_name / f"{agent_name}_journal.json"
            if journal_path.exists():
                with open(journal_path, "r", encoding="utf-8") as f:
                    materials[f"{agent_name}_journal"] = json.load(f)

        # 工程师的设计文档（扫描工作目录下的.md文件）
        for md_file in task_dir.glob("*.md"):
            materials["design_doc"] = md_file.read_text(encoding="utf-8")
            materials["design_doc_name"] = md_file.name
            break  # 取第一个

        # 提取协作交互实例
        materials["collaboration_episodes"] = self._extract_collaboration_episodes(materials)

        return materials

    def _extract_collaboration_episodes(self, materials: Dict) -> List[Dict]:
        """
        从各智能体的上下文日志中提取请求-响应交互实例。

        扫描三种请求类型（data_request, patch_submission, inject_request）
        及其配对的task_report回复，将请求到回复之间的行动片段提取出来。
        """
        episodes = []

        # 请求类型 -> 在哪个智能体的日志中寻找（作为接收方）
        request_routing = {
            "data_request": "explorer",
            "patch_submission": "engineer",
            "inject_request": "engineer",
        }

        for request_type, responder in request_routing.items():
            journal_key = f"{responder}_journal"
            if journal_key not in materials:
                continue

            journal = materials[journal_key]
            messages = journal.get("messages", [])

            i = 0
            while i < len(messages):
                msg = messages[i]

                # 寻找incoming message（以user角色注入的团队消息）
                if (
                    msg.get("role") == "user"
                    and f"({request_type})" in msg.get("content", "")
                ):
                    # 找到请求，向下搜索配对的task_report回复
                    episode_messages = [msg]
                    j = i + 1
                    reply_found = False

                    while j < len(messages):
                        episode_messages.append(messages[j])

                        # 检查是否是发送task_report的工具调用
                        if messages[j].get("role") == "assistant":
                            tool_calls = messages[j].get("tool_calls", [])
                            for tc in tool_calls:
                                if tc.get("function", {}).get("name") == "send_message":
                                    try:
                                        args = json.loads(
                                            tc["function"].get("arguments", "{}")
                                        )
                                        if args.get("msg_type") == "task_report":
                                            reply_found = True
                                    except (json.JSONDecodeError, KeyError):
                                        pass

                        if reply_found:
                            # 包含这条tool_call对应的tool_result
                            if j + 1 < len(messages) and messages[j + 1].get("role") == "tool":
                                episode_messages.append(messages[j + 1])
                            break
                        j += 1

                    if reply_found:
                        episodes.append({
                            "type": request_type,
                            "responder": responder,
                            "messages": episode_messages,
                        })

                    i = j + 1
                else:
                    i += 1

        return episodes

    # ================================================================
    # 提示词构建
    # ================================================================

    def _build_evaluation_prompt(self, materials: Dict) -> str:
        sections = []

        # 任务结果概览
        outcome = materials["outcome"]
        status = "成功" if outcome.get("success") else "失败"
        sections.append(f"## 任务结果：{status}\n")

        if outcome.get("assessment"):
            sections.append(f"**诊断专员的质量评估：**\n{outcome['assessment']}\n")

        if outcome.get("root_cause"):
            sections.append(f"**根因分析：**\n{outcome['root_cause']}\n")

        if outcome.get("confidence") is not None:
            sections.append(f"**置信度：** {outcome['confidence']}\n")

        if outcome.get("terminated_by"):
            sections.append(f"**终止原因：** {outcome['terminated_by']}\n")

        # 代码比对
        if materials.get("initial_code") and materials.get("final_code"):
            sections.append("## 脚本演化\n")
            sections.append("### 初始版本\n")
            sections.append(f"```python\n{materials['initial_code']}\n```\n")
            sections.append("### 最终版本\n")
            sections.append(f"```python\n{materials['final_code']}\n```\n")
        elif materials.get("final_code"):
            sections.append("## 最终脚本\n")
            sections.append(f"```python\n{materials['final_code']}\n```\n")

        # 数据探查报告
        if materials.get("explorer_report"):
            sections.append("## 数据探查员产出\n")
            sections.append(f"```\n{materials['explorer_report']}\n```\n")

        # 工程师设计文档
        if materials.get("design_doc"):
            sections.append(f"## 工程师设计文档（{materials.get('design_doc_name', '')}）\n")
            sections.append(f"{materials['design_doc']}\n")

        # 协作交互实例
        episodes = materials.get("collaboration_episodes", [])
        if episodes:
            sections.append("## 团队协作交互记录\n")
            for idx, episode in enumerate(episodes, 1):
                sections.append(
                    f"### 交互 {idx}：`{episode['type']}` → {episode['responder']}\n"
                )
                sections.append(self._format_episode(episode))
                sections.append("")

        # 诊断专员完整上下文
        diag_journal = materials.get("diagnostician_journal")
        if diag_journal:
            sections.append("## 诊断专员完整行动轨迹\n")
            sections.append(
                "以下为经上下文压缩后的完整消息序列，保留了所有关键的"
                "工具调用、执行结果、团队通信和推理摘要：\n"
            )
            sections.append(f"```json\n{json.dumps(diag_journal.get('messages', []), indent=2, ensure_ascii=False)}\n```\n")

        return "\n".join(sections)

    def _format_episode(self, episode: Dict) -> str:
        """
        将一个请求-响应交互实例转化为可读的文本单元。

        采用带箭头的动作链格式，直观展示从请求到响应的完整流转。
        """
        lines = []
        messages = episode["messages"]

        for msg in messages:
            role = msg.get("role", "")

            if role == "user":
                # incoming message
                content = msg.get("content", "")
                # 截取前500字符避免过长
                preview = content[:500]
                if len(content) > 500:
                    preview += "\n... (truncated)"
                lines.append(f"📨 **收到请求：**\n{preview}\n")

            elif role == "assistant":
                reasoning = msg.get("reasoning_content", "")
                tool_calls = msg.get("tool_calls", [])
                text_content = msg.get("content", "")

                if reasoning:
                    preview = reasoning[:300]
                    if len(reasoning) > 300:
                        preview += "..."
                    lines.append(f"💭 *推理：* {preview}\n")

                for tc in tool_calls:
                    func = tc.get("function", {})
                    name = func.get("name", "?")
                    args = func.get("arguments", "{}")

                    # 压缩过的参数简化显示
                    try:
                        parsed = json.loads(args)
                        if parsed.get("_compressed"):
                            args_display = f"(内容已持久化至 {parsed.get('file_path', 'disk')})"
                        else:
                            args_display = args[:200]
                            if len(args) > 200:
                                args_display += "..."
                    except json.JSONDecodeError:
                        args_display = args[:200]

                    lines.append(f"  🔧 `{name}` → {args_display}")

                if text_content:
                    lines.append(f"  💬 {text_content[:200]}")

            elif role == "tool":
                content = msg.get("content", "")
                preview = content[:300]
                if len(content) > 300:
                    preview += "\n  ... (truncated)"
                lines.append(f"  📋 结果：{preview}\n")

        return "\n".join(lines)

    # ================================================================
    # 系统提示词
    # ================================================================

    def _system_prompt(self) -> str:
        return (
            "你是一位资深的软件工程与地理空间分析评审专家，负责对多智能体协作调试系统"
            "的单次任务执行过程进行全面评估。\n\n"

            "评估应覆盖以下维度，但不必机械地逐条列举——以连贯的分析叙述组织你的观点：\n\n"

            "**交付质量：** 最终脚本是否满足任务的业务目标？代码结构是否遵循了既定规范"
            "（主函数组织、监控装饰器注册、断言插桩）？断言的设计是否切中关键的数据校验点？\n\n"

            "**数据探查效能：** 探测报告是否准确捕获了数据集的核心特征？对字段命名、坐标系、"
            "值域等关键属性的描述是否为后续编码提供了可靠的参照基础？\n\n"

            "**工程规划与实现：** 设计文档展现的架构思路是否与最终实现吻合？初始脚本中"
            "预埋的断言在调试过程中是否发挥了预期的守护作用？\n\n"

            "**协作效率：** 团队成员间的请求-响应交互是否高效？消息传递是否准确命中了"
            "对方的信息需求？有无冗余的沟通往返或被忽视的请求？\n\n"

            "**诊断策略：** 诊断专员的调查路径是否遵循了假设驱动的方法论？工具选择是否"
            "恰当？每一步操作是否都指向了明确的验证目标？整体诊断效率如何？\n\n"

            "**总体评判：** 综合以上各方面，对本次任务执行给出一个整体性的评价，"
            "指出最值得肯定的表现和最需要改进的环节。\n\n"

            "请用中文撰写评估报告。行文应具体、有据——引用特定的代码片段、消息内容、"
            "工具调用或推理步骤来支撑你的判断，而非泛泛而谈。"
        )

    # ================================================================
    # 报告组装
    # ================================================================

    def _compose_report(
        self,
        task_id: int,
        materials: Dict,
        analysis: str,
    ) -> str:
        outcome = materials["outcome"]
        status = "成功" if outcome.get("success") else "失败"

        header = (
            f"# 任务 {task_id} 评估报告\n\n"
            f"- **评估时间：** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"- **任务状态：** {status}\n"
        )

        if outcome.get("confidence") is not None:
            header += f"- **置信度：** {outcome['confidence']}\n"
        if outcome.get("terminated_by"):
            header += f"- **终止方式：** {outcome['terminated_by']}\n"

        header += "\n---\n\n"

        return header + analysis