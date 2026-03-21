# codes/agents/context.py
"""
智能体上下文管理

维护单个智能体的对话消息序列，封装消息追加、工具结果拼接、
上下文压缩等操作。压缩策略围绕"文件系统作为外部记忆"的
理念展开：写入操作完成后擦除参数中的文件内容，读取的文件
通过显式关闭操作替换为路径指针，调试会话关闭后将交互序列
概括为摘要加文件引用。
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional


class AgentContext:
    """
    单个智能体的对话上下文

    内部维护一个符合DeepSeek API消息格式的列表，
    提供面向智能体循环各环节的操作接口。
    """

    def __init__(self, system_prompt: str, agent_name: str):
        self.agent_name = agent_name
        self._messages: List[Dict[str, Any]] = [
            {"role": "system", "content": system_prompt}
        ]
        # 文件路径 -> 关联的tool_call_id列表
        # 追踪同一文件上的读取和追加操作，关闭时统一清理
        self._open_files: Dict[str, List[str]] = {}

    @property
    def messages(self) -> List[Dict[str, Any]]:
        """当前完整的消息序列，传递给API时使用。"""
        return self._messages

    # ================================================================
    # 消息序列操作
    # ================================================================

    def append_user(self, content: str):
        """追加一条用户角色的消息。"""
        self._messages.append({"role": "user", "content": content})

    def append_assistant(self, message: Dict[str, Any]):
        """
        追加模型响应的完整消息体。

        直接接收API返回的message对象，保留tool_calls、
        reasoning_content等所有字段。
        """
        self._messages.append(message)

    def append_tool_result(self, tool_call_id: str, content: str):
        """拼接工具执行结果。"""
        self._messages.append({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": content,
        })

    def inject_skill(self, skill_body: str):
        """
        将技能正文作为用户消息注入上下文。

        技能内容以明确的边界标记包裹，便于模型识别其性质。
        """
        wrapped = (
            "<skill-content>\n"
            f"{skill_body}\n"
            "</skill-content>\n\n"
            "The above skill guide is now active. "
            "Associated tools have been added to your available set."
        )
        self.append_user(wrapped)

    def inject_incoming_message(self, text: str):
        """
        将接收到的智能体间消息注入上下文。

        Args:
            text: Message.to_context_text()的输出
        """
        self.append_user(text)

    # ================================================================
    # 文件视图管理
    # ================================================================

    def track_file_operation(self, file_path: str, tool_call_id: str):
        """
        记录一次与文件关联的操作（读取或追加写入）。

        Args:
            file_path: 操作涉及的文件路径
            tool_call_id: 对应的工具调用标识
        """
        if file_path not in self._open_files:
            self._open_files[file_path] = []
        self._open_files[file_path].append(tool_call_id)

    def is_file_open(self, file_path: str) -> bool:
        """检查指定文件是否有未关闭的上下文视图。"""
        return file_path in self._open_files and len(self._open_files[file_path]) > 0

    def close_file(self, file_path: str) -> str:
        """
        关闭文件的上下文视图，统一清理全部关联操作。

        定位该文件名下所有已追踪的tool_call_id，
        将对应的tool_result内容替换为路径指针，
        同时压缩关联的assistant消息中的参数载荷。
        """
        if file_path not in self._open_files:
            return f"No open file view found for: {file_path}"

        tracked_ids = self._open_files.pop(file_path)
        if not tracked_ids:
            return f"No open file view found for: {file_path}"

        placeholder = f"[File closed: {file_path}. Use read_file to reopen if needed.]"

        for tc_id in tracked_ids:
            # 替换tool_result内容
            for msg in self._messages:
                if (
                    msg.get("role") == "tool"
                    and msg.get("tool_call_id") == tc_id
                ):
                    msg["content"] = placeholder
                    break

            # 压缩对应的assistant消息中的参数
            for msg in self._messages:
                if msg.get("role") != "assistant":
                    continue
                for tc in msg.get("tool_calls", []):
                    if tc.get("id") == tc_id:
                        tc["function"]["arguments"] = json.dumps(
                            {"_compressed": True, "file_path": file_path},
                            ensure_ascii=False,
                        )
                        break

        return f"File view closed: {file_path}"

    # ================================================================
    # 上下文压缩
    # ================================================================

    def compress_write_operation(self, tool_call_id: str, file_path: str):
        """
        压缩一次覆盖式写入的工具调用。

        写入完成后文件内容已落盘，上下文中保留完整参数
        不再有信息价值。定位对应的assistant消息中该tool_call的
        arguments并替换为路径摘要。

        注意：仅用于覆盖式写入。追加写入不触发即时压缩，
        其参数内容在文件关闭时统一清理。
        """
        for msg in self._messages:
            if msg.get("role") != "assistant":
                continue
            for tc in msg.get("tool_calls", []):
                if tc.get("id") == tool_call_id:
                    tc["function"]["arguments"] = json.dumps(
                        {
                            "_compressed": True,
                            "file_path": file_path,
                            "_info": "Arguments trimmed after successful write. File content is on disk."
                        },
                        ensure_ascii=False,
                    )
                    return

    def compress_debug_session(self, summary: str, trace_file: str):
        """
        压缩已结束的PDB调试会话。

        将会话期间积累的全部PDB相关工具调用及其结果
        从消息序列中移除，替换为一条精简的摘要消息。

        Args:
            summary: 调试会话的要点概述
            trace_file: 完整交互轨迹的磁盘路径
        """
        pdb_tool_names = {"execute_pdb_command", "inject_code_block"}
        pdb_call_ids = set()

        for msg in self._messages:
            if msg.get("role") != "assistant":
                continue
            for tc in msg.get("tool_calls", []):
                if tc.get("function", {}).get("name") in pdb_tool_names:
                    pdb_call_ids.add(tc["id"])

        if not pdb_call_ids:
            return

        cleaned = []
        for msg in self._messages:
            # 移除匹配的tool_result消息
            if (
                msg.get("role") == "tool"
                and msg.get("tool_call_id") in pdb_call_ids
            ):
                continue

            # 处理assistant消息中的tool_calls列表
            if msg.get("role") == "assistant" and "tool_calls" in msg:
                filtered = [
                    tc for tc in msg["tool_calls"]
                    if tc.get("id") not in pdb_call_ids
                ]
                if not filtered:
                    # 该消息的全部工具调用都属于调试会话，整条移除
                    continue
                msg["tool_calls"] = filtered

            cleaned.append(msg)

        self._messages = cleaned

        self.append_user(
            f"[Debug session concluded]\n"
            f"Summary: {summary}\n"
            f"Full trace: {trace_file}"
        )

    def extract_reasoning_contents(self) -> List[str]:
        """
        提取所有assistant消息中的推理内容。

        Returns:
            推理文本列表，按时序排列
        """
        contents = []
        for msg in self._messages:
            if msg.get("role") == "assistant":
                reasoning = msg.get("reasoning_content")
                if reasoning:
                    contents.append(reasoning)
        return contents

    def strip_all_reasoning(self):
        """移除所有assistant消息中的reasoning_content字段。"""
        for msg in self._messages:
            if msg.get("role") == "assistant":
                msg.pop("reasoning_content", None)

    def inject_reasoning_summary(self, summary: str, archive_path: str):
        """
        注入推理链的浓缩复盘，替代被移除的原始思维内容。

        以用户消息形式插入，用明确的标记包裹以标识其
        系统级上下文管理操作的性质。

        Args:
            summary: LLM生成的第一人称复盘文本
            archive_path: 原始推理内容的磁盘归档路径
        """
        self.append_user(
            f"<reasoning-summary>\n"
            f"{summary}\n"
            f"</reasoning-summary>\n\n"
            f"Full reasoning archive: {archive_path}"
        )

    def estimate_token_count(self) -> int:
        """
        估算当前消息序列的token总量。

        采用启发式方法：将消息序列序列化为文本后，
        以字符总数除以4作为粗略近似。
        """
        return len(str(self._messages)) // 4

    # ================================================================
    # 快照导出
    # ================================================================

    def snapshot(self) -> List[Dict[str, Any]]:
        """
        导出当前消息序列的深拷贝，用于日志持久化。

        返回的是独立副本，后续对上下文的修改不会影响快照。
        """
        import copy
        return copy.deepcopy(self._messages)