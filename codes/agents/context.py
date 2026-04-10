# codes/agents/context.py
"""
智能体上下文管理

维护单个智能体的对话消息序列，提供消息追加、
技能内容管理、调试会话压缩和推理链压缩等操作。
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

    TAG_SKILL = "skill:"

    def __init__(self, system_prompt: str, agent_name: str):
        self.agent_name = agent_name
        self._messages: List[Dict[str, Any]] = [
            {"role": "system", "content": system_prompt}
        ]
        # 文件路径 -> 最近一次读取结果的内容哈希
        # 用于检测重复读取并返回精简提示
        self._file_read_hashes: Dict[str, str] = {}

    @property
    def messages(self) -> List[Dict[str, Any]]:
        """当前完整的消息序列。"""
        return self._messages

    # ================================================================
    # 消息序列操作
    # ================================================================

    def append_user(self, content: str):
        """追加一条用户角色的消息。"""
        self._messages.append({"role": "user", "content": content})

    def append_assistant(self, message: Dict[str, Any]):
        """追加模型响应的完整消息体。"""
        self._messages.append(message)

    def append_tool_result(self, tool_call_id: str, content: str):
        """拼接工具执行结果。"""
        self._messages.append({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": content,
        })

    # ================================================================
    # 技能内容管理
    # ================================================================

    def inject_skill(self, skill_name: str, skill_body: str):
        """将技能正文作为带标签的用户消息注入上下文。"""
        wrapped = (
            "<skill-content>\n"
            f"{skill_body}\n"
            "</skill-content>\n\n"
            "The above skill guide is now active. "
            "Associated tools have been added to your available set."
        )
        self._messages.append({
            "role": "user",
            "content": wrapped,
            "_tag": f"{self.TAG_SKILL}{skill_name}",
        })

    # ================================================================
    # 文件读取去重
    # ================================================================

    def check_file_read_duplicate(self, file_path: str, content: str) -> bool:
        """
        检查文件内容是否与上次读取完全一致。

        Returns:
            True表示内容重复，调用方应返回精简提示而非全文
        """
        import hashlib
        content_hash = hashlib.md5(content.encode("utf-8")).hexdigest()
        previous = self._file_read_hashes.get(file_path)
        self._file_read_hashes[file_path] = content_hash
        return previous is not None and previous == content_hash

    def clear_file_read_hash(self, file_path: str):
        """清除指定文件的读取哈希记录。"""
        self._file_read_hashes.pop(file_path, None)

    # ================================================================
    # 来信注入
    # ================================================================

    def inject_incoming_message(self, text: str):
        """将接收到的智能体间消息注入上下文。"""
        self.append_user(text)

    # ================================================================
    # 上下文压缩
    # ================================================================

    def compress_debug_session(self, summary: str, trace_file: str):
        """
        压缩已结束的PDB调试会话。

        将会话期间积累的全部PDB相关工具调用及其结果
        从消息序列中移除，替换为一条精简的摘要消息。
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
            if (
                msg.get("role") == "tool"
                and msg.get("tool_call_id") in pdb_call_ids
            ):
                continue

            if msg.get("role") == "assistant" and "tool_calls" in msg:
                filtered = [
                    tc for tc in msg["tool_calls"]
                    if tc.get("id") not in pdb_call_ids
                ]
                if not filtered:
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
        """提取所有assistant消息中的推理内容。"""
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
        """注入推理链的浓缩复盘。"""
        self.append_user(
            f"<reasoning-summary>\n"
            f"{summary}\n"
            f"</reasoning-summary>\n\n"
            f"Full reasoning archive: {archive_path}"
        )

    def estimate_token_count(self) -> int:
        """粗略估算当前消息序列的token总量。"""
        return len(str(self._messages)) // 4

    # ================================================================
    # 快照导出
    # ================================================================

    def snapshot(self) -> List[Dict[str, Any]]:
        """导出当前消息序列的深拷贝。"""
        import copy
        return copy.deepcopy(self._messages)