# codes/agents/message.py
"""
智能体间通信的消息类型体系

所有跨智能体的信息交换都经由此处定义的结构承载。
消息按语义功能划分为若干类别，每个类别携带与其用途匹配的载荷字段。
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class MessageType(Enum):
    """消息的语义类别"""

    # 常规业务通信
    TASK_ASSIGNMENT = "task_assignment"   # 主控节点向成员分派工作指令
    TASK_REPORT = "task_report"          # 成员向其他成员或主控节点交付工作成果
    DATA_REQUEST = "data_request"        # 向其他成员索取特定信息
    PATCH_SUBMISSION = "patch_submission" # 诊断专员向脚本工程师提交修改请求
    INJECT_REQUEST = "inject_request"    # 诊断专员请求插入语句后执行

    # 主控节点的管理指令
    STATUS_INQUIRY = "status_inquiry"    # 询问成员当前进展
    STATUS_REPLY = "status_reply"        # 对进展询问的回复
    TERMINATE = "terminate"              # 主控节点要求成员关闭

    # 终态消息
    TASK_COMPLETE = "task_complete"       # 诊断专员交付最终结论：业务评估与根因分析


@dataclass
class Message:
    """
    智能体间传递的消息实体

    每条消息携带唯一标识、发送方与接收方的身份标签、
    语义类别、自由格式的正文内容，以及可选的结构化载荷。
    """

    msg_type: MessageType
    sender: str
    recipient: str
    content: str
    payload: Dict[str, Any] = field(default_factory=dict)
    msg_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    timestamp: float = field(default_factory=time.time)

    def to_context_text(self) -> str:
        """
        将消息序列化为适合注入智能体上下文的文本形式。

        主控节点和子智能体在收到消息后，将此文本作为用户消息
        拼接到自己的对话序列中，供模型在后续推理时参考。
        """
        import json

        header = f"[Message from {self.sender}] ({self.msg_type.value})"
        parts = [header, self.content]

        if self.payload:
            formatted = json.dumps(self.payload, indent=2, ensure_ascii=False, default=str)
            parts.append(f"Attached data:\n```json\n{formatted}\n```")

        return "\n".join(parts)

    def to_log_dict(self) -> Dict[str, Any]:
        """序列化为可持久化的字典形式，用于日志归档。"""
        return {
            "msg_id": self.msg_id,
            "msg_type": self.msg_type.value,
            "sender": self.sender,
            "recipient": self.recipient,
            "content": self.content,
            "payload_keys": list(self.payload.keys()),
            "timestamp": self.timestamp,
        }