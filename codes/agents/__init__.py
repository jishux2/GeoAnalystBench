# codes/agents/__init__.py
"""
多智能体协作框架

提供从通信原语到协调调度的完整运行时基础设施。
"""

from .message import Message, MessageType
from .channel import AgentChannel, ChannelRegistry
from .context import AgentContext
from .tool_base import ToolSpec, ToolDispatcher
from .skill_registry import SkillRegistry
from .base_agent import BaseAgent
from .coordinator import Coordinator
from .journal import ContextJournal

__all__ = [
    "Message",
    "MessageType",
    "AgentChannel",
    "ChannelRegistry",
    "AgentContext",
    "ToolSpec",
    "ToolDispatcher",
    "SkillRegistry",
    "BaseAgent",
    "Coordinator",
    "ContextJournal",
]