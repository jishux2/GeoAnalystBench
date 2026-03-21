# codes/agents/journal.py
"""
上下文快照的磁盘持久化

在每次上下文发生实质性变更后，将完整的消息序列写入JSON文件。
服务于两个消费场景：意外中断后的状态恢复，以及任务结束后
评估模型所需的行为轨迹素材。

每个子智能体维护独立的日志文件，四份文件共同构成
整个团队协作过程的完整档案。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List
from datetime import datetime


class ContextJournal:
    """
    单个智能体的上下文日志

    采用覆盖式写入策略——每次持久化都输出当前的完整快照，
    而非增量追加。这种设计简化了恢复逻辑（直接加载最新文件
    即可），代价是写入量略大，但对于我们的消息规模完全可接受。
    """

    def __init__(self, output_path: Path, agent_name: str):
        """
        Args:
            output_path: 日志文件路径
            agent_name: 所属智能体名称
        """
        self._path = output_path
        self._agent_name = agent_name
        self._path.parent.mkdir(parents=True, exist_ok=True)

        self._metadata: Dict[str, Any] = {
            "agent_name": agent_name,
            "started_at": datetime.now().isoformat(),
        }

    def persist(self, messages: List[Dict[str, Any]]):
        """
        将消息序列快照写入磁盘。

        Args:
            messages: AgentContext.snapshot()的返回值
        """
        record = {
            **self._metadata,
            "persisted_at": datetime.now().isoformat(),
            "message_count": len(messages),
            "messages": messages,
        }

        with open(self._path, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2, ensure_ascii=False, default=str)

    def finalize(self, outcome: Dict[str, Any], messages: List[Dict[str, Any]]):
        """
        写入最终状态，附带结局摘要。

        Args:
            outcome: 终态信息（成功/失败、最终补丁等）
            messages: 最终的消息序列快照
        """
        record = {
            **self._metadata,
            "completed_at": datetime.now().isoformat(),
            "outcome": outcome,
            "message_count": len(messages),
            "messages": messages,
        }

        with open(self._path, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2, ensure_ascii=False, default=str)