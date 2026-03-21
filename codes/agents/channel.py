# codes/agents/channel.py
"""
基于asyncio.Queue的异步通信通道

每个智能体持有一个专属的收件通道，其他成员通过通道注册表
按名称检索目标通道并投递消息。投递操作非阻塞，接收操作
在队列为空时自动挂起协程，直到新消息到达后唤醒——
天然实现了"有事则醒、无事则眠"的空闲机制，无需轮询。
"""

from __future__ import annotations

import asyncio
from typing import Dict, List, Optional

from .message import Message


class AgentChannel:
    """
    单个智能体的异步收件通道

    封装一个无界队列，提供投递、阻塞接收、
    非阻塞批量排空三种访问模式。
    """

    def __init__(self, owner: str):
        """
        Args:
            owner: 通道所属智能体的名称标识
        """
        self.owner = owner
        self._queue: asyncio.Queue[Message] = asyncio.Queue()

    async def send(self, message: Message):
        """向此通道投递一条消息。非阻塞，立即返回。"""
        await self._queue.put(message)

    async def receive(self, timeout: Optional[float] = None) -> Optional[Message]:
        """
        等待并取出一条消息。

        队列为空时协程挂起，直到有消息到达或超时。
        超时返回None，适用于需要周期性苏醒的场景（如主控节点）。
        """
        try:
            if timeout is not None:
                return await asyncio.wait_for(self._queue.get(), timeout=timeout)
            return await self._queue.get()
        except asyncio.TimeoutError:
            return None

    def drain(self) -> List[Message]:
        """
        非阻塞地取出当前队列中的全部消息。

        用于智能体在工具调用间隙批量检查收件箱。
        队列为空时返回空列表，不挂起。
        """
        messages = []
        while not self._queue.empty():
            try:
                messages.append(self._queue.get_nowait())
            except asyncio.QueueEmpty:
                break
        return messages

    @property
    def pending_count(self) -> int:
        """当前队列中待处理的消息数量。"""
        return self._queue.qsize()


class ChannelRegistry:
    """
    通道注册表

    集中管理所有智能体的收件通道，提供按名称检索和
    消息路由的能力。主控节点在启动团队时创建此注册表，
    所有成员共享同一实例。
    """

    def __init__(self):
        self._channels: Dict[str, AgentChannel] = {}

    def register(self, name: str) -> AgentChannel:
        """
        为指定名称创建并注册一个通道。

        Args:
            name: 智能体的名称标识

        Returns:
            新创建的通道实例

        Raises:
            ValueError: 名称已被占用
        """
        if name in self._channels:
            raise ValueError(f"通道名称已存在：{name}")
        channel = AgentChannel(name)
        self._channels[name] = channel
        return channel

    def get(self, name: str) -> AgentChannel:
        """
        按名称检索通道。

        Args:
            name: 目标智能体的名称标识

        Raises:
            KeyError: 名称未注册
        """
        if name not in self._channels:
            raise KeyError(f"未注册的通道：{name}")
        return self._channels[name]

    async def deliver(self, message: Message):
        """
        将消息路由至接收方的通道。

        根据消息中的recipient字段查找目标通道并投递。

        Raises:
            KeyError: 接收方未注册
        """
        target = self.get(message.recipient)
        await target.send(message)

    @property
    def registered_names(self) -> List[str]:
        """已注册的全部智能体名称。"""
        return list(self._channels.keys())