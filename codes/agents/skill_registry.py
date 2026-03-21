# codes/agents/skill_registry.py
"""
技能注册与生命周期管理

扫描技能目录构建摘要索引，支持按名称加载技能正文并将
关联工具注入调度器，以及卸载技能时的反向清理。
技能的YAML frontmatter中声明name和description用于索引，
正文Markdown在加载时注入智能体上下文。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

from .tool_base import ToolSpec, ToolDispatcher


@dataclass
class SkillEntry:
    """技能的内存表示"""
    name: str
    description: str
    body: str
    directory: Path  # 技能根目录，供定位scripts/和references/


class SkillRegistry:
    """
    技能注册表

    启动时扫描技能目录树建立轻量索引（仅元数据），
    加载时才读取正文并触发工具绑定。每项技能可声明
    一组关联工具的工厂函数，在加载时调用工厂生成
    ToolSpec实例并注入调度器。
    """

    def __init__(self, skills_root: Path):
        """
        Args:
            skills_root: 技能目录的根路径，其下每个子目录包含一个SKILL.md
        """
        self._skills_root = skills_root
        self._index: Dict[str, SkillEntry] = {}
        self._loaded: Dict[str, SkillEntry] = {}

        # 技能名称 -> 工具工厂函数的映射
        # 工厂函数接收技能目录路径，返回ToolSpec列表
        self._tool_factories: Dict[str, Callable[[Path], List[ToolSpec]]] = {}

        self._scan()

    def _scan(self):
        """扫描技能目录，解析SKILL.md的frontmatter构建索引。"""
        if not self._skills_root.exists():
            return

        for skill_file in sorted(self._skills_root.rglob("SKILL.md")):
            text = skill_file.read_text(encoding="utf-8")
            meta, body = self._parse_frontmatter(text)
            name = meta.get("name", skill_file.parent.name)
            description = meta.get("description", "")

            self._index[name] = SkillEntry(
                name=name,
                description=description,
                body=body,
                directory=skill_file.parent,
            )

    def register_tool_factory(
        self,
        skill_name: str,
        factory: Callable[[Path], List[ToolSpec]],
    ):
        """
        为指定技能注册工具工厂。

        工厂函数在技能加载时被调用，接收技能目录路径作为参数，
        返回该技能关联的ToolSpec列表。这些工具随技能加载而
        注入调度器，随技能卸载而移除。

        Args:
            skill_name: 技能名称，须与SKILL.md中的name字段一致
            factory: 工具工厂函数
        """
        self._tool_factories[skill_name] = factory

    def list_descriptions(self) -> str:
        """返回所有已索引技能的名称与摘要，供系统提示词引用。"""
        if not self._index:
            return "(no skills available)"
        lines = []
        for name, entry in self._index.items():
            status = "loaded" if name in self._loaded else "available"
            lines.append(f"  - {name} [{status}]: {entry.description}")
        return "\n".join(lines)

    def load(self, name: str, dispatcher: ToolDispatcher) -> str:
        """
        加载指定技能：将正文返回以注入上下文，同时触发工具绑定。

        Args:
            name: 技能名称
            dispatcher: 当前智能体的工具调度器

        Returns:
            技能正文文本，供调用方拼接到对话序列中

        Raises:
            KeyError: 技能名称未在索引中
        """
        if name not in self._index:
            available = ", ".join(self._index.keys())
            raise KeyError(f"Unknown skill '{name}'. Available: {available}")

        entry = self._index[name]
        self._loaded[name] = entry

        # 如果该技能注册了工具工厂，执行并注入调度器
        factory = self._tool_factories.get(name)
        if factory:
            tools = factory(entry.directory)
            for tool in tools:
                tool.bound_skill = name
            dispatcher.register_batch(tools)

        return entry.body

    def unload(self, name: str, dispatcher: ToolDispatcher) -> str:
        """
        卸载指定技能：从调度器中移除关联工具。

        Args:
            name: 技能名称
            dispatcher: 当前智能体的工具调度器

        Returns:
            确认消息
        """
        if name not in self._loaded:
            return f"Skill '{name}' is not currently loaded."

        dispatcher.unregister_by_skill(name)
        del self._loaded[name]
        return f"Skill '{name}' unloaded. Associated tools removed."

    def is_loaded(self, name: str) -> bool:
        return name in self._loaded

    @staticmethod
    def _parse_frontmatter(text: str):
        """
        分离YAML frontmatter与Markdown正文。

        Returns:
            (metadata_dict, body_text)
        """
        meta = {}
        body = text

        if text.startswith("---"):
            end = text.find("---", 3)
            if end != -1:
                header = text[3:end].strip()
                for line in header.splitlines():
                    if ":" in line:
                        key, value = line.split(":", 1)
                        meta[key.strip()] = value.strip()
                body = text[end + 3:].strip()

        return meta, body