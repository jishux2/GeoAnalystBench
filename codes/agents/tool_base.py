# codes/agents/tool_base.py
"""
工具定义协议与调度框架

建立工具从声明到执行的统一管线：每项工具以ToolSpec描述其
面向模型的接口契约，以Python可调用对象承载实际逻辑。
ToolDispatcher负责管理可用工具集合，支持运行时动态扩充——
技能加载后将绑定的工具注入调度器，使智能体在会话过程中
渐进式地获得新能力。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable, Dict, List, Optional, Union


# 工具处理函数的类型签名：接收关键字参数，返回字典
ToolHandler = Callable[..., Awaitable[Dict[str, Any]]]


@dataclass
class ToolSpec:
    """
    单项工具的完整规格描述

    name和schema构成面向API的声明部分，handler承载执行逻辑，
    二者通过此数据类绑定为不可分割的整体。
    """

    name: str
    description: str
    parameters: Dict[str, Any]
    handler: ToolHandler
    bound_skill: Optional[str] = None  # 若非None，标识此工具由哪项技能引入

    def to_api_schema(self) -> Dict[str, Any]:
        """生成符合DeepSeek API工具声明格式的字典。"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolDispatcher:
    """
    工具调度器

    维护一组可用工具，支持按名称检索与执行，
    并提供动态注册接口以配合技能加载时的工具扩充。
    """

    def __init__(self):
        self._tools: Dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec):
        """注册一项工具。同名覆盖，便于技能卸载后重新加载。"""
        self._tools[spec.name] = spec

    def register_batch(self, specs: List[ToolSpec]):
        """批量注册。"""
        for spec in specs:
            self.register(spec)

    def unregister_by_skill(self, skill_name: str):
        """移除所有隶属于指定技能的工具，用于技能卸载。"""
        to_remove = [
            name for name, spec in self._tools.items()
            if spec.bound_skill == skill_name
        ]
        for name in to_remove:
            del self._tools[name]

    def get(self, name: str) -> Optional[ToolSpec]:
        """按名称查找工具，未找到返回None。"""
        return self._tools.get(name)

    async def execute(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行指定工具并返回结果。

        Args:
            name: 工具名称
            arguments: 模型传入的参数字典

        Returns:
            包含success和result字段的结果字典
        """
        spec = self._tools.get(name)
        if spec is None:
            return {"success": False, "result": f"Unknown tool: {name}"}

        try:
            return await spec.handler(**arguments)
        except Exception as e:
            return {"success": False, "result": f"Tool execution failed: {e}"}

    def api_schemas(self) -> List[Dict[str, Any]]:
        """导出当前全部工具的API声明列表，用于构造请求体。"""
        return [spec.to_api_schema() for spec in self._tools.values()]

    @property
    def available_names(self) -> List[str]:
        """当前已注册的工具名称清单。"""
        return list(self._tools.keys())