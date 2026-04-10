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
from pathlib import Path
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

    维护一组可用工具，支持按名称检索与执行。
    所有工具的返回结果在拼接到上下文前经过体积检查，
    超出预算的结果持久化到磁盘并返回截断预览。
    """

    # 默认的结果体积预算（字符数）
    DEFAULT_RESULT_BUDGET = 30000

    def __init__(self, output_dir: Path = None):
        self._tools: Dict[str, ToolSpec] = {}
        self._result_budgets: Dict[str, int] = {}
        self._output_dir = output_dir
        self._overflow_count = 0

    def register(self, spec: ToolSpec):
        """注册一项工具。同名覆盖。"""
        self._tools[spec.name] = spec

    def register_batch(self, specs: List[ToolSpec]):
        """批量注册。"""
        for spec in specs:
            self.register(spec)

    def set_result_budget(self, tool_name: str, budget: int):
        """为指定工具设置独立的结果体积预算。-1表示不限制。"""
        self._result_budgets[tool_name] = budget

    def get(self, name: str) -> Optional[ToolSpec]:
        """按名称查找工具。"""
        return self._tools.get(name)

    async def execute(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行指定工具并返回结果。

        结果经过体积预算检查，超出阈值的内容持久化到磁盘，
        上下文中仅保留截断预览和磁盘路径指引。
        """
        spec = self._tools.get(name)
        if spec is None:
            return {"success": False, "result": f"Unknown tool: {name}"}

        try:
            result = await spec.handler(**arguments)
        except Exception as e:
            return {"success": False, "result": f"Tool execution failed: {e}"}

        # 体积预算检查
        budget = self._result_budgets.get(name, self.DEFAULT_RESULT_BUDGET)
        if budget >= 0 and result.get("success"):
            result_text = result.get("result", "")
            if len(result_text) > budget:
                result = self._truncate_result(name, result_text, budget)

        return result

    def _truncate_result(self, tool_name: str, full_text: str, budget: int) -> Dict[str, Any]:
        """将超出预算的结果持久化到磁盘，返回截断预览。"""
        self._overflow_count += 1
        overflow_path = self._output_dir / f"tool_overflow_{self._overflow_count}.txt"
        overflow_path.parent.mkdir(parents=True, exist_ok=True)
        overflow_path.write_text(full_text, encoding="utf-8")

        preview_size = min(3000, budget // 4)
        preview = full_text[:preview_size]

        return {
            "success": True,
            "result": (
                f"{preview}\n\n"
                f"[Output truncated: {len(full_text)} chars exceeded budget of {budget}. "
                f"Full content saved to {overflow_path}. "
                f"Use read_file with offset/limit to examine specific sections, "
                f"or grep to locate relevant content.]"
            ),
            "truncated": True,
        }

    def api_schemas(self) -> List[Dict[str, Any]]:
        """导出当前全部工具的API声明列表。"""
        return [spec.to_api_schema() for spec in self._tools.values()]

    @property
    def available_names(self) -> List[str]:
        """当前已注册的工具名称清单。"""
        return list(self._tools.keys())