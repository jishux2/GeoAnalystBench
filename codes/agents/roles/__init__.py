# codes/agents/roles/__init__.py
"""
角色子类

每个角色覆写BaseAgent的扩展点以注入角色专属的
系统提示词和启动行为。工具集的获取统一通过
技能加载机制完成，角色本身不直接定义工具。
"""

from .explorer import DataExplorer
from .engineer import ScriptEngineer
from .diagnostician import Diagnostician

__all__ = ["DataExplorer", "ScriptEngineer", "Diagnostician"]