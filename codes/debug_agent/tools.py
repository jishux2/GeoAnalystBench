"""
Debug Agent工具集定义
"""

from typing import Dict, List, Optional
from .pdb_controller import SubprocessPdbController


class DebugTools:
    """调试工具集"""
    
    def __init__(self, script_path: str, cwd: str, interpreter: str):  # ← 新增参数
        """
        初始化工具集
        
        Args:
            script_path: 待调试脚本路径
            cwd: 工作目录
            interpreter: Python解释器路径
        """
        self.script_path = script_path
        self.cwd = cwd
        self.interpreter = interpreter
        self.controller: Optional[SubprocessPdbController] = None
    
    def get_tool_definitions(self) -> List[Dict]:
        """
        获取工具定义列表（符合DeepSeek API格式）
        
        Returns:
            工具定义的JSON Schema列表
        """
        return [
            {
                "type": "function",
                "function": {
                    "name": "start_proactive_debugging",
                    "description": "Start a proactive debugging session, allowing step-by-step execution from the first line",
                    "parameters": {"type": "object", "properties": {}}
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "start_postmortem_debugging",
                    "description": "Start a post-mortem debugging session, which will be triggered when an unhandled exception occurs",
                    "parameters": {"type": "object", "properties": {}}
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "execute_pdb_command",
                    "description": "Execute a raw PDB command (e.g., 'n', 's', 'c', 'p variable_name', 'l', 'w')",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "command": {
                                "type": "string",
                                "description": "The PDB command to execute"
                            }
                        },
                        "required": ["command"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "execute_code_block",
                    "description": "Inject and execute a custom Python code block in the current debugging context",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "code": {
                                "type": "string",
                                "description": "Multi-line Python code to execute"
                            }
                        },
                        "required": ["code"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "set_breakpoint_by_context",
                    "description": "Set a breakpoint by providing the target code context (ignores indentation differences)",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "target_code": {
                                "type": "string",
                                "description": "The code snippet to locate (provide surrounding context for accuracy)"
                            }
                        },
                        "required": ["target_code"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "close_debugging",
                    "description": "Close the debugging session and clean up resources",
                    "parameters": {"type": "object", "properties": {}}
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "finalize_diagnosis",
                    "description": "Output the final diagnosis and repair plan, ending the agent loop",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "root_cause": {
                                "type": "string",
                                "description": "Natural language description of the root cause"
                            },
                            "repair_plan": {
                                "type": "string",
                                "description": "Actionable repair instructions for the next iteration"
                            },
                            "api_queries": {
                                "type": "array",
                                "description": "List of API documentation queries",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "library": {"type": "string"},
                                        "version": {"type": "string"},
                                        "api_path": {"type": "string"}
                                    }
                                }
                            },
                            "keywords": {
                                "type": "array",
                                "description": "Keywords for code example retrieval",
                                "items": {"type": "string"}
                            },
                            "example_query": {
                                "type": "string",
                                "description": "Semantic query for code example search"
                            }
                        },
                        "required": ["root_cause", "repair_plan"]
                    }
                }
            }
        ]
    
    def execute_tool(self, tool_name: str, arguments: Dict) -> str:
        """
        执行指定的工具调用
        
        Args:
            tool_name: 工具名称
            arguments: 工具参数
        
        Returns:
            工具执行结果
        """
        if tool_name == "start_proactive_debugging":
            return self._start_proactive_debugging()
        
        elif tool_name == "start_postmortem_debugging":
            return self._start_postmortem_debugging()
        
        elif tool_name == "execute_pdb_command":
            return self._execute_pdb_command(arguments["command"])
        
        elif tool_name == "execute_code_block":
            return self._execute_code_block(arguments["code"])
        
        elif tool_name == "set_breakpoint_by_context":
            return self._set_breakpoint_by_context(arguments["target_code"])
        
        elif tool_name == "close_debugging":
            return self._close_debugging()
        
        elif tool_name == "finalize_diagnosis":
            return self._finalize_diagnosis(arguments)
        
        else:
            return f"[ERROR] Unknown tool: {tool_name}"
    
    def _start_proactive_debugging(self) -> str:
        """启动预防性调试"""
        from pathlib import Path
        if self.controller is not None:
            return "[ERROR] Debugging session already active"
        
        try:
            # 转换为绝对路径
            abs_script_path = str(Path(self.script_path).resolve())
            
            self.controller = SubprocessPdbController.start_proactive(
                abs_script_path,  # ← 使用绝对路径
                self.cwd,
                self.interpreter
            )
            return f"Proactive debugging started\nInitial output:\n{self.controller.initial_output}"
        
        except Exception as e:
            return f"[ERROR] Failed to start debugging: {e}"
    
    def _start_postmortem_debugging(self) -> str:
        """启动事后调试"""
        if self.controller is not None:
            return "[ERROR] Debugging session already active"
        
        try:
            self.controller = SubprocessPdbController.start_post_mortem(
                self.script_path,
                self.cwd,
                self.interpreter  # ← 传递解释器路径
            )
            return f"Post-mortem debugging started\nInitial output:\n{self.controller.initial_output}"
        
        except Exception as e:
            return f"[ERROR] Failed to start debugging: {e}"
    
    def _execute_pdb_command(self, command: str) -> str:
        """执行PDB命令"""
        if self.controller is None:
            return "[ERROR] No active debugging session"
        
        return self.controller.send_command(command)
    
    def _execute_code_block(self, code: str) -> str:
        """执行代码块"""
        if self.controller is None:
            return "[ERROR] No active debugging session"
        
        return self.controller.execute_code_block(code)
    
    def _set_breakpoint_by_context(self, target_code: str) -> str:
        """基于上下文设置断点"""
        if self.controller is None:
            return "[ERROR] No active debugging session"
        
        return self.controller.set_breakpoint_by_context(target_code)
    
    def _close_debugging(self) -> str:
        """关闭调试会话"""
        if self.controller is None:
            return "[ERROR] No active debugging session"
        
        self.controller.close()
        self.controller = None
        return "Debugging session closed"
    
    def _finalize_diagnosis(self, arguments: Dict) -> str:
        """输出最终诊断（特殊标记，用于终止Agent循环）"""
        return "[DIAGNOSIS_COMPLETE]"
    
    def cleanup(self):
        """清理资源"""
        if self.controller is not None:
            self.controller.close()
            self.controller = None