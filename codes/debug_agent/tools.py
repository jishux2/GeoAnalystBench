"""
Debug Agent工具集
提供代码执行、文件读取、交互式调试等能力
"""

import asyncio
import json
import os
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Any

from .pdb_controller import PdbSessionController


def _run_script_sync(
    interpreter: str,
    script_path: str,
    cwd: str,
    env: Dict[str, str],
    timeout: int = 300
) -> Dict[str, Any]:
    """
    同步执行脚本（供进程池调用）
    
    独立于类实例，避免序列化时携带不可pickle的对象
    """
    try:
        result = subprocess.run(
            [interpreter, script_path],
            cwd=cwd,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        
        return {
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr
        }
    
    except subprocess.TimeoutExpired:
        return {
            "returncode": -1,
            "stdout": "",
            "stderr": f"Execution timed out (>{timeout}s)"
        }
    
    except Exception as e:
        return {
            "returncode": -1,
            "stdout": "",
            "stderr": str(e)
        }


class DebugToolkit:
    """调试工具集"""
    
    def __init__(
        self,
        script_content: str,
        working_dir: str,
        interpreter: str,
        output_dir: Path,
        executor: Optional[ProcessPoolExecutor] = None
    ):
        """
        初始化工具集
        
        Args:
            script_content: 待调试的脚本内容（不含插桩）
            working_dir: 工作目录
            interpreter: Python解释器路径
            output_dir: 输出目录
            executor: 进程池引用，用于并发执行脚本
        """
        self.script_content = script_content  # 不再调用 _inject_common_helpers
        self.working_dir = working_dir
        self.interpreter = interpreter
        self.output_dir = Path(output_dir)
        self.executor = executor
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self._pdb_session: Optional[PdbSessionController] = None
        self._temp_files: List[Path] = []
    
    def get_tool_definitions(self) -> List[Dict]:
        """获取工具定义列表"""
        return [
            {
                "type": "function",
                "function": {
                    "name": "execute_with_tracing",
                    "description": "Execute the script with error tracking and function call monitoring. Captures detailed runtime context on failure.",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "execute_with_logging",
                    "description": "Execute the script with custom logging statements inserted at specified lines.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "insertions": {
                                "type": "array",
                                "description": "List of logging insertions",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "line_number": {
                                            "type": "integer",
                                            "description": "Line number to insert before"
                                        },
                                        "code": {
                                            "type": "string",
                                            "description": "Python code to insert (written from top-level, indentation will be auto-adjusted)"
                                        }
                                    },
                                    "required": ["line_number", "code"]
                                }
                            }
                        },
                        "required": ["insertions"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "execute_with_postmortem",
                    "description": "Execute the script with post-mortem debugging. When an unhandled exception occurs, drops into PDB at the crash site.",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "start_stepping_debug",
                    "description": "Start an interactive PDB session from the beginning of the script. Allows step-by-step execution.",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read the contents of a file. Use for error traces, call details, or any other file.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "file_path": {
                                "type": "string",
                                "description": "Path to the file (relative to working directory or absolute)"
                            }
                        },
                        "required": ["file_path"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "execute_pdb_command",
                    "description": "Execute a PDB command in the active debugging session.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "command": {
                                "type": "string",
                                "description": "The PDB command to execute (e.g., 'n', 's', 'c', 'p variable', 'l')"
                            }
                        },
                        "required": ["command"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "inject_code_block",
                    "description": "Evaluate a Python code block within an active PDB session. Requires a prior call to start_stepping_debug or execute_with_postmortem. The code runs in the session's runtime scope and does not alter the script itself.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "code": {
                                "type": "string",
                                "description": "Python code to evaluate (e.g., inspect variables, test expressions)"
                            }
                        },
                        "required": ["code"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "set_breakpoint",
                    "description": "Set a breakpoint by matching code context. More robust than line numbers.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "target_code": {
                                "type": "string",
                                "description": "Code snippet to match (whitespace-normalized)"
                            }
                        },
                        "required": ["target_code"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "close_debug_session",
                    "description": "Close the active PDB session and clean up resources.",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "finalize_success",
                    "description": "Conclude the session with a positive outcome. Invoke when the script meets the objectives of the current debugging mode.",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "finalize_with_patch",
                    "description": "Submit the final diagnosis and code fixes. Call this once you have pinpointed the defect and prepared all necessary corrections.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "root_cause": {
                                "type": "string",
                                "description": "Thorough account of the underlying defect—what went wrong and why. Should stand alone without referencing prior rounds."
                            },
                            "patches": {
                                "type": "array",
                                "description": "Exhaustive set of modifications required. Supersedes any earlier version in its entirety.",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "target_code": {
                                            "type": "string",
                                            "description": "Fragment to locate and replace. Transcribe without leading whitespace; indentation handled separately."
                                        },
                                        "replacement_code": {
                                            "type": "string",
                                            "description": "Revised fragment. Likewise, omit leading whitespace."
                                        },
                                        "indent_level": {
                                            "type": "integer",
                                            "description": "Nesting depth in the source file (0=outermost scope, increments per block level).",
                                            "default": 0
                                        }
                                    },
                                    "required": ["target_code", "replacement_code"]
                                }
                            }
                        },
                        "required": ["root_cause", "patches"]
                    }
                }
            }
            # ============================================================
            # 预留：检索相关工具（尚未实现）
            # ============================================================
            # {
            #     "type": "function",
            #     "function": {
            #         "name": "search_api_docs",
            #         "description": "Search API documentation for a specific library/function.",
            #         "parameters": {...}
            #     }
            # },
            # {
            #     "type": "function",
            #     "function": {
            #         "name": "search_code_examples",
            #         "description": "Search for relevant code examples.",
            #         "parameters": {...}
            #     }
            # }
        ]
    
    async def execute_tool(self, tool_name: str, arguments: Dict) -> Dict[str, Any]:
        """执行工具调用"""
        handlers = {
            "execute_with_tracing": self._execute_with_tracing,
            "execute_with_logging": self._execute_with_logging,
            "execute_with_postmortem": self._execute_with_postmortem,
            "start_stepping_debug": self._start_stepping_debug,
            "read_file": self._read_file,
            "execute_pdb_command": self._execute_pdb_command,
            "inject_code_block": self._inject_code_block,
            "set_breakpoint": self._set_breakpoint,
            "close_debug_session": self._close_debug_session,
            "finalize_success": self._finalize_success,
            "finalize_with_patch": self._finalize_with_patch,
        }
        
        handler = handlers.get(tool_name)
        if not handler:
            return {"success": False, "result": f"Unknown tool: {tool_name}"}
        
        try:
            return await handler(arguments)
        except Exception as e:
            return {"success": False, "result": f"Tool execution failed: {e}"}
    
    # ============================================================
    # 一次性执行类方法
    # ============================================================
    
    async def _execute_script_async(
        self,
        script_content: str,
        env_extras: Optional[Dict[str, str]] = None,
        label: str = "script"
    ) -> Dict[str, Any]:
        script_path = self._create_temp_script(script_content)
    
        # 开发阶段：保存注入后的完整脚本供审查
        debug_snapshot = self.output_dir / f"injected_{label}.py"
        debug_snapshot.write_text(script_content, encoding='utf-8')
        
        env = os.environ.copy()
        if env_extras:
            env.update(env_extras)
        
        if self.executor:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.executor,
                _run_script_sync,  # 模块级函数，不带self
                self.interpreter,
                str(script_path),
                self.working_dir,
                env
            )
        else:
            result = _run_script_sync(
                self.interpreter,
                str(script_path),
                self.working_dir,
                env
            )
        
        return result
    
    async def _execute_with_tracing(self, args: Dict) -> Dict[str, Any]:
        """带跟踪的执行"""
        injected_code = self._inject_tracing(self.script_content)
        
        result = await self._execute_script_async(
            injected_code,
            env_extras={'EVAL_OUTPUT_DIR': str(self.output_dir)},
            label="tracing"
        )
        
        success = result["returncode"] == 0
        output = result["stdout"] + result["stderr"]
        
        trace_file = self.output_dir / "error_trace.json"
        trace_hint = ""
        if trace_file.exists():
            trace_hint = f"\n\nError trace saved to: {trace_file}"
        
        return {
            "success": success,
            "result": f"Exit code: {result['returncode']}\n\n{output}{trace_hint}"
        }
    
    async def _execute_with_logging(self, args: Dict) -> Dict[str, Any]:
        """插入日志语句后执行"""
        insertions = args.get("insertions", [])
        
        if not insertions:
            return {"success": False, "result": "No insertions provided"}
        
        try:
            modified_code = self._insert_logging_statements(
                self.script_content,
                insertions
            )
        except Exception as e:
            return {"success": False, "result": f"Failed to insert logging: {e}"}
        
        result = await self._execute_script_async(modified_code, label="logging")
        
        success = result["returncode"] == 0
        output = result["stdout"] + result["stderr"]
        
        return {
            "success": success,
            "result": f"Exit code: {result['returncode']}\n\n{output}"
        }
    
    def _insert_logging_statements(
        self,
        code: str,
        insertions: List[Dict]
    ) -> str:
        """在指定行号前插入日志语句"""
        lines = code.split('\n')
        
        # 按行号降序排列，从后往前插入，避免行号偏移
        sorted_insertions = sorted(
            insertions,
            key=lambda x: x['line_number'],
            reverse=True
        )
        
        for insertion in sorted_insertions:
            line_num = insertion['line_number']
            insert_code = insertion['code']
            
            if line_num < 1 or line_num > len(lines) + 1:
                continue
            
            target_line_idx = line_num - 1
            
            # 获取目标行的缩进
            if target_line_idx < len(lines):
                target_line = lines[target_line_idx]
                indent = len(target_line) - len(target_line.lstrip())
            else:
                indent = 0
            
            # 为插入代码应用相同缩进
            indented_code = self._apply_indentation(insert_code, indent)
            
            lines.insert(target_line_idx, indented_code)
        
        return '\n'.join(lines)
    
    # ============================================================
    # 交互式调试类方法
    # ============================================================
    
    async def _execute_with_postmortem(self, args: Dict) -> Dict[str, Any]:
        """事后调试执行"""
        if self._pdb_session is not None:
            return {"success": False, "result": "A debug session is already active. Close it first."}
        
        script_path = self._create_temp_script(
            self._inject_postmortem_hook(self.script_content)
        )
        
        try:
            self._pdb_session = PdbSessionController.start_script(
                str(script_path),
                self.working_dir,
                self.interpreter
            )
            
            return {
                "success": True,
                "result": f"Post-mortem debugging started.\n\nInitial output:\n{self._pdb_session.initial_output}"
            }
        
        except Exception as e:
            return {"success": False, "result": f"Failed to start post-mortem session: {e}"}
    
    async def _start_stepping_debug(self, args: Dict) -> Dict[str, Any]:
        """启动单步调试"""
        if self._pdb_session is not None:
            return {"success": False, "result": "A debug session is already active. Close it first."}
        
        script_path = self._create_temp_script(self.script_content)
        
        try:
            self._pdb_session = PdbSessionController.start_with_pdb(
                str(script_path),
                self.working_dir,
                self.interpreter
            )
            
            return {
                "success": True,
                "result": f"Step-through debugging started.\n\nInitial output:\n{self._pdb_session.initial_output}"
            }
        
        except Exception as e:
            return {"success": False, "result": f"Failed to start debugging session: {e}"}
    
    async def _read_file(self, args: Dict) -> Dict[str, Any]:
        """读取文件"""
        file_path = args.get("file_path", "")
        
        if not file_path:
            return {"success": False, "result": "file_path is required"}
        
        path = Path(file_path)
        if not path.is_absolute():
            path = Path(self.working_dir) / path
        
        if not path.exists():
            return {"success": False, "result": f"File not found: {path}"}
        
        try:
            content = path.read_text(encoding='utf-8')
            return {"success": True, "result": content}
        
        except Exception as e:
            return {"success": False, "result": f"Failed to read file: {e}"}
    
    async def _execute_pdb_command(self, args: Dict) -> Dict[str, Any]:
        """执行PDB命令"""
        if self._pdb_session is None:
            return {"success": False, "result": "No active debug session"}
        
        command = args.get("command", "")
        if not command:
            return {"success": False, "result": "command is required"}
        
        response = self._pdb_session.send_command(command)
        return {"success": True, "result": response}
    
    async def _inject_code_block(self, args: Dict) -> Dict[str, Any]:
        """注入代码块执行"""
        if self._pdb_session is None:
            return {"success": False, "result": "No active debug session"}
        
        code = args.get("code", "")
        
        if not code:
            return {"success": False, "result": "code is required"}
        
        response = self._pdb_session.execute_code(code)
        return {"success": True, "result": response}
    
    async def _set_breakpoint(self, args: Dict) -> Dict[str, Any]:
        """设置断点"""
        if self._pdb_session is None:
            return {"success": False, "result": "No active debug session"}
        
        target_code = args.get("target_code", "")
        if not target_code:
            return {"success": False, "result": "target_code is required"}
        
        response = self._pdb_session.set_breakpoint_by_context(target_code)
        return {"success": True, "result": response}
    
    async def _close_debug_session(self, args: Dict) -> Dict[str, Any]:
        """关闭调试会话"""
        if self._pdb_session is None:
            return {"success": False, "result": "No active debug session"}
        
        self._pdb_session.close()
        self._pdb_session = None
        
        return {"success": True, "result": "Debug session closed"}
    
    # ============================================================
    # 终结类方法
    # ============================================================
    
    async def _finalize_success(self, args: Dict) -> Dict[str, Any]:
        """标记成功完成"""
        return {
            "success": True,
            "result": "[FINALIZE_SUCCESS]",
            "final": True,
            "diagnosis": None
        }
    
    async def _finalize_with_patch(self, args: Dict) -> Dict[str, Any]:
        """提交诊断和补丁"""
        root_cause = args.get("root_cause", "")
        patches = args.get("patches", [])
        
        if not root_cause:
            return {"success": False, "result": "root_cause is required"}
        
        if not patches:
            return {"success": False, "result": "At least one patch is required"}
        
        processed_patches = []
        for patch in patches:
            indent_level = patch.get("indent_level", 0)
            processed_patches.append({
                "target_code": patch["target_code"],
                "replacement_code": self._apply_indent_level(
                    patch["replacement_code"],
                    indent_level
                )
            })
        
        return {
            "success": True,
            "result": "[FINALIZE_WITH_PATCH]",
            "final": True,
            "diagnosis": {
                "root_cause": root_cause,
                "patches": processed_patches
            }
        }
    
    # ============================================================
    # 辅助方法
    # ============================================================
    
    def cleanup(self):
        """清理资源"""
        if self._pdb_session is not None:
            self._pdb_session.close()
            self._pdb_session = None
        
        for temp_file in self._temp_files:
            try:
                temp_file.unlink()
            except:
                pass
        
        self._temp_files.clear()
    
    def _create_temp_script(self, content: str) -> Path:
        """创建临时脚本文件"""
        temp_file = tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.py',
            delete=False,
            encoding='utf-8'
        )
        temp_file.write(content)
        temp_file.close()
        
        path = Path(temp_file.name)
        self._temp_files.append(path)
        
        return path
    
    def _apply_indentation(self, code: str, spaces: int) -> str:
        """为代码块应用指定数量的空格缩进"""
        if spaces <= 0:
            return code
        
        indent = ' ' * spaces
        lines = code.split('\n')
        
        return '\n'.join(
            indent + line if line.strip() else line
            for line in lines
        )
    
    def _apply_indent_level(self, code: str, indent_level: int) -> str:
        """为代码块应用指定级别的缩进（每级4空格）"""
        return self._apply_indentation(code, indent_level * 4)
    
    # ============================================================
    # 执行模式的代码注入
    # ============================================================
    
    def _inject_tracing(self, code: str) -> str:
        """注入错误跟踪逻辑"""
        tracing_code = self._get_tracing_code()
        return tracing_code + "\n" + code
    
    def _inject_postmortem_hook(self, code: str) -> str:
        """注入事后调试钩子"""
        hook_code = '''
import sys
import pdb
import traceback

def _postmortem_hook(exc_type, exc_value, exc_tb):
    traceback.print_exception(exc_type, exc_value, exc_tb)
    pdb.post_mortem(exc_tb)

sys.excepthook = _postmortem_hook
'''
        return hook_code + "\n" + code
    
    def _get_tracing_code(self) -> str:
        """获取跟踪代码"""
        return '''
import sys
import json
import traceback
import functools
import re
import linecache
import os

# Read output directory from environment variable (set by executor)
EVAL_OUTPUT_DIR = os.environ.get('EVAL_OUTPUT_DIR', '.')

# ============================================================
# Context-aware object summarization for error diagnostics
# ============================================================

def extract_accessed_fields(code_line, var_name):
    """
    Extract field accesses from a line of code for a given variable
    
    Identifies three access patterns:
    - Bracket notation: var_name['field'] or var_name["field"]
    - Dot notation: var_name.field (excludes method calls)
    """
    escaped_var = re.escape(var_name)
    
    # Match var_name['field'] or var_name["field"]
    pattern1 = rf"(?<!\\w){escaped_var}\\['([^']+)'\\]"
    pattern2 = rf'(?<!\\w){escaped_var}\\["([^"]+)"\\]'
    
    # Match var_name.field (but not var_name.method())
    pattern3 = rf"(?<!\\w){escaped_var}\\.(\\w+)(?!\\()"
    
    fields = set()
    
    try:
        fields.update(re.findall(pattern1, code_line))
    except re.error:
        pass
    
    try:
        fields.update(re.findall(pattern2, code_line))
    except re.error:
        pass
    
    try:
        fields.update(re.findall(pattern3, code_line))
    except re.error:
        pass
    
    return list(fields)


def smart_summarize(obj, code_line=None, var_name=None, max_len=200):
    """
    Intelligently summarize objects, prioritizing fields accessed in code
    
    For pandas Series, extracts field access patterns from the error-triggering
    code line and displays those fields first, making diagnostics more relevant
    """
    try:
        obj_type = type(obj).__name__
        
        # Handle pandas Series with context-aware field selection
        if 'Series' in obj_type:
            result = {'_type': 'Series', '_dtype': str(obj.dtype)}
            
            # Extract fields accessed in the error-triggering code line
            priority_fields = []
            if code_line and var_name:
                try:
                    priority_fields = extract_accessed_fields(code_line, var_name)
                except Exception as e:
                    result['_extract_error'] = str(e)
            
            # Display accessed fields first (more relevant for debugging)
            shown_fields = set()
            for field in priority_fields:
                if field in obj.index:
                    try:
                        result[field] = str(obj[field])[:200]
                        shown_fields.add(field)
                    except Exception as e:
                        result[f'{field}_error'] = str(e)
            
            # Fill remaining slots with other fields (max 20 total)
            remaining = 20 - len(shown_fields)
            for field in obj.index:
                if field not in shown_fields and remaining > 0:
                    try:
                        result[str(field)] = str(obj[field])[:100]
                        shown_fields.add(field)
                        remaining -= 1
                    except Exception:
                        pass
            
            if len(obj) > len(shown_fields):
                result['_truncated'] = f'... and {len(obj) - len(shown_fields)} more fields'
            
            return result
        
        # Handle DataFrame/GeoDataFrame
        if 'DataFrame' in obj_type or 'GeoDataFrame' in obj_type:
            info = {
                '_type': obj_type,
                '_shape': f'{obj.shape[0]} rows × {obj.shape[1]} columns',
                '_columns': list(obj.columns)[:10]
            }
            if 'GeoDataFrame' in obj_type and hasattr(obj, 'geometry') and 'geometry' in obj.columns:
                try:
                    geom_types = obj.geometry.geom_type.value_counts().to_dict()
                    info['_geometry_types'] = geom_types
                except Exception:
                    pass
            return info
        
        # Handle collections
        if isinstance(obj, (list, tuple)):
            return f"{obj_type}(len={len(obj)})"
        if isinstance(obj, dict):
            return f"dict(keys={list(obj.keys())[:5]})"
        
        # Default fallback
        return str(obj)[:max_len]
    
    except Exception as e:
        return f"<{type(obj).__name__} object>"


# ============================================================
# Global exception hook with stack frame analysis
# ============================================================

def capture_exception(exc_type, exc_value, exc_traceback):
    """
    Capture exception with detailed context including local variables
    
    Traverses the call stack to collect code context and variable states,
    applying smart summarization to make diagnostics actionable
    """
    stack_info = []
    tb = exc_traceback
    
    while tb is not None:
        frame = tb.tb_frame
        code_line = linecache.getline(frame.f_code.co_filename, tb.tb_lineno).strip()
        
        # Summarize local variables with code context awareness
        frame_locals = {}
        for var_name, var_value in frame.f_locals.items():
            frame_locals[var_name] = smart_summarize(
                var_value,
                code_line=code_line,
                var_name=var_name
            )
        
        stack_info.append({
            'file': frame.f_code.co_filename,
            'function': frame.f_code.co_name,
            'line': tb.tb_lineno,
            'code': code_line,
            'locals': frame_locals
        })
        tb = tb.tb_next
    
    context = {
        'error_type': exc_type.__name__,
        'error_message': str(exc_value),
        'traceback': ''.join(traceback.format_exception(exc_type, exc_value, exc_traceback)),
        'stack_frames': stack_info
    }
    
    trace_file = os.path.join(EVAL_OUTPUT_DIR, 'error_trace.json')
    with open(trace_file, 'w', encoding='utf-8') as f:
        json.dump(context, f, indent=2, ensure_ascii=False)
    
    # Call the original exception handler to print traceback
    sys.__excepthook__(exc_type, exc_value, exc_traceback)

# Register the custom exception hook
sys.excepthook = capture_exception


# ============================================================
# Function call monitoring decorator
# ============================================================

def monitor_call(func_name):
    """
    Decorator to capture function arguments when errors occur
    
    Wraps third-party library functions to log their invocation details,
    providing visibility into failed API calls that try-except might mask
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                arg_summaries = [smart_summarize(arg) for arg in args]
                kwargs_summaries = {k: smart_summarize(v) for k, v in kwargs.items()}
                
                detail = {
                    'function': func_name,
                    'args_summary': arg_summaries,
                    'kwargs_summary': kwargs_summaries,
                    'error': str(e)
                }
                
                detail_file = os.path.join(EVAL_OUTPUT_DIR, 'call_details.json')
                
                # Maintain array structure (read-modify-write)
                if os.path.exists(detail_file):
                    with open(detail_file, 'r', encoding='utf-8') as f:
                        try:
                            data = json.load(f)
                            if not isinstance(data, list):
                                data = []
                        except json.JSONDecodeError:
                            data = []
                else:
                    data = []
                
                data.append(detail)
                
                with open(detail_file, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                
                raise
        return wrapper
    return decorator
'''