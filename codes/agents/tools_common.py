# codes/agents/tools_common.py
"""
跨技能共享的工具handler实现

将文件操作、脚本执行等多个技能都需要的底层能力
收敛为可复用的异步函数。各技能的工厂函数引用这些
实现来构建ToolSpec，避免重复定义。

每个handler遵循统一的签名约定：接收关键字参数，
返回包含success和result字段的字典。额外的元数据
（如file_path、compress_write）通过字典附加字段传递，
供上层的上下文管理逻辑消费。
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional


# ================================================================
# 进程级脚本执行（供进程池调用的模块级函数）
# ================================================================

def _run_command_sync(
    command: list,
    cwd: str,
    env: Dict[str, str],
    timeout: int = 120
) -> Dict[str, Any]:
    try:
        result = subprocess.run(
            command,
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


# ================================================================
# 文件操作handlers
# ================================================================

class FileOperations:
    """
    文件操作的handler集合

    绑定到具体的工作目录和安全边界，供不同技能的
    工厂函数引用。同一个实例可以被多个ToolSpec共享。
    """

    def __init__(self, working_dir: Path, allowed_roots: List[Path]):
        """
        Args:
            working_dir: 工作目录，相对路径的解析基准
            allowed_roots: 允许写入的目录白名单
        """
        self.working_dir = working_dir
        self.allowed_roots = [r.resolve() for r in allowed_roots]

    def _safe_resolve(self, file_path: str) -> Path:
        """解析路径并检查安全边界。"""
        path = Path(file_path)
        if not path.is_absolute():
            path = self.working_dir / path
        resolved = path.resolve()

        if not any(resolved.is_relative_to(root) for root in self.allowed_roots):
            raise PermissionError(f"Access denied: {file_path}")
        return resolved

    async def handle_write_file(
        self, file_path: str, content: str, append: bool = False
    ) -> Dict[str, Any]:
        try:
            resolved = self._safe_resolve(file_path)
            resolved.parent.mkdir(parents=True, exist_ok=True)

            mode = "a" if append else "w"
            with open(resolved, mode, encoding="utf-8") as f:
                f.write(content)

            action = "Appended" if append else "Wrote"
            return {
                "success": True,
                "result": f"{action} {len(content)} bytes to {file_path}",
                "file_path": file_path,
                "compress_write": True,
            }
        except Exception as e:
            return {"success": False, "result": f"Write failed: {e}"}

    async def handle_edit_file(
        self, file_path: str, edits: List[Dict[str, str]]
    ) -> Dict[str, Any]:
        """
        对文件执行一组搜索-替换操作。

        每个edit条目包含search和replace两个字段。
        替换按列表顺序依次执行，每个搜索串必须在
        当前文件内容中恰好匹配一次。

        Args:
            file_path: 目标文件路径
            edits: 搜索-替换对列表
        """
        try:
            resolved = self._safe_resolve(file_path)
            if not resolved.exists():
                return {"success": False, "result": f"File not found: {file_path}"}

            content = resolved.read_text(encoding="utf-8")

            for i, edit in enumerate(edits):
                search = edit["search"]
                replace = edit["replace"]

                count = content.count(search)
                if count == 0:
                    return {
                        "success": False,
                        "result": (
                            f"Edit {i + 1} failed: search text not found in {file_path}\n"
                            f"Search (first 100 chars): {search[:100]}"
                        ),
                    }
                if count > 1:
                    return {
                        "success": False,
                        "result": (
                            f"Edit {i + 1} failed: search text matches {count} locations "
                            f"in {file_path}. Provide more context for unique matching."
                        ),
                    }
                content = content.replace(search, replace, 1)

            resolved.write_text(content, encoding="utf-8")
            return {
                "success": True,
                "result": f"Applied {len(edits)} edit(s) to {file_path}",
                "file_path": file_path,
            }
        except PermissionError as e:
            return {"success": False, "result": str(e)}
        except Exception as e:
            return {"success": False, "result": f"Edit failed: {e}"}

    async def handle_inject_statements(
        self,
        source_path: str,
        output_path: str,
        injections: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        在源文件的指定位置插入语句，另存为新文件。

        每个injection条目包含line_number和code两个字段。
        插入操作基于原始文件的行号，代码的缩进自动对齐
        到目标行的层级。按行号降序处理以避免偏移。

        Args:
            source_path: 原始脚本路径
            output_path: 插入后的输出文件路径
            injections: 插入指令列表
        """
        try:
            source = self._safe_resolve(source_path)
            if not source.exists():
                return {"success": False, "result": f"Source not found: {source_path}"}

            lines = source.read_text(encoding="utf-8").split("\n")

            sorted_injections = sorted(
                injections,
                key=lambda x: x["line_number"],
                reverse=True,
            )

            for inj in sorted_injections:
                line_num = inj["line_number"]
                code = inj["code"]

                if line_num < 1 or line_num > len(lines) + 1:
                    return {
                        "success": False,
                        "result": f"Line number {line_num} out of range (1-{len(lines) + 1})",
                    }

                target_idx = line_num - 1

                # 获取目标行的缩进并应用到插入代码
                if target_idx < len(lines):
                    target_line = lines[target_idx]
                    indent = len(target_line) - len(target_line.lstrip())
                else:
                    indent = 0

                indented_lines = []
                for code_line in code.split("\n"):
                    if code_line.strip():
                        indented_lines.append(" " * indent + code_line)
                    else:
                        indented_lines.append(code_line)

                lines.insert(target_idx, "\n".join(indented_lines))

            output = self._safe_resolve(output_path)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text("\n".join(lines), encoding="utf-8")

            return {
                "success": True,
                "result": (
                    f"Injected {len(injections)} statement(s) into {source_path}, "
                    f"saved to {output_path}"
                ),
                "file_path": output_path,
                "compress_write": True,
            }
        except PermissionError as e:
            return {"success": False, "result": str(e)}
        except Exception as e:
            return {"success": False, "result": f"Injection failed: {e}"}

# ================================================================
# 脚本执行handler
# ================================================================

class ScriptExecutor:
    """
    Python脚本执行的handler封装

    支持普通执行和带追踪钩子的执行两种模式。
    输出流持久化到磁盘，上下文中仅返回路径指针。
    """

    def __init__(
        self,
        interpreter: str,
        working_dir: Path,
        output_dir: Path,
        executor: Optional[ProcessPoolExecutor] = None,
    ):
        self.interpreter = interpreter
        self.working_dir = working_dir
        self.output_dir = output_dir
        self.executor = executor
        self._execution_count = 0

    async def handle_execute_script(
        self,
        file_path: str,
        with_tracing: bool = False,
        args: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        执行指定的Python脚本。

        Args:
            file_path: 脚本文件路径（相对于工作目录或绝对路径）
            with_tracing: 是否注入异常追踪钩子和函数监控装饰器
            args: 传递给脚本的命令行参数列表
        """
        self._execution_count += 1
        run_label = f"run_{self._execution_count}"

        script_path = Path(file_path)
        if not script_path.is_absolute():
            script_path = self.working_dir / script_path

        if not script_path.exists():
            return {"success": False, "result": f"Script not found: {file_path}"}

        # 准备本次执行的输出目录
        run_output_dir = self.output_dir / run_label
        run_output_dir.mkdir(parents=True, exist_ok=True)

        # 读取脚本内容，按需注入追踪代码
        script_content = script_path.read_text(encoding="utf-8")
        if with_tracing:
            script_content = self._inject_tracing(script_content)

        # 写入临时文件供子进程执行
        temp_script = run_output_dir / "executed_script.py"
        temp_script.write_text(script_content, encoding="utf-8")

        # 构造执行命令
        absolute_script_path = str(temp_script.resolve())
        command = [self.interpreter, absolute_script_path]
        if args:
            command.extend(args)

        # 构造环境变量
        env = os.environ.copy()
        relative_output = str(run_output_dir.relative_to(self.working_dir))
        env["EVAL_OUTPUT_DIR"] = relative_output

        # 在进程池或当前进程中执行
        if self.executor:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.executor,
                _run_command_sync,
                command,
                str(self.working_dir),
                env,
                300,
            )
        else:
            result = _run_command_sync(
                command,
                str(self.working_dir),
                env,
                300,
            )

        # 持久化输出流
        stdout_file = run_output_dir / "stdout.txt"
        stderr_file = run_output_dir / "stderr.txt"
        stdout_file.write_text(result["stdout"], encoding="utf-8")
        stderr_file.write_text(result["stderr"], encoding="utf-8")

        # 构造结果摘要
        success = result["returncode"] == 0
        summary_parts = [f"Exit code: {result['returncode']}"]
        summary_parts.append(f"Output directory: {relative_output}")

        if not success:
            # 失败时在摘要中包含stderr的尾部片段辅助快速定位
            stderr_tail = result["stderr"].strip().split("\n")[-10:]
            if stderr_tail:
                summary_parts.append(f"Stderr (last lines):\n" + "\n".join(stderr_tail))

        # 检查诊断文件
        trace_file = run_output_dir / "error_trace.json"
        if trace_file.exists():
            summary_parts.append(
                f"Error trace: {relative_output}/error_trace.json"
            )

        call_file = run_output_dir / "call_details.json"
        if call_file.exists():
            summary_parts.append(
                f"Call details: {relative_output}/call_details.json"
            )

        return {
            "success": success,
            "result": "\n".join(summary_parts),
            "file_path": str(relative_output),
        }

    def _inject_tracing(self, code: str) -> str:
        """注入异常追踪钩子和函数监控装饰器。"""
        tracing_preamble = _get_tracing_code()
        monitor_fallback = (
            "\ntry:\n"
            "    monitor_call\n"
            "except NameError:\n"
            "    def monitor_call(name):\n"
            "        def decorator(func):\n"
            "            return func\n"
            "        return decorator\n"
        )
        return tracing_preamble + "\n" + monitor_fallback + "\n" + code


def _get_tracing_code() -> str:
    """返回完整的追踪代码注入片段。"""
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