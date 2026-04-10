# codes/agents/skill_tools.py
"""
各技能的工具工厂集中定义

每个技能对应一个工厂函数，接收运行时依赖（文件操作实例、
脚本执行器、PDB控制器等）和技能目录路径，返回该技能
关联的ToolSpec列表。

工厂注册在Coordinator初始化阶段完成，与角色子类解耦——
技能自己定义自己带什么工具，角色只决定加载哪个技能。
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List

from .tool_base import ToolSpec

if TYPE_CHECKING:
    from .tools_common import FileOperations, ScriptExecutor
    from .pdb_launcher import PdbLauncher


# ================================================================
# 数据审查技能 (data-inspection)
# ================================================================

def build_data_inspection_tools(
    skill_dir: Path,
    file_ops: FileOperations,
    script_executor: ScriptExecutor,
) -> List[ToolSpec]:
    """
    构建数据审查技能关联的工具集。

    Args:
        skill_dir: 技能目录路径，用于定位预置诊断脚本
        file_ops: 文件操作实例
        script_executor: 脚本执行器实例
    """
    scripts_dir = skill_dir / "scripts"

    return [
        ToolSpec(
            name="write_file",
            description=(
                "Create or overwrite a file, or append content to an existing one. "
                "Use for writing custom inspection scripts, saving the initial exploration "
                "report, or incrementally extending it with findings from follow-up probes."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path relative to working directory.",
                    },
                    "content": {
                        "type": "string",
                        "description": "Text to write or append.",
                    },
                    "append": {
                        "type": "boolean",
                        "description": "When true, content is added to the end of the file rather than replacing it.",
                        "default": False,
                    },
                },
                "required": ["file_path", "content"],
            },
            handler=file_ops.handle_write_file,
        ),
        ToolSpec(
            name="edit_file",
            description=(
                "Apply search-and-replace edits to an existing file. "
                "Each edit must match exactly once in the current content."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the file to edit.",
                    },
                    "edits": {
                        "type": "array",
                        "description": "List of search-replace pairs.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "search": {"type": "string"},
                                "replace": {"type": "string"},
                            },
                            "required": ["search", "replace"],
                        },
                    },
                },
                "required": ["file_path", "edits"],
            },
            handler=file_ops.handle_edit_file,
        ),
        ToolSpec(
            name="execute_script",
            description=(
                "Execute a Python script stored on disk or a code string passed "
                "inline. Staple uses include invoking pre-packaged inspection scripts "
                "bundled with the skill, running focused probes you have assembled, "
                "and performing ad-hoc computations that do not merit a dedicated file. "
                "Output is persisted to disk for file-based runs; inline invocations "
                "return their console output directly in the result."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Script path (mutually exclusive with code).",
                    },
                    "code": {
                        "type": "string",
                        "description": "Python code to execute inline (mutually exclusive with file_path).",
                    },
                    "args": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Command-line arguments forwarded to the script.",
                        "default": [],
                    },
                },
                "required": [],
            },
            handler=lambda file_path=None, code=None, args=None, **kwargs: (
                script_executor.handle_execute_code(code)
                if code else
                script_executor.handle_execute_script(file_path, with_tracing=False, args=args)
            ),
        ),
    ]


# ================================================================
# 脚本工程技能 (script-engineering)
# ================================================================

def build_script_engineering_tools(
    skill_dir: Path,
    file_ops: FileOperations,
) -> List[ToolSpec]:
    """
    构建脚本工程技能关联的工具集。

    工程师的职责收束为数据理解、脚本编写与断言植入，
    交付后即退出活跃状态。工具集仅包含文件写入。
    """
    return [
        ToolSpec(
            name="write_file",
            description=(
                "Create or overwrite a file at a given path. Primary uses include "
                "committing the task script to 'current_script.py' and saving "
                "technical design notes during the planning phase."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path relative to working directory.",
                    },
                    "content": {
                        "type": "string",
                        "description": "File content to write.",
                    },
                },
                "required": ["file_path", "content"],
            },
            handler=file_ops.handle_write_file,
        ),
    ]


# ================================================================
# 代码诊断技能 (code-diagnosis)
# ================================================================

def build_code_diagnosis_tools(
    skill_dir: Path,
    file_ops: FileOperations,
    script_executor: ScriptExecutor,
    pdb_launcher: PdbLauncher,
) -> List[ToolSpec]:
    return [
        ToolSpec(
            name="edit_file",
            description=(
                "Apply search-and-replace edits to an existing file. "
                "Principal use is patching current_script.py to address "
                "identified defects. Each search text must match exactly "
                "once; include sufficient surrounding context for unique targeting."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the file to edit.",
                    },
                    "edits": {
                        "type": "array",
                        "description": "List of search-replace pairs.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "search": {"type": "string"},
                                "replace": {"type": "string"},
                            },
                            "required": ["search", "replace"],
                        },
                    },
                },
                "required": ["file_path", "edits"],
            },
            handler=file_ops.handle_edit_file,
        ),
        ToolSpec(
            name="inject_and_save",
            description=(
                "Insert code statements at specified line numbers in a source file "
                "and save the result to a separate output file. The original file is "
                "not modified. Use for weaving diagnostic print statements or variable "
                "surveillance into a disposable copy—execute it, extract your "
                "observations from the output, and move on without retaining the file "
                "in your working view."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "source_path": {
                        "type": "string",
                        "description": "Path to the original script.",
                    },
                    "output_path": {
                        "type": "string",
                        "description": "Path for the injected version.",
                    },
                    "injections": {
                        "type": "array",
                        "description": "List of insertion directives.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "line_number": {
                                    "type": "integer",
                                    "description": "Line number to insert before (1-based).",
                                },
                                "code": {
                                    "type": "string",
                                    "description": "Code to insert.",
                                },
                            },
                            "required": ["line_number", "code"],
                        },
                    },
                },
                "required": ["source_path", "output_path", "injections"],
            },
            handler=file_ops.handle_inject_statements,
        ),
        ToolSpec(
            name="execute_script",
            description=(
                "Run a Python script from disk or evaluate a code snippet supplied "
                "as a direct parameter. When targeting a file, set with_tracing to "
                "activate exception tracking hooks and function call monitors for "
                "enriched failure analysis. Inline snippets execute without tracing "
                "and return their stdout content directly—suited for quick empirical "
                "checks that do not warrant a persistent script."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Script path to execute (mutually exclusive with code).",
                    },
                    "code": {
                        "type": "string",
                        "description": "Python code to execute inline (mutually exclusive with file_path).",
                    },
                    "with_tracing": {
                        "type": "boolean",
                        "description": "Inject error tracking and function monitoring.",
                        "default": False,
                    },
                    "args": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Command-line arguments forwarded to the script.",
                        "default": [],
                    },
                },
                "required": [],
            },
            handler=lambda file_path=None, code=None, with_tracing=False, args=None, **kwargs: (
                script_executor.handle_execute_code(code, with_tracing=with_tracing)
                if code else
                script_executor.handle_execute_script(file_path, with_tracing=with_tracing, args=args)
            ),
        ),
        ToolSpec(
            name="start_postmortem_debug",
            description=(
                "Launch the task script with a post-mortem hook. When an unhandled "
                "exception occurs, drops into an interactive PDB session at the crash "
                "site. Use execute_pdb_command to inspect the failure state."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "script_path": {
                        "type": "string",
                        "description": "Path to the script to debug.",
                    },
                },
                "required": ["script_path"],
            },
            handler=pdb_launcher.handle_start_postmortem,
        ),
        ToolSpec(
            name="start_stepping_debug",
            description=(
                "Open an interactive PDB session from the first line of the script. "
                "Execution advances only when you issue navigation commands. "
                "Reserve for situations requiring observation of state evolution "
                "across specific code regions."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "script_path": {
                        "type": "string",
                        "description": "Path to the script to debug.",
                    },
                },
                "required": ["script_path"],
            },
            handler=pdb_launcher.handle_start_stepping,
        ),
        ToolSpec(
            name="execute_pdb_command",
            description=(
                "Send a command to the active PDB session and return its output. "
                "Supports the full PDB vocabulary: navigation (n, s, c, r), "
                "inspection (p, pp, w, l), breakpoints (b, cl), and direct "
                "Python evaluation."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "PDB command string.",
                    },
                },
                "required": ["command"],
            },
            handler=pdb_launcher.handle_pdb_command,
        ),
        ToolSpec(
            name="inject_code_block",
            description=(
                "Evaluate a Python code block within the active PDB session's "
                "runtime scope. Side effects persist in the session."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python code to evaluate.",
                    },
                },
                "required": ["code"],
            },
            handler=pdb_launcher.handle_inject_code,
        ),
        ToolSpec(
            name="close_debug_session",
            description=(
                "Terminate the active PDB session and release associated resources. "
                "The full interaction transcript is archived to disk automatically; "
                "provide a summary of your findings to retain a compact reference "
                "in context after the detailed history is compressed away."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "string",
                        "description": (
                            "Key observations and conclusions from this session—"
                            "hypotheses tested, variables examined, outcomes reached."
                        ),
                        "default": "",
                    },
                },
                "required": [],
            },
            handler=pdb_launcher.handle_close_session,
        ),
    ]


# ================================================================
# 工厂注册辅助
# ================================================================

def register_all_skill_factories(
    skill_registry,
    file_ops: FileOperations,
    explorer_executor: ScriptExecutor,
    diagnostician_executor: ScriptExecutor,
    pdb_launcher: PdbLauncher,
):
    """
    在Coordinator初始化阶段统一注册所有技能的工厂函数。

    通过闭包捕获运行时依赖，生成符合SkillRegistry.register_tool_factory
    签名要求的工厂函数（接收skill_dir，返回ToolSpec列表）。
    """
    skill_registry.register_tool_factory(
        "data-inspection",
        lambda skill_dir: build_data_inspection_tools(
            skill_dir, file_ops, explorer_executor
        ),
    )

    skill_registry.register_tool_factory(
        "script-engineering",
        lambda skill_dir: build_script_engineering_tools(
            skill_dir, file_ops
        ),
    )

    skill_registry.register_tool_factory(
        "code-diagnosis",
        lambda skill_dir: build_code_diagnosis_tools(
            skill_dir, file_ops, diagnostician_executor, pdb_launcher
        ),
    )