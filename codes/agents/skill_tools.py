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
                "Run a Python script in the task's working directory. "
                "Use for executing pre-built diagnostic routines from the "
                "skill's scripts/ directory or your own custom inspection scripts. "
                "Output is persisted to disk; the result contains the output directory path."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Script path (relative to working directory or absolute).",
                    },
                    "args": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Command-line arguments forwarded to the script.",
                        "default": [],
                    },
                },
                "required": ["file_path"],
            },
            handler=lambda file_path, args=None, **_: script_executor.handle_execute_script(
                file_path, with_tracing=False, args=args
            ),
        ),
    ]


# ================================================================
# 脚本工程技能 (script-engineering)
# ================================================================

def build_script_engineering_tools(
    skill_dir: Path,
    file_ops: FileOperations,
    script_executor: ScriptExecutor,
) -> List[ToolSpec]:
    """
    构建脚本工程技能关联的工具集。

    与数据审查技能共享文件操作和脚本执行的底层实现，
    但脚本执行额外支持追踪模式。
    """
    return [
        ToolSpec(
            name="write_file",
            description=(
                "Create or overwrite a file at a given path. Primary uses include "
                "committing the task script to 'current_script.py', saving auxiliary "
                "modules, and capturing architectural plans or technical design notes "
                "during the planning phase."
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
        ToolSpec(
            name="edit_file",
            description=(
                "Apply search-and-replace edits to an existing file. "
                "Use for applying patches, inserting assertions, or integrating "
                "logging statements requested by the diagnostician. "
                "Each search text must match exactly once."
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
                "Run a Python script in the task's working directory. "
                "Set with_tracing=true to inject exception tracking hooks and "
                "function call monitors—advisable for task script executions, "
                "though optional for lightweight validation passes. "
                "Output is persisted to disk."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Script path relative to working directory.",
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
                "required": ["file_path"],
            },
            handler=script_executor.handle_execute_script,
        ),
        ToolSpec(
            name="inject_and_save",
            description=(
                "Insert code statements at specified line numbers in a source file "
                "and save the result to a separate output file. The original file is "
                "not modified. Indentation is automatically aligned to the target line. "
                "Use for integrating logging or diagnostic print statements requested "
                "by the diagnostician before re-execution."
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
                                    "description": "Code to insert. Written without indentation; alignment is automatic.",
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
    ]


# ================================================================
# 代码诊断技能 (code-diagnosis)
# ================================================================

def build_code_diagnosis_tools(
    skill_dir: Path,
    file_ops: FileOperations,
    pdb_launcher: PdbLauncher,
) -> List[ToolSpec]:
    """
    构建代码诊断技能关联的工具集。

    Args:
        skill_dir: 技能目录路径
        file_ops: 文件操作实例（诊断专员仅需读写，编辑通过委托完成）
        pdb_launcher: PDB会话启动器，封装调试器的生命周期管理
    """
    return [
        ToolSpec(
            name="write_file",
            description=(
                "Write text content to a file at a given path. Creates the file "
                "if it does not exist, or overwrites it if it does."
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
    engineer_executor: ScriptExecutor,
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
            skill_dir, file_ops, engineer_executor
        ),
    )

    skill_registry.register_tool_factory(
        "code-diagnosis",
        lambda skill_dir: build_code_diagnosis_tools(
            skill_dir, file_ops, pdb_launcher
        ),
    )