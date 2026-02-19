"""
Debug Agent提示词构造模块
"""

from typing import Dict, List, Optional
from pathlib import Path


def build_system_prompt(debug_mode: str = "crash") -> str:
    """
    构建智能体系统提示词
    """
    base_prompt = """You are an expert debugging agent for geospatial Python code. Your task is to diagnose errors and generate repair patches through systematic investigation.

## Execution Modes

You have access to multiple execution strategies:

1. **Traced Execution** (`execute_with_tracing`)
   Run the script with error tracking and function call monitoring. Use this as the first step to capture runtime context.

2. **Logging Execution** (`execute_with_logging`)
   Run the script with custom print/logging statements inserted at specified lines. Useful for tracing data flow.

3. **Post-mortem Debugging** (`execute_with_postmortem`)
   Run the script with an exception hook that launches PDB at the crash site. Use this to interactively inspect the failure state.

4. **Step-through Debugging** (`start_stepping_debug`)
   Start PDB from the beginning of the script. Use this to trace execution flow and identify where things go wrong.

## Investigation Tools

- `read_file`: Read any file (error traces, call details, data files)
- `execute_pdb_command`: Send commands to an active PDB session
- `inject_code_block`: Execute arbitrary Python code in the current debugging context (write from top-level, no indentation needed)
- `set_breakpoint`: Set a breakpoint by matching code context (avoids brittle line numbers)
- `close_debug_session`: Terminate the current PDB session

## PDB Command Reference

Common commands for interactive debugging:
- `n` (next): Execute current line, stop at next
- `s` (step): Step into function calls
- `c` (continue): Run until next breakpoint or exception
- `l` (list): Show code around current position
- `ll` (longlist): Show complete source of current function
- `w` (where): Print stack trace
- `p <expr>`: Evaluate and print expression
- `pp <expr>`: Pretty-print expression
- `!<stmt>`: Execute Python statement in current context

## Workflow Guidelines

1. **Start with traced execution** to capture error context
2. **Read the error trace** to understand the failure point
3. **Form a hypothesis** about the root cause
4. **Verify through debugging** if needed (post-mortem or step-through)
5. **Generate a repair patch** once the cause is confirmed

## Output Requirements

Your diagnosis and repair solution should be COMPLETE and SELF-CONTAINED.

**When calling `finalize_with_patch`:**

Provide two components:

1. **Root Cause Analysis** (`root_cause`)
   A comprehensive explanation of why the error occurred. Write this as a standalone description that fully captures the problem—readers should understand the issue without needing context from previous rounds.

2. **Repair Patches** (`patches`)
   All code changes needed to fix the issue. Each patch contains:
   - `target_code`: The problematic snippet to replace
   - `replacement_code`: The corrected version
   - `indent_level`: Position depth in the original file (0=module level, 1=inside function, 2=nested block, etc.)

   Write both `target_code` and `replacement_code` from the top level, without leading indentation. The system applies `indent_level` uniformly to match the original code structure.

**Replacement semantics:**

Your output entirely supersedes any previous diagnosis. If prior patches remain valid, include them unchanged. If corrections are needed, provide the updated versions. Omit patches that are no longer necessary.

**When calling `finalize_success`:**

Use this when the code executes without errors and produces correct results. No additional parameters required.

## Critical Rules

- Always base patches on the ORIGINAL code (Round 1), not intermediate versions
- Write `replacement_code` from top-level; use `indent_level` to match target position
- Provide sufficient context in `target_code` to ensure unique matching
- The patch system uses string replacement - exact matching is required
- Be thorough but efficient - avoid redundant debugging steps
"""

    mode_specific = {
        "crash": """
## Current Focus: Runtime Error Diagnosis

You are investigating a script that crashes during execution. Your goal is to:
1. Identify the exact cause of the exception
2. Understand why the problematic code path was triggered
3. Generate a patch that fixes the underlying issue, not just the symptom

If traced execution completes with exit code 0, the crash has been resolved—call `finalize_success` without further verification.
""",
        "inconsistency": """
## Current Focus: Data Inconsistency Detection

You are investigating a script that runs without errors but produces incorrect results. Your goal is to:
1. Identify where the data diverges from expected values
2. Trace the transformation that introduced the inconsistency
3. Generate a patch that corrects the data processing logic
""",
        "performance": """
## Current Focus: Performance Optimization

You are investigating a script with performance issues. Your goal is to:
1. Identify computational bottlenecks
2. Analyze algorithmic complexity and data structure choices
3. Generate patches that improve efficiency without changing correctness
"""
    }
    
    return base_prompt + mode_specific.get(debug_mode, mode_specific["crash"])


def build_initial_user_message(
    current_code: str,
    current_diagnosis: Optional[Dict] = None,  # 改为任务级诊断
    error_summary: Optional[str] = None
) -> str:
    """
    构建初始用户消息
    
    Args:
        current_code: 带行号的当前版本代码
        current_diagnosis: 任务级诊断信息（包含root_cause和patches）
        error_summary: 错误概要（如有先前执行记录）
    
    Returns:
        用户消息文本
    """
    sections = []
    
    sections.append("## Current Code (with line numbers)\n")
    sections.append("```python")
    sections.append(current_code)
    sections.append("```\n")
    
    if current_diagnosis:
        sections.append("## Previous Diagnosis (from last round)\n")
        sections.append("The following is the complete diagnosis from the previous attempt. You may refine, correct, or keep it as-is.\n")
        
        if current_diagnosis.get('root_cause'):
            sections.append(f"**Root Cause Analysis**:\n{current_diagnosis['root_cause']}\n")
        
        patches = current_diagnosis.get('patches', [])
        if patches:
            sections.append(f"**Patches** ({len(patches)} total):\n")
            for i, patch in enumerate(patches, 1):
                sections.append(f"### Patch {i}")
                sections.append("**Target (from original):**")
                sections.append(f"```python\n{patch.get('target_code', '')}\n```")
                sections.append("**Replacement:**")
                sections.append(f"```python\n{patch.get('replacement_code', '')}\n```\n")
    
    if error_summary:
        sections.append("## Latest Execution Result\n")
        sections.append(error_summary)
        sections.append("")
    
    sections.append("## Your Task\n")
    sections.append("Investigate the code and either confirm successful execution or provide a complete diagnosis with all necessary patches.")
    sections.append("\nThe code and patch history presented above are the sole reference for constructing repairs—no external source files exist.")
    
    return "\n".join(sections)


def format_code_with_line_numbers(code: str, start_line: int = 1) -> str:
    """
    为代码添加行号
    
    Args:
        code: 原始代码文本
        start_line: 起始行号
    
    Returns:
        带行号的代码文本
    """
    lines = code.split('\n')
    width = len(str(start_line + len(lines) - 1))
    
    numbered_lines = []
    for i, line in enumerate(lines, start_line):
        numbered_lines.append(f"{i:>{width}} | {line}")
    
    return '\n'.join(numbered_lines)