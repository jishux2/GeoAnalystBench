# codes/benchmark/eval_prompt.py
"""
评估prompt的构建

组装系统提示词与两条用户消息，前者定义评判框架、
评分维度和输出schema，后者分别承载参考侧与生成侧
的完整材料（代码、执行状态、产物描述）。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


# 评估结果的JSON schema
EVAL_OUTPUT_SCHEMA = {
    "task_alignment": {
        "score": "1-10",
        "rationale": "string"
    },
    "execution_validity": {
        "score": "1-10",
        "rationale": "string"
    },
    "output_completeness": {
        "score": "1-10",
        "rationale": "string"
    },
    "result_fidelity": {
        "score": "1-10",
        "rationale": "string"
    },
    "visualization_quality": {
        "score": "1-10 or null if no visual output",
        "rationale": "string"
    },
    "code_craftsmanship": {
        "score": "1-10",
        "rationale": "string"
    },
    "overall_score": "1-10",
    "summary": "string",
    "improvements": ["string"],
    "comparative_analysis": {
        "advantages_over_reference": ["string"],
        "disadvantages_versus_reference": ["string"],
        "key_differences": ["string"]
    }
}


def build_system_prompt() -> str:
    """构建评估模型的系统提示词。"""
    import json
    schema_str = json.dumps(EVAL_OUTPUT_SCHEMA, indent=2)

    return f"""You are an expert evaluator for geospatial analysis code. Your task is to assess the quality of a generated Python script against a task specification and, when available, a reference implementation.

## Evaluation Dimensions

Your assessment proceeds along six axes, each probing a distinct facet of quality. These axes form a natural progression: earlier dimensions establish prerequisites that later ones build upon.

**Task Alignment** examines whether the script's analytical logic faithfully addresses the stated objective. Read the code as a proposed solution and judge how closely its processing pipeline mirrors the intent behind the task description—not merely whether it runs, but whether what it computes is what was asked for.

**Execution Validity** shifts attention to the runtime plane. A script that crashes, hangs, or exits with errors cannot deliver value regardless of its conceptual soundness. Evaluate exit status, error messages, and any diagnostic traces to determine whether the program completes its intended control flow without incident.

**Output Completeness** audits the tangible deliverables the script leaves behind. Cross-reference the task requirements against the actual files produced: are all expected artifacts present? Do their formats match what the task prescribed? An incomplete output set—even from an otherwise correct script—represents unfinished work.

**Result Fidelity** looks past the existence of outputs to interrogate their substance. Numeric values, spatial relationships, statistical summaries, and classification outcomes should withstand scrutiny against the task's analytical expectations. Where a reference solution's outputs are available, they anchor this comparison; in their absence, evaluate plausibility and internal consistency.

**Visualization Quality** applies only when the task calls for graphical output. Beyond basic legibility, consider whether the visual encoding effectively communicates the underlying analytical findings—appropriate color scales, meaningful legends, properly labeled axes, and geographic context where warranted. This dimension remains dormant for tasks that produce no visual artifacts.

**Code Craftsmanship** appraises the implementation as a piece of software engineering, independent of its analytical mission. Robustness in the face of edge cases, clarity of variable naming and control flow, economy of computation, and judicious use of library APIs all factor into this assessment.

## Scoring Protocol

Assign each dimension an integer score from 1 to 10. A score of 1 denotes fundamental failure along that axis; 10 represents exemplary performance. The overall score should reflect a holistic judgment rather than a mechanical average—a script that nails every dimension except execution validity, for instance, warrants a low overall score because an unrunnable program delivers no practical value.

For visualization quality, assign null when no graphical output is involved. Where both implementations produce comparable artifacts, let direct side-by-side observation inform your judgment ahead of indirect inference from code alone.

## Reference Material Availability

The reference solution and its outputs serve as calibration anchors, not infallible ground truth. When reference materials are absent or incomplete, evaluate the generated code on its own merits against the task specification. Explicitly note in your rationale when the absence of reference material limits the confidence of a particular judgment.

Implementation conventions such as path resolution strategies, output directory naming, or environment variable usage reflect the execution framework each side operates within—divergences in these mechanical choices do not constitute quality differences and should not influence scoring.

## Output Format

Respond with a single JSON object conforming to the following structure. Do not include any text outside the JSON.

```json
{schema_str}
```"""


def build_reference_message(
    task_text: str,
    ref_code: Optional[str],
    ref_execution: Optional[Dict[str, Any]],
    ref_artifacts: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    构建承载参考侧材料的用户消息。

    当参考代码或执行信息缺失时，以说明性文字替代，
    确保语义完整而非静默留空。
    """
    parts = []

    # 任务描述
    parts.append({
        "type": "text",
        "text": f"## Task Specification\n\n{task_text}",
    })

    # 参考代码
    if ref_code:
        parts.append({
            "type": "text",
            "text": f"## Reference Implementation\n\n```python\n{ref_code}\n```",
        })
    else:
        parts.append({
            "type": "text",
            "text": (
                "## Reference Implementation\n\n"
                "No reference implementation is available for this task. "
                "Evaluate the generated code solely against the task specification."
            ),
        })

    # 参考执行信息
    if ref_execution:
        exec_text = _format_execution_info(ref_execution, "Reference")
        parts.append({
            "type": "text",
            "text": f"## Reference Execution\n\n{exec_text}",
        })
    elif ref_code:
        parts.append({
            "type": "text",
            "text": (
                "## Reference Execution\n\n"
                "The reference script was not executed or its execution "
                "records are unavailable."
            ),
        })

    # 参考产物
    if ref_artifacts:
        artifact_parts = _format_artifacts(ref_artifacts, "Reference")
        parts.extend(artifact_parts)
    elif ref_code:
        parts.append({
            "type": "text",
            "text": (
                "## Reference Outputs\n\n"
                "No output artifacts were produced by the reference implementation."
            ),
        })

    return {"role": "user", "content": parts}


def build_generated_message(
    gen_code: Optional[str],
    gen_execution: Optional[Dict[str, Any]],
    gen_artifacts: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    构建承载生成侧材料的用户消息。
    """
    parts = []

    # 生成代码
    if gen_code:
        parts.append({
            "type": "text",
            "text": f"## Generated Implementation\n\n```python\n{gen_code}\n```",
        })
    else:
        parts.append({
            "type": "text",
            "text": (
                "## Generated Implementation\n\n"
                "No generated code is available for evaluation."
            ),
        })

    # 生成执行信息
    if gen_execution:
        exec_text = _format_execution_info(gen_execution, "Generated")
        parts.append({
            "type": "text",
            "text": f"## Generated Execution\n\n{exec_text}",
        })
    elif gen_code:
        parts.append({
            "type": "text",
            "text": (
                "## Generated Execution\n\n"
                "The generated script was not executed or its execution "
                "records are unavailable."
            ),
        })

    # 生成产物
    if gen_artifacts:
        artifact_parts = _format_artifacts(gen_artifacts, "Generated")
        parts.extend(artifact_parts)
    elif gen_code:
        parts.append({
            "type": "text",
            "text": (
                "## Generated Outputs\n\n"
                "No output artifacts were produced by the generated implementation."
            ),
        })

    return {"role": "user", "content": parts}


def _format_execution_info(execution: Dict[str, Any], label: str) -> str:
    """格式化执行状态信息。"""
    lines = []
    returncode = execution.get("returncode")
    if returncode is not None:
        status = "Success" if returncode == 0 else f"Failed (exit code {returncode})"
        lines.append(f"**Status:** {status}")

    stdout = execution.get("stdout", "").strip()
    if stdout:
        if len(stdout) > 3000:
            stdout = stdout[:3000] + "\n... [truncated]"
        lines.append(f"**Standard Output:**\n```\n{stdout}\n```")
    else:
        lines.append("**Standard Output:** (empty)")

    stderr = execution.get("stderr", "").strip()
    if stderr:
        if len(stderr) > 2000:
            stderr = stderr[:2000] + "\n... [truncated]"
        lines.append(f"**Standard Error:**\n```\n{stderr}\n```")

    return "\n\n".join(lines)


def _format_artifacts(
    artifacts: List[Dict[str, Any]], label: str
) -> List[Dict[str, Any]]:
    """
    将产物列表格式化为content parts。

    图片类型生成image_url块，其余生成文本块。
    """
    parts = []

    text_descriptions = []
    image_items = []

    for art in artifacts:
        art_type = art.get("type", "unknown")
        name = art.get("name", "unnamed")
        content = art.get("content")
        image_url = art.get("image_url")

        if art_type == "image" and image_url:
            image_items.append((name, image_url))
        elif content:
            text_descriptions.append(f"### {name} ({art_type})\n\n{content}")
        else:
            text_descriptions.append(
                f"### {name} ({art_type})\n\n[No content summary available]"
            )

    if text_descriptions:
        parts.append({
            "type": "text",
            "text": f"## {label} Output Artifacts\n\n" + "\n\n".join(text_descriptions),
        })

    for name, url in image_items:
        parts.append({
            "type": "text",
            "text": f"## {label} Visual Output: {name}",
        })
        parts.append({
            "type": "image_url",
            "image_url": {"url": url},
        })

    return parts