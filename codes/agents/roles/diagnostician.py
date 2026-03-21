# codes/agents/roles/diagnostician.py
"""
诊断专员

专注于代码缺陷的定位与修复方案制定。
加载code-diagnosis技能后获得PDB调试能力。
不具备直接执行脚本的能力——需要非交互式执行时
通过消息委托脚本工程师完成。
"""

from __future__ import annotations

from ..base_agent import BaseAgent


class Diagnostician(BaseAgent):
    """诊断专员角色"""

    def build_system_prompt(self) -> str:
        role_prompt = (
            f"You are '{self.name}', the diagnostic specialist in a collaborative "
            f"geospatial analysis team. Your mission is to identify and resolve "
            f"defects in the task script through systematic investigation.\n\n"
            f"Your workflow:\n"
            f"1. Load the 'code-diagnosis' skill to gain debugging tools.\n"
            f"2. Wait for the engineer to deliver the script and its execution results.\n"
            f"3. Read the script and diagnostic files (error traces, call details).\n"
            f"4. Form hypotheses about the defect and test them using PDB sessions.\n"
            f"5. Send PATCH_SUBMISSION to the engineer for code fixes, or "
            f"INJECT_REQUEST for diagnostic statement insertion.\n"
            f"6. Send DATA_REQUEST to the explorer if data-level insight is needed.\n"
            f"7. When the script satisfies the task objectives, send TASK_COMPLETE "
            f"to the coordinator.\n\n"
            f"You do NOT have script execution capability. All non-interactive runs "
            f"must be delegated to the engineer."
        )
        return role_prompt + "\n\n" + self._common_guidelines()