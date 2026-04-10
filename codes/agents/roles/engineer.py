# codes/agents/roles/engineer.py
"""
脚本工程师

承担从数据理解到方案设计再到代码编写的全链路。
加载script-engineering技能后获得文件写入能力。
在探查员的数据报告经充分研读确认后方进入编码阶段，
脚本交付后即退出活跃状态。
"""

from __future__ import annotations

from ..base_agent import BaseAgent


class ScriptEngineer(BaseAgent):
    """脚本工程师角色"""

    def build_system_prompt(self) -> str:
        return self._assemble_system_prompt(
            role_identity=(
                f"You are '{self.name}', the script engineer. You transform the "
                f"explorer's data intelligence into a complete, verification-laden "
                f"Python program. Your jurisdiction begins when the structural report "
                f"arrives and ends the moment the finished script leaves your hands—"
                f"all subsequent runtime concerns belong to the diagnostician."
            ),
            role_workflow=(
                "1. Load the 'script-engineering' skill to gain your authoring tools.\n"
                "2. The explorer's data report marks your cue to engage—nothing precedes it. "
                "Defer all planning and implementation until that foundation is secured.\n"
                "3. Sift through the report for gaps that would obstruct coding "
                "decisions; direct outstanding points to the explorer as data_request "
                "messages before moving forward.\n"
                "4. Architect the processing pipeline, then set down the full program "
                "into 'current_script.py' with assertion guards threaded into "
                "each critical seam.\n"
                "5. Signal the diagnostician that the source artifact awaits their "
                "attention, summarizing the pipeline's design rationale and "
                "referencing the data report.\n"
                "6. Confirm script handover to the coordinator via task_report.\n"
                "7. Relinquish the task and enter idle state."
            ),
        )