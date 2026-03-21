# codes/agents/roles/engineer.py
"""
脚本工程师

承担从方案设计到代码编写再到脚本执行的全链路。
加载script-engineering技能后获得文件操作和脚本执行能力。
编码时主动植入断言语句将业务预期转化为可自动校验的检查点。
"""

from __future__ import annotations

from ..base_agent import BaseAgent


class ScriptEngineer(BaseAgent):
    """脚本工程师角色"""

    def build_system_prompt(self) -> str:
        role_prompt = (
            f"You are '{self.name}', the script engineer in a collaborative "
            f"geospatial analysis team. You own the entire lifecycle of the task "
            f"script: architecture design, implementation, and execution.\n\n"
            f"Your workflow:\n"
            f"1. Load the 'script-engineering' skill to gain coding and execution tools.\n"
            f"2. Upon receiving the task assignment, begin technical planning—identify "
            f"the processing stages, spatial operations, and expected data flow.\n"
            f"3. When the data explorer's report arrives, integrate concrete field names, "
            f"CRS details, and file paths into your design.\n"
            f"4. Write the complete script to 'current_script.py' with assertion "
            f"instrumentation at key pipeline junctions.\n"
            f"5. Execute with tracing enabled and deliver results to the diagnostician.\n"
            f"6. After your first script commit, send a TASK_REPORT to the coordinator "
            f"to trigger baseline backup.\n"
            f"7. Service PATCH_SUBMISSION and INJECT_REQUEST messages from the "
            f"diagnostician by applying edits or producing augmented variants, "
            f"then re-executing and reporting back."
        )
        return role_prompt + "\n\n" + self._common_guidelines()