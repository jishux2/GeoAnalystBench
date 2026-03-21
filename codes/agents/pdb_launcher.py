# codes/agents/pdb_launcher.py
"""
PDB会话启动器

将现有的PdbSessionController适配为技能工具框架所需的
handler接口。封装会话生命周期管理，确保同一时刻仅有
一个活跃会话，并在关闭时触发上下文压缩。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from debug_agent.pdb_controller import PdbSessionController


class PdbLauncher:
    """
    PDB调试会话的handler封装

    每个诊断专员实例持有一个PdbLauncher，
    管理其专属的调试器会话。
    """

    def __init__(
        self,
        interpreter: str,
        working_dir: Path,
    ):
        self.interpreter = interpreter
        self.working_dir = working_dir
        self._session: Optional[PdbSessionController] = None
        self._session_log: list = []

    async def handle_start_postmortem(
        self, script_path: str
    ) -> Dict[str, Any]:
        """启动事后调试会话。"""
        if self._session is not None:
            return {
                "success": False,
                "result": "A debug session is already active. Close it first.",
            }

        resolved = self._resolve_script(script_path)
        if resolved is None:
            return {"success": False, "result": f"Script not found: {script_path}"}

        try:
            injected = self._inject_postmortem_hook(
                resolved.read_text(encoding="utf-8")
            )
            temp_path = self._write_temp_script(injected)

            self._session = await PdbSessionController.start_script(
                str(temp_path),
                str(self.working_dir),
                self.interpreter,
            )
            self._session_log = []

            return {
                "success": True,
                "result": (
                    f"Post-mortem debugging started for {script_path}.\n\n"
                    f"Initial output:\n{self._session.initial_output}"
                ),
            }
        except Exception as e:
            return {"success": False, "result": f"Failed to start session: {e}"}

    async def handle_start_stepping(
        self, script_path: str
    ) -> Dict[str, Any]:
        """启动单步调试会话。"""
        if self._session is not None:
            return {
                "success": False,
                "result": "A debug session is already active. Close it first.",
            }

        resolved = self._resolve_script(script_path)
        if resolved is None:
            return {"success": False, "result": f"Script not found: {script_path}"}

        try:
            self._session = await PdbSessionController.start_with_pdb(
                str(resolved),
                str(self.working_dir),
                self.interpreter,
            )
            self._session_log = []

            return {
                "success": True,
                "result": (
                    f"Step-through debugging started for {script_path}.\n\n"
                    f"Initial output:\n{self._session.initial_output}"
                ),
            }
        except Exception as e:
            return {"success": False, "result": f"Failed to start session: {e}"}

    async def handle_pdb_command(self, command: str) -> Dict[str, Any]:
        """执行PDB命令。"""
        if self._session is None:
            return {"success": False, "result": "No active debug session."}

        response = await self._session.send_command(command)
        self._session_log.append({"command": command, "response": response})

        return {"success": True, "result": response}

    async def handle_inject_code(self, code: str) -> Dict[str, Any]:
        """在调试会话中注入代码块。"""
        if self._session is None:
            return {"success": False, "result": "No active debug session."}

        response = await self._session.execute_code(code)
        self._session_log.append({"injected": code, "response": response})

        return {"success": True, "result": response}

    async def handle_close_session(self, summary: str = "") -> Dict[str, Any]:
        """
        关闭调试会话。

        Args:
            summary: 本次会话的关键发现概述，由调用方在关闭时填写。
        """
        if self._session is None:
            return {"success": False, "result": "No active debug session."}

        await self._session.close()
        self._session = None

        log_copy = list(self._session_log)
        self._session_log = []

        return {
            "success": True,
            "result": "Debug session closed.",
            "session_log": log_copy,
            "summary": summary,
        }

    async def cleanup(self):
        """强制清理残存的会话资源。"""
        if self._session is not None:
            await self._session.close()
            self._session = None

    def _resolve_script(self, script_path: str) -> Optional[Path]:
        path = Path(script_path)
        if not path.is_absolute():
            path = self.working_dir / path
        return path if path.exists() else None

    def _write_temp_script(self, content: str) -> Path:
        import tempfile
        temp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8"
        )
        temp.write(content)
        temp.close()
        return Path(temp.name)

    def _inject_postmortem_hook(self, code: str) -> str:
        hook = (
            "import sys\n"
            "import pdb\n"
            "import traceback\n\n"
            "def _postmortem_hook(exc_type, exc_value, exc_tb):\n"
            "    traceback.print_exception(exc_type, exc_value, exc_tb)\n"
            "    pdb.post_mortem(exc_tb)\n\n"
            "sys.excepthook = _postmortem_hook\n\n"
        )
        return hook + code