# codes/agents/coordinator.py
"""
主控节点

启动并监管三个子智能体的协作流程。自身不直接参与
地理空间分析工作，职责集中于：

- 根据任务元数据初始化团队成员并分发启动指令
- 周期性苏醒检查各成员进展，必要时发送纠偏消息
- 在累积时长触及阈值时强制终止并收集最终产出
- 任务成功完成时有序关闭全部子智能体

主控节点自身也运行在一个协程中，与子智能体共享
同一个asyncio事件循环。
"""

from __future__ import annotations

import asyncio
import difflib
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from concurrent.futures import ProcessPoolExecutor

from .message import Message, MessageType
from .channel import AgentChannel, ChannelRegistry
from .journal import ContextJournal


class Coordinator:
    """
    任务级主控节点

    每个待处理的任务对应一个Coordinator实例，
    由外层编排器创建并启动。
    """

    def __init__(
        self,
        task_id: int,
        api_key: str,
        task_info: Dict[str, Any],
        working_dir: Path,
        output_dir: Path,
        interpreter: str,
        skill_root: Path,
        check_interval: float = 60.0,
        timeout: float = 900.0,
        process_executor: Optional[ProcessPoolExecutor] = None,
        layout: Optional[Any] = None,  # BenchmarkLayout 实例
        dataset_check: Optional[Dict[str, Any]] = None,
    ):
        self.task_id = task_id
        self.api_key = api_key
        self.task_info = task_info
        self.working_dir = working_dir
        self.output_dir = output_dir
        self.interpreter = interpreter
        self.skill_root = skill_root
        self.check_interval = check_interval
        self.timeout = timeout
        self._process_executor = process_executor
        self.layout = layout

        self.script_path = working_dir / "current_script.py"

        self.channel_registry = ChannelRegistry()
        self._coordinator_channel = self.channel_registry.register("coordinator")

        self._agent_tasks: Dict[str, asyncio.Task] = {}

        self._start_time: Optional[float] = None
        self._outcome: Optional[Dict[str, Any]] = None
        self._initial_code_backup: Optional[str] = None

        self._journal = ContextJournal(
            output_dir / "coordinator_journal.json",
            "coordinator",
        )

        self._dataset_check = dataset_check or {"all_present": True, "missing": [], "summary": ""}

    async def run(self) -> Dict[str, Any]:
        """
        主控节点主入口。

        启动子智能体团队后进入监控循环，
        返回任务的最终结果。
        """
        self._start_time = time.time()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 启动子智能体
        await self._launch_team()

        # 监控循环
        try:
            result = await self._supervision_loop()
        except Exception as e:
            result = {
                "success": False,
                "error": str(e),
                "patches": [],
            }

        # 确保所有子智能体已终止
        await self._shutdown_team()

        # 持久化最终状态
        self._outcome = result
        self._journal.finalize(
            result,
            [{"note": "Coordinator does not maintain an LLM context."}],
        )

        return result

    async def _launch_team(self):
        """
        创建并启动三个子智能体。
        """
        from .skill_registry import SkillRegistry
        from .tools_common import FileOperations, ScriptExecutor
        from .pdb_launcher import PdbLauncher
        from .skill_tools import register_all_skill_factories

        skill_registry = SkillRegistry(self.skill_root)

        file_ops = FileOperations(
            working_dir=self.working_dir,
            allowed_roots=[self.working_dir, self.skill_root],
        )

        explorer_executor = ScriptExecutor(
            interpreter=self.interpreter,
            working_dir=self.working_dir,
            output_dir=self.output_dir / "explorer",
            executor=self._process_executor,
        )

        diagnostician_executor = ScriptExecutor(
            interpreter=self.interpreter,
            working_dir=self.working_dir,
            output_dir=self.output_dir / "diagnostician",
            executor=self._process_executor,
        )

        pdb_launcher = PdbLauncher(
            interpreter=self.interpreter,
            working_dir=self.working_dir,
        )
        self._pdb_launcher = pdb_launcher

        register_all_skill_factories(
            skill_registry, file_ops,
            explorer_executor,
            diagnostician_executor, pdb_launcher,
        )

        agents = self._build_agent_configs(skill_registry)

        # 投递初始指令——探查员立即获得完整任务描述，
        # 工程师和诊断专员仅收到待命通知
        await self._dispatch_initial_instructions()

        # 启动协程
        for agent_name, agent_instance in agents.items():
            task = asyncio.create_task(
                agent_instance.run(),
                name=f"agent-{agent_name}",
            )
            self._agent_tasks[agent_name] = task

    def _build_agent_configs(self, skill_registry) -> Dict:
        from .roles import DataExplorer, ScriptEngineer, Diagnostician

        source = self.task_info.get("source", "")

        common_kwargs = {
            "api_key": self.api_key,
            "channel_registry": self.channel_registry,
            "skill_registry": skill_registry,
            "working_dir": self.working_dir,
            "temperature": 0.7,
            "layout": self.layout,
        }

        agents = {
            "explorer": DataExplorer(
                name="explorer",
                output_dir=self.output_dir / "explorer",
                **common_kwargs,
            ),
            "engineer": ScriptEngineer(
                name="engineer",
                output_dir=self.output_dir / "engineer",
                **common_kwargs,
            ),
            "diagnostician": Diagnostician(
                name="diagnostician",
                output_dir=self.output_dir / "diagnostician",
                **common_kwargs,
            ),
        }

        for agent in agents.values():
            agent._task_source = source

        return agents

    async def _dispatch_initial_instructions(self):
        """
        向三个子智能体投递启动阶段的情境通报。

        探查员作为流程的首发环节，立即获得完整的任务描述
        并着手数据摸底。工程师和诊断专员各自收到一条待命
        通知，明确告知它们的介入时机取决于上游交付物的到位。

        内容严格限于此刻的团队动态和各成员的首要动作，
        不重复角色提示词中的职责描述或技能文档中的方法论。
        """
        task_description = self.task_info.get("full_text", "")

        # 路径使用指导，附加在每条分发指令末尾
        path_guidance = ""
        source = self.task_info.get("source", "")
        if self.layout and source:
            dataset_abs = self.layout.dataset_dir(source).resolve()
            path_guidance = (
                f"\n\nWhen accessing dataset files, use the absolute path "
                f"provided in your context ({dataset_abs}) rather than "
                f"attempting to reconstruct relative paths from directory "
                f"nesting—relative traversals from the task working directory "
                f"are error-prone and strongly discouraged."
            )

        dataset_warning = ""
        if not self._dataset_check["all_present"]:
            dataset_warning = (
                f"\n\n⚠ Data availability notice: {self._dataset_check['summary']}"
            )

        await self.channel_registry.deliver(Message(
            msg_type=MessageType.TASK_ASSIGNMENT,
            sender="coordinator",
            recipient="explorer",
            content=(
                "A geospatial analysis task has been assigned to the team. As the "
                "first member to engage, survey the dataset resources and compile "
                "an inventory of their schemas, spatial attributes, and quality "
                "characteristics. The engineer is standing by for your contributions "
                "and will not begin implementation until your report is in hand. "
                "Direct your attention toward the dimensions most consequential "
                "for the operations this task prescribes.\n\n"
                f"Task specification:\n{task_description}"
                f"{path_guidance}"
                f"{dataset_warning}"
            ),
        ))

        await self.channel_registry.deliver(Message(
            msg_type=MessageType.TASK_ASSIGNMENT,
            sender="coordinator",
            recipient="engineer",
            content=(
                "The team has commenced work on a geospatial analysis task. The "
                "explorer is currently conducting a meticulous canvass of the "
                "dataset and will relay the findings once the examination concludes. "
                "Remain idle until that material reaches you—your design and coding "
                "effort should proceed only after you have internalized the data "
                "landscape and clarified any residual uncertainties with the explorer.\n\n"
                f"Task specification:\n{task_description}"
                f"{path_guidance}"
            ),
        ))

        await self.channel_registry.deliver(Message(
            msg_type=MessageType.TASK_ASSIGNMENT,
            sender="coordinator",
            recipient="diagnostician",
            content=(
                "The collaborative pipeline for a geospatial analysis task has been "
                "set in motion. The explorer is assembling a data characterization "
                "report while the engineer awaits that input before drafting the "
                "script. You will be called upon once the engineer signals that the "
                "source artifact has been deposited at the agreed-upon location. "
                "Suspend activity until that notification arrives.\n\n"
                f"Task specification:\n{task_description}"
                f"{path_guidance}"
            ),
        ))

    async def _supervision_loop(self) -> Dict[str, Any]:
        while True:
            elapsed = time.time() - self._start_time

            if elapsed >= self.timeout:
                return await self._force_termination()

            msg = await self._coordinator_channel.receive(timeout=30.0)

            if msg is not None:
                result = self._process_coordinator_message(msg)
                if result is not None:
                    return result

            crash_result = self._check_agent_health()
            if crash_result is not None:
                return crash_result

    def _check_agent_health(self) -> Optional[Dict[str, Any]]:
        """
        检查子任务是否有异常终止的情况。

        遍历所有已注册的子任务，若发现某个Task已完成且
        携带异常，提取异常信息并返回失败结果。
        """
        for agent_name, task in self._agent_tasks.items():
            if task.done() and not task.cancelled():
                exc = task.exception()
                if exc is not None:
                    return {
                        "success": False,
                        "error": f"Agent '{agent_name}' crashed: {type(exc).__name__}: {exc}",
                        "patches": self._generate_diff(),
                        "terminated_by": "agent_crash",
                    }
        return None

    def _process_coordinator_message(self, msg: Message) -> Optional[Dict[str, Any]]:
        """处理抵达主控节点的消息。"""
        # 子智能体崩溃通知
        if (
            msg.msg_type == MessageType.TASK_REPORT
            and msg.payload.get("agent_crash")
        ):
            return {
                "success": False,
                "error": f"Agent '{msg.sender}' reported fatal error: {msg.content}",
                "patches": self._generate_diff(),
                "terminated_by": "agent_crash",
            }

        # 探查员上报数据不可行
        if (
            msg.msg_type == MessageType.TASK_REPORT
            and msg.payload.get("task_infeasible")
        ):
            return {
                "success": False,
                "error": msg.content,
                "patches": [],
                "terminated_by": "data_infeasible",
                "infeasibility_reason": msg.payload.get("reason", msg.content),
            }

        if (
            msg.msg_type == MessageType.TASK_REPORT
            and msg.sender == "engineer"
            and self._initial_code_backup is None
            and self.script_path.exists()
        ):
            self._initial_code_backup = self.script_path.read_text(encoding="utf-8")
            backup_path = self.output_dir / "initial_code.py"
            backup_path.write_text(self._initial_code_backup, encoding="utf-8")

        if msg.msg_type == MessageType.TASK_COMPLETE:
            result = {
                "success": True,
                "patches": self._generate_diff(),
                "assessment": msg.content,
                "root_cause": msg.payload.get("root_cause", ""),
            }
            # 超时场景下诊断专员会附带置信度
            if "confidence" in msg.payload:
                result["success"] = False
                result["confidence"] = msg.payload["confidence"]
                result["terminated_by"] = "timeout"
            return result

        if msg.msg_type == MessageType.STATUS_REPLY:
            self._log_status_update(msg)

        return None

    def _log_status_update(self, msg: Message):
        """将诊断专员的进展回复持久化到监控日志。"""
        log_path = self.output_dir / "status_log.jsonl"
        import json
        entry = {
            "timestamp": msg.timestamp,
            "sender": msg.sender,
            "content": msg.content,
            "payload": msg.payload,
        }
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")

    async def _probe_team_status(self):
        """向诊断专员发送状态询问。"""
        try:
            await self.channel_registry.deliver(Message(
                msg_type=MessageType.STATUS_INQUIRY,
                sender="coordinator",
                recipient="diagnostician",
                content="Please provide a brief progress update on your current investigation.",
            ))
        except KeyError:
            pass  # 诊断专员可能尚未启动或已退出

    async def _force_termination(self) -> Dict[str, Any]:
        """超时后强制终止：要求诊断专员交付当前最佳结论。"""
        try:
            await self.channel_registry.deliver(Message(
                msg_type=MessageType.TERMINATE,
                sender="coordinator",
                recipient="diagnostician",
                content=(
                    "Time budget exhausted. If you have applied a patch that has not "
                    "yet been verified through re-execution, perform one final traced "
                    "run before submitting. Then deliver your current assessment via "
                    "task_complete, attaching a confidence score (0.0-1.0) in the "
                    "payload that reflects how rigorously your conclusions have been "
                    "corroborated."
                ),
            ))

            msg = await self._coordinator_channel.receive(timeout=120.0)
            if msg and msg.msg_type == MessageType.TASK_COMPLETE:
                return self._process_coordinator_message(msg)
        except (KeyError, Exception):
            pass

        return {
            "success": False,
            "patches": [],
            "confidence": 0.0,
            "assessment": "",
            "root_cause": "Forced termination due to timeout; no final assessment received.",
            "terminated_by": "timeout",
        }

    async def _shutdown_team(self):
        """向所有存活的子智能体发送终止指令并等待退出。"""
        for agent_name in list(self._agent_tasks.keys()):
            try:
                await self.channel_registry.deliver(Message(
                    msg_type=MessageType.TERMINATE,
                    sender="coordinator",
                    recipient=agent_name,
                    content="Task concluded. Shutting down.",
                ))
            except KeyError:
                pass

        if self._agent_tasks:
            tasks = list(self._agent_tasks.values())
            done, pending = await asyncio.wait(tasks, timeout=15.0)
            for t in pending:
                t.cancel()

        # 清理可能残存的PDB会话
        # pdb_launcher在_launch_team中创建，需要保存为实例属性
        if hasattr(self, '_pdb_launcher') and self._pdb_launcher:
            await self._pdb_launcher.cleanup()

    def _generate_diff(self) -> List[Dict[str, str]]:
        """
        比对初始代码与当前版本，生成差异记录。
        """
        if self._initial_code_backup is None:
            return []

        if not self.script_path.exists():
            return []

        current_code = self.script_path.read_text(encoding="utf-8")

        if current_code == self._initial_code_backup:
            return []

        diff = list(difflib.unified_diff(
            self._initial_code_backup.splitlines(keepends=True),
            current_code.splitlines(keepends=True),
            fromfile="initial",
            tofile="patched",
        ))

        return [{
            "diff": "".join(diff),
            "initial_code": self._initial_code_backup,
            "final_code": current_code,
        }]