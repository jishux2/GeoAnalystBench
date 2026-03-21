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
        task_config: Dict[str, Any],
        working_dir: Path,
        output_dir: Path,
        interpreter: str,
        skill_root: Path,
        check_interval: float = 60.0,
        timeout: float = 900.0,
        process_executor: Optional[ProcessPoolExecutor] = None,
    ):
        """
        Args:
            task_id: 任务编号
            api_key: DeepSeek API密钥
            task_config: 任务元数据（prompt、技术栈属性等）
            working_dir: 任务工作目录
            output_dir: 输出目录
            interpreter: Python解释器路径
            skill_root: 技能目录根路径
            check_interval: 主控节点检查间隔（秒）
            timeout: 全局超时阈值（秒）
        """
        self.task_id = task_id
        self.api_key = api_key
        self.task_config = task_config
        self.working_dir = working_dir
        self.output_dir = output_dir
        self.interpreter = interpreter
        self.skill_root = skill_root
        self.check_interval = check_interval
        self.timeout = timeout
        self._process_executor = process_executor

        # 脚本的约定落盘路径
        self.script_path = working_dir / "current_script.py"

        # 通信基础设施
        self.channel_registry = ChannelRegistry()
        self._coordinator_channel = self.channel_registry.register("coordinator")

        # 子智能体任务句柄
        self._agent_tasks: Dict[str, asyncio.Task] = {}

        # 状态追踪
        self._start_time: Optional[float] = None
        self._outcome: Optional[Dict[str, Any]] = None
        self._initial_code_backup: Optional[str] = None  # 延后到首次落盘时赋值

        # 日志
        self._journal = ContextJournal(
            output_dir / "coordinator_journal.json",
            "coordinator",
        )

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

        engineer_executor = ScriptExecutor(
            interpreter=self.interpreter,
            working_dir=self.working_dir,
            output_dir=self.output_dir / "engineer",
            executor=self._process_executor,
        )

        pdb_launcher = PdbLauncher(
            interpreter=self.interpreter,
            working_dir=self.working_dir,
        )
        self._pdb_launcher = pdb_launcher

        register_all_skill_factories(
            skill_registry, file_ops, explorer_executor, engineer_executor, pdb_launcher
        )

        agents = self._build_agent_configs(skill_registry)

        # 先注册通道（_build_agent_configs中BaseAgent.__init__已完成）
        # 再投递初始指令，确保消息在协程启动前就位
        await self._dispatch_initial_instructions()

        # 最后启动协程
        for agent_name, agent_instance in agents.items():
            task = asyncio.create_task(
                agent_instance.run(),
                name=f"agent-{agent_name}",
            )
            self._agent_tasks[agent_name] = task

    def _build_agent_configs(self, skill_registry) -> Dict:
        from .roles import DataExplorer, ScriptEngineer, Diagnostician

        common_kwargs = {
            "api_key": self.api_key,
            "channel_registry": self.channel_registry,
            "skill_registry": skill_registry,
            "working_dir": self.working_dir,
            "temperature": 0.7,
        }

        return {
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

    async def _dispatch_initial_instructions(self):
        """
        向三个子智能体投递启动时的情境通报与即刻行动指引。

        内容严格限于此刻的团队动态和各成员的首要动作，
        不重复角色提示词中的职责描述或技能文档中的方法论。
        """
        task_description = self.task_config.get("prompt", {}).get("full_text", "")

        await self.channel_registry.deliver(Message(
            msg_type=MessageType.TASK_ASSIGNMENT,
            sender="coordinator",
            recipient="explorer",
            content=(
                "The team is now active on the following geospatial analysis task. "
                "The engineer is simultaneously drafting an architectural plan and "
                "will depend on your structural findings to finalize implementation. "
                "Prioritize the dataset dimensions most relevant to the operations "
                "this task demands.\n\n"
                f"Task specification:\n{task_description}"
            ),
        ))

        await self.channel_registry.deliver(Message(
            msg_type=MessageType.TASK_ASSIGNMENT,
            sender="coordinator",
            recipient="engineer",
            content=(
                "The team is now active on the following geospatial analysis task. "
                "The explorer is concurrently profiling the dataset and will forward "
                "a structural report once complete. Begin with abstract architectural "
                "reasoning—processing stages, operation sequencing, data flow "
                "topology—while awaiting those concrete details.\n\n"
                f"Task specification:\n{task_description}"
            ),
        ))

        await self.channel_registry.deliver(Message(
            msg_type=MessageType.TASK_ASSIGNMENT,
            sender="coordinator",
            recipient="diagnostician",
            content=(
                "The team is now active on the following geospatial analysis task. "
                "The explorer is profiling the dataset and the engineer is developing "
                "the initial script. Your involvement begins when the engineer delivers "
                "the executable artifact and its runtime output. Hold idle until that "
                "handoff arrives.\n\n"
                f"Task specification:\n{task_description}"
            ),
        ))

    async def _supervision_loop(self) -> Dict[str, Any]:
        """
        周期性检查团队状态的监控循环。

        每个周期中：
        1. 消费主控节点收件箱中的消息
        2. 判定是否达成成功条件或触及超时
        3. 未达终态则继续等待
        """
        while True:
            elapsed = time.time() - self._start_time

            # 超时强制终止
            if elapsed >= self.timeout:
                return await self._force_termination()

            # 消费收件箱
            msg = await self._coordinator_channel.receive(
                timeout=self.check_interval
            )

            if msg is not None:
                result = self._process_coordinator_message(msg)
                if result is not None:
                    return result

            # 超时后仍无终态消息，发送状态询问
            # （仅当距上次检查已过一个完整周期时触发）
            if msg is None and elapsed > self.check_interval:
                await self._probe_team_status()

    def _process_coordinator_message(self, msg: Message) -> Optional[Dict[str, Any]]:
        """
        处理抵达主控节点的消息。
        """
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
                    "Time budget exhausted. Submit your current assessment of the "
                    "script's status via TASK_COMPLETE, including a confidence score "
                    "(0.0-1.0) in the payload reflecting how thoroughly your findings "
                    "have been validated."
                ),
            ))

            msg = await self._coordinator_channel.receive(timeout=30.0)
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