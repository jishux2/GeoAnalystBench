# codes/agents/base_agent.py
"""
子智能体协程骨架

定义所有子智能体共享的运行时框架：消息循环、工具调度、
收件箱检查、空闲挂起与唤醒、上下文日志持久化。
具体的角色行为（系统提示词、初始工具集、技能绑定）
由子类或配置注入，骨架本身不包含任何业务逻辑。
"""

from __future__ import annotations

import json
import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional

from .message import Message, MessageType
from .channel import AgentChannel, ChannelRegistry
from .context import AgentContext
from .tool_base import ToolSpec, ToolDispatcher
from .skill_registry import SkillRegistry
from .journal import ContextJournal


class BaseAgent:
    """
    子智能体的运行时骨架

    子类通过覆写configure()提供角色特定的系统提示词和初始工具，
    通过覆写on_activated()执行启动后的首要动作（如主动加载技能）。
    骨架负责驱动"调用API→执行工具→检查收件箱→持久化日志"的
    核心循环，并在收到TERMINATE消息时有序退出。
    """

    def __init__(
        self,
        name: str,
        api_key: str,
        channel_registry: ChannelRegistry,
        skill_registry: SkillRegistry,
        working_dir: Path,
        output_dir: Path,
        temperature: float = 0.7,
        context_threshold: int = 50000,
    ):
        self.name = name
        self.api_key = api_key
        self.channel_registry = channel_registry
        self.skill_registry = skill_registry
        self.working_dir = working_dir
        self.output_dir = output_dir
        self.temperature = temperature

        # 通信通道
        self._channel: AgentChannel = channel_registry.register(name)

        # 工具调度器（基础工具在_setup_base_tools中注册）
        self.dispatcher = ToolDispatcher()

        # 上下文与日志（在run()中初始化，因为需要子类先提供系统提示词）
        self.context: Optional[AgentContext] = None
        self.journal: Optional[ContextJournal] = None

        # 生命周期状态
        self._terminated = False
        self._turn_count = 0

        # 上下文压缩配置
        self.context_threshold = context_threshold

        # 计数器
        self._debug_session_count = 0
        self._reasoning_archive_count = 0

    async def run(self):
        """
        智能体主入口。

        初始化上下文、注册基础工具、调用子类的配置钩子，
        然后进入核心循环。循环退出后执行清理并持久化最终状态。
        """
        from deepseek.deepseek_client import DeepSeekClient

        system_prompt = self.build_system_prompt()
        self.context = AgentContext(system_prompt, self.name)
        self.journal = ContextJournal(
            self.output_dir / f"{self.name}_journal.json",
            self.name,
        )

        self._setup_base_tools()
        self.configure()

        # 子类可在此执行启动后的首要动作
        await self.on_activated()

        async with DeepSeekClient(self.api_key) as client:
            while not self._terminated:

                # 检查收件箱，将新消息注入上下文
                self._drain_inbox()

                # 第二层：阈值触发的自动压缩
                if self.context.estimate_token_count() > self.context_threshold:
                    await self._execute_context_compression(client)
                    self._persist_snapshot()

                # 调用API
                response_data = await client.chat_completion_with_tools(
                    messages=self.context.messages,
                    tools=self.dispatcher.api_schemas(),
                    temperature=self.temperature,
                    max_tokens=8192,
                    thinking={"type": "enabled"},
                )

                message = response_data["choices"][0]["message"]
                self.context.append_assistant(message)

                tool_calls = message.get("tool_calls")

                if not tool_calls:
                    # 模型给出纯文本回复——不等同于空闲，
                    # 继续下一轮循环让模型决定后续行动
                    self._persist_snapshot()
                    continue

                idle_requested = False
                compression_requested = False

                for tc in tool_calls:
                    tool_name = tc["function"]["name"]
                    arguments = json.loads(tc["function"]["arguments"])

                    result = await self.dispatcher.execute(tool_name, arguments)

                    self.context.append_tool_result(
                        tool_call_id=tc["id"],
                        content=result.get("result", ""),
                    )

                    # 技能加载：延迟注入正文到tool_result之后
                    if result.get("inject_skill_body"):
                        self.context.inject_skill(result["inject_skill_body"])

                    # 文件读取：追踪并检查唯一性（由handler层处理）
                    if tool_name == "read_file" and result.get("file_path"):
                        self.context.track_file_operation(
                            result["file_path"], tc["id"]
                        )

                    # 写入操作的压缩决策
                    if result.get("compress_write"):
                        self.context.compress_write_operation(
                            tc["id"], result["file_path"]
                        )
                    elif tool_name == "write_file" and result.get("file_path"):
                        # 追加写入：不即时压缩，但追踪以便关闭时统一清理
                        if arguments.get("append"):
                            self.context.track_file_operation(
                                result["file_path"], tc["id"]
                            )

                    # 调试会话关闭时：持久化交互日志并压缩上下文
                    if tool_name == "close_debug_session" and result.get("session_log"):
                        self._debug_session_count += 1
                        trace_path = self.output_dir / f"debug_session_{self._debug_session_count}.json"
                        trace_path.write_text(
                            json.dumps(result["session_log"], indent=2, ensure_ascii=False),
                            encoding="utf-8",
                        )
                        summary = result.get("summary", "")
                        if not summary:
                            summary = f"Debug session with {len(result['session_log'])} interactions."
                        self.context.compress_debug_session(summary, str(trace_path))

                    # 第三层：手动压缩标记
                    if result.get("trigger_compression"):
                        compression_requested = True

                    if tool_name == "go_idle":
                        idle_requested = True

                self._persist_snapshot()

                # 执行手动压缩（需要在工具结果都已拼接后进行）
                if compression_requested:
                    await self._execute_context_compression(client)
                    self._persist_snapshot()

                if self._terminated:
                    break

                if idle_requested:
                    await self._enter_idle()

        # 最终持久化
        outcome = {"terminated_by": "normal"}
        self.journal.finalize(outcome, self.context.snapshot())

    # ================================================================
    # 子类扩展点
    # ================================================================

    def build_system_prompt(self) -> str:
        """
        子类覆写以提供角色专属的系统提示词。

        默认实现仅返回通用行为规范，缺少角色定位。
        生产环境中所有角色子类都应覆写此方法。
        """
        return self._common_guidelines()

    def _common_guidelines(self) -> str:
        """所有角色共享的行为规范与资源索引。"""
        skills_overview = self.skill_registry.list_descriptions()
        return (
            "## Working Environment\n\n"
            f"Your task directory is rooted at the workspace path assigned to the current "
            f"job. All relative file references resolve against this root. The dataset/ "
            f"subdirectory houses source data files; outputs/ collects execution artifacts "
            f"and diagnostic records, organized by team member.\n\n"

            "## Team Structure\n\n"
            "You operate within a four-member unit assembled for each geospatial analysis "
            "task. The **explorer** profiles dataset resources and surfaces structural "
            "intelligence. The **engineer** architects, implements, and executes the task "
            "script. The **diagnostician** investigates runtime defects and orchestrates "
            "repairs. A supervisory **coordinator** oversees the collective timeline and "
            "intervenes when progress stalls or the time budget expires. Address teammates "
            "by these identifiers when routing messages.\n\n"

            f"## Available Skills\n\n"
            f"Load a skill to activate domain-specific guidance and tools:\n"
            f"{skills_overview}\n\n"

            "## Context Discipline\n\n"
            "Your context window is a finite resource. Adopt these habits to preserve it:\n"
            "- After reading a file, close it with close_file once you no longer need "
            "its content visible. The file remains on disk for re-reading if needed later.\n"
            "- When a debug session concludes, close it promptly. The interaction history "
            "is archived to disk automatically; keeping it open wastes context capacity.\n"
            "- Prefer sending file paths in messages rather than inlining large text blocks. "
            "Recipients can read the file at their discretion.\n"
            "- If you notice your reasoning becoming sluggish or earlier details growing "
            "hazy after a long stretch of tool interactions, invoke compact_context to "
            "archive verbose reasoning traces and reclaim working space. The operation "
            "preserves a distilled summary of your analytical trajectory while shedding "
            "the raw intermediate bulk.\n"
            "- The runtime infrastructure automatically trims input arguments from certain "
            "tool calls after the authored material is safely on disk—overwrites are trimmed "
            "immediately, while appends are batched for cleanup when the target file is closed. "
            "If you encounter a prior invocation whose parameters appear abbreviated, this is "
            "expected housekeeping; the content resides intact in the file system and can be "
            "retrieved via read_file at any time.\n\n"

            "## Reasoning Economy\n\n"
            "Externalize your analytical steps through tool calls rather than extended "
            "internal deliberation. A single executed command that yields a concrete "
            "observation outweighs lengthy speculative reasoning. When a hypothesis "
            "forms, test it immediately; when evidence suffices for a conclusion, "
            "commit your findings without rehearsing alternatives.\n\n"

            "## Communication Protocol\n\n"
            "Use send_message to exchange information with teammates. When composing a "
            "message, the level of structural discipline you apply to the payload should "
            "reflect two factors: the communicative role the message serves in the workflow, "
            "and the nature of the recipient that will consume it.\n\n"

            "Messages whose full meaning resides in natural language—status updates, "
            "completion notifications, exploratory requests—need no structured payload. "
            "Express the intent clearly in the content field and leave payload empty.\n\n"

            "When a message carries machine-actionable data alongside its textual narrative "
            "(edit sequences, insertion directives, diagnostic parameters), attach that data "
            "as a JSON structure in the payload field. If the recipient is another agent, "
            "treat the payload schema as a recommended convention rather than a rigid contract—"
            "your counterpart can interpret reasonable variations. Your skill guide specifies "
            "suggested shapes for each message type you are expected to send.\n\n"

            "If the recipient is the coordinator—a deterministic process, not a language model—"
            "payload structure becomes a binding obligation. The coordinator extracts values by "
            "fixed key names; deviations cause silent data loss. Your skill guide will call out "
            "the exact keys and types required for coordinator-bound messages.\n\n"

            "When you have no immediate work, call go_idle to suspend until a new message "
            "arrives. Avoid busy-waiting or polling—idle suspension is automatic and costs "
            "nothing."
        )

    def configure(self):
        """
        子类覆写以注册角色专属的初始工具或执行其他配置。

        在基础工具注册完毕后、主循环启动前调用。
        """
        pass

    async def on_activated(self):
        """
        子类覆写以在启动后执行首要动作。

        例如数据探查员可在此主动加载数据审查技能。
        在configure()之后、主循环之前调用。
        """
        pass

    # ================================================================
    # 基础工具注册
    # ================================================================

    def _setup_base_tools(self):
        """注册所有智能体共享的基础工具集。"""
        base_tools = [
            ToolSpec(
                name="load_skill",
                description=(
                    "Load a specialized skill by name. Injects the skill's knowledge "
                    "into your context and activates associated tools."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Skill name to load."},
                    },
                    "required": ["name"],
                },
                handler=self._handle_load_skill,
            ),
            ToolSpec(
                name="unload_skill",
                description="Unload a previously loaded skill and remove its associated tools.",
                parameters={
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Skill name to unload."},
                    },
                    "required": ["name"],
                },
                handler=self._handle_unload_skill,
            ),
            ToolSpec(
                name="send_message",
                description=(
                    "Send a message to another team member. The message will be placed "
                    "in their inbox for processing at their next check cycle."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "to": {"type": "string", "description": "Recipient agent name."},
                        "content": {"type": "string", "description": "Message text."},
                        "msg_type": {
                            "type": "string",
                            "description": "Message category.",
                            "enum": [t.value for t in MessageType],
                            "default": "task_report",
                        },
                        "payload": {
                            "description": (
                                "Optional structured data attachment. Accepts any valid JSON "
                                "structure—objects, arrays, nested combinations. Use for "
                                "transmitting edit sequences, injection directives, or other "
                                "machine-readable content that complements the free-text message body."
                            ),
                            "default": {},
                        },
                    },
                    "required": ["to", "content"],
                },
                handler=self._handle_send_message,
            ),
            ToolSpec(
                name="go_idle",
                description=(
                    "Signal that you have no immediate work to do. "
                    "Suspends execution until a new message arrives in your inbox."
                ),
                parameters={
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
                handler=self._handle_go_idle,
            ),
            ToolSpec(
                name="read_file",
                description=(
                    "Read the contents of a file. Use for examining data files, "
                    "diagnostic outputs, skill references, or any text resource. "
                    "Enable line numbering for code files to establish positional "
                    "references for injection directives or patch targeting."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "Absolute path or path relative to working directory.",
                        },
                        "with_line_numbers": {
                            "type": "boolean",
                            "description": "Prepend sequential line numbers to each line of output.",
                            "default": False,
                        },
                    },
                    "required": ["file_path"],
                },
                handler=self._handle_read_file,
            ),
            ToolSpec(
                name="close_file",
                description=(
                    "Release a previously read file from your context window. "
                    "The content is replaced with a path pointer; re-read if needed later."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "Path of the file to close.",
                        },
                    },
                    "required": ["file_path"],
                },
                handler=self._handle_close_file,
            ),
            ToolSpec(
                name="compact_context",
                description=(
                    "Compress your context window by archiving raw reasoning traces "
                    "to disk and replacing them with a concise retrospective summary. "
                    "Invoke when you sense your context is becoming saturated—responses "
                    "slowing, earlier details fading, or after a long sequence of "
                    "tool interactions. The operation preserves your analytical narrative "
                    "while reclaiming space occupied by verbose intermediate reasoning."
                ),
                parameters={
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
                handler=self._handle_compact_context,
            ),
        ]
        self.dispatcher.register_batch(base_tools)

    # ================================================================
    # 基础工具处理方法
    # ================================================================

    async def _handle_load_skill(self, name: str) -> Dict[str, Any]:
        try:
            body = self.skill_registry.load(name, self.dispatcher)
            new_tools = [
                t for t in self.dispatcher.available_names
                if (spec := self.dispatcher.get(t)) and spec.bound_skill == name
            ]
            tool_list = ", ".join(new_tools) if new_tools else "(no additional tools)"

            entry = self.skill_registry._loaded.get(name)
            skill_dir = str(entry.directory.resolve()) if entry else "(unknown)"

            return {
                "success": True,
                "result": (
                    f"Skill '{name}' loaded. New tools available: {tool_list}\n"
                    f"Skill directory: {skill_dir}"
                ),
                "inject_skill_body": body,
            }
        except KeyError as e:
            return {"success": False, "result": str(e)}

    async def _handle_unload_skill(self, name: str) -> Dict[str, Any]:
        result = self.skill_registry.unload(name, self.dispatcher)
        return {"success": True, "result": result}

    async def _handle_send_message(
        self,
        to: str,
        content: str,
        msg_type: str = "task_report",
        payload = None,
    ) -> Dict[str, Any]:
        try:
            msg = Message(
                msg_type=MessageType(msg_type),
                sender=self.name,
                recipient=to,
                content=content,
                payload=payload if payload is not None else {},
            )
            await self.channel_registry.deliver(msg)
            return {"success": True, "result": f"Message delivered to {to}."}
        except (KeyError, ValueError) as e:
            return {"success": False, "result": f"Delivery failed: {e}"}

    async def _handle_go_idle(self) -> Dict[str, Any]:
        return {"success": True, "result": "Entering idle state. Will resume when a message arrives."}

    async def _handle_read_file(
        self, file_path: str, with_line_numbers: bool = False
    ) -> Dict[str, Any]:
        path = Path(file_path)
        if not path.is_absolute():
            path = self.working_dir / path

        allowed_roots = [self.working_dir.resolve()]
        skills_root = self.skill_registry._skills_root.resolve()
        if skills_root.exists():
            allowed_roots.append(skills_root)

        resolved = path.resolve()
        if not any(resolved.is_relative_to(root) for root in allowed_roots):
            return {"success": False, "result": f"Access denied: {file_path}"}

        if not resolved.exists():
            return {"success": False, "result": f"File not found: {file_path}"}

        # 唯一性守卫：同一文件不允许重复打开
        if self.context.is_file_open(file_path):
            return {
                "success": False,
                "result": (
                    f"File '{file_path}' is already open in your context. "
                    f"Close it with close_file before re-reading to ensure "
                    f"you see the latest version."
                ),
            }

        try:
            content = resolved.read_text(encoding="utf-8")

            if with_line_numbers:
                lines = content.split("\n")
                width = len(str(len(lines)))
                content = "\n".join(
                    f"{i:>{width}} | {line}"
                    for i, line in enumerate(lines, 1)
                )

            return {
                "success": True,
                "result": content,
                "file_path": file_path,
            }
        except Exception as e:
            return {"success": False, "result": f"Read failed: {e}"}

    async def _handle_close_file(self, file_path: str) -> Dict[str, Any]:
        result = self.context.close_file(file_path)
        return {"success": True, "result": result}

    async def _handle_compact_context(self) -> Dict[str, Any]:
        """手动触发的上下文压缩。需要访问当前的API客户端。"""
        # 标记待执行，实际压缩在主循环中完成（因为需要client引用）
        return {
            "success": True,
            "result": "Context compression scheduled.",
            "trigger_compression": True,
        }

    # ================================================================
    # 收件箱与空闲管理
    # ================================================================

    def _drain_inbox(self):
        """
        批量消费收件箱中的待处理消息，注入上下文。

        在每轮API调用前执行，确保模型在决策时能看到
        所有已到达的消息。TERMINATE消息触发终止标记。
        """
        messages = self._channel.drain()
        for msg in messages:
            if msg.msg_type == MessageType.TERMINATE:
                self._terminated = True
                self.context.inject_incoming_message(
                    f"[TERMINATE] Shutdown directive from {msg.sender}: {msg.content}"
                )
                continue
            self.context.inject_incoming_message(msg.to_context_text())

    async def _enter_idle(self):
        """
        进入空闲状态，挂起直到收到新消息。

        空闲标志着当前工作阶段结束，此时清理累积的
        推理内容以释放上下文空间。
        """
        self._persist_snapshot()

        msg = await self._channel.receive()
        if msg is None:
            return

        if msg.msg_type == MessageType.TERMINATE:
            self._terminated = True
            self.context.inject_incoming_message(
                f"[TERMINATE] Shutdown directive from {msg.sender}: {msg.content}"
            )
        else:
            self.context.inject_incoming_message(msg.to_context_text())

    async def _execute_context_compression(self, client) -> str:
        """
        执行推理链压缩：归档、摘要、替换。

        1. 提取所有历史推理内容并转储到磁盘
        2. 发起独立的LLM调用，生成第一人称的浓缩复盘
        3. 移除原始推理内容，注入复盘文本

        Args:
            client: 当前活跃的DeepSeekClient实例

        Returns:
            压缩操作的结果描述
        """
        reasoning_contents = self.context.extract_reasoning_contents()

        if not reasoning_contents:
            return "No reasoning content to compress."

        # 归档原始推理内容
        self._reasoning_archive_count += 1
        archive_path = self.output_dir / f"reasoning_archive_{self._reasoning_archive_count}.json"
        archive_path.write_text(
            json.dumps(reasoning_contents, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        # 发起摘要请求
        combined = "\n\n---\n\n".join(reasoning_contents)
        summary_prompt = (
            "The following are your internal reasoning traces from the current work session, "
            "listed in chronological order. Produce a first-person retrospective that captures "
            "the essential thread: what key observations you made, which hypotheses you formed "
            "and how they evolved, what pivotal decisions you reached and the evidence behind them, "
            "and where you currently stand in the investigation. Omit mechanical details like "
            "variable listings or routine tool output parsing—focus on the analytical narrative "
            "that would let you resume effectively if the detailed traces were no longer available.\n\n"
            f"{combined}"
        )

        try:
            summary = await client.chat_completion(
                messages=[
                    {"role": "user", "content": summary_prompt}
                ],
                temperature=0.3,
                max_tokens=2048,
            )
        except Exception:
            # 摘要调用失败时退回到简单清理
            self.context.strip_all_reasoning()
            return f"Reasoning archived to {archive_path}. Summary generation failed; raw traces removed."

        # 执行替换
        self.context.strip_all_reasoning()
        self.context.inject_reasoning_summary(summary, str(archive_path))

        return f"Compressed {len(reasoning_contents)} reasoning blocks. Archive: {archive_path}"

    # ================================================================
    # 持久化
    # ================================================================

    def _persist_snapshot(self):
        """将当前上下文快照写入磁盘。"""
        if self.journal:
            self.journal.persist(self.context.snapshot())