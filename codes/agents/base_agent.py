# codes/agents/base_agent.py
"""
子智能体协程骨架

定义所有子智能体共享的运行时框架：消息循环、工具调度、
收件箱检查、空闲挂起与唤醒、上下文持久化。
"""

from __future__ import annotations

import json
import asyncio
import subprocess
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
    """

    # ripgrep二进制路径（可通过类属性覆盖）
    RIPGREP_PATH = "rg"

    # 读取工具的限流阈值
    READ_FILE_SIZE_LIMIT = 256 * 1024      # 256KB，整体读取时的文件大小上限
    READ_RESULT_TOKEN_BUDGET = 25000        # 读取结果的token预算（字符数/4估算）
    READ_RESULT_CHAR_BUDGET = 100000        # 对应的字符数预算

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

        self._channel: AgentChannel = channel_registry.register(name)
        self.dispatcher = ToolDispatcher(output_dir=output_dir)
        self.context: Optional[AgentContext] = None
        self.journal: Optional[ContextJournal] = None

        self._terminated = False
        self._turn_count = 0
        self.context_threshold = context_threshold

        self._debug_session_count = 0
        self._reasoning_archive_count = 0

    async def run(self):
        """智能体主入口。"""
        from deepseek.deepseek_client import DeepSeekClient

        system_prompt = self.build_system_prompt()
        self.context = AgentContext(system_prompt, self.name)
        self.journal = ContextJournal(
            self.output_dir / f"{self.name}_journal.json",
            self.name,
        )

        self._setup_base_tools()
        self.configure()

        try:
            await self.on_activated()

            async with DeepSeekClient(self.api_key) as client:
                while not self._terminated:

                    self._drain_inbox()

                    if self.context.estimate_token_count() > self.context_threshold:
                        await self._execute_context_compression(client)
                        self._persist_snapshot()

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

                        if result.get("inject_skill"):
                            skill_name = result["inject_skill"]["name"]
                            skill_body = result["inject_skill"]["body"]
                            self.context.inject_skill(skill_name, skill_body)

                        # 调试会话关闭
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

                        if result.get("trigger_compression"):
                            compression_requested = True

                        if tool_name == "go_idle":
                            idle_requested = True

                    self._persist_snapshot()

                    if compression_requested:
                        await self._execute_context_compression(client)
                        self._persist_snapshot()

                    if self._terminated:
                        break

                    if idle_requested:
                        await self._enter_idle()

        except Exception as e:
            import traceback
            error_summary = f"{type(e).__name__}: {e}"
            try:
                await self.channel_registry.deliver(Message(
                    msg_type=MessageType.TASK_REPORT,
                    sender=self.name,
                    recipient="coordinator",
                    content=(
                        f"Agent '{self.name}' encountered a fatal error and is "
                        f"shutting down: {error_summary}"
                    ),
                    payload={
                        "agent_crash": True,
                        "error_type": type(e).__name__,
                        "error_message": str(e),
                        "traceback": traceback.format_exc(),
                    },
                ))
            except Exception:
                pass
            raise

        finally:
            outcome = {"terminated_by": "normal" if self._terminated else "exception"}
            if self.journal and self.context:
                self.journal.finalize(outcome, self.context.snapshot())

    # ================================================================
    # 子类扩展点
    # ================================================================

    def build_system_prompt(self) -> str:
        return self._assemble_system_prompt(
            role_identity="You are a team member in a collaborative geospatial analysis unit.",
            role_workflow="Await instructions from the coordinator.",
        )

    def _assemble_system_prompt(self, role_identity: str, role_workflow: str) -> str:
        skills_overview = self.skill_registry.list_descriptions()

        return (
            "## Working Environment\n\n"
            "Each task occupies an isolated directory under the evaluation workspace. "
            "The dataset/ subdirectory houses source data files; outputs/ collects "
            "execution artifacts and diagnostic records, partitioned by team member.\n\n"

            "## Team Composition\n\n"
            "Four members collaborate on each geospatial analysis task. "
            "The **explorer** surveys dataset resources and surfaces structural "
            "intelligence. The **engineer** architects and implements the task "
            "script, embedding verification checkpoints throughout the processing "
            "pipeline. The **diagnostician** assumes ownership of the executable "
            "artifact post-delivery, conducting runtime investigation and driving "
            "the repair cycle to resolution. A supervisory **coordinator** governs "
            "the collective timeline and intervenes when progress stalls or the "
            "time budget expires. Address teammates by these identifiers when "
            "routing messages.\n\n"

            "## Your Role\n\n"
            f"{role_identity}\n\n"

            f"## Skill Repertoire\n\n"
            f"Load a skill to activate domain-specific guidance and tools:\n"
            f"{skills_overview}\n\n"

            "## Operational Sequence\n\n"
            f"{role_workflow}\n\n"

            "## File Access Model\n\n"
            "Use read_file to examine any text file within the workspace. "
            "The tool supports reading entire files or specific line ranges "
            "via offset and limit parameters—prefer targeted reads over "
            "loading entire large files when you only need a particular section. "
            "Use grep to locate content by pattern before reading, forming an "
            "efficient search-then-read workflow.\n\n"
            "When a file's content has not changed since your last read, the "
            "tool returns a brief confirmation rather than the full text again, "
            "conserving context space.\n\n"
            "Direct reading of files under the dataset/ directory is restricted. "
            "Source data files are often too voluminous for context and require "
            "format-aware inspection. Access their content through the explorer's "
            "diagnostic scripts or purpose-built probes instead.\n\n"

            "## Context Stewardship\n\n"
            "Your context window is a finite resource. Adopt these practices to "
            "sustain its capacity over extended work sessions:\n"
            "- Prefer targeted line-range reads over full-file reads when you "
            "only need a specific section. Use grep to pinpoint locations first.\n"
            "- Conclude debug sessions promptly after extracting your findings. "
            "The interaction transcript is archived to disk automatically; "
            "lingering sessions consume space without contributing fresh insight.\n"
            "- Prefer transmitting file paths in messages rather than inlining "
            "bulky text. Recipients retrieve the content at their own discretion.\n"
            "- When your reasoning begins to feel sluggish or earlier details "
            "grow indistinct after a prolonged stretch of tool interactions, "
            "invoke compact_context to archive verbose reasoning traces and "
            "reclaim working space. The operation distills your analytical "
            "trajectory into a concise retrospective while shedding the raw "
            "intermediate bulk.\n\n"

            "## Reasoning Economy\n\n"
            "Externalize your analytical steps through tool calls rather than "
            "extended internal deliberation. A single executed command that "
            "yields a concrete observation outweighs lengthy speculative "
            "reasoning. When a hypothesis forms, test it immediately; when "
            "evidence suffices for a conclusion, commit your findings without "
            "rehearsing alternatives.\n\n"

            "## Communication Protocol\n\n"
            "Use send_message to exchange information with teammates. Each "
            "message comprises a free-text content field and an optional "
            "structured payload. Composing a message involves two sequential "
            "determinations.\n\n"
            "The first concerns substance: what information the message must "
            "carry. This follows from the communicative role the message plays "
            "in the workflow—a reply to an inquiry conveys both a direct answer "
            "and any attendant signals the recipient needs to act on; a handoff "
            "notification furnishes the artifact's location alongside a synopsis "
            "of the work that produced it. Identify every discrete informational "
            "obligation the message's function imposes before deciding how to "
            "encode any of them.\n\n"
            "The second concerns form: how each piece of information should be "
            "organized within the message envelope. This is governed not by who "
            "receives the message, but by the processing pathway each fragment "
            "will traverse upon arrival. Fragments destined for semantic "
            "interpretation—situational context, analytical narratives, "
            "natural-language answers—belong in the content field, where a "
            "language model can absorb them fluidly. Fragments that must be "
            "extracted programmatically—values keyed by fixed names, sequences "
            "consumed by deterministic logic—demand the predictable structure "
            "of a payload object. A single message may well contain both kinds: "
            "its textual body addressing the recipient's reasoning faculties "
            "while its payload feeds an automated handler that shares the same "
            "inbox.\n\n"
            "When you have no immediate work, call go_idle to suspend until "
            "a new message arrives. Idle suspension is automatic and costs "
            "nothing; avoid busy-waiting or polling."
        )

    def configure(self):
        pass

    async def on_activated(self):
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
                                "structure. Use for transmitting machine-readable content that "
                                "complements the free-text message body."
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
                    "Read a file's content. Supports full reads and targeted line-range "
                    "reads via offset and limit parameters. When the content is unchanged "
                    "since your last read of the same file, returns a brief confirmation "
                    "instead of the full text. Large files that exceed the size limit "
                    "must be read in segments."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "Absolute path or path relative to working directory.",
                        },
                        "offset": {
                            "type": "integer",
                            "description": "Starting line number (1-based). Omit to start from the beginning.",
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of lines to read from the offset. Omit to read to the end.",
                        },
                    },
                    "required": ["file_path"],
                },
                handler=self._handle_read_file,
            ),
            ToolSpec(
                name="grep",
                description=(
                    "Search file contents using ripgrep. Returns matching lines with "
                    "file paths, line numbers, and optional surrounding context. "
                    "Use --files mode to list files matching a glob pattern without "
                    "searching content."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "pattern": {
                            "type": "string",
                            "description": "Search pattern (regex by default, literal with fixed_strings=true).",
                        },
                        "path": {
                            "type": "string",
                            "description": "Directory or file to search. Defaults to working directory.",
                            "default": ".",
                        },
                        "glob": {
                            "type": "string",
                            "description": "Filter files by glob pattern, e.g. '*.py'.",
                        },
                        "fixed_strings": {
                            "type": "boolean",
                            "description": "Treat pattern as literal text rather than regex.",
                            "default": False,
                        },
                        "ignore_case": {
                            "type": "boolean",
                            "description": "Case-insensitive matching.",
                            "default": False,
                        },
                        "context_lines": {
                            "type": "integer",
                            "description": "Number of context lines before and after each match.",
                            "default": 0,
                        },
                        "max_count": {
                            "type": "integer",
                            "description": "Maximum number of matches per file.",
                            "default": 50,
                        },
                        "list_files": {
                            "type": "boolean",
                            "description": "List matching file paths only (--files mode). Pattern is used as glob filter.",
                            "default": False,
                        },
                    },
                    "required": ["pattern"],
                },
                handler=self._handle_grep,
            ),
            ToolSpec(
                name="compact_context",
                description=(
                    "Compress your context by archiving raw reasoning traces "
                    "to disk and replacing them with a concise retrospective. "
                    "Invoke when responses slow or earlier details grow hazy."
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

        # 读取工具豁免结果预算（自带限流）
        self.dispatcher.set_result_budget("read_file", -1)

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
                "inject_skill": {"name": name, "body": body},
            }
        except KeyError as e:
            return {"success": False, "result": str(e)}

    async def _handle_send_message(
        self,
        to: str,
        content: str,
        msg_type: str = "task_report",
        payload=None,
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
        self, file_path: str, offset: int = None, limit: int = None
    ) -> Dict[str, Any]:
        path = Path(file_path)
        if not path.is_absolute():
            path = self.working_dir / path

        # dataset/目录读取管控
        dataset_dir = (self.working_dir / "dataset").resolve()
        resolved = path.resolve()
        if resolved.is_relative_to(dataset_dir):
            return {
                "success": False,
                "result": (
                    "Direct reading of dataset files is restricted. "
                    "Source data files are often too large for context and require "
                    "format-aware inspection. Use the explorer's diagnostic scripts "
                    "or write a custom probe to extract the specific information you need."
                ),
            }

        allowed_roots = [self.working_dir.resolve()]
        skills_root = self.skill_registry._skills_root.resolve()
        if skills_root.exists():
            allowed_roots.append(skills_root)

        if not any(resolved.is_relative_to(root) for root in allowed_roots):
            return {"success": False, "result": f"Access denied: {file_path}"}

        if not resolved.exists():
            return {"success": False, "result": f"File not found: {file_path}"}

        # 第一层限流：整体读取时检查文件大小
        is_full_read = offset is None and limit is None
        if is_full_read:
            file_size = resolved.stat().st_size
            if file_size > self.READ_FILE_SIZE_LIMIT:
                return {
                    "success": False,
                    "result": (
                        f"File too large for full read: {file_size:,} bytes "
                        f"(limit: {self.READ_FILE_SIZE_LIMIT:,}). "
                        f"Use offset and limit parameters to read specific sections, "
                        f"or use grep to locate content of interest first."
                    ),
                }

        try:
            all_lines = resolved.read_text(encoding="utf-8").split("\n")
            total_lines = len(all_lines)

            # 应用行范围
            if offset is not None:
                start = max(0, offset - 1)  # 转为0-based
            else:
                start = 0

            if limit is not None:
                end = min(start + limit, total_lines)
            else:
                end = total_lines

            selected = all_lines[start:end]

            # 附加行号
            width = len(str(end))
            numbered = "\n".join(
                f"{i:>{width}} | {line}"
                for i, line in enumerate(selected, start + 1)
            )

            # 第二层限流：内容字符数检查
            if len(numbered) > self.READ_RESULT_CHAR_BUDGET:
                return {
                    "success": False,
                    "result": (
                        f"Read result too large: {len(numbered):,} chars "
                        f"(budget: {self.READ_RESULT_CHAR_BUDGET:,}). "
                        f"Narrow your read range with offset/limit, "
                        f"or use grep to locate specific content first."
                    ),
                }

            # 重复内容检测
            content_for_hash = "\n".join(selected)
            if self.context.check_file_read_duplicate(file_path, content_for_hash):
                range_desc = ""
                if offset is not None or limit is not None:
                    range_desc = f" (lines {start + 1}-{end})"
                return {
                    "success": True,
                    "result": (
                        f"File '{file_path}'{range_desc} unchanged since last read "
                        f"({total_lines} total lines)."
                    ),
                }

            range_info = f"Showing lines {start + 1}-{end} of {total_lines}"
            return {
                "success": True,
                "result": f"{range_info}\n{numbered}",
            }
        except Exception as e:
            return {"success": False, "result": f"Read failed: {e}"}

    async def _handle_grep(
        self,
        pattern: str,
        path: str = ".",
        glob: str = None,
        fixed_strings: bool = False,
        ignore_case: bool = False,
        context_lines: int = 0,
        max_count: int = 50,
        list_files: bool = False,
    ) -> Dict[str, Any]:
        # 验证路径存在性时用绝对路径
        search_path = Path(path)
        if not search_path.is_absolute():
            search_path = self.working_dir / search_path

        if not search_path.exists():
            return {"success": False, "result": f"Path not found: {path}"}

        # 传给ripgrep的保持用户原始输入，因为cwd已经是working_dir
        cmd = [self.RIPGREP_PATH]

        if list_files:
            cmd.append("--files")
            if pattern and pattern != ".":
                cmd.extend(["-g", pattern])
        else:
            cmd.extend(["--line-number", "--column"])
            cmd.extend(["--max-columns", "300", "--max-columns-preview"])
            cmd.extend(["--max-count", str(max_count)])
            cmd.append("--color=never")

            if fixed_strings:
                cmd.append("--fixed-strings")
            if ignore_case:
                cmd.append("--ignore-case")
            if context_lines > 0:
                cmd.extend(["-C", str(context_lines)])
            if glob:
                cmd.extend(["-g", glob])

            cmd.append(pattern)

        cmd.append(path)  # 用原始路径而非拼接后的绝对路径

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=30,
                cwd=str(self.working_dir),
            )

            output = result.stdout.strip()
            if result.returncode == 0:
                if not output:
                    return {"success": True, "result": "No matches found."}
                return {"success": True, "result": output}
            elif result.returncode == 1:
                return {"success": True, "result": "No matches found."}
            else:
                error = result.stderr.strip()
                return {"success": False, "result": f"grep failed: {error}"}
        except FileNotFoundError:
            return {
                "success": False,
                "result": "ripgrep (rg) not found. Ensure it is installed and accessible.",
            }
        except subprocess.TimeoutExpired:
            return {"success": False, "result": "grep timed out after 30 seconds."}
        except Exception as e:
            return {"success": False, "result": f"grep failed: {e}"}

    async def _handle_compact_context(self) -> Dict[str, Any]:
        return {
            "success": True,
            "result": "Context compression scheduled.",
            "trigger_compression": True,
        }

    # ================================================================
    # 收件箱与空闲管理
    # ================================================================

    def _drain_inbox(self):
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
        reasoning_contents = self.context.extract_reasoning_contents()

        if not reasoning_contents:
            return "No reasoning content to compress."

        self._reasoning_archive_count += 1
        archive_path = self.output_dir / f"reasoning_archive_{self._reasoning_archive_count}.json"
        archive_path.write_text(
            json.dumps(reasoning_contents, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

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
            self.context.strip_all_reasoning()
            return f"Reasoning archived to {archive_path}. Summary generation failed; raw traces removed."

        self.context.strip_all_reasoning()
        self.context.inject_reasoning_summary(summary, str(archive_path))

        return f"Compressed {len(reasoning_contents)} reasoning blocks. Archive: {archive_path}"

    # ================================================================
    # 持久化
    # ================================================================

    def _persist_snapshot(self):
        if self.journal:
            self.journal.persist(self.context.snapshot())