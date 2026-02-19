"""
Debug Agent核心循环
"""

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

from .tools import DebugToolkit
from .prompts import build_system_prompt, build_initial_user_message, format_code_with_line_numbers
from .trace_logger import TraceLogger


class DebugAgent:
    """调试智能体"""
    
    def __init__(
        self,
        api_key: str,
        script_content: str,
        working_dir: str,
        interpreter: str,
        output_dir: Path,
        current_diagnosis: Optional[Dict] = None,  # 改为任务级诊断
        error_summary: Optional[str] = None,        # 新增：上次执行的错误概要
        debug_mode: str = "crash",
        max_turns: int = 20,
        temperature: float = 0.7,
        executor: Optional[ProcessPoolExecutor] = None
    ):
        """
        初始化智能体
        
        Args:
            api_key: DeepSeek API密钥
            script_content: 原始代码（第1轮生成的纯净代码）
            working_dir: 工作目录
            interpreter: Python解释器路径
            output_dir: 输出目录
            current_diagnosis: 任务级诊断（包含root_cause和patches）
            error_summary: 上次执行的错误概要
            debug_mode: 调试模式
            max_turns: 最大交互轮次
            temperature: 采样温度
            executor: 进程池引用
        """
        self.api_key = api_key
        self.script_content = script_content
        self.working_dir = working_dir
        self.interpreter = interpreter
        self.output_dir = Path(output_dir)
        self.current_diagnosis = current_diagnosis
        self.error_summary = error_summary
        self.debug_mode = debug_mode
        self.max_turns = max_turns
        self.temperature = temperature
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.toolkit = DebugToolkit(
            script_content=script_content,
            working_dir=working_dir,
            interpreter=interpreter,
            output_dir=self.output_dir,
            executor=executor  # 传递进程池
        )
        
        self.logger = TraceLogger(self.output_dir / "debug_trace.json")
        
        self.messages: List[Dict] = []
        self.turn_count = 0
        self.final_result: Optional[Dict] = None
    
    async def run(self) -> Dict[str, Any]:
        """
        运行智能体循环
        
        Returns:
            最终结果，包含：
            - success: 是否成功
            - diagnosis: 诊断信息（如失败）
            - patches: 修复补丁列表（如失败）
        """
        from deepseek.deepseek_client import DeepSeekClient
        
        self._initialize_messages()
        
        # 开发阶段：保存初始提示词
        prompt_path = self.output_dir / "initial_prompt.md"
        prompt_path.write_text(
            '\n\n---\n\n'.join(msg.get('content', '') for msg in self.messages),
            encoding='utf-8'
        )
        
        async with DeepSeekClient(self.api_key) as client:
            while self.turn_count < self.max_turns:
                self.turn_count += 1
                
                response_data = await client.chat_completion_with_tools(
                    messages=self.messages,
                    tools=self.toolkit.get_tool_definitions(),
                    temperature=self.temperature,
                    max_tokens=8192,
                    thinking={"type": "enabled"}
                )
                
                message = response_data["choices"][0]["message"]
                tool_calls = message.get("tool_calls")
                
                self.messages.append(message)
                
                if not tool_calls:
                    self.logger.log_turn(
                        turn_num=self.turn_count,
                        reasoning=message.get("reasoning_content"),
                        tool_calls=None,
                        tool_results=None,
                        response_text=message.get("content")
                    )
                    
                    print(f"[Turn {self.turn_count}] No tool calls - unexpected termination")
                    break
                
                tool_results = []
                should_terminate = False
                
                for tool_call in tool_calls:
                    tool_name = tool_call["function"]["name"]
                    arguments = json.loads(tool_call["function"]["arguments"])
                    
                    result = await self.toolkit.execute_tool(tool_name, arguments)
                    tool_results.append(result)
                    
                    self.messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "content": result.get("result", "")
                    })
                    
                    if result.get("final"):
                        should_terminate = True
                        
                        if tool_name == "finalize_success":
                            self.final_result = {"success": True}
                        
                        elif tool_name == "finalize_with_patch":
                            self.final_result = {
                                "success": False,
                                **result.get("diagnosis", {})
                            }
                
                self.logger.log_turn(
                    turn_num=self.turn_count,
                    reasoning=message.get("reasoning_content"),
                    tool_calls=tool_calls,
                    tool_results=[r.get("result", "") for r in tool_results],
                    response_text=message.get("content")
                )
                
                if should_terminate:
                    break
        
        self.toolkit.cleanup()
        
        if self.final_result is None:
            self.final_result = {
                "success": False,
                "root_cause": "Agent terminated without providing a conclusion",
                "patches": []
            }
        
        self.logger.finalize(self.final_result)
        
        return self.final_result
    
    def _initialize_messages(self):
        """构建初始消息序列"""
        system_prompt = build_system_prompt(self.debug_mode)
        
        code_with_lines = format_code_with_line_numbers(self.script_content)
        
        user_message = build_initial_user_message(
            current_code=code_with_lines,
            current_diagnosis=self.current_diagnosis,  # 改为任务级诊断
            error_summary=self.error_summary
        )
        
        self.messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]