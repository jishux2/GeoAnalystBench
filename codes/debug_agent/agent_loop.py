"""
Debug Agent核心循环
"""

import json
from pathlib import Path
from typing import Dict, Optional
from .tools import DebugTools
from .trace_logger import TraceLogger


class DebugAgent:
    """基于工具调用的调试智能体"""
    
    def __init__(
        self,
        api_key: str,
        script_path: str,
        cwd: str,
        interpreter: str,  # ← 新增参数
        error_context: Dict,
        output_dir: Path,
        max_turns: int = 20,
        temperature: float = 0.7
    ):
        """
        初始化Debug Agent
        
        Args:
            api_key: DeepSeek API密钥
            script_path: 待调试脚本路径
            cwd: 工作目录
            interpreter: Python解释器路径
            error_context: 错误上下文
            output_dir: 输出目录
            max_turns: 最大交互轮次
            temperature: 采样温度
        """
        self.api_key = api_key
        self.script_path = script_path
        self.cwd = cwd
        self.error_context = error_context
        self.max_turns = max_turns
        self.temperature = temperature
        
        self.tools = DebugTools(script_path, cwd, interpreter)  # ← 传递解释器
        self.logger = TraceLogger(output_dir / "debug_trace.json")
        
        self.messages = []
        self.turn_count = 0
        self.final_diagnosis = None
    
    async def run(self) -> Dict:
        """
        运行Agent循环
        
        Returns:
            最终诊断结果
        """
        from deepseek.deepseek_client import DeepSeekClient
        
        self._initialize_messages()
        
        async with DeepSeekClient(self.api_key) as client:
            while self.turn_count < self.max_turns:
                self.turn_count += 1
                
                response_data = await client.chat_completion_with_tools(
                    messages=self.messages,
                    tools=self.tools.get_tool_definitions(),
                    temperature=self.temperature,
                    max_tokens=8192,
                    thinking={"type": "enabled"}
                )
                
                message = response_data["choices"][0]["message"]
                
                reasoning = message.get("reasoning_content")
                content = message.get("content")
                tool_calls = message.get("tool_calls")
                
                # 构建完整的assistant消息（保留reasoning_content）
                assistant_msg = {
                    "role": "assistant",
                    "content": content or ""  # content不能为None
                }
                
                # 如果有思考内容，必须保留
                if reasoning:
                    assistant_msg["reasoning_content"] = reasoning
                
                # 如果有工具调用，添加到消息中
                if tool_calls:
                    assistant_msg["tool_calls"] = tool_calls
                
                self.messages.append(assistant_msg)
                
                # 如果没有工具调用，循环结束
                if not tool_calls:
                    self.logger.log_turn(
                        self.turn_count,
                        reasoning,
                        None,
                        None,
                        content
                    )
                    break
                
                # 执行工具调用
                tool_results = []
                diagnosis_complete = False
                
                for tool_call in tool_calls:
                    tool_name = tool_call["function"]["name"]
                    arguments = json.loads(tool_call["function"]["arguments"])
                    
                    if tool_name == "finalize_diagnosis":
                        self.final_diagnosis = arguments
                        diagnosis_complete = True
                        result = "[DIAGNOSIS_COMPLETE]"
                    else:
                        result = self.tools.execute_tool(tool_name, arguments)
                    
                    tool_results.append(result)
                    
                    self.messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "content": result
                    })
                
                self.logger.log_turn(
                    self.turn_count,
                    reasoning,
                    tool_calls,
                    tool_results,
                    content
                )
                
                if diagnosis_complete:
                    break
        
        self.tools.cleanup()
        
        if not self.final_diagnosis:
            self.final_diagnosis = {
                "root_cause": "Agent terminated without providing diagnosis",
                "repair_plan": "Unable to generate repair plan"
            }
        
        self.logger.finalize(self.final_diagnosis)
        
        return self.final_diagnosis
    
    def _initialize_messages(self):
        """构建初始系统提示词"""
        system_prompt = self._build_system_prompt()
        
        self.messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": self._format_error_context()}
        ]

        print(self._format_error_context())
    
    def _build_system_prompt(self) -> str:
        """构建系统提示词"""
        return """You are an expert debugging agent. Your task is to diagnose code errors through interactive debugging.

Available debugging modes:
- **Proactive debugging**: Step through code from the beginning to identify issues
- **Post-mortem debugging**: Analyze the state when an exception occurred

PDB command reference:
- `n` (next): Execute current line, stop at next line
- `s` (step): Step into function calls
- `c` (continue): Continue execution until next breakpoint
- `l` (list): Show current code context
- `w` (where): Print stack trace
- `p <expr>`: Print expression value
- `!<code>`: Execute arbitrary Python code

Debugging workflow:
1. Review the error context provided
2. Decide which debugging mode to use
3. Use PDB commands or code injection to explore the runtime state
4. Identify the root cause through systematic investigation
5. Call `finalize_diagnosis` with a detailed repair plan

Important guidelines:
- Provide actionable repair instructions, not just error descriptions
- Include specific code changes needed
- Be thorough but efficient with debugging steps
"""
    
    def _format_error_context(self) -> str:
        """格式化错误上下文"""
        sections = ["=== Error Context ===\n"]
        
        error_trace = self.error_context.get("error_trace", {})
        sections.append(f"Error Type: {error_trace.get('error_type', 'Unknown')}")
        sections.append(f"Error Message: {error_trace.get('error_message', '')}\n")
        
        if error_trace.get("stack_frames"):
            sections.append("Stack Trace:")
            for frame in error_trace["stack_frames"]:
                sections.append(f"\nFile: {frame.get('file', 'unknown')}, Line {frame.get('line', '?')}")
                sections.append(f"Function: {frame.get('function', 'unknown')}")
                sections.append(f"Code: {frame.get('code', '')}")
                
                if frame.get("locals"):
                    sections.append("Local Variables:")
                    sections.append(json.dumps(frame["locals"], indent=2, ensure_ascii=False))
        
        call_details = self.error_context.get("call_details", [])
        if call_details:
            sections.append("\n=== Function Call Details ===")
            for detail in call_details:
                sections.append(f"\nFunction: {detail.get('function', 'unknown')}")
                sections.append(f"Error: {detail.get('error', '')}")
                sections.append(json.dumps({
                    'args': detail.get('args_summary', []),
                    'kwargs': detail.get('kwargs_summary', {})
                }, indent=2, ensure_ascii=False))
        
        return "\n".join(sections)
    
    @staticmethod
    def _extract_reasoning(response: str) -> Optional[str]:
        """从响应中提取思考内容（如果存在）"""
        return None
    
    @staticmethod
    def _extract_tool_calls(response: str) -> Optional[list]:
        """从响应中提取工具调用"""
        return None