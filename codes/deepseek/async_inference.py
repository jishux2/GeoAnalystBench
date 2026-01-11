# codes/deepseek/async_inference.py
"""
DeepSeek API异步推理引擎（精简版）
仅提供三个核心API调用接口，不负责任务管理和状态追踪
"""

import asyncio
from .deepseek_client import DeepSeekClient, DeepSeekAPIError


class AsyncInferenceEngine:
    """异步推理引擎"""
    
    def __init__(
        self,
        api_key: str,
        temperature: float = 0.7,
        enable_thinking: bool = True
    ):
        """
        初始化引擎
        
        Args:
            api_key: DeepSeek API密钥
            temperature: 采样温度
            enable_thinking: 是否启用思考模式
        """
        self.api_key = api_key
        self.temperature = temperature
        self.enable_thinking = enable_thinking
        
        self.client: DeepSeekClient = None
    
    async def __aenter__(self):
        """异步上下文管理器入口"""
        self.client = DeepSeekClient(self.api_key)
        await self.client.__aenter__()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器退出"""
        if self.client:
            await self.client.__aexit__(exc_type, exc_val, exc_tb)
    
    async def generate_initial_code(self, prompt: str) -> str:
        """
        生成首轮完整代码
        
        Args:
            prompt: 完整提示词文本
        
        Returns:
            生成的Python代码（包含markdown代码块标记）
        
        Raises:
            DeepSeekAPIError: API调用失败
        """
        # 优先使用前缀续写优化输出格式
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": "```python\n", "prefix": True}
        ]
        
        try:
            response = await self.client.chat_completion(
                messages=messages,
                temperature=self.temperature,
                max_tokens=8192,
                stop=["```"],
                use_prefix=True
            )
            
            # 补全代码块标记
            return "```python\n" + response + "\n```"
        
        except DeepSeekAPIError as e:
            # 前缀续写失败时降级到标准模式
            if e.status_code in [400, 422]:
                return await self._generate_with_thinking(prompt)
            raise
    
    async def generate_patch(self, prompt: str) -> dict:
        """
        生成代码补丁（JSON格式）
        
        Args:
            prompt: 完整提示词文本
        
        Returns:
            包含target_code和replacement_code的字典
        
        Raises:
            DeepSeekAPIError: API调用失败或JSON解析失败
        """
        return await self.client.generate_code_patch(
            prompt,
            self.temperature,
            self.enable_thinking
        )
    
    async def diagnose(self, prompt: str) -> dict:
        """
        执行错误诊断分析（JSON格式）
        
        Args:
            prompt: 完整提示词文本
        
        Returns:
            包含root_cause、api_queries、keywords、example_query的字典
        
        Raises:
            DeepSeekAPIError: API调用失败或JSON解析失败
        """
        return await self.client.diagnose_error(
            prompt,
            self.temperature,
            self.enable_thinking
        )
    
    async def _generate_with_thinking(self, prompt: str) -> str:
        """
        使用思考模式生成代码（内部辅助方法）
        
        Args:
            prompt: 完整提示词
        
        Returns:
            生成的代码文本
        """
        messages = [{"role": "user", "content": prompt}]
        
        url = f"{self.client.BASE_URL}/chat/completions"
        
        payload = {
            "model": "deepseek-chat",
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": 8192,
            "stream": False
        }
        
        if self.enable_thinking:
            payload["thinking"] = {"type": "enabled"}
        
        async with self.client.session.post(url, json=payload) as response:
            response_data = await response.json()
            
            if response.status == 200:
                return response_data["choices"][0]["message"]["content"]
            
            error_info = response_data.get("error", {})
            raise DeepSeekAPIError(
                response.status,
                error_info.get("message", "未知错误"),
                error_info.get("type", "unknown_error")
            )