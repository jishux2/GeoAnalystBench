# codes/deepseek/deepseek_client.py
"""
DeepSeek API 客户端封装
提供异步请求接口、错误处理和特殊功能支持
"""

import asyncio
import aiohttp
import json
from typing import Optional, Dict, Any, List
from datetime import datetime


class DeepSeekAPIError(Exception):
    """DeepSeek API错误基类"""
    def __init__(self, status_code: int, message: str, error_type: str = "unknown"):
        self.status_code = status_code
        self.message = message
        self.error_type = error_type
        super().__init__(f"[{status_code}] {error_type}: {message}")


class DeepSeekClient:
    """DeepSeek API异步客户端"""
    
    # API端点配置
    BASE_URL = "https://api.deepseek.com"
    BETA_URL = "https://api.deepseek.com/beta"
    
    # 错误码分类
    RETRYABLE_ERRORS = {429, 500, 503}
    
    def __init__(self, api_key: str, max_retries: int = 3, timeout: int = 120):
        """
        初始化客户端
        
        Args:
            api_key: DeepSeek API密钥
            max_retries: 可重试错误的最大重试次数
            timeout: 单次请求超时时间（秒）
        """
        self.api_key = api_key
        self.max_retries = max_retries
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def __aenter__(self):
        """异步上下文管理器入口"""
        self.session = aiohttp.ClientSession(
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            },
            timeout=self.timeout
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器退出"""
        if self.session:
            await self.session.close()
    
    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: str = "deepseek-chat",
        temperature: float = 0.7,
        max_tokens: int = 4096,
        stop: Optional[List[str]] = None,
        use_prefix: bool = False
    ) -> str:
        """
        调用对话补全API
        
        Args:
            messages: 对话消息列表
            model: 模型名称
            temperature: 采样温度
            max_tokens: 最大输出token数
            stop: 停止序列
            use_prefix: 是否启用前缀续写功能
        
        Returns:
            模型生成的文本内容
        
        Raises:
            DeepSeekAPIError: API调用失败时抛出
        """
        try:
            url = f"{self.BETA_URL if use_prefix else self.BASE_URL}/chat/completions"
        
            # 添加调试输出
            print(f"[调试] 正在请求API: {url}")
            
            payload = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": False
            }
            
            if stop:
                payload["stop"] = stop
            
            # 执行带重试的请求
            for attempt in range(self.max_retries + 1):
                try:
                    async with self.session.post(url, json=payload) as response:
                        response_data = await response.json()
                        
                        if response.status == 200:
                            print(f"[调试] 收到响应，状态码200")
                            # print(f"[调试] 响应内容: {response_data}")
                            content = response_data["choices"][0]["message"]["content"]
                            print(f"[调试] 提取到内容长度: {len(content)}")
                            return content
                        
                        # 处理错误响应
                        error_info = response_data.get("error", {})
                        error_msg = error_info.get("message", "未知错误")
                        error_type = error_info.get("type", "unknown_error")
                        
                        # 判断是否可重试
                        if response.status in self.RETRYABLE_ERRORS and attempt < self.max_retries:
                            wait_time = 2 ** attempt  # 指数退避
                            await asyncio.sleep(wait_time)
                            continue
                        
                        raise DeepSeekAPIError(response.status, error_msg, error_type)
                
                except asyncio.TimeoutError:
                    if attempt < self.max_retries:
                        await asyncio.sleep(2 ** attempt)
                        continue
                    raise DeepSeekAPIError(408, "请求超时", "timeout_error")
                
                except aiohttp.ClientError as e:
                    if attempt < self.max_retries:
                        await asyncio.sleep(2 ** attempt)
                        continue
                    raise DeepSeekAPIError(0, f"网络错误: {str(e)}", "network_error")
            
            raise DeepSeekAPIError(500, "达到最大重试次数", "max_retries_exceeded")
        except Exception as e:
            print(f"[错误] chat_completion未捕获异常: {type(e).__name__} - {str(e)}")
            raise
    
    async def generate_workflow(self, prompt: str, temperature: float = 0.7) -> str:
        """
        生成工作流（使用前缀续写优化输出）
        
        Args:
            prompt: 完整提示词
            temperature: 采样温度
        
        Returns:
            工作流代码
        """
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": "tasks = [", "prefix": True}
        ]
        
        try:
            response = await self.chat_completion(
                messages=messages,
                temperature=temperature,
                max_tokens=8192,
                use_prefix=True
            )
            return "tasks = [" + response
        
        except DeepSeekAPIError:
            # 前缀续写失败时回退到普通模式
            messages = [{"role": "user", "content": prompt}]
            return await self.chat_completion(
                messages=messages,
                temperature=temperature,
                max_tokens=8192
            )
    
    async def generate_code(self, prompt: str, temperature: float = 0.7) -> str:
        """
        生成代码（使用前缀续写确保输出纯代码）
        
        Args:
            prompt: 完整提示词
            temperature: 采样温度
        
        Returns:
            Python代码
        """
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": "```python\n", "prefix": True}
        ]
        
        try:
            response = await self.chat_completion(
                messages=messages,
                temperature=temperature,
                max_tokens=8192,
                stop=["```"],
                use_prefix=True
            )
            return "```python\n" + response + "\n```"
        
        except DeepSeekAPIError:
            # 前缀续写失败时回退到普通模式
            messages = [{"role": "user", "content": prompt}]
            return await self.chat_completion(
                messages=messages,
                temperature=temperature,
                max_tokens=8192
            )