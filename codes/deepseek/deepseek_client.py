# codes/deepseek/deepseek_client.py
"""
DeepSeek API 客户端封装
提供异步请求接口、错误处理和JSON格式响应解析
"""

import asyncio
import aiohttp
import json
from typing import Optional, Dict, Any, List


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
    BETA_URL = "https://api.deepseek.com/beta"  # 前缀续写功能需要使用Beta端点
    
    # 错误码分类
    RETRYABLE_ERRORS = {429, 500, 503}  # 这些状态码表示临时性问题，值得重试
    
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
        """
        异步上下文管理器入口
        
        作用：创建aiohttp会话，避免每次请求都创建新连接
        用法：async with DeepSeekClient(api_key) as client: ...
        """
        self.session = aiohttp.ClientSession(
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            },
            timeout=self.timeout
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器退出，关闭会话"""
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
        
        注意：这个方法保留是因为generate_initial_code需要用到前缀续写功能
        
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
        # 前缀续写需要访问Beta端点，常规对话使用标准端点
        url = f"{self.BETA_URL if use_prefix else self.BASE_URL}/chat/completions"
        
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False
        }
        
        if stop:
            payload["stop"] = stop
        
        # 执行带指数退避的重试策略
        for attempt in range(self.max_retries + 1):
            try:
                async with self.session.post(url, json=payload) as response:
                    response_data = await response.json()
                    
                    if response.status == 200:
                        return response_data["choices"][0]["message"]["content"]
                    
                    # 处理错误响应
                    error_info = response_data.get("error", {})
                    error_msg = error_info.get("message", "未知错误")
                    error_type = error_info.get("type", "unknown_error")
                    
                    # 仅对服务端临时性错误进行重试，客户端错误直接抛出
                    if response.status in self.RETRYABLE_ERRORS and attempt < self.max_retries:
                        wait_time = 2 ** attempt  # 2秒、4秒、8秒递增
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
    
    async def generate_code_patch(
        self,
        prompt: str,
        temperature: float = 0.7,
        enable_thinking: bool = True
    ) -> dict:
        """
        生成代码补丁（JSON格式响应）
        
        Args:
            prompt: 完整提示词
            temperature: 采样温度
            enable_thinking: 是否启用思考模式
        
        Returns:
            包含target_code和replacement_code的字典
        
        Raises:
            DeepSeekAPIError: API调用失败或JSON解析失败
        """
        messages = [{"role": "user", "content": prompt}]
        
        url = f"{self.BASE_URL}/chat/completions"
        
        payload = {
            "model": "deepseek-chat",
            "messages": messages,
            "temperature": temperature,
            "max_tokens": 8192,
            "stream": False
        }
        
        if enable_thinking:
            payload["thinking"] = {"type": "enabled"}
        
        # 执行带重试的请求
        for attempt in range(self.max_retries + 1):
            try:
                async with self.session.post(url, json=payload) as response:
                    response_data = await response.json()
                    
                    if response.status == 200:
                        content = response_data["choices"][0]["message"]["content"]
                        
                        # 解析JSON响应
                        try:
                            cleaned = content.strip()
                            # 移除可能的markdown代码块标记
                            if cleaned.startswith('```'):
                                lines = cleaned.split('\n')
                                # 提取代码块内容（去掉首尾的```标记）
                                if lines[0].startswith('```'):
                                    lines = lines[1:]
                                if lines and lines[-1].strip() == '```':
                                    lines = lines[:-1]
                                cleaned = '\n'.join(lines)
                            
                            return json.loads(cleaned)
                        
                        except json.JSONDecodeError as e:
                            raise DeepSeekAPIError(
                                422,
                                f"补丁响应不是有效JSON: {str(e)}\n原始内容: {content[:300]}",
                                "json_decode_error"
                            )
                    
                    # 错误处理
                    error_info = response_data.get("error", {})
                    error_msg = error_info.get("message", "未知错误")
                    error_type = error_info.get("type", "unknown_error")
                    
                    if response.status in self.RETRYABLE_ERRORS and attempt < self.max_retries:
                        wait_time = 2 ** attempt
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
    
    async def diagnose_error(
        self,
        prompt: str,
        temperature: float = 0.7,
        enable_thinking: bool = True
    ) -> dict:
        """
        错误诊断分析（JSON格式响应）
        
        Args:
            prompt: 完整提示词
            temperature: 采样温度
            enable_thinking: 是否启用思考模式
        
        Returns:
            包含root_cause、api_queries、keywords、example_query的字典
        
        Raises:
            DeepSeekAPIError: API调用失败或JSON解析失败
        """
        # 与generate_code_patch逻辑完全相同，复用实现
        return await self.generate_code_patch(prompt, temperature, enable_thinking)