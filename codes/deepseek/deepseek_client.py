# codes/deepseek/deepseek_client.py
"""
DeepSeek API 客户端封装
提供异步HTTP通信基础设施，不涉及业务层逻辑
"""

import asyncio
import aiohttp
from typing import Optional, Dict, List


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
        thinking: Optional[Dict[str, str]] = None,
        use_prefix: bool = False
    ) -> str:
        """
        对话补全，返回生成的文本内容

        Args:
            messages: 对话消息列表
            model: 模型名称
            temperature: 采样温度
            max_tokens: 最大输出token数
            stop: 停止序列
            thinking: 思考模式配置
            use_prefix: 是否启用前缀续写（切换至Beta端点）

        Returns:
            模型生成的文本内容
        """
        payload = self._build_payload(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop,
            thinking=thinking
        )

        response_data = await self._request_with_retry(payload, use_beta=use_prefix)
        return response_data["choices"][0]["message"]["content"]

    async def chat_completion_with_tools(
        self,
        messages: List[Dict[str, str]],
        tools: List[Dict],
        model: str = "deepseek-chat",
        temperature: float = 0.7,
        max_tokens: int = 8192,
        thinking: Optional[Dict[str, str]] = None
    ) -> Dict:
        """
        支持工具调用的对话补全，返回完整响应对象

        Args:
            messages: 对话消息列表
            tools: 工具定义列表
            model: 模型名称
            temperature: 采样温度
            max_tokens: 最大输出token数
            thinking: 思考模式配置

        Returns:
            完整的API响应对象
        """
        payload = self._build_payload(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
            thinking=thinking
        )

        return await self._request_with_retry(payload, use_beta=False)

    def _build_payload(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float,
        max_tokens: int,
        stop: Optional[List[str]] = None,
        tools: Optional[List[Dict]] = None,
        thinking: Optional[Dict[str, str]] = None
    ) -> Dict:
        """组装请求体，仅包含非空字段"""
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False
        }

        if stop:
            payload["stop"] = stop
        if tools:
            payload["tools"] = tools
        if thinking:
            payload["thinking"] = thinking

        return payload

    async def _request_with_retry(self, payload: Dict, use_beta: bool = False) -> Dict:
        """
        执行带指数退避的重试请求

        Args:
            payload: 请求体
            use_beta: 是否使用Beta端点

        Returns:
            解析后的响应JSON

        Raises:
            DeepSeekAPIError: 不可恢复的API错误
        """
        url = f"{self.BETA_URL if use_beta else self.BASE_URL}/chat/completions"

        for attempt in range(self.max_retries + 1):
            try:
                async with self.session.post(url, json=payload) as response:
                    response_data = await response.json()

                    if response.status == 200:
                        return response_data

                    error_info = response_data.get("error", {})
                    error_msg = error_info.get("message", "未知错误")
                    error_type = error_info.get("type", "unknown_error")

                    if response.status in self.RETRYABLE_ERRORS and attempt < self.max_retries:
                        await asyncio.sleep(2 ** attempt)
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