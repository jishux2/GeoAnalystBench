# codes/benchmark/eval_client.py
"""
评估模型的API客户端

对接本地反向代理端点，通过流式调用获取评估结果。
协议兼容OpenAI Chat Completions，但content字段
支持文本与图片混合的内容块数组。
"""

from __future__ import annotations

import json
import aiohttp
from typing import Any, Dict, List, Optional


class EvalClient:
    """
    评估模型的异步流式客户端

    专为单轮评估调用设计，不涉及工具调用或多轮交互。
    每次调用以流式方式接收响应并拼接为完整文本返回。
    """

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8317",
        api_key: str = "sk-ECc3M9K2lRIOsN5Of",
        model: str = "gpt-5.4",
        timeout: int = 300,
    ):
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            timeout=self.timeout,
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def evaluate(
        self,
        messages: List[Dict[str, Any]],
        temperature: float = 0.3,
        max_tokens: int = 4096,
    ) -> str:
        """
        发起流式评估调用，拼接并返回完整的响应文本。

        Args:
            messages: 消息序列，content字段可为字符串或内容块数组
            temperature: 采样温度
            max_tokens: 最大输出token数

        Returns:
            模型生成的完整文本
        """
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }

        url = f"{self.base_url}/v1/chat/completions"

        collected = []
        async with self.session.post(url, json=payload) as response:
            if response.status != 200:
                error_text = await response.text()
                raise RuntimeError(
                    f"Evaluation API returned {response.status}: {error_text}"
                )

            async for line in response.content:
                decoded = line.decode("utf-8").strip()
                if not decoded or not decoded.startswith("data: "):
                    continue
                data_str = decoded[6:]
                if data_str == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                    delta = chunk["choices"][0].get("delta", {})
                    content = delta.get("content", "")
                    if content:
                        collected.append(content)
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue

        return "".join(collected)