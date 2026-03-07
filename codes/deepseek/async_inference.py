# codes/deepseek/async_inference.py
"""
DeepSeek API异步推理引擎
将业务侧的生成需求转化为API调用，处理前缀续写、降级回退、响应提取等协议层细节
"""

from .deepseek_client import DeepSeekClient, DeepSeekAPIError


class AsyncInferenceEngine:
    """
    异步推理引擎
    
    封装从提示词到可用文本的完整转换链路，
    屏蔽端点选择、格式约束、容错降级等底层交互机制
    """
    
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
        
        优先通过Beta端点的前缀续写功能约束输出格式，
        失败时自动降级至标准对话补全模式
        
        Args:
            prompt: 完整提示词文本
        
        Returns:
            生成的Python代码（包含markdown代码块标记）
        
        Raises:
            DeepSeekAPIError: API调用失败
        """
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
            
            return "```python\n" + response + "\n```"
        
        except DeepSeekAPIError as e:
            if e.status_code in [400, 422]:
                messages = [{"role": "user", "content": prompt}]
                
                thinking = {"type": "enabled"} if self.enable_thinking else None
                
                response = await self.client.chat_completion(
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=8192,
                    thinking=thinking
                )
                
                return response
            
            raise