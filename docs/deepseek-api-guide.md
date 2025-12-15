**基于DeepSeek API的高并发推理方案**

DeepSeek作为国内大语言模型领域的重要参与者，其API服务在定价策略与并发能力方面展现出显著的竞争优势。平台采用按token计量的收费模式：

- 输入端依据缓存命中情况实施差异化定价
  - 缓存命中：0.2元/百万token
  - 缓存未命中：2元/百万token
- 输出端统一执行3元/百万token的标准

这一费率结构意味着完成GeoAnalystBench基准测试所需的1200次模型调用，即便考虑最坏情况（全部缓存未命中），总支出仍可控制在个位数人民币范围内。

更具吸引力的是其并发政策——平台未对单用户施加RPM（每分钟请求数）或TPM（每分钟token处理量）硬性限制，仅在服务端负载过高时触发503状态码提示用户稍后重试。这一特性为批量推理场景带来了独特优势：研究者无需编写复杂的限流逻辑，可直接采用高并发策略充分利用API资源，将原本需要数小时的顺序调用压缩至二三十分钟完成。

**服务端点与模型版本**

DeepSeek当前维护两套API端点，分别对应不同的功能支持范围：

| 端点类型 | URL | 用途 |
|---------|-----|------|
| 标准端点 | `https://api.deepseek.com` | 常规对话补全服务 |
| Beta端点 | `https://api.deepseek.com/beta` | 前缀续写等实验性特性 |

两套端点在响应格式、错误处理等方面保持一致，差异仅体现在所支持的请求参数集合上。

模型层面提供三个可选标识符：

- `deepseek-chat` — 非推理模式的DeepSeek-V3.2版本，适用于需要直接输出结果的场景
- `deepseek-reasoner` — 启用思考链机制，模型会在生成最终答案前展示中间推理过程
- `deepseek-reasoner-speciale` — 专为推理优化的变体，仅支持思考模式且输出长度上限扩展至128K

就GeoAnalystBench的推理需求而言，工作流构建与代码生成均属确定性任务，无需借助显式推理链来提升准确度，故采用`deepseek-chat`即可满足要求。该模型接受最大128K的输入上下文，默认输出限额为4K token但可通过`max_tokens`参数调整至8K，这一容量足以容纳完整的空间分析代码实现。

**请求体结构解析**

DeepSeek声称与OpenAI保持API兼容，实际调用时确实可复用OpenAI SDK的客户端实例，仅需修改`base_url`参数指向DeepSeek端点。标准对话补全请求包含以下核心字段：

`messages` — 承载完整的对话历史，每个元素需指定`role`（system/user/assistant）与`content`属性。系统提示通常置于首位，用户查询与助手响应交替排列。

`model` — 指明目标模型标识符，前述三个可选值对应不同的能力配置。

`temperature` — 控制采样随机性，取值0-2之间，数值越高输出越具创造性但稳定性下降。基准测试中统一设为0.7以平衡多样性与可复现性。

`max_tokens` — 限定单次响应的最大长度，需根据任务特性合理设置。工作流生成通常在2K以内，代码实现则可能突破5K。

`stop` — 定义停止序列，模型生成内容匹配该序列时立即中断输出。这一机制在强制模型遵循特定格式时极为有效，后文将详述其在代码块提取中的应用。

`stream` — 布尔值决定是否采用流式传输，设为false时服务端会等待生成完成后一次性返回完整结果。

除上述OpenAI标准参数外，DeepSeek引入了`thinking`扩展字段来控制推理模式行为。该参数接受包含`type`属性的对象，可选值为`enabled`或`disabled`。由于本项目选用非推理模式的`deepseek-chat`，此参数实际不会生效，故在构造请求时可直接省略。

**前缀续写机制详解**

Beta端点提供的前缀续写功能允许开发者在`messages`数组的末尾追加一个`role`为`assistant`的消息，并通过设置`prefix: true`标记来指示模型将该消息内容视为已生成的前缀，继续补全后续部分。这一设计本质上是在强制指定输出的开头片段，通过减少模型自由发挥空间来提升格式遵从度。

以代码生成为例，传统做法是在用户消息中要求"请用Python实现"，但模型往往会先输出"好的，我来为您编写代码"之类的确认语句，随后才进入代码块。采用前缀续写后的处理流程：

1. 在assistant消息中预填markdown代码块的起始标记` ```python\n `
2. 配合`stop`参数设为`["```"]`
3. 模型输出纯粹的代码内容，省略所有额外解释

工作流构建场景同样受益于此机制。由于提示词要求输出NetworkX绘图代码，模型需先定义`tasks`列表再进行图结构操作。通过预填`tasks = [`作为前缀，可引导模型立即进入任务枚举环节，避免生成诸如"首先我们定义工作流步骤"的冗余说明。

实施前缀续写需注意两点约束：
- 标记了`prefix: true`的消息必须位于`messages`数组的末尾位置
- 必须将`base_url`切换至Beta端点，标准端点会拒绝包含该标记的请求并返回参数错误

考虑到Beta端点的实验性质，代码实现中应设计降级策略——当前缀续写请求失败时自动回退到常规模式，确保服务可用性不受单一特性影响。

**错误码体系与重试策略**

API调用过程可能遭遇多种异常情况，平台通过HTTP状态码传递错误类型信息：

**客户端错误（不可重试）**
- 400 — 请求体格式畸形
- 401 — API密钥无效或缺失
- 402 — 账户余额不足（需人工充值）
- 422 — 参数值不符合约束条件

此类错误源于调用方配置不当，重试无法解决问题，应直接抛出异常供上层处理。

**服务端错误（可重试）**
- 429 — 请求速率超限（尽管DeepSeek宣称无固定RPM限制，突发流量仍可能触发保护机制）
- 500 — 服务器内部故障
- 503 — 服务端过载

这三类错误通过等待后重试往往能够恢复，适合纳入自动重试逻辑。

实践中采用指数退避策略效果较佳：
- 首次失败后等待2秒
- 第二次失败等待4秒
- 第三次失败等待8秒
- 最多重试3轮

等待时长呈指数增长可有效缓解服务端压力，同时避免因固定间隔导致的请求集中。若达到最大重试次数仍未成功，则将错误信息记录并标记该任务为失败状态，后续可通过增量推理机制针对性补偿。

**系统架构与模块组成**

基于DeepSeek API构建的异步推理系统采用模块化设计，将API交互、结果持久化、任务编排等职责明确划分至独立组件。完整的目录布局如下：

```
GeoAnalystBench/
├─ codes/
│  ├─ deepseek/                       # DeepSeek专属推理模块
│  │  ├─ __init__.py
│  │  ├─ deepseek_client.py           # HTTP通信层封装
│  │  ├─ async_inference.py           # 异步任务协调中枢
│  │  └─ results_manager.py           # 数据存储与索引维护
│  ├─ run_deepseek_inference.py       # 推理系统主程序
│  └─ prompt_generation.py            # 提示词批量生成工具
│
├─ prompts/                           # 提示词集中存放区
│  ├─ code_prompts.csv
│  └─ workflow_prompts.csv
│
├─ results/                           # 推理输出归档目录
│  └─ deepseek/
│     ├─ workflow_responses.csv
│     └─ code_responses.csv
│
└─ dataset/                           # 基准测试数据源
   └─ GeoAnalystBench.csv
```

该架构将DeepSeek相关实现隔离在独立的`deepseek/`子包中，与原有的Ollama推理路径完全解耦，便于后续针对不同模型特性进行定制化开发。

**组件职责划分**

**1. `deepseek_client.py` - API客户端接口**

屏蔽与DeepSeek API的底层交互细节，暴露简洁的异步调用接口。模块定义了`DeepSeekClient`类及配套的异常类型`DeepSeekAPIError`，前者专注于请求构造、响应解析、错误重试等环节，后者承载结构化的故障信息（HTTP状态码、错误分类、详细描述）供上层捕获。

```python
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
    
    async def generate_workflow(self, prompt: str, temperature: float = 0.7) -> str:
        """
        生成工作流（使用前缀续写优化输出）
        
        通过在assistant消息中预填"tasks = ["，引导模型直接输出任务列表代码，
        减少不必要的解释性文本，提升输出质量的稳定性
        
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
            # 前缀续写依赖Beta端点，若失败则降级到标准模式
            messages = [{"role": "user", "content": prompt}]
            return await self.chat_completion(
                messages=messages,
                temperature=temperature,
                max_tokens=8192
            )
    
    async def generate_code(self, prompt: str, temperature: float = 0.7) -> str:
        """
        生成代码（使用前缀续写确保输出纯代码）
        
        预填Python代码块起始标记并设置stop序列，强制模型输出格式化代码，
        避免在代码前后添加额外的解释或说明
        
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
                stop=["```"],  # 遇到代码块结束标记时停止生成
                use_prefix=True
            )
            return "```python\n" + response + "```"
        
        except DeepSeekAPIError:
            # 降级处理保证基本功能可用
            messages = [{"role": "user", "content": prompt}]
            return await self.chat_completion(
                messages=messages,
                temperature=temperature,
                max_tokens=8192
            )
```

类内实现了三个关键方法：`chat_completion`处理通用对话补全请求，内嵌指数退避重试策略；`generate_workflow`和`generate_code`分别针对流程图生成与脚本编写场景，借助前缀续写特性规约输出格式。所有方法均返回协程对象，需配合`await`关键字使用。

**2. `results_manager.py` - 持久化与状态追踪**

负责推理产出的文件归档及完成进度索引构建。`ResultsManager`类管控两份CSV文档（`workflow_responses.csv`与`code_responses.csv`）的写入操作，同时在内存中保有已完成任务的集合索引，为增量执行与断点恢复提供基础。

```python
# codes/deepseek/results_manager.py
"""
推理结果管理器
负责结果的持久化存储、增量更新和状态追踪
"""

# 标准库
import csv
import os
import re
import asyncio
from pathlib import Path
from typing import Dict, Set, Tuple, Optional
from datetime import datetime

# 第三方库
import pandas as pd
import aiofiles


class ResultsManager:
    """推理结果管理器"""
    
    def __init__(self, output_dir: str = "results/deepseek"):
        """
        初始化管理器
        
        Args:
            output_dir: 结果输出目录
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.workflow_file = self.output_dir / "workflow_responses.csv"
        self.code_file = self.output_dir / "code_responses.csv"
        
        # 异步锁必须在事件循环内创建，此处先初始化为None
        # 实际的Lock对象会在首次异步调用时由事件循环创建
        self.async_lock = None
        
        # 已完成任务的索引集合
        self.completed_tasks: Dict[str, Set[Tuple[int, str, int]]] = {
            'workflow': set(),
            'code': set()
        }
        
        # 初始化输出文件
        self._initialize_files()
        
        # 加载已有结果
        self._load_existing_results()
    
    def _initialize_files(self):
        """初始化CSV文件结构"""
        header = [
            'task_id', 'response_id', 'prompt_type', 'response_type',
            'Arcpy', 'llm_model', 'response_content', 'task_length',
            'error_info', 'timestamp'
        ]
        
        for file_path in [self.workflow_file, self.code_file]:
            # 处理文件不存在、为空或缺失表头的情况
            if not file_path.exists() or file_path.stat().st_size == 0:
                with open(file_path, 'w', newline='', encoding='utf-8') as f:
                    csv.writer(f).writerow(header)
            else:
                # 检查已存在文件的首行是否为有效表头
                with open(file_path, 'r', encoding='utf-8') as f:
                    first_line = f.readline().strip()
                    # 若首行不是表头（可能因手动删除部分数据导致），则补充表头
                    if first_line and not first_line.startswith('task_id'):
                        f.seek(0)
                        content = f.read()
                        with open(file_path, 'w', newline='', encoding='utf-8') as fw:
                            csv.writer(fw).writerow(header)
                            fw.write(content)
    
    def _load_existing_results(self):
        """加载已有的推理结果，构建完成状态索引"""
        for task_type, file_path in [
            ('workflow', self.workflow_file),
            ('code', self.code_file)
        ]:
            if not file_path.exists() or file_path.stat().st_size == 0:
                continue
            
            try:
                df = pd.read_csv(file_path)
                for _, row in df.iterrows():
                    # 从response_id中解析重复序号，格式为{task_id}{type}{repeat_idx}
                    match = re.search(r'(\d+)(workflow|code)(\d+)$', str(row['response_id']))
                    if match:
                        task_id = int(match.group(1))
                        repeat_idx = int(match.group(3))
                        prompt_type = row['prompt_type']
                        
                        # 仅将成功的结果标记为已完成，错误结果需要重新执行
                        if pd.isna(row.get('error_info')) or not row.get('error_info'):
                            self.completed_tasks[task_type].add(
                                (task_id, prompt_type, repeat_idx)
                            )
            
            except Exception as e:
                print(f"警告：加载{file_path}时出错：{e}")
    
    def is_completed(
        self,
        task_type: str,
        task_id: int,
        prompt_type: str,
        repeat_idx: int
    ) -> bool:
        """
        检查特定任务是否已完成
        
        Args:
            task_type: 'workflow' 或 'code'
            task_id: 任务编号
            prompt_type: 提示词类型
            repeat_idx: 重复序号（0-2）
        
        Returns:
            是否已完成
        """
        return (task_id, prompt_type, repeat_idx) in self.completed_tasks[task_type]
    
    def calculate_workflow_length(self, workflow_text: str) -> int:
        """
        从工作流文本中提取步骤数量
        
        通过正则表达式精确定位tasks列表的范围，避免误统计代码中其他位置的字符串
        （如matplotlib参数、图表标题等）
        """
        import re
        
        # 匹配整个tasks = [...] 结构，re.DOTALL使.能匹配换行符
        pattern = r'tasks\s*=\s*\[(.*?)\]'
        match = re.search(pattern, workflow_text, re.DOTALL)
        
        if not match:
            return 0
        
        # 仅在列表内容中统计引号包裹的字符串数量
        tasks_content = match.group(1)
        task_pattern = r'"[^"]*"'
        tasks = re.findall(task_pattern, tasks_content)
        
        return len(tasks)
    
    async def save_result(
        self,
        task_type: str,
        task_id: int,
        response_id: str,
        prompt_type: str,
        arcpy: bool,
        response_content: str,
        error_info: Optional[str] = None
    ):
        """
        保存单个推理结果（异步版本）
        
        Args:
            task_type: 'workflow' 或 'code'
            task_id: 任务编号
            response_id: 响应唯一标识
            prompt_type: 提示词类型
            arcpy: 是否使用ArcPy
            response_content: 模型响应内容
            error_info: 错误信息（如有）
        """
        file_path = self.workflow_file if task_type == 'workflow' else self.code_file
        
        # 计算工作流长度
        task_length = 'none'
        if task_type == 'workflow' and not error_info:
            task_length = self.calculate_workflow_length(response_content)
        
        row = [
            task_id, response_id, prompt_type, task_type, arcpy,
            'deepseek-chat', response_content if not error_info else '',
            task_length, error_info or '', datetime.now().isoformat()
        ]
        
        # 在内存中构建CSV格式的行，避免csv.writer直接操作异步文件对象
        import csv
        import io
        output = io.StringIO()
        csv.writer(output).writerow(row)
        line = output.getvalue()
        
        # 使用异步锁保护文件写入，防止并发写入导致数据交错
        # 虽然写入是串行的，但配合异步I/O，等待锁的协程不会阻塞事件循环
        async with self.async_lock:
            async with aiofiles.open(file_path, 'a', encoding='utf-8') as f:
                await f.write(line)
        
        # 更新完成状态索引（在锁外执行，因为操作的是内存数据结构）
        if not error_info:
            match = re.search(r'(\d+)(workflow|code)(\d+)$', response_id)
            if match:
                repeat_idx = int(match.group(3))
                self.completed_tasks[task_type].add(
                    (task_id, prompt_type, repeat_idx)
                )
    
    def get_statistics(self, task_type: str) -> Dict:
        """
        获取当前完成统计信息
        
        Args:
            task_type: 'workflow' 或 'code'
        
        Returns:
            包含统计数据的字典
        """
        total_tasks = 50 * 4 * 3  # 50任务 × 4配置 × 3重复
        completed = len(self.completed_tasks[task_type])
        
        return {
            'total': total_tasks,
            'completed': completed,
            'remaining': total_tasks - completed,
            'progress_pct': (completed / total_tasks * 100) if total_tasks > 0 else 0
        }
```

`save_result`方法采用异步文件I/O实现，通过`asyncio.Lock`机制防止并发写入冲突。`calculate_workflow_length`运用正则表达式从生成的代码中抽取任务步骤计数。`_load_existing_results`在启动阶段扫描历史记录，将已成功完成的任务标识注入索引，供后续查询。

**3. `async_inference.py` - 并发调度引擎**

统筹整个推理流程的执行编排，涵盖任务清单构建、并发度控制、进度可视化、异常隔离等环节。`AsyncInferenceEngine`类接收API凭证与并发参数，初始化客户端及结果管理器实例，通过信号量机制限定同时在途的请求数量。

```python
# codes/deepseek/async_inference.py
"""
异步推理引擎
实现并发调度、进度追踪和批量处理
"""

import asyncio
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple
from tqdm.asyncio import tqdm
from datetime import datetime

from .deepseek_client import DeepSeekClient, DeepSeekAPIError
from .results_manager import ResultsManager


class AsyncInferenceEngine:
    """异步推理引擎"""
    
    def __init__(
        self,
        api_key: str,
        max_concurrent: int = 30,
        temperature: float = 0.7
    ):
        """
        初始化引擎
        
        Args:
            api_key: DeepSeek API密钥
            max_concurrent: 最大并发请求数，建议范围20-100
            temperature: 采样温度
        """
        self.api_key = api_key
        self.max_concurrent = max_concurrent
        self.temperature = temperature
        
        self.client: DeepSeekClient = None
        self.results_manager = ResultsManager()
        
        # 通过信号量控制同时发送的请求数量
        self.semaphore = asyncio.Semaphore(max_concurrent)
        
        # 统计信息
        self.stats = {
            'success': 0,
            'failed': 0,
            'retried': 0
        }
    
    def _load_prompts(self, task_type: str) -> pd.DataFrame:
        """加载提示词文件"""
        file_path = f"prompts/{task_type}_prompts.csv"
        if not Path(file_path).exists():
            raise FileNotFoundError(f"提示词文件不存在：{file_path}")
        return pd.read_csv(file_path)
    
    def _build_task_list(
        self,
        task_type: str,
        prompts_df: pd.DataFrame
    ) -> List[Dict]:
        """
        构建待执行任务列表（仅包含未完成的任务）
        
        通过与已完成记录比对，实现增量推理和断点续跑
        
        Args:
            task_type: 'workflow' 或 'code'
            prompts_df: 提示词数据框
        
        Returns:
            任务配置列表
        """
        tasks = []
        
        for _, row in prompts_df.iterrows():
            task_id = row['task_id']
            prompt_type = self._get_prompt_type(row)
            
            # 每个配置需要3次独立推理以评估稳定性
            for repeat_idx in range(3):
                # 检查该任务是否已成功完成
                if self.results_manager.is_completed(
                    task_type, task_id, prompt_type, repeat_idx
                ):
                    continue
                
                tasks.append({
                    'task_type': task_type,
                    'task_id': task_id,
                    'prompt_type': prompt_type,
                    'prompt_content': row['prompt_content'],
                    'arcpy': row['Arcpy'],
                    'repeat_idx': repeat_idx,
                    'response_id': f"{task_id}{task_type}{repeat_idx}"
                })
        
        return tasks
    
    def _get_prompt_type(self, row: pd.Series) -> str:
        """根据配置确定提示词类型"""
        if row['domain_knowledge'] and row['dataset']:
            return 'domain_and_dataset'
        elif row['domain_knowledge']:
            return 'domain'
        elif row['dataset']:
            return 'dataset'
        else:
            return 'original'
    
    async def _execute_single_task(self, task_config: Dict, pbar: tqdm):
        """
        执行单个推理任务
        
        通过信号量限制并发数，避免本地资源耗尽或触发API限流
        
        Args:
            task_config: 任务配置字典
            pbar: 进度条对象
        """
        async with self.semaphore:
            task_type = task_config['task_type']
            prompt = task_config['prompt_content']
            
            try:
                # 根据任务类型选择对应的生成方法
                if task_type == 'workflow':
                    response = await self.client.generate_workflow(
                        prompt, self.temperature
                    )
                else:
                    response = await self.client.generate_code(
                        prompt, self.temperature
                    )
                
                # 推理成功后立即写入文件，实现实时增量保存
                await self.results_manager.save_result(
                    task_type=task_type,
                    task_id=task_config['task_id'],
                    response_id=task_config['response_id'],
                    prompt_type=task_config['prompt_type'],
                    arcpy=task_config['arcpy'],
                    response_content=response
                )
                
                self.stats['success'] += 1
            
            except DeepSeekAPIError as e:
                # 将错误信息写入结果文件，便于后续分析失败原因
                await self.results_manager.save_result(
                    task_type=task_type,
                    task_id=task_config['task_id'],
                    response_id=task_config['response_id'],
                    prompt_type=task_config['prompt_type'],
                    arcpy=task_config['arcpy'],
                    response_content='',
                    error_info=str(e)
                )
                
                self.stats['failed'] += 1
            
            except Exception as e:
                # 捕获所有未预期的异常，避免单个任务失败导致整体中断
                error_msg = f"未知错误：{type(e).__name__} - {str(e)}"
                await self.results_manager.save_result(
                    task_type=task_type,
                    task_id=task_config['task_id'],
                    response_id=task_config['response_id'],
                    prompt_type=task_config['prompt_type'],
                    arcpy=task_config['arcpy'],
                    response_content='',
                    error_info=error_msg
                )
                
                self.stats['failed'] += 1
            
            finally:
                pbar.update(1)
    
    async def run_inference(self, task_type: str):
        """
        执行特定类型的批量推理
        
        Args:
            task_type: 'workflow' 或 'code'
        """
        print(f"\n{'='*60}")
        print(f"开始执行{task_type}推理任务")
        print(f"{'='*60}\n")
        
        # 在事件循环内初始化异步锁，确保锁对象绑定到正确的事件循环
        if self.results_manager.async_lock is None:
            self.results_manager.async_lock = asyncio.Lock()
        
        # 加载提示词
        prompts_df = self._load_prompts(task_type)
        
        # 构建待执行任务列表
        tasks = self._build_task_list(task_type, prompts_df)
        
        if not tasks:
            print(f"所有{task_type}任务已完成，无需执行！")
            return
        
        print(f"待执行任务数：{len(tasks)}")
        print(f"最大并发数：{self.max_concurrent}")
        print(f"开始时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # 重置统计信息
        self.stats = {'success': 0, 'failed': 0, 'retried': 0}
        
        # 初始化客户端并执行并发推理
        async with DeepSeekClient(self.api_key) as client:
            self.client = client
            
            # 创建进度条并并发执行所有任务
            with tqdm(total=len(tasks), desc=f"{task_type}推理进度") as pbar:
                await asyncio.gather(*[
                    self._execute_single_task(task, pbar)
                    for task in tasks
                ])
        
        # 输出统计信息
        print(f"\n{'='*60}")
        print(f"{task_type}推理完成")
        print(f"{'='*60}")
        print(f"成功：{self.stats['success']}")
        print(f"失败：{self.stats['failed']}")
        print(f"成功率：{self.stats['success']/(len(tasks))*100:.2f}%")
        print(f"结束时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # 显示整体进度
        stats = self.results_manager.get_statistics(task_type)
        print(f"整体进度：{stats['completed']}/{stats['total']} "
              f"({stats['progress_pct']:.2f}%)")
        print(f"剩余任务：{stats['remaining']}\n")
    
    async def run_all(self):
        """依次执行工作流和代码推理"""
        await self.run_inference('workflow')
        await self.run_inference('code')
```

`_build_task_list`方法遍历提示词配置表，过滤已完成项后生成待处理清单。`_execute_single_task`将单次推理请求包裹在异常捕获逻辑中，确保个别故障不会传播至整体流程。`run_inference`编排完整的执行序列，利用`asyncio.gather`并发调度全部任务协程。

**4. `run_deepseek_inference.py` - 命令行入口**

提供可执行脚本接口，从环境变量提取API密钥，实例化推理引擎并依次启动工作流与代码生成任务。脚本内含基本的参数校验与异常处理逻辑，支持用户通过Ctrl+C中断执行。

```python
# codes/run_deepseek_inference.py
"""
DeepSeek API推理执行入口
支持增量推理、并发控制和错误恢复
"""

import asyncio
import os
import sys
from pathlib import Path

# 添加deepseek模块到路径
sys.path.insert(0, str(Path(__file__).parent))

from deepseek.async_inference import AsyncInferenceEngine


def main():
    """主执行函数"""
    # 从环境变量获取API密钥
    api_key = os.getenv("DEEPSEEK_API_KEY")
    
    if not api_key:
        print("错误：未设置DEEPSEEK_API_KEY环境变量")
        print("请执行：export DEEPSEEK_API_KEY='your_api_key_here'")
        return
    
    print("="*60)
    print("GeoAnalystBench - DeepSeek API 推理系统")
    print("="*60)
    print(f"API密钥：{'*' * (len(api_key) - 8)}{api_key[-8:]}")
    print("="*60)
    
    # 配置推理引擎
    engine = AsyncInferenceEngine(
        api_key=api_key,
        max_concurrent=30,  # 可根据实际情况调整
        temperature=0.7
    )
    
    # 执行推理
    try:
        asyncio.run(engine.run_all())
        print("\n所有推理任务执行完毕！")
    
    except KeyboardInterrupt:
        print("\n\n推理被用户中断")
        print("已完成的结果已保存，下次运行将自动跳过这些任务")
    
    except Exception as e:
        print(f"\n执行过程中发生错误：{e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
```

**配套工具更新**

`prompt_generation.py`经过路径配置调整，现将生成的提示词文件统一写入`prompts/`目录，与代码文件物理隔离。修改涉及函数默认参数及目录创建逻辑，确保首次运行时自动建立必要的文件夹结构。

**推理结果的组织形式**

异步推理流程的产出归档于`results/deepseek/`目录，以两份CSV表格形式分别记录流程图构建与脚本编写任务的完整输出。文档结构采用统一的字段配置：

| 字段名 | 含义说明 |
|--------|----------|
| `task_id` | 对应基准测试集中的任务序号 |
| `response_id` | 全局唯一标识符，格式为`{task_id}{type}{repeat_idx}` |
| `prompt_type` | 提示词配置分类，可能值包括original、domain、dataset、domain_and_dataset |
| `response_type` | 区分workflow与code两类任务 |
| `Arcpy` | 布尔值，指示该任务依赖闭源库或开源方案 |
| `llm_model` | 当前固定为deepseek-chat |
| `response_content` | 模型输出的原始文本 |
| `task_length` | 工作流包含的步骤总数，代码任务此项为none |
| `error_info` | 异常发生时记录诊断信息，正常情况下留空 |
| `timestamp` | ISO格式时间戳，标记写入发生的时刻 |

并发写入环境下，文件内各条记录的排列次序反映的是任务实际完成的先后关系，而非按编号或配置类型的逻辑分组。这种次序的不可预测性不构成使用障碍——每行都携带足够的标识属性，分析阶段可根据实际需要对数据框重新排序或切片。

`response_id`采用三段式结构实现全局唯一标识：前缀部分由任务编号与类型拼接而成，尾部的数字索引（0至2）则对应同一配置下的三轮独立推理。以`5workflow1`为例，解读为第5号任务在工作流生成模式下的第二次执行。

容错与重试机制依赖`error_info`字段的状态判断。API调用顺利完成时该栏位保持空白；遭遇网络中断、服务端异常或参数校验失败时，会捕获并存储具体的错误类型与描述文本。包含错误信息的条目不会进入已完成索引，下轮执行时会被自动识别并重新尝试，直至获得有效响应或触发人工介入条件。

**数据流转路径**

完整的推理流程遵循以下数据流向：

1. `prompt_generation.py`从`dataset/GeoAnalystBench.csv`读取任务描述，衍生四种配置组合的提示词，写入`prompts/`目录
2. `run_deepseek_inference.py`启动时，`AsyncInferenceEngine`载入提示词文件
3. `ResultsManager`扫描`results/deepseek/`下的历史记录，重建完成状态索引
4. 引擎依据索引筛除已完成任务，构建待处理清单
5. `DeepSeekClient`并发发起API请求，接收模型响应
6. `ResultsManager`将结果实时追加至对应CSV文档
7. 全部任务完成后，输出统计摘要并退出

该流程设计确保了数据的单向流动与状态的一致性保障，任何环节的中断都不会导致已完成工作的丢失。