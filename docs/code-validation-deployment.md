**GeoCode Validator部署与操作实务**

GeoCode Validator是面向地理空间分析代码自动化验证需求设计的测试框架。其核心职责在于检验大语言模型输出的Python脚本能否在真实数据集上正常执行，通过模拟实际运行环境捕获错误信息并汇总统计结果。

系统具备四项关键能力：从推理产出的CSV文件中批量抽取程序代码，为脚本自动注入运行时必需的辅助逻辑（目录创建、后端配置等），在隔离环境中并发调度多个任务，最终生成涵盖成功率、错误分布、类别表现等维度的评估报告。这套流程将原本需要手动逐个测试的过程转化为可重复的自动化验证，显著降低大规模代码评测的操作成本。

适用场景包括模型生成代码的质量评估、不同提示策略的效果对比、以及特定任务类别的错误模式挖掘。通过灵活的筛选机制，使用者可按方法论分类、技术栈类型或任务编号组合过滤待测对象，聚焦于特定研究维度展开分析。

**环境准备与资源配置**

开始使用前需完成三项基础准备：构建独立的Python执行环境、组织评测所需的数据资源、配置解释器路径映射。

首先是地理空间处理环境的搭建。为避免与项目主环境产生依赖冲突，系统采用独立虚拟环境承载`geopandas`、`rasterio`等专用库。运行配置脚本即可完成环境创建与依赖批量安装：

```bash
python codes/setup_evaluation_env.py
```

该脚本执行后将在项目根目录生成`venv_gis_opensource`文件夹，其中包含地理数据处理所需的全部工具包。脚本同时会将解释器的绝对路径写入`codes/evaluator_config.json`，供后续任务调度时读取使用。若安装过程中出现网络超时或包冲突，可参考输出日志定位具体库并手动重试。

数据资源层面需要准备两类文件。第一类是模型推理产出的结果集，即`results/deepseek/code_responses.csv`，该文件记录了各任务在不同提示配置下的生成代码及元数据。第二类是评测工作空间`evaluation_workspace`，其内部按任务编号划分子目录，每个子目录包含专家参考实现、数据集文件及输出目录等结构。典型布局如下：

```
evaluation_workspace/
├─ 1/
│  ├─ UrbanHeat.py          # 专家代码
│  ├─ dataset/              # 地理数据源
│  │  ├─ CensusBlock.geojson
│  │  └─ Temperature.geojson
│  ├─ generated/            # 模型生成脚本存放处（自动创建）
│  ├─ outputs/              # 执行产物归档目录（自动创建）
│  └─ evaluation.json       # 任务配置与状态记录（自动创建）
├─ 2/
└─ ...
```

其中`dataset`目录及专家代码需预先准备，带标注"自动创建"的部分由系统在初始化阶段生成。需特别注意数据集文件命名应与提示词描述保持一致，否则脚本在引用路径时将触发找不到文件的错误。

解释器配置完成后，`evaluator_config.json`内容应类似：

```json
{
  "opensource_interpreter": "E:/GeoAnalystBench/venv_gis_opensource/Scripts/python.exe",
  "arcgis_interpreter": null
}
```

开源环境路径由配置脚本自动填充，闭源环境（ArcGIS Pro）需手动补充对应的Python解释器位置。当前阶段若仅测试开源任务，可暂时忽略`arcgis_interpreter`字段。

**源码包的内部构造**

验证系统的实现逻辑分布于`codes/evaluator/`子包之下，各Python模块承担明确划分的职责范畴：

```
codes/
├─ evaluator/
│  ├─ __init__.py
│  ├─ workspace_manager.py        # 任务索引与代码落盘
│  ├─ code_executor.py             # 子进程隔离执行控制器
│  └─ evaluation_reporter.py       # 分组统计与序列化接口
├─ evaluator_config.json           # 运行时环境的定位声明
├─ setup_evaluation_env.py         # 虚拟环境自动构建脚本
└─ run_evaluation.py               # 顶层流程入口
```

这套模块体系将原始推理数据转化为可执行脚本、启动隔离进程完成验证、最终输出结构化评估的全过程拆解为松耦合的组件。各文件通过导入依赖建立协作关系，而非通过硬编码路径或全局状态传递信息。配置文件采用JSON格式存储环境参数，支持在不修改源码的前提下切换技术栈或调整资源限制。

**任务调度与验证执行**

评测流程的启动依赖主控脚本`run_evaluation.py`的编排协调。该脚本串联了任务筛选、代码提取、并发调度、报告生成四个环节，将分散的功能模块整合为连贯的自动化链路。使用者仅需在脚本中设定目标范围与运行策略，即可完成从原始推理产物到结构化统计结果的全流程转换。

**1. `workspace_manager.py` - 工作空间与代码抽取引擎**

```python
# codes/evaluator/workspace_manager.py
"""
评测工作空间管理器
负责目录结构初始化和模型生成代码的物理化
"""

import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Set


class WorkspaceManager:
    """评测工作空间管理器"""
    
    # 六种空间分析方法论分类
    CATEGORY_REVERSE_MAP = {
        'Detecting and quantifying patterns': 'DP',
        'Determining how places are related': 'DR',
        'Finding the best locations and paths': 'F',
        'Making predictions': 'S',
        'Measuring size, shape, and distribution': 'M',
        'Understanding where': 'U'
    }
    
    def __init__(
        self,
        workspace_root: str = "evaluation_workspace",
        dataset_path: str = "dataset/GeoAnalystBench.csv",
        responses_path: str = "results/deepseek/code_responses.csv"
    ):
        """
        初始化管理器
        
        Args:
            workspace_root: 评测工作空间根目录
            dataset_path: 基准测试集路径
            responses_path: 模型响应结果路径
        """
        self.workspace_root = Path(workspace_root)
        self.dataset_path = Path(dataset_path)
        self.responses_path = Path(responses_path)
        
        # 加载任务元数据
        self.tasks_df = pd.read_csv(dataset_path)
        self.responses_df = None
        
        # 任务配置索引
        self.task_configs: Dict[int, Dict] = {}
        self._build_task_index()
    
    def _build_task_index(self):
        """构建任务配置索引"""
        for idx, row in self.tasks_df.iterrows():
            task_id = idx + 1  # 任务ID从1开始
            
            # 解析方法论类别
            categories = []
            for cat_col in ['Task Categories1', 'Task Categories2', 'Task Categories3']:
                if pd.notna(row.get(cat_col)):
                    cat_full = row[cat_col].strip()
                    # 转换为缩写
                    cat_abbr = self.CATEGORY_REVERSE_MAP.get(cat_full)
                    if cat_abbr:
                        categories.append(cat_abbr)
            
            self.task_configs[task_id] = {
                'id': task_id,
                'title': row['Task'],
                'is_opensource': row['Open Source'] == 'T',
                'categories': categories,
                'reference_code': row['CodeString'],
                'workflow_length': row['Task Length']
            }
    
    def filter_tasks(
        self,
        opensource_only: bool = False,
        categories: Optional[List[str]] = None,
        task_ids: Optional[List[int]] = None
    ) -> List[int]:
        """
        按条件筛选任务
        
        Args:
            opensource_only: 仅包含开源任务
            categories: 方法论类别列表（如['DR', 'F']）
            task_ids: 明确指定的任务ID列表
        
        Returns:
            符合条件的任务ID列表
        """
        selected = set(self.task_configs.keys())
        
        # 应用开源过滤
        if opensource_only:
            selected = {
                tid for tid in selected
                if self.task_configs[tid]['is_opensource']
            }
        
        # 应用类别过滤
        if categories:
            selected = {
                tid for tid in selected
                if any(cat in self.task_configs[tid]['categories'] for cat in categories)
            }
        
        # 应用ID过滤
        if task_ids:
            selected = selected & set(task_ids)
        
        return sorted(selected)
    
    def inject_instrumentation(self, code: str) -> str:
        """
        向模型生成的代码注入增强版插桩逻辑
        
        该方法在代码提取后、写入文件前执行，将复杂的调试机制统一注入，
        避免在提示词中暴露过多实现细节。插桩包含两个核心组件：
        
        1. 上下文感知的异常钩子：通过解析出错代码行，优先展示被访问的变量字段
        2. 函数调用监控装饰器：记录第三方库函数的参数详情
        
        Args:
            code: 从响应中提取的原始Python代码
        
        Returns:
            注入插桩逻辑后的完整代码
        """
        # 完整的插桩代码块，采用英文注释以保持生成代码的风格一致性
        instrumentation_code = '''
import sys
import json
import traceback
import functools
import re
import linecache

# Context-aware object summarization for error diagnostics
def extract_accessed_fields(code_line, var_name):
    """Extract field accesses from a line of code for a given variable"""
    escaped_var = re.escape(var_name)
    
    # Match var_name['field'] or var_name["field"]
    pattern1 = rf"(?<!\\w){escaped_var}\\['([^']+)'\\]"
    pattern2 = rf'(?<!\\w){escaped_var}\\["([^"]+)"\\]'
    
    # Match var_name.field (but not var_name.method())
    pattern3 = rf"(?<!\\w){escaped_var}\\.(\\w+)(?!\\()"
    
    fields = set()
    
    try:
        fields.update(re.findall(pattern1, code_line))
    except re.error:
        pass
    
    try:
        fields.update(re.findall(pattern2, code_line))
    except re.error:
        pass
    
    try:
        fields.update(re.findall(pattern3, code_line))
    except re.error:
        pass
    
    return list(fields)


def smart_summarize(obj, code_line=None, var_name=None, max_len=200):
    """Intelligently summarize objects, prioritizing fields accessed in code"""
    try:
        obj_type = type(obj).__name__
        
        # Handle pandas Series with context-aware field selection
        if 'Series' in obj_type:
            result = {'_type': 'Series', '_dtype': str(obj.dtype)}
            
            # Extract fields accessed in the error-triggering code line
            priority_fields = []
            if code_line and var_name:
                try:
                    priority_fields = extract_accessed_fields(code_line, var_name)
                except Exception as e:
                    result['_extract_error'] = str(e)
            
            # Display accessed fields first
            shown_fields = set()
            for field in priority_fields:
                if field in obj.index:
                    try:
                        result[field] = str(obj[field])[:200]
                        shown_fields.add(field)
                    except Exception as e:
                        result[f'{field}_error'] = str(e)
            
            # Fill remaining slots with other fields (max 20 total)
            remaining = 20 - len(shown_fields)
            for field in obj.index:
                if field not in shown_fields and remaining > 0:
                    try:
                        result[str(field)] = str(obj[field])[:100]
                        shown_fields.add(field)
                        remaining -= 1
                    except Exception:
                        pass
            
            if len(obj) > len(shown_fields):
                result['_truncated'] = f'... and {len(obj) - len(shown_fields)} more fields'
            
            return result
        
        # Handle DataFrame/GeoDataFrame
        if 'DataFrame' in obj_type or 'GeoDataFrame' in obj_type:
            info = {
                '_type': obj_type,
                '_shape': f'{obj.shape[0]} rows × {obj.shape[1]} columns',
                '_columns': list(obj.columns)[:10]
            }
            if 'GeoDataFrame' in obj_type and hasattr(obj, 'geometry') and 'geometry' in obj.columns:
                try:
                    geom_types = obj.geometry.geom_type.value_counts().to_dict()
                    info['_geometry_types'] = geom_types
                except Exception:
                    pass
            return info
        
        # Handle collections
        if isinstance(obj, (list, tuple)):
            return f"{obj_type}(len={len(obj)})"
        if isinstance(obj, dict):
            return f"dict(keys={list(obj.keys())[:5]})"
        
        # Default fallback
        return str(obj)[:max_len]
    
    except Exception as e:
        return f"<{type(obj).__name__} object>"


# Enhanced global exception hook with stack frame analysis
def capture_exception(exc_type, exc_value, exc_traceback):
    """Capture exception with detailed context including local variables"""
    stack_info = []
    tb = exc_traceback
    
    while tb is not None:
        frame = tb.tb_frame
        code_line = linecache.getline(frame.f_code.co_filename, tb.tb_lineno).strip()
        
        # Summarize local variables with code context awareness
        frame_locals = {}
        for var_name, var_value in frame.f_locals.items():
            frame_locals[var_name] = smart_summarize(
                var_value,
                code_line=code_line,
                var_name=var_name
            )
        
        stack_info.append({
            'file': frame.f_code.co_filename,
            'function': frame.f_code.co_name,
            'line': tb.tb_lineno,
            'code': code_line,
            'locals': frame_locals
        })
        tb = tb.tb_next
    
    context = {
        'error_type': exc_type.__name__,
        'error_message': str(exc_value),
        'traceback': ''.join(traceback.format_exception(exc_type, exc_value, exc_traceback)),
        'stack_frames': stack_info
    }
    
    with open('error_trace.json', 'w', encoding='utf-8') as f:
        json.dump(context, f, indent=2, ensure_ascii=False)
    
    sys.__excepthook__(exc_type, exc_value, exc_traceback)

sys.excepthook = capture_exception


# Function call monitoring decorator
def monitor_call(func_name):
    """Decorator to capture function arguments when errors occur"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                arg_summaries = [smart_summarize(arg) for arg in args]
                kwargs_summaries = {k: smart_summarize(v) for k, v in kwargs.items()}
                
                detail = {
                    'function': func_name,
                    'args_summary': arg_summaries,
                    'kwargs_summary': kwargs_summaries,
                    'error': str(e)
                }
                
                with open('call_details.json', 'a', encoding='utf-8') as f:
                    json.dump(detail, f, indent=2, ensure_ascii=False)
                    f.write('\\n')
                raise
        return wrapper
    return decorator

'''
        
        # 定位第一个import语句的位置，将插桩代码插在最前面
        # 这样确保异常钩子在任何业务逻辑执行前就已注册
        import re
        lines = code.split('\n')
        insert_pos = 0
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped and not stripped.startswith('#'):
                # 跳过开头的注释和空行，找到第一个实际语句
                if re.match(r'^(import |from )', stripped):
                    insert_pos = i
                    break
        
        # 在找到的位置插入插桩代码
        lines.insert(insert_pos, instrumentation_code)
        
        return '\n'.join(lines)
    
    def extract_code_from_response(self, response_text: str) -> str:
        """
        从模型响应中提取Python代码并注入调试机制
        
        处理流程包含三个阶段：
        1. 识别并剥离Markdown代码块标记
        2. 注入上下文感知的插桩逻辑（异常钩子与函数监控）
        3. 补充运行时辅助代码（目录创建、后端配置等）
        
        Args:
            response_text: 模型返回的原始文本
        
        Returns:
            经过增强处理的可执行Python代码
        """
        import re
        
        # 提取markdown代码块，优先选择最长匹配以应对多块情况
        pattern = r'```(?:python)?\s*(.*?)```'
        matches = re.findall(pattern, response_text, re.DOTALL)
        
        if matches:
            code = max(matches, key=len).strip()
        else:
            # 未检测到围栏标记时，将全文视为纯代码
            code = response_text.strip()
        
        # 统一换行符格式，消除跨平台差异
        code = code.replace('\r\n', '\n').replace('\r', '\n')
        
        # 注入插桩逻辑，该步骤在所有其他修改之前执行
        # 确保调试机制优先于业务代码加载
        code = self.inject_instrumentation(code)
        
        # 检测输出路径引用，自动补充目录创建逻辑
        # 避免因目录不存在导致的文件写入失败
        if 'pred_results/' in code:
            if 'makedirs' not in code and 'mkdir' not in code.lower():
                lines = code.split('\n')
                insert_pos = 0
                
                # 在import语句之后插入目录创建代码
                for i, line in enumerate(lines):
                    if line.strip().startswith(('import ', 'from ')):
                        insert_pos = i + 1
                
                dir_code = [
                    '',
                    '# 确保输出目录存在',
                    "import os",
                    "os.makedirs('pred_results', exist_ok=True)",
                    ''
                ]
                
                lines = lines[:insert_pos] + dir_code + lines[insert_pos:]
                code = '\n'.join(lines)
        
        # 为matplotlib设置非交互式后端，规避并发环境下的GUI资源竞争
        # Agg后端仅生成图像文件，不依赖显示设备
        if 'matplotlib.pyplot' in code and 'matplotlib.use' not in code:
            # 找到matplotlib.pyplot的导入位置
            lines = code.split('\n')
            
            for i, line in enumerate(lines):
                # 匹配 "import matplotlib.pyplot as plt" 或 "import matplotlib.pyplot"
                if 'import matplotlib.pyplot' in line:
                    # 在这一行之前插入后端设置
                    backend_code = [
                        'import matplotlib',
                        "matplotlib.use('Agg')  # 设置后端避免并发内存问题"
                    ]
                    lines = lines[:i] + backend_code + lines[i:]
                    code = '\n'.join(lines)
                    break
        
        return code
    
    def setup_workspace(
        self,
        task_ids: Optional[List[int]] = None,
        prompt_type: str = "domain_and_dataset",
        force_overwrite: bool = False
    ):
        """
        初始化评测工作空间
        
        为指定任务创建目录结构并提取生成代码
        
        Args:
            task_ids: 需要设置的任务ID列表，None表示全部任务
            prompt_type: 使用的提示词类型
            force_overwrite: 是否覆盖已存在的代码文件
        """
        # 加载模型响应数据
        if self.responses_df is None:
            if not self.responses_path.exists():
                raise FileNotFoundError(f"响应文件不存在：{self.responses_path}")
            self.responses_df = pd.read_csv(self.responses_path)
        
        # 确定待处理任务
        if task_ids is None:
            task_ids = list(self.task_configs.keys())
        
        print(f"\n开始初始化工作空间...")
        print(f"任务数量：{len(task_ids)}")
        print(f"提示词类型：{prompt_type}")
        print(f"覆盖模式：{'是' if force_overwrite else '否'}\n")
        
        success_count = 0
        skip_count = 0
        error_count = 0
        
        for task_id in task_ids:
            try:
                task_dir = self.workspace_root / str(task_id)
                task_dir.mkdir(parents=True, exist_ok=True)
                
                # 创建子目录
                (task_dir / "generated").mkdir(exist_ok=True)
                (task_dir / "outputs" / prompt_type / "results").mkdir(parents=True, exist_ok=True)
                
                # 提取并保存生成代码
                code_path = task_dir / "generated" / f"{prompt_type}.py"
                
                if code_path.exists() and not force_overwrite:
                    skip_count += 1
                    continue
                
                # 从响应数据中查找对应记录
                mask = (
                    (self.responses_df['task_id'] == task_id) &
                    (self.responses_df['prompt_type'] == prompt_type) &
                    (self.responses_df['error_info'].isna() | (self.responses_df['error_info'] == ''))
                )
                
                matching_responses = self.responses_df[mask]
                
                if matching_responses.empty:
                    print(f"警告：任务{task_id}未找到有效的{prompt_type}响应")
                    error_count += 1
                    continue
                
                # 取第一条有效响应（三次重复中的任意一次）
                response_text = matching_responses.iloc[0]['response_content']
                code = self.extract_code_from_response(response_text)
                
                # 写入代码文件
                with open(code_path, 'w', encoding='utf-8', newline='\n') as f:
                    f.write(code)
                
                # 创建任务级配置文件
                config = {
                    'task_id': task_id,
                    'title': self.task_configs[task_id]['title'],
                    'categories': self.task_configs[task_id]['categories'],
                    'is_opensource': self.task_configs[task_id]['is_opensource'],
                    'prompt_type': prompt_type,
                    'code_extracted': True,
                    'execution_status': 'pending'
                }
                
                config_path = task_dir / "evaluation.json"
                with open(config_path, 'w', encoding='utf-8') as f:
                    json.dump(config, f, indent=2, ensure_ascii=False)
                
                success_count += 1
            
            except Exception as e:
                print(f"错误：任务{task_id}设置失败 - {e}")
                error_count += 1
        
        print(f"\n工作空间初始化完成")
        print(f"成功：{success_count}")
        print(f"跳过：{skip_count}")
        print(f"失败：{error_count}")
    
    def get_task_info(self, task_id: int) -> Dict:
        """获取任务完整信息"""
        if task_id not in self.task_configs:
            raise ValueError(f"无效的任务ID：{task_id}")
        
        return self.task_configs[task_id].copy()
    
    def list_pending_tasks(self, prompt_type: str = "domain_and_dataset") -> List[int]:
        """列出待执行的任务"""
        pending = []
        
        for task_id in self.task_configs.keys():
            task_dir = self.workspace_root / str(task_id)
            config_path = task_dir / "evaluation.json"
            
            if not config_path.exists():
                continue
            
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            if config.get('execution_status') == 'pending':
                pending.append(task_id)
        
        return pending
```

该模块承载三项关键能力。首要职责是从CSV应答记录中批量提取模型编写的Python程序，借助正则匹配定位Markdown围栏标记的边界位置并剥离格式包装，获得可直接执行的源码文本。其次是建立任务元数据的结构化索引，将基准数据集中的方法论归属、技术栈标识等属性字段解析入库，为随后的多维度筛选操作提供查询基础。

索引构建环节需协调一处表示层级的不一致：原始数据采用完整术语标注分析类别（诸如`Determining how places are related`），但筛选接口识别的是简化缩写（对应`DR`）。模块通过内置的双向映射机制完成词汇转换，令外部调用无需介入底层编码细节。

代码提取流程包含三个层次的增强处理。最先执行的是插桩逻辑的植入——将上下文感知的异常钩子与函数监控装饰器嵌入程序开头，确保调试基础设施优先于业务代码加载。随后针对输出路径缺失风险，在侦测到`pred_results`引用时自动补充目录创建语句。最后应对可视化库的并发冲突，为`matplotlib`强制指定非交互式后端，消除GUI资源竞争隐患。上述转换均于写入文件前集中完成，规避运行期动态干预引入的额外开销。

**2. `code_executor.py` - 并发调度与环境隔离机制**

```python
# codes/evaluator/code_executor.py
"""
代码执行引擎
负责在隔离环境中运行生成代码并收集执行结果
"""

import subprocess
import json
import time
from pathlib import Path
from typing import Dict, Optional, List
from datetime import datetime
import asyncio
from concurrent.futures import ProcessPoolExecutor
import sys


class ExecutionResult:
    """代码执行结果"""
    
    def __init__(
        self,
        task_id: int,
        success: bool,
        duration: float,
        stdout: str = "",
        stderr: str = "",
        error_type: Optional[str] = None,
        error_message: Optional[str] = None
    ):
        self.task_id = task_id
        self.success = success
        self.duration = duration
        self.stdout = stdout
        self.stderr = stderr
        self.error_type = error_type
        self.error_message = error_message
    
    def to_dict(self) -> Dict:
        """转换为字典格式"""
        return {
            'task_id': self.task_id,
            'success': self.success,
            'duration': self.duration,
            'stdout': self.stdout,
            'stderr': self.stderr,
            'error_type': self.error_type,
            'error_message': self.error_message,
            'timestamp': datetime.now().isoformat()
        }


class CodeExecutor:
    """代码执行引擎"""
    
    def __init__(self, workspace_root="evaluation_workspace", timeout=300, max_workers=4):
        self.workspace_root = Path(workspace_root)
        self.timeout = timeout
        self.max_workers = max_workers
        
        # 加载解释器配置
        config_path = Path("codes/evaluator_config.json")
        if not config_path.exists():
            raise FileNotFoundError(
                "解释器配置文件不存在！请先运行: python codes/setup_evaluation_env.py"
            )
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        self.opensource_interpreter = config['opensource_interpreter']
        self.arcgis_interpreter = config.get('arcgis_interpreter')
        
        print(f"开源环境解释器: {self.opensource_interpreter}")
        if self.arcgis_interpreter:
            print(f"ArcGIS环境解释器: {self.arcgis_interpreter}")
        else:
            print("ArcGIS环境未配置（闭源任务将无法执行）")
        
        # 验证解释器
        self._verify_interpreter(self.opensource_interpreter)
    
    def _get_interpreter_for_task(self, task_id: int) -> str:
        """根据任务类型选择合适的解释器"""
        task_dir = self.workspace_root / str(task_id)
        config_path = task_dir / "evaluation.json"
        
        if not config_path.exists():
            # 如果没有配置文件，默认用开源解释器
            return self.opensource_interpreter
        
        with open(config_path, 'r', encoding='utf-8') as f:
            task_config = json.load(f)
        
        if task_config.get('is_opensource', True):
            return self.opensource_interpreter
        else:
            if not self.arcgis_interpreter:
                raise RuntimeError(
                    f"任务{task_id}需要ArcGIS环境，但未配置ArcGIS解释器"
                )
            return self.arcgis_interpreter
    
    def _verify_interpreter(self, interpreter_path: str):
        """验证Python解释器是否可用"""
        try:
            result = subprocess.run(
                [interpreter_path, '--version'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode != 0:
                raise RuntimeError(f"解释器不可用：{interpreter_path}")
            
            print(f"  ✓ {result.stdout.strip()}")
        
        except Exception as e:
            raise RuntimeError(f"解释器验证失败 {interpreter_path}: {e}")
    
    def execute_single_task(
        self,
        task_id: int,
        prompt_type: str = "domain_and_dataset"
    ) -> ExecutionResult:
        """
        执行单个任务的生成代码
        
        Args:
            task_id: 任务ID
            prompt_type: 提示词类型
        
        Returns:
            执行结果对象
        """
        task_dir = self.workspace_root / str(task_id)
        code_path = task_dir / "generated" / f"{prompt_type}.py"
        output_dir = task_dir / "outputs" / prompt_type
        log_path = output_dir / "execution.log"
        
        # 检查代码文件是否存在
        if not code_path.exists():
            return ExecutionResult(
                task_id=task_id,
                success=False,
                duration=0,
                error_type="FileNotFoundError",
                error_message=f"代码文件不存在：{code_path}"
            )
        
        # 确保输出目录存在
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 根据任务选择解释器
        try:
            interpreter = self._get_interpreter_for_task(task_id)
        except RuntimeError as e:
            return ExecutionResult(
                task_id=task_id,
                success=False,
                duration=0,
                error_type="EnvironmentError",
                error_message=str(e)
            )

        # 转换为绝对路径（关键修复）
        code_path = code_path.resolve()

        # 构建执行命令
        cmd = [interpreter, str(code_path)]  # 使用选择的解释器
        
        start_time = time.time()
        
        try:
            # 在任务目录下执行代码
            result = subprocess.run(
                cmd,
                cwd=str(task_dir),  # 关键：设置工作目录为任务目录
                capture_output=True,
                text=True,
                timeout=self.timeout,
                encoding='utf-8',
                errors='replace'  # 处理编码错误
            )
            
            duration = time.time() - start_time
            
            # 将输出写入日志
            with open(log_path, 'w', encoding='utf-8') as f:
                f.write(f"=== 执行时间：{datetime.now().isoformat()} ===\n")
                f.write(f"=== 耗时：{duration:.2f}秒 ===\n\n")
                f.write("=== STDOUT ===\n")
                f.write(result.stdout)
                f.write("\n\n=== STDERR ===\n")
                f.write(result.stderr)
            
            # 判断执行是否成功
            success = result.returncode == 0
            error_type = None
            error_message = None
            
            if not success:
                error_type = "RuntimeError"
                error_message = result.stderr[:500] if result.stderr else "未知错误"
            
            return ExecutionResult(
                task_id=task_id,
                success=success,
                duration=duration,
                stdout=result.stdout,
                stderr=result.stderr,
                error_type=error_type,
                error_message=error_message
            )
        
        except subprocess.TimeoutExpired:
            duration = time.time() - start_time
            return ExecutionResult(
                task_id=task_id,
                success=False,
                duration=duration,
                error_type="TimeoutError",
                error_message=f"执行超时（>{self.timeout}秒）"
            )
        
        except Exception as e:
            duration = time.time() - start_time
            return ExecutionResult(
                task_id=task_id,
                success=False,
                duration=duration,
                error_type=type(e).__name__,
                error_message=str(e)
            )
    
    def execute_batch(
        self,
        task_ids: List[int],
        prompt_type: str = "domain_and_dataset",
        use_concurrent: bool = True
    ) -> List[ExecutionResult]:
        """
        批量执行任务
        
        Args:
            task_ids: 任务ID列表
            prompt_type: 提示词类型
            use_concurrent: 是否使用并发执行
        
        Returns:
            执行结果列表
        """
        print(f"\n开始执行代码验证...")
        print(f"任务数量：{len(task_ids)}")
        print(f"并发模式：{'是' if use_concurrent else '否'}")
        print(f"超时设置：{self.timeout}秒\n")
        
        results = []
        
        if use_concurrent and len(task_ids) > 1:
            # 并发执行
            with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {
                    executor.submit(self.execute_single_task, tid, prompt_type): tid
                    for tid in task_ids
                }
                
                from concurrent.futures import as_completed
                from tqdm import tqdm
                
                for future in tqdm(as_completed(futures), total=len(task_ids), desc="执行进度"):
                    try:
                        result = future.result()
                        results.append(result)
                        
                        # 更新任务配置
                        self._update_task_status(result, prompt_type)
                    
                    except Exception as e:
                        task_id = futures[future]
                        print(f"任务{task_id}执行异常：{e}")
                        results.append(ExecutionResult(
                            task_id=task_id,
                            success=False,
                            duration=0,
                            error_type="ExecutorError",
                            error_message=str(e)
                        ))
        
        else:
            # 串行执行
            from tqdm import tqdm
            
            for task_id in tqdm(task_ids, desc="执行进度"):
                result = self.execute_single_task(task_id, prompt_type)
                results.append(result)
                self._update_task_status(result, prompt_type)
        
        # 输出统计信息
        success_count = sum(1 for r in results if r.success)
        print(f"\n执行完成")
        print(f"成功：{success_count}/{len(results)}")
        print(f"失败：{len(results) - success_count}")
        
        return results
    
    def _update_task_status(self, result: ExecutionResult, prompt_type: str):
        """更新任务执行状态"""
        task_dir = self.workspace_root / str(result.task_id)
        config_path = task_dir / "evaluation.json"
        
        if not config_path.exists():
            return
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            config['execution_status'] = 'success' if result.success else 'failed'
            config['execution_duration'] = result.duration
            config['error_type'] = result.error_type
            config['error_message'] = result.error_message
            config['last_execution'] = datetime.now().isoformat()
            
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
        
        except Exception as e:
            print(f"警告：更新任务{result.task_id}状态失败 - {e}")
```

此组件封装了脚本执行的底层细节，核心技术要点包括进程池管理、解释器动态选择、路径解析策略三个层面。

进程池采用`ProcessPoolExecutor`实现并发调度，借助信号量机制限定同步运行的作业上限。每个子进程在独立内存空间中启动Python解释器，加载待测脚本并捕获标准输出与错误流。这种隔离机制确保单个任务的崩溃不会影响整体执行进度。

解释器选择基于任务元数据中的技术栈标识自动切换。开源任务路由至虚拟环境的Python实例，闭源任务则定向到ArcGIS Pro内置解释器。路径映射关系从配置文件动态读取，支持灵活调整而无需修改代码。

路径处理层面需协调两类需求：主进程通过绝对路径定位待执行脚本避免查找歧义，子进程工作目录设为任务文件夹使内部相对引用（如`dataset/xxx.geojson`）正确解析。前者通过`Path.resolve()`转换实现，后者依赖`subprocess.run`的`cwd`参数设定。

**3. `evaluation_reporter.py` - 多维度统计与报告输出**

```python
# codes/evaluator/evaluation_reporter.py
"""
评估报告生成器
汇总执行结果并生成分析报告
"""

import json
import pandas as pd
from pathlib import Path
from typing import List, Dict
from datetime import datetime
from collections import defaultdict


class EvaluationReporter:
    """评估报告生成器"""
    
    def __init__(self, workspace_root: str = "evaluation_workspace"):
        self.workspace_root = Path(workspace_root)
    
    def generate_summary_report(
        self,
        output_path: str = "results/evaluation_summary.json"
    ) -> Dict:
        """
        生成评估摘要报告
        
        Returns:
            包含统计信息的字典
        """
        # 收集所有任务的执行状态
        task_results = []
        
        for task_dir in sorted(self.workspace_root.iterdir()):
            if not task_dir.is_dir():
                continue
            
            config_path = task_dir / "evaluation.json"
            if not config_path.exists():
                continue
            
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                task_results.append(config)
            except Exception as e:
                print(f"警告：读取{config_path}失败 - {e}")
        
        if not task_results:
            return {'error': '未找到任何评估结果'}
        
        # 统计分析
        total = len(task_results)
        success = sum(1 for r in task_results if r.get('execution_status') == 'success')
        failed = sum(1 for r in task_results if r.get('execution_status') == 'failed')
        pending = sum(1 for r in task_results if r.get('execution_status') == 'pending')
        
        # 按类别统计
        category_stats = defaultdict(lambda: {'total': 0, 'success': 0, 'failed': 0})
        for result in task_results:
            for cat in result.get('categories', []):
                category_stats[cat]['total'] += 1
                if result.get('execution_status') == 'success':
                    category_stats[cat]['success'] += 1
                elif result.get('execution_status') == 'failed':
                    category_stats[cat]['failed'] += 1
        
        # 错误类型统计
        error_types = defaultdict(int)
        for result in task_results:
            if result.get('error_type'):
                error_types[result['error_type']] += 1
        
        # 构建报告
        report = {
            'generated_at': datetime.now().isoformat(),
            'overall': {
                'total_tasks': total,
                'success': success,
                'failed': failed,
                'pending': pending,
                'success_rate': f"{success/total*100:.2f}%" if total > 0 else "0%"
            },
            'by_category': {
                cat: {
                    'total': stats['total'],
                    'success': stats['success'],
                    'failed': stats['failed'],
                    'success_rate': f"{stats['success']/stats['total']*100:.2f}%" if stats['total'] > 0 else "0%"
                }
                for cat, stats in category_stats.items()
            },
            'error_distribution': dict(error_types),
            'opensource_vs_closed': self._analyze_opensource_split(task_results)
        }
        
        # 保存报告
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"\n评估报告已保存至：{output_path}")
        
        return report
    
    def _analyze_opensource_split(self, task_results: List[Dict]) -> Dict:
        """分析开源/闭源任务的执行差异"""
        opensource = {'total': 0, 'success': 0}
        closed = {'total': 0, 'success': 0}
        
        for result in task_results:
            if result.get('is_opensource'):
                opensource['total'] += 1
                if result.get('execution_status') == 'success':
                    opensource['success'] += 1
            else:
                closed['total'] += 1
                if result.get('execution_status') == 'success':
                    closed['success'] += 1
        
        return {
            'opensource': {
                **opensource,
                'success_rate': f"{opensource['success']/opensource['total']*100:.2f}%" if opensource['total'] > 0 else "0%"
            },
            'closed_source': {
                **closed,
                'success_rate': f"{closed['success']/closed['total']*100:.2f}%" if closed['total'] > 0 else "0%"
            }
        }
    
    def print_summary(self, report: Dict):
        """在终端打印摘要信息"""
        print("\n" + "="*60)
        print("评估结果摘要")
        print("="*60)
        
        overall = report['overall']
        print(f"\n整体统计：")
        print(f"  总任务数：{overall['total_tasks']}")
        print(f"  成功：{overall['success']} ({overall['success_rate']})")
        print(f"  失败：{overall['failed']}")
        print(f"  待执行：{overall['pending']}")
        
        print(f"\n按类别统计：")
        for cat, stats in report['by_category'].items():
            cat_name = {
                'DP': '模式检测',
                'DR': '位置关联',
                'F': '路径优化',
                'M': '形态测量',
                'S': '空间插值',
                'U': '位置理解'
            }.get(cat, cat)
            
            print(f"  {cat_name}（{cat}）：{stats['success']}/{stats['total']} ({stats['success_rate']})")
        
        if report.get('error_distribution'):
            print(f"\n常见错误类型：")
            sorted_errors = sorted(
                report['error_distribution'].items(),
                key=lambda x: x[1],
                reverse=True
            )
            for error_type, count in sorted_errors[:5]:
                print(f"  {error_type}：{count}次")
```

报告生成器遍历工作空间收集各任务的执行状态，按成功率、类别表现、错误类型分布等维度聚合数据。统计逻辑支持历史累积模式，即纳入所有已完成验证的任务而非仅限当次运行范围，便于跟踪长期进展。

类别统计需处理一对多映射关系——单个任务可能同时归属多个方法论分类，聚合时会被计入所有相关维度。错误类型分布通过字典计数实现，按频次降序排列后展示高发问题，辅助快速定位系统性缺陷。

输出格式包括JSON结构化数据和终端可读摘要两种形式。前者保留完整统计细节供程序化分析，后者将缩写映射为中文描述并截取高频错误TOP5，优化人工审阅体验。

**4. 执行参数配置与启动**

主控脚本中通过调整筛选条件和执行参数控制验证范围：

```python
# codes/run_evaluation.py
"""
GeoCode Validator - 代码执行与评估系统
自动化验证模型生成代码的可执行性
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from evaluator.workspace_manager import WorkspaceManager
from evaluator.code_executor import CodeExecutor
from evaluator.evaluation_reporter import EvaluationReporter


def main():
    """主执行流程"""
    print("="*60)
    print("GeoCode Validator - 代码执行评估系统")
    print("="*60)
    
    # 初始化管理器
    workspace_mgr = WorkspaceManager()
    
    # 第一步：筛选任务
    # 示例：仅执行开源任务中DR和F类别的任务
    print("\n正在筛选任务...")
    task_ids = workspace_mgr.filter_tasks(
        opensource_only=True,
        categories=['DR', 'F']
    )
    
    print(f"筛选结果：共{len(task_ids)}个任务")
    print(f"任务ID：{task_ids}\n")
    
    if not task_ids:
        print("未找到符合条件的任务，退出")
        return
    
    # 第二步：初始化工作空间
    workspace_mgr.setup_workspace(
        task_ids=task_ids,
        prompt_type="domain_and_dataset",
        force_overwrite=True
    )
    
    # 第三步：执行代码
    executor = CodeExecutor(
        timeout=300,  # 5分钟超时
        max_workers=4  # 并发数
    )
    
    results = executor.execute_batch(
        task_ids=task_ids,
        prompt_type="domain_and_dataset",
        use_concurrent=True  # 启用并发
    )
    
    # 第四步：生成报告
    reporter = EvaluationReporter()
    report = reporter.generate_summary_report()
    reporter.print_summary(report)
    
    print("\n评估流程完成！")


if __name__ == "__main__":
    main()
```

筛选接口支持三种组合方式。按技术栈过滤时传入`opensource_only=True`限定开源任务；按方法论维度传入类别缩写列表如`categories=['DR', 'F']`；按编号明确指定时使用`task_ids=[1, 5, 10]`。三类条件可叠加使用取交集。

执行器初始化时需设定两项关键参数。`timeout`控制单任务最长运行时限，默认300秒适配大部分地理处理操作，复杂的栅格叠加分析可酌情延长。`max_workers`决定并发度，建议设为CPU核心数的1-2倍，过高可能引发内存竞争导致`matplotlib`渲染失败。

并发开关通过`use_concurrent`参数切换。启用时采用进程池并行执行显著缩短总耗时，但在调试阶段或遭遇并发异常时可临时改为串行模式逐个排查问题。

**常见问题与诊断策略**

代码验证过程中遭遇的异常可归纳为三类核心模式：数据资源定位失败、依赖库缺失或版本不兼容、工具链特性引发的运行时冲突。针对这些问题的诊断与修复需要结合执行日志、错误堆栈及任务特性综合判断。

**数据集文件命名偏差**

最高频的失败原因是脚本引用的文件路径与实际布局不匹配。模型严格按照提示词中的信息生成路径字符串，但基准数据集在文档记载与物理文件之间存在系统性偏离。

错误模式涵盖多个层面。部分任务涉及词汇简化或单复数调整（如`CensusBlock.geojson`在实际环境中被简写为`block.geojson`），栅格数据的大小写规范未能统一（`landCover.tif`在文档中标注为全小写形式`landcover.tif`），个别案例出现拼写疏漏与分隔符差异（`berlin-neighbourhoods.geojson`被记录成`berling_neighbourhoods.geojson`）。极端情况下甚至发生目录路径本身配置错误，导致整个数据集无法定位。

诊断方法是运行检查脚本扫描工作空间，对照数据集描述与实际文件清单的差异：

```python
# 占位符：文件名一致性检查脚本
```

发现不匹配后直接修正实际文件名使之符合描述，而非修改生成代码中的路径引用。这是因为批量重命名文件比逐个编辑脚本更高效，且保持了与提示词描述的对应关系。

**依赖库覆盖不全**

环境配置阶段虽然安装了主流地理处理工具，但部分专用库需根据任务特性按需补充。错误信息通常直接指明缺失模块：

```
ModuleNotFoundError: No module named 'pykrige'
```

此类问题通过在虚拟环境中追加安装即可解决：

```bash
venv_gis_opensource/Scripts/pip.exe install pykrige
```

值得注意的是某些元包（如`pysal`）会自动携带多个子包，安装时应选择顶层包而非逐个安装依赖项。

**API版本迁移滞后**

开源库在重构过程中常调整模块结构或弃用旧接口，而模型训练数据可能包含过时的API用法。典型案例是`pysal`库在v2.0拆分为多个独立子包后，原有的`pysal.lib.weights`路径已失效，正确导入应改为`from libpysal.weights import ...`。

这类错误的堆栈信息会明确显示导入路径无效：

```
ModuleNotFoundError: No module named 'pysal.lib.weights'
```

修复方式是参照当前库版本的官方文档，将生成代码中的导入语句替换为新的模块路径。若此类问题频繁出现，说明训练数据未及时跟进生态演进，需在提示工程阶段补充版本约束信息。

**并发环境资源竞争**

启用并发模式时，`matplotlib`在处理大尺寸栅格数据的可视化环节可能触发内存分配异常。尽管数据规模本身不大（如29MB数组），但GUI后端在多进程环境下存在状态竞争，导致看似充足的内存无法被正常申请。

错误信息表现为：

```
numpy._core._exceptions._ArrayMemoryError: Unable to allocate 29.1 MiB
```

系统已在代码提取阶段自动注入`matplotlib.use('Agg')`后端配置，理论上可规避该问题。若仍偶发异常，可临时关闭并发模式改为串行执行，或降低`max_workers`参数减轻资源压力。

**日志分析与问题定位**

每个任务的执行日志存储于`outputs/{prompt_type}/execution.log`，包含完整的标准输出与错误流。STDOUT部分记录程序打印的进度信息，STDERR部分保留异常堆栈。诊断时应优先检查STDERR末尾的错误类型和触发位置，结合生成代码的对应行号定位根因。

对于成功执行的任务，可通过检查`pred_results`目录验证输出文件是否符合预期。若文件生成但内容异常（如空白图像、数值全零的栅格），说明算法逻辑存在问题而非环境配置缺陷，需要进一步分析代码实现与数据特性的匹配度。

