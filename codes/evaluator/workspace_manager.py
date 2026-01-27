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
    
    # 空间分析方法论分类的标准描述
    TASK_CATEGORIES = {
        'DP': 'detecting and quantifying patterns',
        'DR': 'determining how places are related',
        'F': 'finding the best locations and paths',
        'M': 'measuring size, shape, and distribution',
        'S': 'spatial interpolation and predictive modeling',
        'U': 'understanding where'
    }
    
    # 数据集中采用完整描述，通过反向查找建立缩写索引
    # 首字母大写是为了匹配CSV中"Making predictions"这类格式
    CATEGORY_REVERSE_MAP = {
        desc.capitalize(): abbr 
        for abbr, desc in TASK_CATEGORIES.items()
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
            
            # 从三个类别列中提取方法论标签
            # CSV存储的是完整英文描述，需转换为缩写形式以便筛选
            categories = []
            for cat_col in ['Task Categories1', 'Task Categories2', 'Task Categories3']:
                if pd.notna(row.get(cat_col)):
                    cat_full = row[cat_col].strip()
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
import os

# Read output directory from environment variable (set by executor)
EVAL_OUTPUT_DIR = os.environ.get('EVAL_OUTPUT_DIR', '.')

# ============================================================
# Context-aware object summarization for error diagnostics
# ============================================================

def extract_accessed_fields(code_line, var_name):
    """
    Extract field accesses from a line of code for a given variable
    
    Identifies three access patterns:
    - Bracket notation: var_name['field'] or var_name["field"]
    - Dot notation: var_name.field (excludes method calls)
    """
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
    """
    Intelligently summarize objects, prioritizing fields accessed in code
    
    For pandas Series, extracts field access patterns from the error-triggering
    code line and displays those fields first, making diagnostics more relevant
    """
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
            
            # Display accessed fields first (more relevant for debugging)
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


# ============================================================
# Global exception hook with stack frame analysis
# ============================================================

def capture_exception(exc_type, exc_value, exc_traceback):
    """
    Capture exception with detailed context including local variables
    
    Traverses the call stack to collect code context and variable states,
    applying smart summarization to make diagnostics actionable
    """
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
    
    trace_file = os.path.join(EVAL_OUTPUT_DIR, 'error_trace.json')
    with open(trace_file, 'w', encoding='utf-8') as f:
        json.dump(context, f, indent=2, ensure_ascii=False)
    
    # Call the original exception handler to print traceback
    sys.__excepthook__(exc_type, exc_value, exc_traceback)

# Register the custom exception hook
sys.excepthook = capture_exception


# ============================================================
# Function call monitoring decorator
# ============================================================

def monitor_call(func_name):
    """
    Decorator to capture function arguments when errors occur
    
    Wraps third-party library functions to log their invocation details,
    providing visibility into failed API calls that try-except might mask
    """
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
                
                detail_file = os.path.join(EVAL_OUTPUT_DIR, 'call_details.json')
                
                # Maintain array structure (read-modify-write)
                if os.path.exists(detail_file):
                    with open(detail_file, 'r', encoding='utf-8') as f:
                        try:
                            data = json.load(f)
                            if not isinstance(data, list):
                                data = []
                        except json.JSONDecodeError:
                            data = []
                else:
                    data = []
                
                data.append(detail)
                
                with open(detail_file, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                
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
        
        # 尝试提取markdown代码块，支持带python标记或纯反引号两种格式
        pattern = r'```(?:python)?\s*(.*?)```'
        matches = re.findall(pattern, response_text, re.DOTALL)
        
        if matches:
            # 取最长的代码块（避免误提取示例片段）
            code = max(matches, key=len).strip()
        else:
            # 未找到代码块标记，假定全文为代码
            code = response_text.strip()
        
        # Windows环境下CSV可能引入CRLF，需统一转换为LF
        # 避免文本模式写入时的双重转换导致`\r\r\n`异常
        code = code.replace('\r\n', '\n').replace('\r', '\n')
        
        # 注入插桩逻辑，该步骤在所有其他修改之前执行
        # 确保调试机制优先于业务代码加载
        code = self.inject_instrumentation(code)
        
        # 模型生成代码通常直接写入相对路径，但不会主动创建目录
        # 检测到pred_results引用时自动注入目录创建逻辑
        if 'pred_results/' in code:
            if 'makedirs' not in code and 'mkdir' not in code.lower():
                lines = code.split('\n')
                insert_pos = 0
                
                # 定位import语句块的末尾
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
        
        # matplotlib在并发环境下处理大型栅格数据时可能出现内存分配失败
        # 强制使用Agg后端可规避GUI组件带来的额外开销和状态竞争
        if 'matplotlib.pyplot' in code and 'matplotlib.use' not in code:
            # 找到matplotlib.pyplot的导入位置
            lines = code.split('\n')
            
            for i, line in enumerate(lines):
                # 匹配 "import matplotlib.pyplot as plt" 或 "import matplotlib.pyplot"
                if 'import matplotlib.pyplot' in line:
                    # 在这一行之前插入后端设置
                    backend_code = [
                        'import matplotlib',
                        "matplotlib.use('Agg')  # 非交互式后端，适配批处理场景"
                    ]
                    lines = lines[:i] + backend_code + lines[i:]
                    code = '\n'.join(lines)
                    break
        
        return code