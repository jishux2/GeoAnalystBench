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
                # 明确指定newline='\n'防止Windows平台自动转换换行符
                # 若CSV本身为CRLF格式，pandas读取后内部已是\r\n
                # 文本模式默认行为会再次转换导致\r\r\n，引发格式异常
                # （注：就输出场景而言，newline=''与当前设置效果一致，均可阻止平台相关的换行符映射）
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