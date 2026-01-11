# codes/evaluator/code_executor.py
"""
代码执行引擎
负责在隔离环境中运行生成代码并收集执行结果
"""

import os
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
    
    def __init__(
        self,
        workspace_root: str = "evaluation_workspace",
        timeout: int = 300,
        max_workers: int = 4
    ):
        """
        初始化执行器
        
        Args:
            workspace_root: 工作空间根目录
            timeout: 单个任务超时时间（秒）
            max_workers: 最大并发执行数
        """
        self.workspace_root = Path(workspace_root)
        self.timeout = timeout
        self.max_workers = max_workers
        
        # 从配置文件读取解释器路径，避免硬编码
        # 开源环境由setup_evaluation_env.py创建，闭源环境需手动配置
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
        
        # 验证开源环境的可用性，闭源环境延迟到实际使用时检查
        self._verify_interpreter(self.opensource_interpreter)
    
    def _get_interpreter_for_task(self, task_id: int) -> str:
        """
        根据任务类型选择合适的解释器
        
        开源任务使用独立虚拟环境（geopandas/rasterio等）
        闭源任务依赖ArcGIS Pro内置Python环境
        """
        task_dir = self.workspace_root / str(task_id)
        config_path = task_dir / "evaluation.json"
        
        # 配置文件不存在时默认使用开源解释器
        # 这种情况通常发生在手动创建任务目录的测试场景
        if not config_path.exists():
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
        round_num: int = 1  # 新增轮次参数
    ) -> ExecutionResult:
        """
        执行单个任务的生成代码
        
        Args:
            task_id: 任务ID
            round_num: 当前执行轮次（1-3）
        
        Returns:
            执行结果对象
        """
        task_dir = self.workspace_root / str(task_id)
        
        # 路径调整：引入轮次层级
        code_path = task_dir / "generated" / f"round_{round_num}_code.py"
        output_dir = task_dir / "outputs" / f"round_{round_num}"
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
        
        # 清空本轮次的调试文件
        call_details_file = output_dir / "call_details.json"
        error_trace_file = output_dir / "error_trace.json"
        
        if call_details_file.exists():
            call_details_file.write_text('[]', encoding='utf-8')
        if error_trace_file.exists():
            error_trace_file.unlink()
        
        # 根据任务属性选择对应的Python环境
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
        
        # 转换为绝对路径避免cwd引起的路径重复
        # subprocess设置cwd后，相对路径会基于新工作目录再次拼接
        # 导致"evaluation_workspace/1/evaluation_workspace/1/..."的错误
        code_path = code_path.resolve()
        
        # 构建执行命令
        cmd = [interpreter, str(code_path)]
        
        # 准备环境变量：传递相对于task_dir的输出目录路径
        env = os.environ.copy()
        relative_output_dir = f"outputs/round_{round_num}"
        env['EVAL_OUTPUT_DIR'] = relative_output_dir
        
        start_time = time.time()
        
        try:
            # cwd设为任务目录，使代码内的相对路径（如dataset/xxx.geojson）正确解析
            # 生成代码通常假定与数据集在同一层级，设置工作目录后无需修改代码
            result = subprocess.run(
                cmd,
                cwd=str(task_dir),
                env=env,  # 传递包含输出目录信息的环境变量
                capture_output=True,
                text=True,
                timeout=self.timeout,
                encoding='utf-8',
                errors='replace'  # 遇到无法解码的字节时用替换字符，避免崩溃
            )
            
            duration = time.time() - start_time
            
            # 将完整输出写入日志文件
            with open(log_path, 'w', encoding='utf-8') as f:
                f.write(f"=== 执行时间：{datetime.now().isoformat()} ===\n")
                f.write(f"=== 轮次：{round_num} ===\n")
                f.write(f"=== 耗时：{duration:.2f}秒 ===\n\n")
                f.write("=== STDOUT ===\n")
                f.write(result.stdout)
                f.write("\n\n=== STDERR ===\n")
                f.write(result.stderr)
            
            # 返回码为0表示正常退出，非零表示运行时错误
            success = result.returncode == 0
            error_type = None
            error_message = None
            
            if not success:
                error_type = "RuntimeError"
                # 截取stderr前500字符作为错误摘要，完整信息在日志文件中
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
        round_num: int,  # 改为接收round_num而非prompt_type
        use_concurrent: bool = True
    ) -> List[ExecutionResult]:
        """
        批量执行任务
        
        Args:
            task_ids: 任务ID列表
            round_num: 当前轮次
            use_concurrent: 是否使用并发执行
        
        Returns:
            执行结果列表
        """
        print(f"\n开始执行代码验证...")
        print(f"任务数量：{len(task_ids)}")
        print(f"当前轮次：{round_num}")
        print(f"并发模式：{'是' if use_concurrent else '否'}")
        print(f"超时设置：{self.timeout}秒\n")
        
        results = []
        
        if use_concurrent and len(task_ids) > 1:
            # 使用进程池实现并发执行
            # ProcessPoolExecutor在Windows上通过spawn创建子进程
            # 每个子进程拥有独立的内存空间和Python解释器实例
            with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {
                    executor.submit(self.execute_single_task, tid, round_num): tid
                    for tid in task_ids
                }
                
                from concurrent.futures import as_completed
                from tqdm import tqdm
                
                results_dict = {}  # ← 改用字典
                
                for future in tqdm(as_completed(futures), total=len(task_ids), desc="执行进度"):
                    try:
                        result = future.result()
                        task_id = futures[future]  # ← 获取对应的task_id
                        results_dict[task_id] = result  # ← 按task_id存储
                    
                    except Exception as e:
                        # 捕获execute_single_task自身抛出的意外异常
                        # 正常的执行失败已转换为ExecutionResult，不会走到这里
                        task_id = futures[future]
                        print(f"任务{task_id}执行异常：{e}")
                        results_dict[task_id] = ExecutionResult(
                            task_id=task_id,
                            success=False,
                            duration=0,
                            error_type="ExecutorError",
                            error_message=str(e)
                        )
                
                # 按task_ids顺序重建结果列表
                results = [results_dict[tid] for tid in task_ids]
        
        else:
            # 串行模式：逐个执行任务
            # 适用于调试场景或需要完全隔离执行环境的情况
            from tqdm import tqdm
            
            for task_id in tqdm(task_ids, desc="执行进度"):
                result = self.execute_single_task(task_id, round_num)
                results.append(result)
        
        # 输出统计信息
        success_count = sum(1 for r in results if r.success)
        print(f"\n执行完成")
        print(f"成功：{success_count}/{len(results)}")
        print(f"失败：{len(results) - success_count}")
        
        return results