# codes/benchmark/reference_runner.py
"""
参考代码批量执行器

继承通用批处理框架，为每个任务启动独立的 Python 子进程
执行参考脚本，注入 TASK_INPUT_ROOT 和 TASK_OUTPUT_ROOT
环境变量，收集退出码与输出流并归档至任务目录。

执行前自动清空输出目录以确保产物纯净，执行后扫描产物
清单写入状态记录，为下游评估提供即取即用的素材。
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional

from .batch_runner import BatchRunner, StatusRegistry, TaskStatus
from .task_index import TaskIndex
from .workspace_setup import BenchmarkLayout


def _run_script_sync(
    command: List[str],
    cwd: str,
    env: Dict[str, str],
    timeout: int = 300,
) -> Dict[str, Any]:
    """
    在子进程中同步执行脚本。

    模块级函数，供进程池序列化调用。
    """
    env["PYTHONIOENCODING"] = "utf-8"
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=timeout,
        )
        return {
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
    except subprocess.TimeoutExpired:
        return {
            "returncode": -1,
            "stdout": "",
            "stderr": f"Execution timed out (>{timeout}s)",
        }
    except Exception as e:
        return {
            "returncode": -1,
            "stdout": "",
            "stderr": str(e),
        }


class ReferenceRunner(BatchRunner):
    """
    参考代码执行器

    遍历任务清单，对每个拥有参考脚本的任务执行一次
    带环境变量注入的子进程调用，将执行状态和产物信息
    持久化至任务目录下的状态文件。
    """

    STDOUT_FILENAME = "stdout.txt"
    STDERR_FILENAME = "stderr.txt"

    def __init__(
        self,
        task_index: TaskIndex,
        layout: BenchmarkLayout,
        interpreter: str,
        max_concurrent: int = 4,
        script_timeout: int = 300,
        executor: Optional[ProcessPoolExecutor] = None,
    ):
        """
        Args:
            task_index: 任务索引实例
            layout: 目录布局实例
            interpreter: Python 解释器路径
            max_concurrent: 最大并发数
            script_timeout: 单脚本执行超时（秒）
            executor: 可选的进程池，为 None 时在当前进程中执行
        """
        registry = StatusRegistry(layout.ground_truth_root)
        super().__init__(task_index, registry, max_concurrent)

        self.layout = layout
        self.interpreter = interpreter
        self.script_timeout = script_timeout
        self._executor = executor

    async def _execute_single(
        self, task_id: str, entry: Dict[str, Any]
    ) -> Dict[str, Any]:
        """执行单个任务的参考脚本。"""
        source = entry["source"]

        # 检查参考脚本是否存在
        script_path = self.layout.ground_truth_script(source, task_id)
        if not script_path.exists():
            print(f"  [{source}/{task_id}] skipped (no reference script)")
            return {
                "success": False,
                "status": TaskStatus.NO_CODE,
                "error": "No reference script available",
            }

        # 清空输出目录以确保产物纯净
        output_dir = self.layout.ground_truth_output(source, task_id)
        if output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 构造环境变量
        env = os.environ.copy()
        env["TASK_INPUT_ROOT"] = self.layout.env_input_root(source)
        env["TASK_OUTPUT_ROOT"] = self.layout.env_output_root_ground_truth(
            source, task_id
        )
        env["MPLBACKEND"] = "Agg"

        # 工作目录设为任务目录
        task_dir = self.layout.ground_truth_task_dir(source, task_id)
        command = [self.interpreter, str(script_path.resolve())]

        # 执行
        print(f"  [{source}/{task_id}] running...")
        if self._executor:
            loop = asyncio.get_event_loop()
            run_result = await loop.run_in_executor(
                self._executor,
                _run_script_sync,
                command,
                str(task_dir),
                env,
                self.script_timeout,
            )
        else:
            run_result = _run_script_sync(
                command,
                str(task_dir),
                env,
                self.script_timeout,
            )

        # 持久化输出流
        stdout_path = task_dir / self.STDOUT_FILENAME
        stderr_path = task_dir / self.STDERR_FILENAME
        stdout_path.write_text(run_result["stdout"], encoding="utf-8")
        stderr_path.write_text(run_result["stderr"], encoding="utf-8")

        # 扫描产物清单
        output_files = self._scan_output_files(output_dir)

        success = run_result["returncode"] == 0

        return {
            "success": success,
            "returncode": run_result["returncode"],
            "output_files": output_files,
            "error": run_result["stderr"].strip() if not success else None,
        }

    def _scan_output_files(self, output_dir: Path) -> List[Dict[str, Any]]:
        """
        扫描输出目录下的产物文件。

        返回每个文件的名称、大小和后缀，供状态记录
        和下游评估消费。
        """
        if not output_dir.exists():
            return []

        files = []
        for item in sorted(output_dir.iterdir()):
            if item.is_file():
                files.append({
                    "name": item.name,
                    "size": item.stat().st_size,
                    "suffix": item.suffix.lower(),
                })
        return files