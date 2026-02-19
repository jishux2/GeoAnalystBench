"""
基于子进程的PDB程序化控制器
支持预防性调试和事后调试两种模式
"""

import sys
import subprocess
import os
import tempfile
import threading
import queue
import time
from typing import List, Optional
from pathlib import Path


class SubprocessPdbController:
    """统一的PDB子进程控制器"""
    
    PDB_PROMPT = '(Pdb) '
    
    @classmethod
    def start_proactive(
        cls,
        script_path: str,
        cwd: Optional[str] = None,
        interpreter: Optional[str] = None  # ← 新增参数
    ) -> 'SubprocessPdbController':
        """
        启动预防性调试会话
        
        Args:
            script_path: 要调试的Python脚本路径
            cwd: 工作目录
            interpreter: Python解释器路径，默认使用sys.executable
        
        Returns:
            已初始化的控制器实例
        """
        python_exe = interpreter or sys.executable
        command = [python_exe, '-m', 'pdb', script_path]
        return cls(command, cwd, is_temp=False)
    
    @classmethod
    def start_post_mortem(
        cls,
        script_path: str,
        cwd: Optional[str] = None,
        interpreter: Optional[str] = None  # ← 新增参数
    ) -> 'SubprocessPdbController':
        """
        启动事后调试会话
        
        在脚本中注入异常钩子，发生未捕获异常时启动PDB
        
        Args:
            script_path: 脚本路径
            cwd: 工作目录
            interpreter: Python解释器路径
        
        Returns:
            已初始化的控制器实例
        """
        with open(script_path, 'r', encoding='utf-8') as f:
            original_code = f.read()
        
        hook_code = """
import sys, pdb, traceback
def _pdb_excepthook(exc_type, exc_value, tb):
    traceback.print_exception(exc_type, exc_value, tb)
    pdb.post_mortem(tb)
sys.excepthook = _pdb_excepthook
"""
        
        injected_script = f"{hook_code}\ndef main():\n"
        injected_script += "".join([f"    {line}\n" for line in original_code.splitlines()])
        injected_script += "\nif __name__ == '__main__':\n    main()\n"
        
        temp_file = tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.py', encoding='utf-8')
        temp_file.write(injected_script)
        temp_file.close()
        
        python_exe = interpreter or sys.executable
        command = [python_exe, temp_file.name]
        return cls(command, cwd, is_temp=True, temp_path=temp_file.name)
    
    def __init__(self, command: List[str], cwd: Optional[str], is_temp: bool = False, temp_path: Optional[str] = None):
        """
        初始化PDB子进程和通信机制
        
        Args:
            command: 启动命令
            cwd: 工作目录
            is_temp: 是否使用临时文件
            temp_path: 临时文件路径
        """
        self._command = command
        self._cwd = cwd
        self._is_temp = is_temp
        self._temp_path = temp_path
        
        self._output_queue = queue.Queue()
        self._process = None
        self._reader_thread = None
        
        self._start_process()
    
    def _start_process(self):
        """启动子进程并初始化通信线程"""
        try:
            child_env = os.environ.copy()
            child_env["PYTHONUTF8"] = "1"
            
            self._process = subprocess.Popen(
                self._command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding='utf-8',
                env=child_env,
                cwd=self._cwd,
                bufsize=1
            )
            
            self._reader_thread = threading.Thread(target=self._reader_thread_func, daemon=True)
            self._reader_thread.start()
            
            self.initial_output = self._read_until_prompt()
            
        except Exception as e:
            self.close()
            raise RuntimeError(f"启动PDB子进程失败: {e}")
    
    def _reader_thread_func(self):
        """后台线程持续读取子进程输出"""
        while self._process and self._process.stdout:
            try:
                char = self._process.stdout.read(1)
                if char:
                    self._output_queue.put(char)
                else:
                    break
            except:
                break
    
    def _read_until_prompt(self) -> str:
        """读取输出直到遇到PDB提示符"""
        output = []
        while True:
            try:
                char = self._output_queue.get(timeout=10)
                output.append(char)
                if "".join(output).endswith(self.PDB_PROMPT):
                    break
            except queue.Empty:
                if self._process.poll() is not None:
                    return "".join(output).strip()
                raise TimeoutError("未在10秒内收到PDB响应")
        
        full_output = "".join(output)
        return full_output.removesuffix(self.PDB_PROMPT).strip()
    
    def send_command(self, command: str) -> str:
        """
        发送单条命令并获取响应
        
        Args:
            command: PDB命令
        
        Returns:
            命令执行结果
        """
        if self._process.poll() is not None:
            return "[ERROR] Session terminated"
        
        try:
            self._process.stdin.write(command + '\n')
            self._process.stdin.flush()
        except (OSError, BrokenPipeError) as e:
            return f"[ERROR] Pipe broken: {e}"
        
        response = self._read_until_prompt()
        return response
    
    def send_commands(self, commands: str) -> List[str]:
        """
        发送多条命令
        
        Args:
            commands: 换行分隔的命令列表
        
        Returns:
            所有响应的列表
        """
        command_list = [cmd.strip() for cmd in commands.strip().split('\n') if cmd.strip()]
        responses = []
        
        for cmd in command_list:
            response = self.send_command(cmd)
            responses.append(response)
            if "[ERROR]" in response or self._process.poll() is not None:
                break
        
        return responses
    
    def execute_code_block(self, code_block: str) -> str:
        """
        在当前上下文中执行代码块
        
        Args:
            code_block: 多行Python代码
        
        Returns:
            执行结果
        """
        command = f"!exec({repr(code_block)})"
        return self.send_command(command)
    
    def set_breakpoint_by_context(self, target_code: str) -> str:
        """
        基于代码上下文设置断点
        
        Args:
            target_code: 目标代码片段（包含上下文）
        
        Returns:
            断点设置结果
        """
        normalized_target = self._normalize_code(target_code)
        
        list_output = self.send_command('l')
        
        lines = list_output.split('\n')
        for line in lines:
            parts = line.split(None, 1)
            if len(parts) < 2:
                continue
            
            try:
                line_num = int(parts[0].strip('->*'))
                code = parts[1] if len(parts) > 1 else ''
                
                normalized_code = self._normalize_code(code)
                
                if normalized_target in normalized_code:
                    return self.send_command(f'b {line_num}')
            
            except (ValueError, IndexError):
                continue
        
        return "[ERROR] Target code not found in current context"
    
    @staticmethod
    def _normalize_code(code: str) -> str:
        """标准化代码以忽略缩进差异"""
        return ''.join(code.split())
    
    def close(self):
        """关闭调试会话并清理资源"""
        if self._process and self._process.poll() is None:
            try:
                self._process.stdin.write('q\n')
                self._process.stdin.flush()
                self._process.wait(timeout=2)
            except:
                self._process.terminate()
        
        if self._reader_thread and self._reader_thread.is_alive():
            self._reader_thread.join(timeout=1)
        
        if self._is_temp and self._temp_path and os.path.exists(self._temp_path):
            os.remove(self._temp_path)