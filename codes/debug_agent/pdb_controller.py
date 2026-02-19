"""
PDB会话控制器
封装与PDB子进程的交互逻辑
"""

import subprocess
import threading
import queue
import os
from typing import Optional, List, Tuple
from pathlib import Path


class PdbSessionController:
    """PDB会话控制器"""
    
    PDB_PROMPT = '(Pdb) '
    READ_TIMEOUT = 30
    
    def __init__(
        self,
        process: subprocess.Popen,
        initial_output: str = ""
    ):
        """
        初始化控制器
        
        Args:
            process: PDB子进程
            initial_output: 启动时的初始输出
        """
        self._process = process
        self.initial_output = initial_output
        
        self._output_queue: queue.Queue = queue.Queue()
        self._reader_thread: Optional[threading.Thread] = None
        
        self._start_reader()
    
    @classmethod
    def start_with_pdb(
        cls,
        script_path: str,
        cwd: str,
        interpreter: str
    ) -> 'PdbSessionController':
        """
        以PDB模式启动脚本（单步调试）
        
        Args:
            script_path: 脚本路径
            cwd: 工作目录
            interpreter: Python解释器路径
        
        Returns:
            初始化完成的控制器实例
        """
        command = [interpreter, '-m', 'pdb', script_path]
        
        process = cls._create_process(command, cwd)
        controller = cls(process)
        
        controller.initial_output, _ = controller._read_until_prompt()
        
        return controller
    
    @classmethod
    def start_script(
        cls,
        script_path: str,
        cwd: str,
        interpreter: str
    ) -> 'PdbSessionController':
        """
        直接启动脚本（用于事后调试，脚本需自带钩子）
        
        Args:
            script_path: 脚本路径
            cwd: 工作目录
            interpreter: Python解释器路径
        
        Returns:
            初始化完成的控制器实例
        """
        command = [interpreter, script_path]
        
        process = cls._create_process(command, cwd)
        controller = cls(process)
        
        controller.initial_output, _ = controller._read_until_prompt()
        
        return controller
    
    @classmethod
    def _create_process(cls, command: List[str], cwd: str) -> subprocess.Popen:
        """创建子进程"""
        env = os.environ.copy()
        env["PYTHONUTF8"] = "1"
        
        return subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8',
            env=env,
            cwd=cwd,
            bufsize=1
        )
    
    def send_command(self, command: str) -> str:
        """
        发送PDB命令并获取响应
        
        Args:
            command: PDB命令
        
        Returns:
            命令执行结果
        """
        if self._process.poll() is not None:
            return "[SESSION_TERMINATED] The debugging session has ended."
        
        try:
            self._process.stdin.write(command + '\n')
            self._process.stdin.flush()
        except (OSError, BrokenPipeError) as e:
            return f"[ERROR] Communication failed: {e}"
        
        response, is_alive = self._read_until_prompt()
        
        if not is_alive:
            if response:
                return response + "\n[SESSION_TERMINATED]"
            return "[SESSION_TERMINATED]"
        
        return response
    
    def execute_code(self, code: str) -> str:
        """
        在当前上下文执行Python代码
        
        Args:
            code: Python代码
        
        Returns:
            执行结果
        """
        escaped_code = repr(code)
        command = f"!exec({escaped_code})"
        return self.send_command(command)
    
    def set_breakpoint_by_context(self, target_code: str) -> str:
        """
        基于代码上下文设置断点
        
        Args:
            target_code: 目标代码片段
        
        Returns:
            设置结果
        """
        list_output = self.send_command('ll')
        
        normalized_target = self._normalize_whitespace(target_code)
        
        for line in list_output.split('\n'):
            parts = line.split(None, 1)
            if len(parts) < 2:
                continue
            
            line_num_str = parts[0].lstrip(' ').rstrip('>')
            
            try:
                line_num = int(line_num_str)
                code_part = parts[1] if len(parts) > 1 else ''
                
                if normalized_target in self._normalize_whitespace(code_part):
                    return self.send_command(f'b {line_num}')
            
            except ValueError:
                continue
        
        return "[NOT_FOUND] Could not locate the target code in current context."
    
    def close(self):
        """
        关闭会话并清理资源
        
        处理嵌套会话：预防性调试中触发异常会进入事后调试模式，
        需要连续发送退出命令才能完全终止进程
        """
        if self._process and self._process.poll() is None:
            for _ in range(2):
                try:
                    self._process.stdin.write('q\n')
                    self._process.stdin.flush()
                    self._process.wait(timeout=0.5)
                    break
                except (OSError, BrokenPipeError):
                    break
                except subprocess.TimeoutExpired:
                    continue
            
            if self._process.poll() is None:
                self._process.terminate()
                try:
                    self._process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    pass
        
        if self._reader_thread and self._reader_thread.is_alive():
            self._reader_thread.join(timeout=1)
    
    def _start_reader(self):
        """启动后台读取线程"""
        self._reader_thread = threading.Thread(
            target=self._reader_loop,
            daemon=True
        )
        self._reader_thread.start()
    
    def _reader_loop(self):
        """后台持续读取子进程输出"""
        while self._process.poll() is None:
            try:
                char = self._process.stdout.read(1)
                if char:
                    self._output_queue.put(char)
                else:
                    break
            except:
                break
    
    def _read_until_prompt(self) -> Tuple[str, bool]:
        """
        读取输出直到遇到PDB提示符或检测到会话终止
        
        采用观测型轮询：以短间隔交替检查输出队列与读取线程状态，
        在会话终止时立即返回，规避无效的阻塞等待
        
        Returns:
            元组(响应内容, 会话是否存活)
        """
        buffer = []
        
        while True:
            if not self._reader_thread.is_alive():
                # 线程已退出，排空队列后返回
                while True:
                    try:
                        buffer.append(self._output_queue.get_nowait())
                    except queue.Empty:
                        break
                return ''.join(buffer).strip(), False
            
            try:
                char = self._output_queue.get(timeout=0.05)
                buffer.append(char)
                
                current = ''.join(buffer)
                if current.endswith(self.PDB_PROMPT):
                    return current[:-len(self.PDB_PROMPT)].strip(), True
            
            except queue.Empty:
                continue
    
    @staticmethod
    def _normalize_whitespace(text: str) -> str:
        """标准化空白字符以便匹配"""
        return ''.join(text.split())