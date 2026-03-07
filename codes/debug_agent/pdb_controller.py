"""
PDB异步会话控制器
封装与PDB子进程的交互逻辑，基于asyncio实现非阻塞通信
"""

import asyncio
import os
from typing import Optional, Tuple
from pathlib import Path


class PdbSessionController:
    """PDB异步会话控制器"""

    PDB_PROMPT = '(Pdb) '

    def __init__(
        self,
        process: asyncio.subprocess.Process,
        initial_output: str = ""
    ):
        """
        初始化控制器

        Args:
            process: asyncio子进程实例
            initial_output: 启动阶段的初始输出
        """
        self._process = process
        self.initial_output = initial_output

    @classmethod
    async def start_with_pdb(
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
        process = await cls._create_process(
            [interpreter, '-m', 'pdb', script_path],
            cwd
        )
        controller = cls(process)
        controller.initial_output, _ = await controller._read_until_prompt()
        return controller

    @classmethod
    async def start_script(
        cls,
        script_path: str,
        cwd: str,
        interpreter: str
    ) -> 'PdbSessionController':
        """
        直接启动脚本（事后调试模式，脚本需自带异常钩子）

        Args:
            script_path: 脚本路径
            cwd: 工作目录
            interpreter: Python解释器路径

        Returns:
            初始化完成的控制器实例
        """
        process = await cls._create_process(
            [interpreter, script_path],
            cwd
        )
        controller = cls(process)
        controller.initial_output, _ = await controller._read_until_prompt()
        return controller

    @classmethod
    async def _create_process(
        cls,
        command: list,
        cwd: str
    ) -> asyncio.subprocess.Process:
        """创建异步子进程"""
        env = os.environ.copy()
        env["PYTHONUTF8"] = "1"

        return await asyncio.create_subprocess_exec(
            *command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=cwd,
            env=env
        )

    async def send_command(self, command: str) -> str:
        """
        发送PDB命令并获取响应

        Args:
            command: PDB命令字符串

        Returns:
            命令执行结果
        """
        if self._process.returncode is not None:
            return "[SESSION_TERMINATED] The debugging session has ended."

        try:
            self._process.stdin.write((command + '\n').encode('utf-8'))
            await self._process.stdin.drain()
        except (OSError, BrokenPipeError, ConnectionResetError) as e:
            return f"[ERROR] Communication failed: {e}"

        response, is_alive = await self._read_until_prompt()

        if not is_alive:
            suffix = "\n[SESSION_TERMINATED]"
            return (response + suffix) if response else "[SESSION_TERMINATED]"

        return response

    async def execute_code(self, code: str) -> str:
        """
        在当前调试上下文中执行Python代码

        通过exec()将多行代码压缩为单条PDB指令，
        确保复杂逻辑也能在交互环境中完整执行

        Args:
            code: Python代码片段

        Returns:
            执行结果
        """
        escaped_code = repr(code)
        return await self.send_command(f"exec({escaped_code})")

    async def close(self):
        """
        关闭会话并释放资源

        采用三级退出策略：
        1. 发送quit命令，等待进程自行终止
        2. 若超时未退出（可能处于嵌套调试会话），再次发送quit
        3. 若仍未退出，强制终止进程
        """
        if self._process.returncode is not None:
            return

        # 第一次尝试：常规退出
        if await self._try_quit(timeout=1.0):
            return

        # 第二次尝试：应对嵌套会话（如事后调试中触发的二级PDB）
        if await self._try_quit(timeout=1.0):
            return

        # 兜底：强制终止
        self._process.terminate()
        try:
            await asyncio.wait_for(self._process.wait(), timeout=2.0)
        except asyncio.TimeoutError:
            self._process.kill()

    async def _try_quit(self, timeout: float) -> bool:
        """
        尝试发送quit命令并等待进程退出

        Args:
            timeout: 等待退出的超时时间（秒）

        Returns:
            进程是否已退出
        """
        if self._process.returncode is not None:
            return True

        try:
            self._process.stdin.write(b'q\n')
            await self._process.stdin.drain()
        except (OSError, BrokenPipeError, ConnectionResetError):
            return self._process.returncode is not None

        try:
            await asyncio.wait_for(self._process.wait(), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            return False

    async def _read_until_prompt(self) -> Tuple[str, bool]:
        """
        持续读取输出直至遇到PDB提示符或检测到会话终止

        逐字节读取并在累积缓冲区中匹配提示符尾缀，
        PDB的提示符不以换行结束，因此无法依赖行级读取接口

        Returns:
            元组(响应文本, 会话是否存活)
        """
        buffer = bytearray()
        prompt_bytes = self.PDB_PROMPT.encode('utf-8')

        while True:
            try:
                chunk = await asyncio.wait_for(
                    self._process.stdout.read(1),
                    timeout=60.0
                )
            except asyncio.TimeoutError:
                text = buffer.decode('utf-8', errors='replace').strip()
                return text, self._process.returncode is None

            if not chunk:
                # 流关闭，进程已退出
                text = buffer.decode('utf-8', errors='replace').strip()
                return text, False

            buffer.extend(chunk)

            if buffer.endswith(prompt_bytes):
                text = buffer[:-len(prompt_bytes)].decode('utf-8', errors='replace').strip()
                return text, True