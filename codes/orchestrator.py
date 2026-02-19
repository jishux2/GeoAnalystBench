# codes/orchestrator.py
"""
迭代修复流程编排器
统筹智能体调试循环的启动、状态管理与结果记录
"""

import asyncio
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import List, Dict, Optional, Tuple

from evaluator.dialogue_manager import DialogueManager
from evaluator.workspace_manager import WorkspaceManager
from prompt_builder import InitialPromptBuilder
from debug_agent import DebugAgent
from debug_agent.prompts import format_code_with_line_numbers


class IterativeRepairOrchestrator:
    """迭代修复编排器"""
    
    def __init__(
        self,
        api_key: str,
        max_rounds: int = 3,
        max_concurrent: int = 4,
        temperature: float = 0.7,
        enable_thinking: bool = True,
        workspace_root: str = "evaluation_workspace",
        debug_max_turns: int = 20
    ):
        """
        初始化编排器
        
        Args:
            api_key: DeepSeek API密钥
            max_rounds: 最大迭代轮次
            max_concurrent: 最大并发任务数
            temperature: 采样温度
            enable_thinking: 是否启用思考模式
            workspace_root: 工作空间根目录
            debug_max_turns: 智能体单次调试的最大交互轮次
        """
        self.api_key = api_key
        self.max_rounds = max_rounds
        self.max_concurrent = max_concurrent
        self.temperature = temperature
        self.enable_thinking = enable_thinking
        self.workspace_root = workspace_root
        self.debug_max_turns = debug_max_turns
        
        self.dialogue_mgr = DialogueManager(workspace_root)
        self.workspace_mgr = WorkspaceManager(workspace_root)
        self.initial_prompt_builder = InitialPromptBuilder()
        
        # 加载解释器配置
        self._load_interpreter_config()
        
        self.executor: Optional[ProcessPoolExecutor] = None
        
        self.stats = {
            'total_tasks': 0,
            'success': 0,
            'failed': 0
        }
    
    async def run(self, task_ids: List[int]):
        """
        执行完整的迭代修复流程
        
        Args:
            task_ids: 待处理的任务ID列表
        """
        print("=" * 60)
        print("迭代修复系统启动")
        print("=" * 60)
        print(f"任务数量：{len(task_ids)}")
        print(f"最大轮次：{self.max_rounds}")
        print(f"并发数：{self.max_concurrent}")
        print(f"思考模式：{'启用' if self.enable_thinking else '禁用'}")
        print(f"智能体最大交互轮次：{self.debug_max_turns}")
        print("=" * 60)
        
        self.stats['total_tasks'] = len(task_ids)
        
        with ProcessPoolExecutor(max_workers=self.max_concurrent) as executor:
            self.executor = executor
            
            await self._initialize_tasks(task_ids)
            
            for round_num in range(1, self.max_rounds + 1):
                tasks_for_round = self._get_tasks_for_round(round_num, task_ids)
                
                if not tasks_for_round:
                    print(f"\n第{round_num}轮无待处理任务")
                    continue
                
                await self._run_round(round_num, tasks_for_round)
            
            self.executor = None
        
        self._print_final_stats(task_ids)
    
    async def _initialize_tasks(self, task_ids: List[int]):
        """
        初始化任务的对话历史
        
        Args:
            task_ids: 任务ID列表
        """
        print(f"\n初始化{len(task_ids)}个任务...")
        
        for task_id in task_ids:
            existing = self.dialogue_mgr.get_history(task_id)
            if existing:
                print(f"  任务{task_id}：已存在，跳过")
                continue
            
            prompt_dict = self.initial_prompt_builder.build(task_id)
            task_config = self.workspace_mgr.task_configs.get(task_id, {})
            
            self.dialogue_mgr.initialize(
                task_id=task_id,
                initial_prompt=prompt_dict,
                max_rounds=self.max_rounds,
                categories=task_config.get('categories', []),
                is_opensource=task_config.get('is_opensource', True)
            )
            
            print(f"  任务{task_id}：初始化完成")
        
        print("初始化阶段完成\n")
    
    async def _run_round(self, round_num: int, task_ids: List[int]):
        """
        执行单轮迭代
        
        Args:
            round_num: 当前轮次
            task_ids: 本轮待处理的任务列表
        """
        print(f"\n{'=' * 60}")
        print(f"第{round_num}轮：{'初始代码生成与调试' if round_num == 1 else '迭代修复'}")
        print(f"{'=' * 60}")
        print(f"待处理任务：{len(task_ids)}个 - {task_ids}\n")
        
        if round_num == 1:
            await self._run_round_1(task_ids)
        else:
            await self._run_round_n(round_num, task_ids)
        
        success_count = len(self.dialogue_mgr.get_tasks_by_status('success', scope=task_ids))
        print(f"\n第{round_num}轮完成：成功{success_count}/{self.stats['total_tasks']}")
    
    async def _run_round_1(self, task_ids: List[int]):
        """
        执行第1轮：代码生成与智能体调试
        
        Args:
            task_ids: 任务列表
        """
        need_generate = []
        have_code = []
        
        for task_id in task_ids:
            round_1 = self.dialogue_mgr.get_round_data(task_id, 1)
            if not round_1 or not round_1.get('generated_code'):
                need_generate.append(task_id)
            else:
                have_code.append(task_id)
        
        if need_generate:
            print(f"步骤1：生成代码（{len(need_generate)}个任务）...")
            from deepseek.async_inference import AsyncInferenceEngine
            
            async with AsyncInferenceEngine(
                api_key=self.api_key,
                temperature=self.temperature,
                enable_thinking=self.enable_thinking
            ) as engine:
                gen_tasks = [
                    self._generate_initial_code(engine, tid)
                    for tid in need_generate
                ]
                await asyncio.gather(*gen_tasks)
        
        all_for_debug = have_code + need_generate
        if all_for_debug:
            print(f"\n步骤2：智能体调试（{len(all_for_debug)}个任务）...")
            agent_tasks = [
                self._run_debug_agent(tid, round_num=1)
                for tid in all_for_debug
            ]
            await asyncio.gather(*agent_tasks)
    
    async def _run_round_n(self, round_num: int, task_ids: List[int]):
        """
        执行第N轮（N>=2）：基于补丁的迭代修复
        
        Args:
            round_num: 当前轮次
            task_ids: 任务列表
        """
        for task_id in task_ids:
            round_data = self.dialogue_mgr.get_round_data(task_id, round_num)
            if not round_data:
                self.dialogue_mgr.add_round(task_id, round_num)
        
        print(f"启动智能体调试（{len(task_ids)}个任务）...")
        agent_tasks = [
            self._run_debug_agent(tid, round_num)
            for tid in task_ids
        ]
        await asyncio.gather(*agent_tasks)
        
        failed_tasks = [
            tid for tid in task_ids
            if self.dialogue_mgr.get_history(tid)['status'] == 'pending'
        ]
        
        if failed_tasks and round_num >= self.max_rounds:
            print(f"\n达到最大轮次，标记{len(failed_tasks)}个任务为失败")
            for task_id in failed_tasks:
                self.dialogue_mgr.mark_failed(task_id, reason="max_rounds_reached")
    
    async def _generate_initial_code(self, engine, task_id: int):
        """
        生成首轮代码
        
        Args:
            engine: 推理引擎实例
            task_id: 任务ID
        """
        history = self.dialogue_mgr.get_history(task_id)
        prompt = history['rounds'][0]['prompt']['full_text']
        
        try:
            response = await engine.generate_initial_code(prompt)
            extracted_code = self._extract_code_from_response(response)
            
            self.dialogue_mgr.update_round(
                task_id,
                round_num=1,
                generated_code=extracted_code
            )
            
            print(f"  ✓ 任务{task_id}代码生成完成")
        
        except Exception as e:
            print(f"  ✗ 任务{task_id}代码生成失败：{e}")
            self.dialogue_mgr.mark_failed(task_id, reason=str(e))
    
    async def _run_debug_agent(self, task_id: int, round_num: int):
        """
        启动智能体进行调试和修复
        
        Args:
            task_id: 任务ID
            round_num: 当前轮次
        """
        history = self.dialogue_mgr.get_history(task_id)
        task_dir = Path(self.workspace_root) / str(task_id)
        output_dir = task_dir / "outputs" / f"round_{round_num}"
        
        original_code = history['rounds'][0].get('generated_code')
        if not original_code:
            print(f"  ✗ 任务{task_id}缺少代码，跳过")
            return
        
        # 先对原始代码注入通用辅助逻辑，建立稳定基线
        base_code = self._inject_common_helpers(original_code)
        
        # 在预处理后的基线上应用补丁
        current_diagnosis = history.get('diagnosis')
        script_content = self._apply_patches(base_code, current_diagnosis)
        
        # 保存快照供审查
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "current_code.py").write_text(script_content, encoding='utf-8')
        
        try:
            interpreter = self._get_interpreter_for_task(task_id)
        except RuntimeError as e:
            print(f"  ✗ 任务{task_id}环境错误：{e}")
            self.dialogue_mgr.update_round(
                task_id,
                round_num,
                execution={'status': 'error', 'error_message': str(e)}
            )
            return
        
        try:
            agent = DebugAgent(
                api_key=self.api_key,
                script_content=script_content,
                working_dir=str(task_dir),
                interpreter=interpreter,
                output_dir=output_dir,
                current_diagnosis=current_diagnosis,
                error_summary=None,
                debug_mode="crash",
                max_turns=self.debug_max_turns,
                temperature=self.temperature,
                executor=self.executor
            )
            
            result = await agent.run()
            
            if result.get('success'):
                self.dialogue_mgr.update_round(
                    task_id,
                    round_num,
                    execution={'status': 'success'}
                )
                self.dialogue_mgr.mark_success(task_id)
                print(f"  ✓ 任务{task_id}执行成功")
            
            else:
                diagnosis = {
                    'root_cause': result.get('root_cause', ''),
                    'patches': result.get('patches', [])
                }
                
                self.dialogue_mgr.update_diagnosis(task_id, diagnosis)
                self.dialogue_mgr.update_round(
                    task_id,
                    round_num,
                    execution={'status': 'diagnosed'}
                )
                
                root_cause_preview = diagnosis['root_cause'][:60] + '...' if len(diagnosis['root_cause']) > 60 else diagnosis['root_cause']
                print(f"  ✗ 任务{task_id}诊断完成：{root_cause_preview}")
        
        except Exception as e:
            print(f"  ✗ 任务{task_id}智能体异常：{e}")
            self.dialogue_mgr.update_round(
                task_id,
                round_num,
                execution={'status': 'error', 'error_message': str(e)}
            )
    
    def _get_tasks_for_round(self, round_num: int, candidate_tasks: List[int]) -> List[int]:
        """
        筛选需要执行指定轮次的任务
        
        Args:
            round_num: 轮次编号
            candidate_tasks: 候选任务列表
        
        Returns:
            需要执行本轮的任务ID列表
        """
        tasks = []
        
        for task_id in candidate_tasks:
            history = self.dialogue_mgr.get_history(task_id)
            
            if not history:
                if round_num == 1:
                    tasks.append(task_id)
                continue
            
            if history['status'] in ['success', 'failed']:
                continue
            
            if round_num == 1:
                if not self._is_round_completed(task_id, 1):
                    tasks.append(task_id)
            else:
                prev_completed = self._is_round_completed(task_id, round_num - 1)
                current_incomplete = not self._is_round_completed(task_id, round_num)
                
                if prev_completed and current_incomplete:
                    tasks.append(task_id)
        
        return sorted(tasks)
    
    def _is_round_completed(self, task_id: int, round_num: int) -> bool:
        """
        判断指定轮次是否已完结
        
        完结条件：execution.status为diagnosed
        
        Args:
            task_id: 任务编号
            round_num: 轮次编号
        
        Returns:
            该轮次是否已完结
        """
        round_data = self.dialogue_mgr.get_round_data(task_id, round_num)
        
        if not round_data:
            return False
        
        return round_data.get('execution', {}).get('status') == 'diagnosed'
    
    def _apply_patches(self, original_code: str, diagnosis: Optional[Dict]) -> str:
        """
        将补丁应用到原始代码
        
        Args:
            original_code: 首轮生成的原始代码
            diagnosis: 任务级诊断（包含patches列表）
        
        Returns:
            应用补丁后的代码
        """
        if not diagnosis:
            return original_code
        
        patches = diagnosis.get('patches', [])
        if not patches:
            return original_code
        
        current_code = original_code
        
        for patch in patches:
            target = patch.get('target_code', '')
            replacement = patch.get('replacement_code', '')
            
            if target and target in current_code:
                current_code = current_code.replace(target, replacement, 1)
        
        return current_code
    
    def _inject_common_helpers(self, code: str) -> str:
        """注入通用辅助逻辑"""
        lines = code.split('\n')
        
        # monitor_call降级桩：当跟踪模式未注入完整实现时，
        # 提供一个透传原函数的空操作版本，避免NameError
        fallback_monitor = [
            '',
            'try:',
            '    monitor_call',
            'except NameError:',
            '    def monitor_call(name):',
            '        def decorator(func):',
            '            return func',
            '        return decorator',
            ''
        ]
        
        # 插入到文件开头
        lines = fallback_monitor + lines
        code = '\n'.join(lines)
        lines = code.split('\n')
        
        # matplotlib后端配置
        if 'matplotlib.pyplot' in code and 'matplotlib.use' not in code:
            mpl_pos = self._find_matplotlib_import(lines)
            if mpl_pos:
                line_idx, indent = mpl_pos
                indent_str = ' ' * indent
                backend_code = [
                    f'{indent_str}import matplotlib',
                    f'{indent_str}matplotlib.use("Agg")'
                ]
                lines = lines[:line_idx] + backend_code + lines[line_idx:]
        
        # 目录创建
        if 'pred_results/' in code and 'makedirs' not in code:
            insert_pos = self._find_last_top_level_import(lines)
            dir_code = [
                '',
                'import os',
                'os.makedirs("pred_results", exist_ok=True)',
                ''
            ]
            lines = lines[:insert_pos] + dir_code + lines[insert_pos:]
        
        return '\n'.join(lines)
    
    def _find_last_top_level_import(self, lines: List[str]) -> int:
        """
        定位文件中最后一个顶层导入语句的下一行位置
        
        扫描全文，返回最后一个零缩进import语句之后的行索引，
        作为注入辅助代码的插入点
        
        Args:
            lines: 代码行列表
        
        Returns:
            插入位置的行索引
        """
        last_import_pos = 0
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            
            # 跳过空行和注释
            if not stripped or stripped.startswith('#'):
                continue
            
            # 检测顶层导入（无前导空白）
            if line and line[0] not in (' ', '\t'):
                if stripped.startswith(('import ', 'from ')):
                    last_import_pos = i + 1
        
        return last_import_pos


    def _find_matplotlib_import(self, lines: List[str]) -> Optional[Tuple[int, int]]:
        """
        定位matplotlib.pyplot导入语句的位置和缩进
        
        Args:
            lines: 代码行列表
        
        Returns:
            元组(行索引, 缩进空格数)，未找到返回None
        """
        for i, line in enumerate(lines):
            stripped = line.strip()
            
            if 'import matplotlib.pyplot' in stripped or 'from matplotlib import pyplot' in stripped:
                # 计算该行的缩进空格数
                indent = len(line) - len(line.lstrip())
                return (i, indent)
        
        return None
    
    def _extract_code_from_response(self, response_text: str) -> str:
        """
        从模型响应中提取Python代码
        
        Args:
            response_text: 模型返回的原始文本
        
        Returns:
            提取的代码
        """
        import re
        
        pattern = r'```(?:python)?\s*(.*?)```'
        matches = re.findall(pattern, response_text, re.DOTALL)
        
        if matches:
            code = max(matches, key=len).strip()
        else:
            code = response_text.strip()
        
        return code.replace('\r\n', '\n').replace('\r', '\n')
    
    def _load_interpreter_config(self):
        """
        从配置文件加载解释器路径
        
        开源环境由setup_evaluation_env.py创建，闭源环境需手动配置
        """
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
        
        self._verify_interpreter(self.opensource_interpreter)

    def _verify_interpreter(self, interpreter_path: str):
        """验证Python解释器是否可用"""
        import subprocess
        
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

    def _get_interpreter_for_task(self, task_id: int) -> str:
        """
        根据任务技术栈属性选择合适的解释器
        
        Args:
            task_id: 任务ID
        
        Returns:
            解释器路径
        """
        history = self.dialogue_mgr.get_history(task_id)
        
        if not history or 'metadata' not in history:
            return self.opensource_interpreter
        
        is_opensource = history['metadata'].get('is_opensource', True)
        
        if is_opensource:
            return self.opensource_interpreter
        else:
            if not self.arcgis_interpreter:
                raise RuntimeError(f"任务{task_id}需要ArcGIS环境，但未配置解释器")
            return self.arcgis_interpreter
    
    def _print_final_stats(self, task_ids: List[int]):
        """输出最终统计"""
        success_count = len(self.dialogue_mgr.get_tasks_by_status('success', scope=task_ids))
        failed_count = len(self.dialogue_mgr.get_tasks_by_status('failed', scope=task_ids))
        
        self.stats['success'] = success_count
        self.stats['failed'] = failed_count
        
        print("\n" + "=" * 60)
        print("迭代修复完成")
        print("=" * 60)
        print(f"总任务数：{self.stats['total_tasks']}")
        print(f"成功：{success_count}")
        print(f"失败：{failed_count}")
        
        if self.stats['total_tasks'] > 0:
            rate = success_count / self.stats['total_tasks'] * 100
            print(f"成功率：{rate:.1f}%")
        
        print("=" * 60)