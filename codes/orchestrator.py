# codes/orchestrator.py
"""
迭代修复流程编排器
统筹提示词构建、推理调用、代码执行、错误诊断的完整链路
"""

import asyncio
import json
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime

from evaluator.dialogue_manager import DialogueManager
from evaluator.workspace_manager import WorkspaceManager
from evaluator.code_executor import CodeExecutor
from prompt_builder import InitialPromptBuilder, IterativePromptBuilder


class IterativeRepairOrchestrator:
    """迭代修复编排器"""
    
    def __init__(
        self,
        api_key: str,
        max_rounds: int = 3,
        max_concurrent: int = 4,
        temperature: float = 0.7,
        enable_thinking: bool = True,
        workspace_root: str = "evaluation_workspace"
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
        """
        self.api_key = api_key
        self.max_rounds = max_rounds
        self.max_concurrent = max_concurrent
        self.temperature = temperature
        self.enable_thinking = enable_thinking
        
        # 初始化各模块
        self.dialogue_mgr = DialogueManager(workspace_root)
        self.workspace_mgr = WorkspaceManager(workspace_root)
        self.executor = CodeExecutor(workspace_root)
        
        self.initial_prompt_builder = InitialPromptBuilder()
        self.iterative_prompt_builder = IterativePromptBuilder()
        
        # 推理引擎（延迟初始化）
        self.inference_engine = None
        
        # 并发控制
        self.semaphore = asyncio.Semaphore(max_concurrent)
        
        # 统计信息
        self.stats = {
            'total_tasks': 0,
            'success': 0,
            'failed': 0,
            'max_rounds_reached': 0
        }
    
    async def initialize_tasks(self, task_ids: List[int]):
        """初始化任务的对话历史"""
        print(f"\n初始化{len(task_ids)}个任务的对话历史...")
        
        for task_id in task_ids:
            existing = self.dialogue_mgr.get_history(task_id)
            if existing:
                print(f"任务{task_id}已存在对话历史，跳过")
                continue
            
            # 构建首轮提示词
            prompt_dict = self.initial_prompt_builder.build(task_id)
            
            # 从workspace_mgr获取元数据
            task_config = self.workspace_mgr.task_configs[task_id]
            
            # 传递元数据至对话历史
            self.dialogue_mgr.initialize(
                task_id=task_id,
                initial_prompt=prompt_dict,
                max_rounds=self.max_rounds,
                categories=task_config['categories'],
                is_opensource=task_config['is_opensource']
            )
            
            print(f"任务{task_id}初始化完成")
        
        print("初始化阶段完成\n")
    
    async def run_round_1(self, task_ids: List[int]):
        """
        执行第1轮：代码生成与验证
        
        Args:
            task_ids: 待处理的任务ID列表
        """
        print(f"\n{'='*60}")
        print(f"第1轮：初始代码生成")
        print(f"{'='*60}\n")
        
        # 不再需要内部过滤，直接执行
        print(f"待处理任务：{len(task_ids)}个")

        # 阶段1：并发生成代码
        print("阶段1：生成代码...")
        tasks = [self._generate_round_1_code(tid) for tid in task_ids]
        await asyncio.gather(*tasks)
        
        # 阶段2：提取代码并注入插桩
        print("\n阶段2：代码提取与插桩...")
        for task_id in task_ids:
            await self._extract_and_inject_code(task_id, round_num=1)
        
        # 阶段3：并发执行代码
        print("\n阶段3：执行代码...")
        execution_results = await self._execute_codes(task_ids, round_num=1)
        
        # 阶段4：对失败任务进行诊断
        failed_tasks = [
            task_id for task_id, result in zip(task_ids, execution_results)
            if not result.success
        ]
        
        if failed_tasks:
            print(f"\n阶段4：诊断{len(failed_tasks)}个失败任务...")
            diagnosis_tasks = [self._diagnose_error(tid, round_num=1) for tid in failed_tasks]
            await asyncio.gather(*diagnosis_tasks)
        
        # # 阶段5：检索辅助信息（为第2轮准备）
        # if failed_tasks:
        #     print(f"\n阶段5：为失败任务检索辅助信息...")
        #     # TODO: 实现检索逻辑
        #     # 当前暂时跳过，留待知识库模块完成后补充
        #     pass
        
        # 更新统计
        success_count = len(task_ids) - len(failed_tasks)
        print(f"\n第1轮完成：成功{success_count}/{len(task_ids)}")
    
    async def run_round_n(self, round_num: int, task_ids: List[int]):
        """执行第N轮（N>=2）：补丁生成与验证"""
        print(f"\n{'='*60}")
        print(f"第{round_num}轮：迭代修复")
        print(f"{'='*60}\n")
        
        if not task_ids:
            print(f"无待处理任务")
            return
        
        print(f"待修复任务：{len(task_ids)}个 - {task_ids}")
        
        # 阶段1：初始化新轮次并执行检索
        print("\n阶段1：初始化新轮次并检索辅助信息...")
        for task_id in task_ids:
            round_data = self.dialogue_mgr.get_round_data(task_id, round_num)
            if not round_data:
                # 获取上一轮的诊断结果
                prev_round = self.dialogue_mgr.get_round_data(task_id, round_num - 1)
                diagnosis = prev_round.get('diagnosis') if prev_round else None
                
                # TODO: 根据诊断结果执行检索
                # retrieved_docs = self.retrieve_api_docs(diagnosis['api_queries'])
                # retrieved_examples = self.retrieve_examples(diagnosis['keywords'], diagnosis['example_query'])
                
                # 当前暂时传入空列表
                retrieved_docs = []
                retrieved_examples = []
                
                self.dialogue_mgr.add_round(
                    task_id, 
                    round_num,
                    retrieved_docs=retrieved_docs,
                    retrieved_examples=retrieved_examples
                )
        
        # 阶段2：并发生成补丁
        print("\n阶段2：生成代码补丁...")
        patch_tasks = [self._generate_patch(tid, round_num) for tid in task_ids]
        await asyncio.gather(*patch_tasks)
        
        # 阶段3：应用补丁并写入文件
        print("\n阶段3：应用补丁...")
        for task_id in task_ids:
            await self._apply_and_save_patch(task_id, round_num)
        
        # 阶段4：并发执行代码
        print("\n阶段4：执行代码...")
        execution_results = await self._execute_codes(task_ids, round_num)
        
        # 阶段5：对失败任务进行诊断（仅当不是最后一轮）
        failed_tasks = [
            task_id for task_id, result in zip(task_ids, execution_results)
            if not result.success
        ]
        
        if failed_tasks:
            if round_num < self.max_rounds:
                print(f"\n阶段5：诊断{len(failed_tasks)}个失败任务...")
                diagnosis_tasks = [self._diagnose_error(tid, round_num) for tid in failed_tasks]
                await asyncio.gather(*diagnosis_tasks)
            else:
                # 达到最大轮次，标记为失败
                print(f"\n达到最大轮次，标记{len(failed_tasks)}个任务为失败")
                for task_id in failed_tasks:
                    self.dialogue_mgr.mark_failed(task_id, reason="max_rounds_reached")
        
        # 统计
        success_count = len(task_ids) - len(failed_tasks)
        print(f"\n第{round_num}轮完成：成功{success_count}/{len(task_ids)}")
    
    async def run(self, task_ids: List[int]):
        """
        执行完整的迭代修复流程
        
        Args:
            task_ids: 待处理的任务ID列表
        """
        print("="*60)
        print("迭代修复系统启动")
        print("="*60)
        print(f"任务数量：{len(task_ids)}")
        print(f"最大轮次：{self.max_rounds}")
        print(f"并发数：{self.max_concurrent}")
        print(f"思考模式：{'启用' if self.enable_thinking else '禁用'}")
        print("="*60)
        
        self.stats['total_tasks'] = len(task_ids)
        
        # 修正：使用async with初始化推理引擎
        from deepseek.async_inference import AsyncInferenceEngine
        
        async with AsyncInferenceEngine(
            api_key=self.api_key,
            temperature=self.temperature,
            enable_thinking=self.enable_thinking
        ) as engine:
            self.inference_engine = engine
            
            # 初始化任务
            await self.initialize_tasks(task_ids)
            
            # 按轮次执行，每轮只处理需要该轮次的任务
            for round_num in range(1, self.max_rounds + 1):
                # 传入task_ids限定筛选范围
                tasks_for_this_round = self._get_tasks_for_round(round_num, task_ids)
                
                if not tasks_for_this_round:
                    print(f"\n第{round_num}轮无待处理任务\n")
                    continue
                
                if round_num == 1:
                    await self.run_round_1(tasks_for_this_round)
                else:
                    await self.run_round_n(round_num, tasks_for_this_round)
        
        # 输出最终统计
        self._print_final_stats()
    
    # ========== 内部辅助方法 ==========
    
    async def _generate_round_1_code(self, task_id: int):
        """生成首轮代码（带并发控制）"""
        async with self.semaphore:
            history = self.dialogue_mgr.get_history(task_id)
            prompt = history['rounds'][0]['prompt']['full_text']
            
            try:
                code = await self.inference_engine.generate_initial_code(prompt)
                
                # 更新对话历史
                self.dialogue_mgr.update_round(
                    task_id,
                    round_num=1,
                    generated_code=code
                )
                
                print(f"✓ 任务{task_id}代码生成完成")
            
            except Exception as e:
                print(f"✗ 任务{task_id}代码生成失败：{e}")
                self.dialogue_mgr.mark_failed(task_id, reason=str(e))
    
    async def _extract_and_inject_code(self, task_id: int, round_num: int):
        """提取代码并注入插桩逻辑"""
        history = self.dialogue_mgr.get_history(task_id)
        
        if round_num == 1:
            response_text = history['rounds'][0]['generated_code']
        else:
            # 应用补丁后的完整代码
            response_text = self.dialogue_mgr.apply_patch(task_id, round_num)
        
        # 提取并注入
        code = self.workspace_mgr.extract_code_from_response(response_text)
        
        # 写入文件
        task_dir = Path(self.workspace_mgr.workspace_root) / str(task_id)
        code_path = task_dir / "generated" / f"round_{round_num}_code.py"
        code_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(code_path, 'w', encoding='utf-8') as f:
            f.write(code)
    
    async def _execute_codes(self, task_ids: List[int], round_num: int) -> List:
        """并发执行代码"""
        
        loop = asyncio.get_event_loop()
        
        results = await loop.run_in_executor(
            None,
            self.executor.execute_batch,
            task_ids,
            round_num,
            True
        )
        
        # 根据结果更新对话历史
        for task_id, result in zip(task_ids, results):
            if result.success:
                self.dialogue_mgr.update_round(
                    task_id,
                    round_num,
                    execution={'status': 'success', 'duration': result.duration}
                )
                self.dialogue_mgr.mark_success(task_id)
                print(f"✓ 任务{task_id}执行成功")
            
            else:
                # 构建基础执行记录
                execution_record = result.to_execution_record()
                
                # 仅对运行时失效读取诊断文件
                if result.is_runtime_failure():
                    task_dir = Path(self.workspace_mgr.workspace_root) / str(task_id)
                    output_dir = task_dir / "outputs" / f"round_{round_num}"
                    
                    error_trace = None
                    call_details = None
                    
                    error_file = output_dir / "error_trace.json"
                    if error_file.exists():
                        with open(error_file, 'r', encoding='utf-8') as f:
                            error_trace = json.load(f)
                    
                    call_file = output_dir / "call_details.json"
                    if call_file.exists():
                        with open(call_file, 'r', encoding='utf-8') as f:
                            call_details = json.load(f)
                    
                    # 仅在实际读取到数据时填充字段
                    if error_trace:
                        execution_record['error_trace'] = error_trace
                    if call_details:
                        execution_record['call_details'] = call_details
                
                # 对前置条件缺失不读取诊断文件，保持字段为None
                
                self.dialogue_mgr.update_round(
                    task_id,
                    round_num,
                    execution=execution_record
                )
                
                print(f"✗ 任务{task_id}执行失败：{result.error_type}")
        
        return results
    
    async def _diagnose_error(self, task_id: int, round_num: int):
        """错误诊断（带并发控制）"""
        async with self.semaphore:
            history = self.dialogue_mgr.get_history(task_id)
            round_data = self.dialogue_mgr.get_round_data(task_id, round_num)
             
            # 构建诊断提示词
            prompt = self.iterative_prompt_builder.build_diagnosis_prompt(
                dialogue_history=history,
                round_num=round_num,
                error_trace=round_data['execution']['error_trace'],
                call_details=round_data['execution']['call_details'] or []
            )
            
            print(f"prompt: \n{prompt}")
            
            try:
                diagnosis = await self.inference_engine.diagnose(prompt)
                
                # 更新对话历史
                self.dialogue_mgr.update_round(
                    task_id,
                    round_num,
                    diagnosis=diagnosis
                )
                
                print(f"✓ 任务{task_id}诊断完成")
            
            except Exception as e:
                print(f"✗ 任务{task_id}诊断失败：{e}")
    
    async def _generate_patch(self, task_id: int, round_num: int):
        """生成补丁（带并发控制）"""
        async with self.semaphore:
            history = self.dialogue_mgr.get_history(task_id)
            
            # 构建补丁生成提示词
            prompt = self.iterative_prompt_builder.build_code_prompt(
                dialogue_history=history,
                round_num=round_num
            )
            
            print(f"prompt: \n{prompt}")
            
            try:
                patch = await self.inference_engine.generate_patch(prompt)
                
                # 更新对话历史
                self.dialogue_mgr.update_round(
                    task_id,
                    round_num,
                    generated_patch=patch
                )
                
                print(f"✓ 任务{task_id}补丁生成完成")
            
            except Exception as e:
                print(f"✗ 任务{task_id}补丁生成失败：{e}")
    
    async def _apply_and_save_patch(self, task_id: int, round_num: int):
        """应用补丁并保存"""
        # 应用补丁得到完整代码
        patched_code = self.dialogue_mgr.apply_patch(task_id, round_num)
        
        # 提取并注入插桩
        await self._extract_and_inject_code(task_id, round_num)
    
    def _get_tasks_for_round(self, round_num: int, candidate_tasks: List[int]) -> List[int]:
        """
        从候选任务中筛选需要执行指定轮次的任务
        
        Args:
            round_num: 轮次编号
            candidate_tasks: 候选任务ID列表（来自run方法传入的task_ids）
        
        Returns:
            需要执行本轮的任务ID列表
        
        筛选规则：
            - status='success' → 跳过（已完成）
            - status='failed' → 跳过（已达最大轮次失败）
            - 第1轮：检查是否需要生成代码
            - 第N轮（N>=2）：检查上一轮是否失败
        """
        tasks = []
        
        for task_id in candidate_tasks:
            history = self.dialogue_mgr.get_history(task_id)
            
            if not history:
                # 未初始化的任务，从第1轮开始
                if round_num == 1:
                    tasks.append(task_id)
                continue
            
            # 已完成或已失败的任务跳过
            if history['status'] in ['success', 'failed']:
                continue
            
            current_round = history['current_round']
            
            # 第1轮：检查是否需要生成代码
            if round_num == 1:
                round_data = self.dialogue_mgr.get_round_data(task_id, 1)
                
                # 如果第1轮记录不存在，或者生成代码为空，或者执行状态是pending
                if not round_data:
                    tasks.append(task_id)
                elif not round_data.get('generated_code'):
                    tasks.append(task_id)
                elif round_data['execution']['status'] == 'pending':
                    tasks.append(task_id)
            
            # 第N轮（N>=2）：检查是否需要继续迭代
            elif round_num >= 2:
                # 检查是否已有本轮记录
                current_round_data = self.dialogue_mgr.get_round_data(task_id, round_num)
                
                if current_round_data:
                    # 本轮已执行过，检查执行状态
                    if current_round_data['execution']['status'] == 'pending':
                        tasks.append(task_id)
                else:
                    # 本轮未执行，检查上一轮是否失败
                    prev_round = self.dialogue_mgr.get_round_data(task_id, round_num - 1)
                    if prev_round and prev_round['execution']['status'] == 'failed':
                        tasks.append(task_id)
        
        return sorted(tasks)
    
    def _print_final_stats(self):
        """输出最终统计"""
        success_tasks = self.dialogue_mgr.get_tasks_by_status('success')
        failed_tasks = self.dialogue_mgr.get_tasks_by_status('failed')
        
        print("\n" + "="*60)
        print("迭代修复完成")
        print("="*60)
        print(f"总任务数：{self.stats['total_tasks']}")
        print(f"成功：{len(success_tasks)}")
        print(f"失败：{len(failed_tasks)}")
        print(f"成功率：{len(success_tasks)/self.stats['total_tasks']*100:.2f}%")
        print("="*60)