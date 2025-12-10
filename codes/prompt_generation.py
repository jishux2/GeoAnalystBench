"""
GeoAnalystBench 提示词生成器
基于任务数据集为50个地理分析任务生成不同配置的提示词组合
"""

import pandas as pd
import csv


# ============================================================
# 文本格式化工具
# ============================================================

def wrap_text(text, max_line_length=80):
    """将长文本按词边界折行，每行不超过指定字符数"""
    if not isinstance(text, str):
        return str(text)
    
    words = text.split()
    lines = []
    current_line = []
    current_length = 0
    
    for word in words:
        word_length = len(word) + 1  # 加上空格
        if current_length + word_length > max_line_length and current_line:
            lines.append(' '.join(current_line))
            current_line = [word]
            current_length = word_length
        else:
            current_line.append(word)
            current_length += word_length
    
    if current_line:
        lines.append(' '.join(current_line))
    
    return '\n'.join(lines)


def format_multiline_text(text):
    """对多行文本的每一行进行折行处理"""
    if not isinstance(text, str):
        return str(text)
    
    formatted_lines = [wrap_text(line) for line in text.split('\n')]
    return '\n'.join(formatted_lines)


# ============================================================
# 提示词模板构建
# ============================================================

def build_workflow_prompt(task, instruction, domain_knowledge=None, dataset_desc=None,
                         include_example=True):
    """
    构建工作流生成任务的提示词
    
    Args:
        task: 任务简要描述
        instruction: 详细指令
        domain_knowledge: 可选的领域知识说明
        dataset_desc: 可选的数据集描述
        include_example: 是否包含输出示例
    """
    sections = {
        "Task": task,
        "Instruction": instruction,
    }
    
    if domain_knowledge:
        sections["Domain Knowledge"] = domain_knowledge
    if dataset_desc:
        sections["Dataset Description"] = dataset_desc
    
    prompt = "As a Geospatial data scientist, you will generate a workflow to a proposed task.\n"
    
    for key, value in sections.items():
        prompt += f"\n[{key}]:\n{value}\n"
    
    prompt += "\n[Key Notes]:"
    prompt += "\n1. Use **automatic reasoning** and clearly explain each step (Chain of Thoughts approach)."
    prompt += "\n2. Using **NetworkX** package for visualization."
    prompt += "\n3. Using 'dot' for graph visualization layout."
    prompt += "\n4. Multiple subtasks can be proceeded correspondingly because all of their outputs will be inputs for the next subtask."
    prompt += "\n5. Limiting your output to code, no extra information."
    prompt += "\n6. Only codes for workflow, no implementation.\n"
    
    if include_example:
        example = '''
"""
tasks = ["task1", "task2", "task3"]

G = nx.DiGraph()
for i in range(len(tasks) - 1):
    G.add_edge(tasks[i], tasks[i + 1])
pos = nx.drawing.nx_pydot.graphviz_layout(G, prog="dot")
plt.figure(figsize=(15, 8))
nx.draw(G, pos, with_labels=True, node_size=3000, node_color='lightblue',
        font_size=10, font_weight='bold', arrowsize=20)
plt.title("Workflow for Analyzing Urban Heat Using Kriging Interpolation", fontsize=14)
plt.show()
"""
'''
        prompt += "\n[Expected Sample Output Begin]"
        prompt += example
        prompt += "\n[Expected Sample Output End]"
    
    return prompt


def build_code_prompt(task, instruction, use_arcpy, domain_knowledge=None,
                     dataset_desc=None, include_example=True):
    """
    构建代码实现任务的提示词
    
    Args:
        task: 任务简要描述
        instruction: 详细指令
        use_arcpy: 是否使用ArcPy库
        domain_knowledge: 可选的领域知识说明
        dataset_desc: 可选的数据集描述
        include_example: 是否包含输出示例
    """
    sections = {
        "Task": task,
        "Instruction": instruction,
    }
    
    if domain_knowledge:
        sections["Domain Knowledge"] = domain_knowledge
    if dataset_desc:
        sections["Dataset Description"] = dataset_desc
    
    prompt = "As a Geospatial data scientist, generate a python file to solve the proposed task.\n"
    
    for key, value in sections.items():
        prompt += f"\n[{key}]:\n{value}\n"
    
    prompt += "\n[Key Notes]:"
    prompt += "\n1. Use **automatic reasoning** and clearly explain each subtask before performing it (ReAct approach)."
    prompt += "\n2. Using latest python packages for code generation."
    prompt += "\n3. Put all code under main function, no helper functions."
    prompt += "\n4. Limit your output to code, no extra information."
    
    if use_arcpy:
        prompt += "\n5. Use latest **Arcpy** functions only."
    else:
        prompt += "\n5. Use latest open source python packages only."
    
    prompt += "\n"
    
    if include_example:
        example = '''
"""
import packages

def main():
    path = "path"
    data = loaddata()
    # code for subtask1
    # code for subtask2
    # code for final task

if __name__ == "__main__":
    main()
"""
'''
        prompt += "\n[Expected Sample Output Begin]"
        prompt += example
        prompt += "\n[Expected Sample Output End]"
    
    return prompt


# ============================================================
# 主生成流程
# ============================================================

def generate_all_prompts(dataset_path='dataset/GeoAnalystBench.csv',
                        code_output='codes/code_prompts.csv',
                        workflow_output='codes/workflow_prompts.csv'):
    """为所有任务生成完整的提示词矩阵"""
    
    tasks_df = pd.read_csv(dataset_path)
    
    # 初始化输出文件
    header = ['task_id', 'type', 'domain_knowledge', 'dataset', 'Arcpy', 'prompt_content']
    
    with open(code_output, 'w', newline='', encoding='utf-8') as f:
        csv.writer(f).writerow(header)
    
    with open(workflow_output, 'w', newline='', encoding='utf-8') as f:
        csv.writer(f).writerow(header)
    
    # 为每个任务生成四种配置的提示词
    for idx, row in tasks_df.iterrows():
        task_id = idx + 1
        task = wrap_text(row["Task"])
        instruction = wrap_text(row["Instruction"])
        domain_knowledge = wrap_text(row["Domain Knowledge"])
        dataset_desc = format_multiline_text(row["Dataset Description"])
        use_arcpy = row["Open Source"] != 'T'
        
        # 四种配置组合：(包含领域知识, 包含数据集描述)
        configs = [
            (False, False),
            (True, False),
            (False, True),
            (True, True)
        ]
        
        for include_domain, include_dataset in configs:
            code_prompt = build_code_prompt(
                task=task,
                instruction=instruction,
                use_arcpy=use_arcpy,
                domain_knowledge=domain_knowledge if include_domain else None,
                dataset_desc=dataset_desc if include_dataset else None,
                include_example=True
            )
            
            workflow_prompt = build_workflow_prompt(
                task=task,
                instruction=instruction,
                domain_knowledge=domain_knowledge if include_domain else None,
                dataset_desc=dataset_desc if include_dataset else None,
                include_example=True
            )
            
            # 写入代码提示词
            with open(code_output, 'a', newline='', encoding='utf-8') as f:
                csv.writer(f).writerow([
                    task_id, 'code', include_domain, include_dataset,
                    use_arcpy, code_prompt
                ])
            
            # 写入工作流提示词
            with open(workflow_output, 'a', newline='', encoding='utf-8') as f:
                csv.writer(f).writerow([
                    task_id, 'workflow', include_domain, include_dataset,
                    use_arcpy, workflow_prompt
                ])
        
        print(f"已生成任务 {task_id}/50 的提示词")
    
    print("\n提示词生成完成！")
    print(f"代码提示词: {code_output}")
    print(f"工作流提示词: {workflow_output}")


if __name__ == "__main__":
    generate_all_prompts()