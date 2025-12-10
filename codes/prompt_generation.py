"""
提示词生成脚本
用于从基准数据集生成用于工作流推导和代码实现的完整提示词集合
"""

import pandas as pd
import csv


def add_line_breaks(long_string, char_limit=80):
    """限制每行字符数为80"""
    words = long_string.split()
    new_string = ""
    char_count = 0
    for word in words:
        new_string += word + " "
        char_count += len(word)
        if char_count > char_limit:
            new_string += "\n"
            char_count = 0
    return new_string


def long_line_break(long_string):
    """处理过长字符串，对每行进行换行处理"""
    result = ""
    if isinstance(long_string, str):
        for line in long_string.split("\n"):
            new_line = add_line_breaks(line)
            result += new_line + "\n"
    else:
        result = str(long_string)
    return result


def workflow_template(IDs=None, tasks=None, instructions=None, zeroShot=False, 
                     domainKnowledges=None, datasets=None):
    """生成工作流推导的提示词模板"""
    if tasks is None and instructions is None:
        print("Task or Instruction is necessary")
        return None
    
    prompt = {
        "Task": tasks,
        "Instruction": instructions,
        "Domain Knowledge": domainKnowledges,
        "Dataset Description": datasets
    }

    template = """As a Geospatial data scientist, you will generate a workflow to a proposed task.\n"""
    
    for key, value in prompt.items():
        if value is not None:
            template += f"\n[{key}]: \n{value}"

    sample = """ \n\"\"\"
  tasks = ["task1", "task2", "task3"]

  G = nx.DiGraph()
  for i in range(len(tasks) - 1):
      G.add_edge(tasks[i], tasks[i + 1])
  pos = nx.drawing.nx_pydot.graphviz_layout(G, prog="dot")
  plt.figure(figsize=(15, 8))
  nx.draw(G, pos, with_labels=True, node_size=3000, node_color='lightblue', font_size=10, font_weight='bold', arrowsize=20)
  plt.title("Workflow for Analyzing Urban Heat Using Kriging Interpolation", fontsize=14)
  plt.show()\n\"\"\"
  """

    template += '\n[Key Notes]:'
    template += '\n1.Use **automatic reasoning** and clearly explain each step (Chain of Thoughts approach).'
    template += '\n2.Using **NetworkX* package for visualization.'
    template += '\n3.Using \'dot\' for graph visualization layout.'
    template += '\n4.Multiple subtasks can be proceeded correspondingly because'
    template += '\nall of their outputs will be inputs for the next subtask.'
    template += "\n5.Limiting your output to code, no extra information."
    template += '\n6.Only codes for workflow, no implementation.'
    template += '\n'
    
    if zeroShot is False:
        template += "\n[Expected Sample Output Begin]"
        template += "\n" + sample
        template += "[Expected Sample Output End]"
    
    return template


def code_template(IDs=None, tasks=None, instructions=None, zeroShot=False, 
                 domainKnowledges=None, datasets=None, Arcpy=False):
    """生成代码实现的提示词模板"""
    if tasks is None and instructions is None:
        print("Task or Instruction is necessary")
        return None
    
    prompt = {
        "Task": tasks,
        "Instruction": instructions,
        "Domain Knowledge": domainKnowledges,
        "Dataset Description": datasets
    }

    template = """As a Geospatial data scientist, generate a python file to solve the proposed task.\n"""
    
    for key, value in prompt.items():
        if value is not None:
            template += f"\n[{key}]: \n{value}"

    sample = """ \"\"\"
    import packages

    def main():
      path = "path"
      data = loaddata()
      #code for subtask1
      #code for subtask2
      #code for final task

    if __name__ == "__main__":
      main()
  \"\"\"
  """
    
    template += '\n\n[Key Notes]:'
    template += '\n1.Use **automatic reasoning** and clearly explain each subtask before performing it (ReAct approach).'
    template += '\n2.Using latest python packages for code generation'
    template += "\n3.Put all code under main function, no helper functions"
    template += "\n4.Limit your output to code, no extra information."
    
    if Arcpy is True:
        template += "\n5.Use latest **Arcpy** functions only"
    else:
        template += "\n5.Use latest open source python packages only"
    
    template += '\n'
    
    if zeroShot is False:
        template += "\n[Expected Sample Output Begin]"
        template += "\n" + sample
        template += "[Expected Sample Output End]"
    
    return template


def generate_prompts(dataset_path='dataset/GeoAnalystBench.csv', 
                    code_output='codes/code_prompts.csv',
                    workflow_output='codes/workflow_prompts.csv'):
    """主函数：为50个任务生成所有提示词组合"""
    
    data = pd.read_csv(dataset_path)
    
    with open(code_output, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['task_id', 'type', 'domain_knowledge', 'dataset', 'Arcpy', 'prompt_content'])

    with open(workflow_output, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['task_id', 'type', 'domain_knowledge', 'dataset', 'Arcpy', 'prompt_content'])

    for id in range(50):
        row = data.iloc[id]
        task = row["Task"]
        instruction = row["Instruction"]
        domainKnowledge = row["Domain Knowledge"]
        dataset = row["Dataset Description"]
        Arcpy = row["Open Source"] != 'T'

        instruction = add_line_breaks(instruction)
        task = add_line_breaks(task)
        domainKnowledge = add_line_breaks(domainKnowledge)
        dataset = long_line_break(dataset)

        combinations = [
            (False, False),
            (True, False),
            (False, True),
            (True, True)
        ]

        for domain, dataset_included in combinations:
            code_params = {
                'tasks': task,
                'instructions': instruction,
                'zeroShot': True,
                'Arcpy': Arcpy
            }

            workflow_params = {
                'tasks': task,
                'instructions': instruction,
                'zeroShot': False
            }

            if domain:
                code_params['domainKnowledges'] = domainKnowledge
                workflow_params['domainKnowledges'] = domainKnowledge
            if dataset_included:
                code_params['datasets'] = dataset
                workflow_params['datasets'] = dataset

            code_prompt = code_template(**code_params)
            workflow_prompt = workflow_template(**workflow_params)

            with open(code_output, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([id+1, 'code', domain, dataset_included, Arcpy, code_prompt])
            
            with open(workflow_output, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([id+1, 'workflow', domain, dataset_included, Arcpy, workflow_prompt])
        
        print(f"已生成任务 {id+1}/50 的提示词")

    print("提示词生成完成！")
    print(f"代码提示词已保存至：{code_output}")
    print(f"工作流提示词已保存至：{workflow_output}")


if __name__ == "__main__":
    generate_prompts()