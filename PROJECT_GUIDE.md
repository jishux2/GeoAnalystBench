我们要复现的项目是**GeoAnalystBench**，这是一个专门用于评估大语言模型在空间分析工作流构建和代码生成方面能力的GeoAI基准测试系统。该项目汇集了50个经GIS领域专家严格验证的真实地理处理任务，旨在检验不同LLM能否从自然语言描述中准确推导出空间分析流程并生成可执行的Python脚本。

先看一下项目的整体架构。以下目录结构反映了当前实际布局，相较原始版本的主要变化在于：将Jupyter Notebook文件转换为标准Python脚本，并补充了依赖配置文件。

```
GeoAnalystBench/
│
├─ LICENSE                                    # 开源协议声明
├─ README.md                                  # 项目文档，涵盖基准测试说明、任务清单及实践案例
├─ PROJECT_GUIDE.md                           # 完整复现指南（本文档），详述项目架构、核心逻辑及实施流程
├─ requirements.txt                           # Python依赖包配置清单
│
├─ case_study/                                # 实证研究材料
│  └─ figures/                                # 可视化成果
│     ├─ elk.png                              # 麋鹿活动范围原始数据呈现
│     ├─ ElkAI.png                            # 基于AI的麋鹿迁徙模式预测结果
│     ├─ traffic.png                          # 车祸地点空间分布原图
│     └─ TrafficAI.png                        # 模型识别的事故热点区域
│
├─ codes/                                     # 核心实现代码
│  ├─ Inference.py                            # 调度多个LLM完成推理任务的主控脚本
│  ├─ prompt_generation.py                    # 为50个任务构建不同组合的提示词
│  └─ utils.py                                # 通用函数库，封装API调用、文本解析及指标计算
│
├─ dataset/                                   # 实验数据
│  └─ GeoAnalystBench.csv                     # 包含50道题目的评测集，记录任务描述、参考代码等字段
│
└─ figures/                                   # 框架图示
   └─ framework.png                           # 研究设计的整体架构
```

接下来梳理核心组成部分及其功能。

**数据集构成**

基准测试的核心载体是`dataset/GeoAnalystBench.csv`文件，其中详尽记录了全部50个任务的完整信息。每条记录涵盖以下关键维度：

| 字段名 | 说明 |
|--------|------|
| ID | 任务的唯一标识符 |
| Open or Closed Source | 标注采用开源（T）或闭源（F）库 |
| Task | 任务的简要描述 |
| Instruction/Prompt | 完成任务所需的自然语言指令 |
| Domain Knowledge | 相关领域的专业知识要点 |
| Dataset Description | 涉及数据的名称、格式、描述及关键字段 |
| Human Designed Workflow | 专家设计的工作流步骤编号 |
| Task Length | 标准工作流的步骤数量 |
| Code | 对应任务的参考Python实现 |

这些字段在后续的提示词构造和模型推理环节中都会被引用。

*数据集另通过Task Categories1-3字段标注各任务的方法论归属，采用ESRI空间分析分类框架的六个维度：理解位置（U）、测量形态分布（M）、确定位置关联（DR）、寻找最优方案（F）、检测量化模式（DP）、空间插值预测（S）。每个任务可归入至多三个类别以反映其复合性质。*

**任务概览**

50个精选任务横跨多个地理处理领域，从城市热岛效应分析到野生动物迁徙追踪，从森林退化预测到交通事故聚集识别，类型丰富多样。以下列举部分代表性任务：

| ID | 任务名称 | 来源 |
|----|----------|------|
| 1 | 查找威斯康星州麦迪逊的热岛和高风险人群 | Analyze urban heat using kriging |
| 2 | 查找汉密尔顿未来的公交站点位置 | Assess access to public transit |
| 3 | 使用卫星图像评估蒙大拿州的火灾疤痕和野火影响 | Assess burn scars with satellite imagery |
| 4 | 识别需要保护的地下水脆弱区域 | Identify groundwater vulnerable areas |
| 5 | 在保护隐私的同时可视化儿童血铅水平升高的数据 | De-identify health data |
| ... | ... | ... |
| 43 | 使用动物GPS轨迹对活动范围进行建模，以了解它们的位置以及随时间的移动方式 | Model animal home range（后文案例1详述） |
| ... | ... | ... |
| 46 | 识别高峰时段车祸的热点 | Determine the most dangerous roads for drivers（案例研究2涉及） |
| ... | ... | ... |
| 50 | 预测海草栖息地 | Predict seagrass habitats with machine learning |

*（中间任务省略，数据集文件中包含全部50个任务的详细信息）*

**典型案例解析**

为深入理解项目的运作机制，我们剖析两个代表性案例。

*案例1：麋鹿活动范围识别（任务43）*

该任务要求利用GPS跟踪记录识别2009年阿尔伯塔省西南部麋鹿种群的活动范围，进而分析其空间利用模式和移动聚类特征。

所用数据集为`dataset/Elk_in_Southwestern_Alberta_2009.geojson`，其中记录了麋鹿移动轨迹点，核心字段包括`OBJECTID`、`timestamp`、`long`、`lat`、`individual`、`geometry`等坐标与时间信息。

针对此任务设计了两类提示词模板。第一类用于引导模型生成工作流框架：

```
作为地理空间数据科学家，你将为提议的任务生成工作流。

[任务]：
使用动物GPS轨迹对活动范围进行建模，以了解它们的位置以及随时间的移动方式。

[指令]：
你的任务是使用提供的数据集分析和可视化麋鹿的移动。目标是使用空间分析技术估计活动范围并评估栖息地偏好，包括最小边界几何（凸包）、核密度估计和基于密度的聚类（DBSCAN）。该分析将生成存储在"dataset/elk_home_range.gdb"和"dataset/"中的空间输出。

[领域知识]：
"活动范围"可以定义为动物正常生活并找到生存所需物品的区域。"最小边界几何"创建一个要素类，其中包含代表指定最小边界几何的多边形。"凸包"是可以包围一组对象的最小凸多边形。"核密度制图"计算并可视化给定区域中要素的密度。"DBSCAN"根据密度标准对点进行聚类。

[数据集描述]：
dataset/Elk_in_Southwestern_Alberta_2009.geojson：用于存储2009年阿尔伯塔省西南部麋鹿移动点的geojson文件。

列：'OBJECTID'、'timestamp'、'long'、'lat'、'individual'、'geometry'等

[关键要点]：
1. 使用自动推理并清楚地解释每个步骤（思维链方法）
2. 使用NetworkX包进行可视化
3. 使用'dot'进行图形可视化布局
4. 可以相应地进行多个子任务
5. 将输出限制为代码，不要提供额外信息
6. 仅用于工作流的代码，不要实现

[预期示例输出开始]
"""
tasks = [Task1, Task2, Task3]
G = nx.DiGraph()
for i in range(len(tasks) - 1):
    G.add_edge(tasks[i], tasks[i + 1])
pos = nx.drawing.nx_pydot.graphviz_layout(G, prog="dot")
plt.figure(figsize=(15, 8))
nx.draw(G, pos, with_labels=True, node_size=3000, node_color='lightblue', font_size=10, font_weight='bold', arrowsize=20)
plt.title("Workflow", fontsize=14)
plt.show()
"""
[预期示例输出结束]
```

第二类则要求模型直接输出可执行的完整Python代码：

```
作为地理空间数据科学家，生成一个python文件来解决提议的任务。

[任务]：
使用动物GPS轨迹对活动范围进行建模。

[指令]：
（同上）

[领域知识]：
（同上）

[数据集描述]：
（同上）

[关键要点]：
1. 使用自动推理并在执行之前清楚地解释每个子任务（ReAct方法）
2. 使用最新的python包进行代码生成
3. 将所有代码放在main函数下，不要使用辅助函数
4. 将输出限制为代码，不要提供额外信息
5. 仅使用最新的Arcpy函数
```

*案例2：车祸热点分析（任务46）*

该案例聚焦于定位2010年至2015年间佛罗里达州布里瓦德县高峰时段交通事故的空间热点区域。

分析依赖三个配套数据文件：`dataset/roads.shp`描绘路网结构，`dataset/crashes.shp`标注事故发生位置，`dataset/nwswm360ft.swm`提供网络空间权重矩阵。

相应的提示词示例如下：

```
[任务]：
识别高峰时段车祸的热点

[指令]：
你的任务是识别2010年至2015年佛罗里达州布里瓦德县高峰时段车祸的热点。第一步是根据高峰时区选择所有车祸。创建所选车祸数据的副本。然后，将车祸点捕捉到路网并与道路进行空间连接。根据连接数据计算车祸率，并使用热点分析获得车祸热点图作为结果。

[领域知识]：
我们将工作日下午3点到5点之间的时区交通视为高峰。对于捕捉过程，道路上推荐的缓冲区为0.25英里。热点分析寻找聚集在一起的高车祸率，基于路网的准确距离测量至关重要。

[数据集描述]：
dataset/crashes.shp：2010年至2015年佛罗里达州布里瓦德县的车祸位置
dataset/roads.shp：布里瓦德县的路网
dataset/nwswm360ft.swm：网络空间权重矩阵文件
```

透过这两个实例可以清晰看出，项目的核心评测逻辑在于：通过组合不同配置的提示词（是否附带领域知识、是否提供数据集详情）测试各类LLM的响应表现，随后将模型生成的工作流与代码同专家设计的标准方案进行比对，从而量化评估其准确性与实用性。

**环境配置与依赖管理**

项目的技术栈依赖声明于根目录的`requirements.txt`文件，其中列举了运行所需的全部Python软件包：

```txt
pandas>=2.0.0
numpy>=1.24.0
langchain-openai>=0.1.0
langchain-anthropic>=0.1.0
langchain-google-genai>=1.0.0
ollama>=0.1.0
```

这些依赖覆盖了数据处理基础库（`pandas`、`numpy`）、多家LLM服务商的API封装（`langchain`系列包）及本地模型接入工具（`ollama`）。即便当前仅使用本地推理方式，商业模型的SDK亦一并安装，便于后续灵活切换。

搭建独立虚拟环境可避免与系统Python产生依赖冲突：

```bash
cd GeoAnalystBench
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

若选择本地部署路线，需额外配置`ollama`服务并拉取目标模型：

```bash
ollama pull deepseek-r1:7b
```

此处明确指定7B参数规模的Q4_K_M量化版本，以对齐原论文评估实验的配置基准。Ollama平台的`:latest`当前指向8.2B参数变体，两个规格在资源消耗与运行效率上有所不同，命令中附加具体后缀可保障复现环境与研究原型的对应关系。

**项目核心代码文件**

整个评测系统的运转依赖三个核心Python脚本的协同配合。它们在架构层面各司其职：基础层负责将不同供应商的API调用统一为标准化接口，中间环节专注于根据配置规则批量构造输入样本，顶层控制器则编排整个推理流程并汇总各模型的响应。这种模块化的设计将从原始任务描述到LLM输出结果的全链路处理转化为高度自动化的执行过程。

**1. `utils.py` - 工具函数库**

该模块封装了贯穿项目全程的各类通用函数，涵盖多个商业闭源模型（GPT、Claude、Gemini）和本地开源模型（基于Ollama平台托管）的统一接口。此外还实现了任务列表的文本解析、工作流长度的自动计算等辅助功能，为上层推理脚本提供了标准化的调用方式。

```python
"""
GeoAnalystBench 工具函数库
提供LLM接口统一封装、文本解析和评估指标计算
"""

import re
import csv
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import GoogleGenerativeAI
import ollama


# ============================================================
# 文本解析工具
# ============================================================

def extract_task_list(text):
    """从模型输出中提取任务列表，自动过滤掉纯数字编号行"""
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    return [line for line in lines if not re.match(r'^\d+\.$', line)]


def calculate_workflow_length(workflow_text):
    """
    从工作流文本中提取步骤数量
    通过匹配"数字+句点"模式识别步骤编号，返回最大编号值
    """
    max_step = 0
    
    for line in workflow_text.split("\n"):
        i = 0
        while i < len(line):
            if line[i].isdigit():
                # 处理两位数编号
                if i + 1 < len(line) and line[i + 1].isdigit():
                    if i + 2 < len(line) and line[i + 2] == '.':
                        max_step = max(max_step, int(line[i:i+2]))
                        i += 2
                # 处理单位数编号
                elif i + 1 < len(line) and line[i + 1] == '.':
                    num = int(line[i])
                    if num <= 10:  # 避免误识别其他数字
                        max_step = max(max_step, num)
                    i += 1
                else:
                    i += 1
            else:
                i += 1
    
    return max_step


def calculate_length_mae(ground_truth_df, predictions_df):
    """
    计算工作流长度预测的平均绝对误差
    
    Args:
        ground_truth_df: 包含专家标注的DataFrame，需包含'id'和'task_length'列
        predictions_df: 包含模型预测的DataFrame，需包含'task_id'和'task_length'列
    
    Returns:
        float: 平均绝对误差值
    """
    total_error = 0
    count = 0
    
    for _, gt_row in ground_truth_df.iterrows():
        task_predictions = predictions_df[predictions_df["task_id"] == gt_row["id"]]
        
        for _, pred_row in task_predictions.iterrows():
            total_error += abs(pred_row["task_length"] - gt_row["task_length"])
            count += 1
    
    return total_error / count if count > 0 else 0


# ============================================================
# LLM统一接口层
# ============================================================

def call_gpt(prompt, temperature=0.7, max_tokens=None, timeout=None):
    """调用OpenAI GPT系列模型"""
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout
    )
    return llm.invoke(prompt).content


def call_claude(prompt, temperature=0.7, max_tokens=None, timeout=None):
    """调用Anthropic Claude系列模型"""
    llm = ChatAnthropic(
        model="claude-3-5-sonnet-20241022",
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout
    )
    return llm.invoke(prompt).content


def call_gemini(prompt, temperature=0.7, max_tokens=None, timeout=None):
    """调用Google Gemini系列模型"""
    llm = GoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout
    )
    return llm.invoke(prompt)


def call_ollama(prompt, model='deepseek-r1:7b', temperature=0.7):
    """
    调用本地Ollama服务托管的模型
    自动处理DeepSeek-R1的<think>标记，提取实际回答内容
    """
    response = ollama.generate(
        model=model,
        options={"temperature": temperature},
        prompt=prompt
    )
    
    result = response['response']
    
    # DeepSeek-R1会输出<think>推理过程</think>实际回答的格式
    if '</think>' in result:
        return result.split("</think>", 1)[1].strip()
    
    return result


# ============================================================
# 批量推理调度
# ============================================================

def batch_inference(task_type, prompt_csv, output_csv, model_config):
    """
    批量执行模型推理任务
    
    Args:
        task_type: 'workflow' 或 'code'
        prompt_csv: 提示词文件路径
        output_csv: 输出结果保存路径
        model_config: 字典格式，包含以下键：
            - 'provider': 'gpt'/'claude'/'gemini'/'ollama'
            - 'model_name': 模型标识符（ollama专用）
            - 'temperature': 采样温度
    """
    prompts_df = pd.read_csv(prompt_csv)
    provider = model_config['provider']
    temperature = model_config.get('temperature', 0.7)
    
    # 初始化输出文件
    with open(output_csv, "w", newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'task_id', 'response_id', 'prompt_type', 'response_type',
            'Arcpy', 'llm_model', 'response_content', 'task_length'
        ])
    
    # 逐个处理提示词
    total = len(prompts_df)
    for idx, row in prompts_df.iterrows():
        if idx > 0:
            print('\r' + ' ' * 80 + '\r', end='')
        print(f'进度: {idx + 1}/{total} | 任务ID: {row["task_id"]}', end='', flush=True)
        
        prompt = row['prompt_content']
        
        # 每个提示词请求3次以评估稳定性
        responses = []
        for _ in range(3):
            if provider == 'gpt':
                response = call_gpt(prompt, temperature)
            elif provider == 'claude':
                response = call_claude(prompt, temperature)
            elif provider == 'gemini':
                response = call_gemini(prompt, temperature)
            elif provider == 'ollama':
                response = call_ollama(
                    prompt,
                    model=model_config.get('model_name', 'deepseek-r1:7b'),
                    temperature=temperature
                )
            else:
                raise ValueError(f"不支持的模型提供商: {provider}")
            
            responses.append(response)
        
        # 确定提示词类型标签
        if row['domain_knowledge'] and row['dataset']:
            prompt_type = 'domain_and_dataset'
        elif row['domain_knowledge']:
            prompt_type = 'domain'
        elif row['dataset']:
            prompt_type = 'dataset'
        else:
            prompt_type = 'original'
        
        # 写入响应结果
        with open(output_csv, "a", newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            for i, response in enumerate(responses):
                task_length = calculate_workflow_length(response) if task_type == 'workflow' else 'none'
                
                writer.writerow([
                    row['task_id'],
                    f"{row['task_id']}{task_type}{i}",
                    prompt_type,
                    task_type,
                    row['Arcpy'],
                    provider,
                    response,
                    task_length
                ])
    
    print()  # 换行
```

**2. `prompt_generation.py` - 提示词矩阵构建**

此脚本负责从基准数据集中读取任务信息，并为每个任务衍生出四种配置组合的提示词变体。通过控制是否包含领域知识和数据集描述两个维度，系统地生成了用于工作流推导和代码实现的完整提示词集合，最终输出为`code_prompts.csv`和`workflow_prompts.csv`两个文件。

```python
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
```

值得一提的是，若需在正式执行前快速验证流程可行性，可临时调整任务处理范围。在`generate_all_prompts`函数中定位到遍历数据集的循环语句`for idx, row in tasks_df.iterrows():`，将其改为`for idx, row in tasks_df.head(5).iterrows():`即可将处理对象限定为前5个任务。这样生成的提示词文件仅包含20条记录，对应的推理时长可缩短至30分钟到2小时，便于在短时间内确认整套系统能否正常运转。待测试通过后恢复原始代码，重新生成完整的提示词集合并启动全量推理即可。

**3. `Inference.py` - 模型推理调度中枢**

作为整个评测流程的执行入口，该脚本批量调用前述工具函数，向不同LLM发起推理请求。当前配置优先使用本地`ollama`模型（`deepseek-r1`），商业模型（GPT-4、Claude、Gemini）的调用接口已完整保留但以注释形式存在，可根据需要随时启用。

脚本会读取生成好的提示词文件，依次请求模型完成工作流设计和代码编写任务，并将响应结果结构化地存储到对应的CSV文件中。

```python
"""
GeoAnalystBench 模型推理调度器
批量调用不同LLM完成工作流设计和代码生成任务
"""

import os
from utils import batch_inference


# ============================================================
# API密钥配置（商业模型需要）
# ============================================================

# 使用商业模型时，请取消注释并填入有效密钥
# os.environ["OPENAI_API_KEY"] = "your_openai_key_here"
# os.environ["ANTHROPIC_API_KEY"] = "your_claude_key_here"
# os.environ["GOOGLE_API_KEY"] = "your_gemini_key_here"


# ============================================================
# 模型配置
# ============================================================

MODELS = {
    'ollama_deepseek': {
        'provider': 'ollama',
        'model_name': 'deepseek-r1:7b',
        'temperature': 0.7,
        'output_suffix': 'ollama_deepseek-r1-7b'
    },
    'gpt4': {
        'provider': 'gpt',
        'temperature': 0.7,
        'output_suffix': 'gpt4'
    },
    'claude': {
        'provider': 'claude',
        'temperature': 0.7,
        'output_suffix': 'claude'
    },
    'gemini': {
        'provider': 'gemini',
        'temperature': 0.7,
        'output_suffix': 'gemini'
    },
}


# ============================================================
# 推理执行函数
# ============================================================

def run_inference_for_model(model_key, run_code=True, run_workflow=True):
    """
    为指定模型执行推理任务
    
    Args:
        model_key: MODELS字典中的模型标识
        run_code: 是否执行代码生成任务
        run_workflow: 是否执行工作流生成任务
    """
    if model_key not in MODELS:
        raise ValueError(f"未知的模型配置: {model_key}")
    
    config = MODELS[model_key]
    output_suffix = config['output_suffix']
    
    print(f"\n{'='*60}")
    print(f"开始使用 {model_key} 进行推理")
    print(f"{'='*60}")
    
    if run_code:
        print(f"\n[1/2] 执行代码生成任务...")
        batch_inference(
            task_type='code',
            prompt_csv='codes/code_prompts.csv',
            output_csv=f'codes/code_responses_{output_suffix}.csv',
            model_config=config
        )
    
    if run_workflow:
        print(f"\n[2/2] 执行工作流生成任务...")
        batch_inference(
            task_type='workflow',
            prompt_csv='codes/workflow_prompts.csv',
            output_csv=f'codes/workflow_responses_{output_suffix}.csv',
            model_config=config
        )
    
    print(f"\n{model_key} 推理完成！")
    print(f"{'='*60}\n")


# ============================================================
# 主执行流程
# ============================================================

def main():
    """
    主执行入口
    默认使用本地Ollama模型，如需测试其他模型请修改此处
    """
    print("GeoAnalystBench 模型推理系统")
    print(f"{'='*60}")
    
    # 当前配置：仅使用本地Ollama模型
    run_inference_for_model('ollama_deepseek')
    
    # 使用其他模型的示例（需先配置API密钥）：
    # run_inference_for_model('gpt4')
    # run_inference_for_model('claude')
    # run_inference_for_model('gemini')
    
    print("\n所有推理任务执行完毕！")


if __name__ == "__main__":
    main()
```

**完整复现流程**

基准测试的实施按照固定的先后顺序展开，前序操作产生的数据文件会被后续程序读取利用。全过程需要依次调用两个独立脚本，每个脚本完成任务后均会向`codes/`文件夹写入特定格式的CSV表格。

首先运行提示词构建程序：

```bash
python codes/prompt_generation.py
```

该步骤会在`codes/`目录下创建两份结构化数据表：

- `code_prompts.csv`：汇集200条代码生成提示词
- `workflow_prompts.csv`：收录200条工作流构建提示词

二者采用统一的表格结构，各列含义如下：

| 字段名 | 说明 |
|--------|------|
| task_id | 任务编号，取值范围1-50 |
| type | 提示词类型，标识为`code`或`workflow` |
| domain_knowledge | 布尔值，标记是否包含领域知识 |
| dataset | 布尔值，标记是否包含数据集描述 |
| Arcpy | 布尔值，标识任务是否需要使用闭源库 |
| prompt_content | 完整的提示词文本内容 |

每个任务基于`domain_knowledge`与`dataset`两个维度的组合衍生出四种配置版本，因此50个任务共产生200条记录。

随后启动推理调度器：

```bash
python codes/Inference.py
```

此阶段将批量处理先前构建的提示词样本，通过API接口获取LLM的应答内容。脚本预设使用本地部署的`ollama`服务，如需接入商业平台，须在配置区填入有效密钥并解除相关代码注释。执行期间，终端会实时展示任务进度编号。

推理工作完结后，`codes/`文件夹内将新增两份记录模型响应的CSV文档，命名遵循`{任务类型}_responses_{模型标识}.csv`规范。以使用`ollama`的`deepseek-r1:7b`模型为例，所得文件为：

- `code_responses_ollama_deepseek-r1-7b.csv`
- `workflow_responses_ollama_deepseek-r1-7b.csv`

这些文件记录了模型的全部响应内容，各字段定义为：

| 字段名 | 说明 |
|--------|------|
| task_id | 对应的任务编号 |
| response_id | 响应的唯一标识符，格式为`{task_id}{type}{序号}` |
| prompt_type | 提示词配置类型，可能值为`original`、`domain`、`dataset`、`domain_and_dataset` |
| response_type | 响应类型，与任务类型保持一致 |
| Arcpy | 标识是否使用闭源库 |
| llm_model | 执行推理的模型名称 |
| response_content | 模型返回的完整文本 |
| task_length | 工作流步骤数量（仅工作流任务有效，代码任务此项为`none`） |

值得注意的是，每个任务配置会请求三次独立推理（由`utils.py`中的循环控制），因此最终文件包含600条记录（50个任务×4种配置×3次重复）。其中`task_length`通过正则表达式自动提取工作流中的步骤编号，为后续评估提供量化依据。

至此，原始数据已转化为可供分析的结构化记录。研究者可基于这些输出计算各模型在不同配置下的准确率、工作流长度偏差等指标，并与数据集中专家设计的标准方案进行对比验证。

**Ollama本地部署与项目集成实践**

在GeoAnalystBench的技术架构中，Ollama承担着关键的模型运行支撑功能。对其工作机制建立完整认知是推动复现流程顺畅进行的关键所在。

**架构定位与运作原理**

Ollama采用客户端-服务器分离设计，本质上是一套完整的本地LLM运行框架。首次执行任意`ollama`指令时，系统会自动在后台启动HTTP服务进程，默认监听`127.0.0.1:11434`端口。此后该服务持续运行，负责模型加载、推理计算等核心任务。用户在终端输入的各类命令实则充当客户端角色，经由本地接口完成交互。

这种架构带来的直接优势在于，任何具备HTTP通信能力的程序均可调用已部署的模型资源。项目中的Python脚本正是利用这一特性，通过`ollama`库直接与后台服务对接，跳过了命令行界面这一中转层。

**安装配置流程**

各操作系统的安装方式存在差异。macOS与Linux环境可执行官方提供的一键安装脚本，Windows平台则需下载对应的exe安装程序。针对Windows用户，若希望调整程序主体的落盘位置，可在命令行界面通过附加`/DIR`参数的方式实现，例如`.\OllamaSetup.exe /DIR="D:\Ollama"`，从而将接近5GB的核心组件置于非系统分区。完成安装后，通过`ollama pull`指令拉取相应模型，关联文件会自动保存至默认存储路径。

模型文件的存放位置遵循系统约定：Windows环境下位于`C:\Users\<用户名>\.ollama\models`，Linux系统通常为`~/.ollama/models`。由于大型模型占用空间可观，需要调整该路径以避免系统分区容量告急。Windows版本较新的Ollama在图形界面中提供了"Model location"选项，可直接浏览并选定目标文件夹，该方式在当前版本中优先级更高。传统的`OLLAMA_MODELS`环境变量方案在部分场景下可能不被采纳，但仍可作为辅助手段保留。

Windows环境下若搭载NVIDIA显卡，需确保GPU加速机制正常启用。通过新建系统环境变量`OLLAMA_GPU_LAYER`并赋值为`cuda`，同时设置`CUDA_VISIBLE_DEVICES`来选定具体设备（多GPU情况下按`nvidia-smi`输出的编号填写，通常为`0`），即可令模型优先加载至显存而非占用大量内存空间。配置生效需重启Ollama服务，可借助任务管理器或`nvidia-smi`工具验证显存使用情况。

**量化策略与资源需求**

Ollama的模型库采用GGUF量化格式，这是一种将浮点参数压缩为低位整数的技术方案。量化级别直接影响文件体积与推理质量的平衡点：

- Q2/Q3级别可将模型压缩至原始大小的20-30%，但精度损失明显
- Q4量化在体积与性能间达到较优平衡，7B参数模型压缩后约需3.5-4GB存储
- Q5/Q6级别保留更多细节，文件尺寸随之增加至5-7GB范围

运行时的内存占用与磁盘文件大小基本相当，但需额外预留1-2GB空间用于推理中间结果和上下文缓存。以`deepseek-r1:7b`的Q4版本为例，实际运行时建议系统保有5-6GB可用内存。若配备独立显卡，Ollama会优先利用显存加速计算，显存不足时自动回退至系统内存，或采用混合加载策略。

针对16GB内存+4GB显存的典型配置，7B规模的Q4量化模型可流畅运行，但13B及以上参数量的模型会面临显存溢出，系统内存需求突破9GB阈值，运行期间需关闭非必要进程以维持稳定性。

**服务启动与验证**

项目代码中使用的`ollama`库会自动管理后台服务的生命周期。但若需手动操作，可通过以下方式触发服务启动：

```bash
ollama serve
```

该命令会阻塞当前终端并持续输出日志。也可执行任意查询指令如`ollama list`来隐式唤醒服务进程。

验证部署状态最直接的方法是访问API健康检查端点。打开浏览器访问`http://localhost:11434`，若显示"Ollama is running"消息，表明服务已正常就绪。此时即可通过Python代码直接调用已下载的模型，无需预先执行`ollama run`启动交互式会话。

**与项目代码的衔接**

回顾`utils.py`中的`call_ollama`函数实现：

```python
def call_ollama(prompt, model='deepseek-r1:7b', temperature=0.7):
    response = ollama.generate(
        model=model,
        options={"temperature": temperature},
        prompt=prompt
    )
    result = response['response']
    if '</think>' in result:
        return result.split("</think>", 1)[1].strip()
    return result
```

此函数借助`ollama.generate()`方法通过本地API接口发起生成操作，参数`model`指定使用的模型标识符。DeepSeek-R1系列模型会在输出中包裹`<think>...</think>`标记来呈现推理过程，代码逻辑依靠文本分割来提取实际回答内容。这一细节处理确保了返回结果与其他商业模型保持一致的格式规范。

执行`Inference.py`时，脚本会连续执行数百轮模型调用。得益于Ollama服务的持久化运行特性，模型仅在首次调用时加载进内存，后续请求复用已初始化的模型实例，避免了重复加载带来的时间开销。整个推理过程中，Ollama服务进程在后台静默处理所有请求，Python脚本通过库封装的接口完成同步或异步通信，两者协作构成了完整的本地推理链路。