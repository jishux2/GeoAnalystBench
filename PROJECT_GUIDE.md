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
├─ docs/                                      # 专项技术指引
│  ├─ ollama-integration.md                   # Ollama平台部署流程与运行时调优
│  ├─ deepseek-api-guide.md                   # DeepSeek异步推理架构与接口应用
│  └─ code-validation-deployment.md           # 自动化验证框架的搭建与运维实践
│
├─ case_study/                                # 实证研究材料
│  └─ figures/                                # 可视化成果
│     ├─ elk.png                              # 麋鹿活动范围原始数据呈现
│     ├─ ElkAI.png                            # 基于AI的麋鹿迁徙模式预测结果
│     ├─ traffic.png                          # 车祸地点空间分布原图
│     └─ TrafficAI.png                        # 模型识别的事故热点区域
│
├─ codes/                                     # 核心实现代码
│  ├─ prompt_generation.py                    # 为50个任务构建不同组合的提示词
│  ├─ Inference.py                            # 调度多个LLM完成推理任务的主控脚本
│  └─ utils.py                                # 通用函数库，封装API调用、文本解析及指标计算
│
├─ prompts/                                   # 生成的提示词集合
│  ├─ code_prompts.csv                        # 代码实现任务的提示词矩阵
│  └─ workflow_prompts.csv                    # 工作流构建任务的提示词矩阵
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

**1. `prompt_generation.py` - 提示词矩阵构建**

此脚本负责从基准数据集中读取任务信息，并为每个任务衍生出四种配置组合的提示词变体。通过控制是否包含领域知识和数据集描述两个维度，系统地生成了用于工作流推导和代码实现的完整提示词集合，最终输出为`prompts/code_prompts.csv`和`prompts/workflow_prompts.csv`两个文件。

```python
"""
GeoAnalystBench 提示词生成器
从任务数据集出发，为50道地理分析题目编制多类参数组合的交互文本
"""

import pandas as pd
import csv
from pathlib import Path


# ============================================================
# 文本格式化工具
# ============================================================

def wrap_text(text, max_line_length=80):
    """
    将冗长文本沿词汇边界拆分成多行
    
    Args:
        text: 待处理的原始字符串
        max_line_length: 每行允许的字符数上限
    
    Returns:
        按指定宽度折叠后的文本
    """
    # 实现省略


def format_multiline_text(text):
    """
    逐行应用折叠规则重组多段文本
    
    Args:
        text: 携带换行符的源内容
    
    Returns:
        各段落均已格式化的结果
    """
    # 实现省略


# ============================================================
# 提示词模板构建
# ============================================================

def build_workflow_prompt(...):
    """组装工作流生成场景所需的引导文本"""
    # 实现省略


def build_code_prompt(...):
    """构造代码编写任务对应的指令模板"""
    # 实现省略


# ============================================================
# 主生成流程
# ============================================================

def generate_all_prompts(
    dataset_path='dataset/GeoAnalystBench.csv',
    code_output='prompts/code_prompts.csv',
    workflow_output='prompts/workflow_prompts.csv'
):
    """
    为全部任务批量生成提示词矩阵
    
    字段来源与构造规则：
    
    - task_id: 由数据集行索引递增计算（idx + 1）
    - type: 依据输出目标分别标注为'code'或'workflow'
    - domain_knowledge/dataset: 通过配置组合的遍历逻辑赋予布尔值
    - Arcpy: 读取'Open Source'列内容并执行逻辑非运算（'T'对应False）
    - prompt_content: 将数据集相关列传入模板函数，结合参数生成完整文本
    
    代码生成提示词的结构样式：
    
    '''
    As a Geospatial data scientist, generate a python file to solve the proposed task.
    
    [Task]:
    {源自'Task'字段，由wrap_text执行分行}
    
    [Instruction]:
    {抽取'Instruction'内容，同样经wrap_text折行}
    
    [Domain Knowledge]:  # 依domain_knowledge值决定是否出现
    {提取'Domain Knowledge'列，应用wrap_text处理}
    
    [Dataset Description]:  # 由dataset参数控制纳入与否
    {采用'Dataset Description'数据，通过format_multiline_text转换}
    
    [Key Notes]:
    1. Use **automatic reasoning** and clearly explain each subtask before performing it (ReAct approach).
    2. Using latest python packages for code generation.
    3. Put all code under main function, no helper functions.
    4. Limit your output to code, no extra information.
    5. Use latest **Arcpy** functions only.  # Arcpy为False时切换为开源库指引
    
    [Expected Sample Output Begin]
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
    [Expected Sample Output End]
    '''
    
    工作流提示词呈现的差异特征：
    - 角色声明调整为"generate a workflow"形式
    - Key Notes转向要求NetworkX图形化表达
    - 示例部分展现tasks列表定义及可视化脚本
    
    配置空间的展开策略：
    各任务借助双层循环派生4种参数组合：
    - (False, False) - 原始精简版
    - (True, False) - 融入领域专业知识
    - (False, True) - 嵌入数据集说明
    - (True, True) - 集成全部辅助信息
    
    50项任务 × 4类配置 × 2个维度 = 共计400条记录
    """
    
    Path('prompts').mkdir(exist_ok=True)
    
    tasks_df = pd.read_csv(dataset_path)
    
    # 初始化输出文件
    header = ['task_id', 'type', 'domain_knowledge', 'dataset', 'Arcpy', 'prompt_content']
    
    with open(code_output, 'w', newline='', encoding='utf-8') as f:
        csv.writer(f).writerow(header)
    
    with open(workflow_output, 'w', newline='', encoding='utf-8') as f:
        csv.writer(f).writerow(header)
    
    # 为每个任务生成四种配置的提示词
    for idx, row in tasks_df.iterrows():
        # 构建逻辑省略
        ...
    
    print("\n提示词生成完成！")
    print(f"代码提示词: {code_output}")
    print(f"工作流提示词: {workflow_output}")


if __name__ == "__main__":
    generate_all_prompts()
```

值得一提的是，若需在正式执行前快速验证流程可行性，可临时调整任务处理范围。在`generate_all_prompts`函数中定位到遍历数据集的循环语句`for idx, row in tasks_df.iterrows():`，将其改为`for idx, row in tasks_df.head(5).iterrows():`即可将处理对象限定为前5个任务。这样生成的提示词文件仅包含20条记录，对应的推理时长可缩短至30分钟到2小时，便于在短时间内确认整套系统能否正常运转。待测试通过后恢复原始代码，重新生成完整的提示词集合并启动全量推理即可。

**2. `utils.py` - 工具函数库**

> *该模块在原始实现中封装了多个商业模型（GPT、Claude、Gemini）及本地Ollama模型的统一调用接口，并提供了文本解析、工作流长度计算等辅助功能。由于后续开发中采用了针对特定平台优化的异步推理架构，原有的同步调用方式及部分存在缺陷的实现逻辑（如`calculate_workflow_length`方法）已被完全替代，此处不再展示具体代码。详细的API集成方案可参阅`docs/`目录下的专项文档。*

**3. `Inference.py` - 模型推理调度中枢**

> *作为原始方案中的执行入口，该脚本通过顺序调用`utils.py`中的接口函数，依次向各类LLM发起推理请求。这种串行架构虽然具备通用性，但在处理大规模任务时存在明显的效率瓶颈。项目后续引入的并发推理系统已从根本上重构了任务调度逻辑，原实现不再参与实际执行流程，故此处省略代码展示。*

**完整复现流程**

基准测试的实施按照固定的先后顺序展开，前序操作产生的数据文件会被后续程序读取利用。全过程需要依次调用两个独立脚本，每个脚本完成任务后均会向`prompts/`文件夹写入特定格式的CSV表格。

首先运行提示词构建程序：

```bash
python codes/prompt_generation.py
```

该步骤会在`prompts/`目录下创建两份结构化数据表：

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

每个任务基于`domain_knowledge`与`dataset`两个维度的组合延展为四种配置版本，因此50个任务共产生200条记录。

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

