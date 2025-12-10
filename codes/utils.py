"""
工具函数库
提供LLM API调用、文本解析和评估指标计算等核心功能
"""

import re
import csv
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import GoogleGenerativeAI
import ollama


# ==================== 文本处理工具 ====================

def extract_task_list(text):
    """从文本中提取任务列表，过滤掉编号行"""
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    return [line for line in lines if not re.match(r'^\d+\.', line)]


def find_task_length(workflow_text):
    """从工作流文本中提取最大步骤编号"""
    max_number = 0
    
    for line in workflow_text.split("\n"):
        i = 0
        while i < len(line):
            if line[i].isdigit():
                # 检查是否为两位数
                if i + 1 < len(line) and line[i + 1].isdigit():
                    if i + 2 < len(line) and line[i + 2] == '.':
                        max_number = max(max_number, int(line[i:i+2]))
                        i += 2
                # 单位数且后跟句点
                elif i + 1 < len(line) and line[i + 1] == '.':
                    num = int(line[i])
                    if num <= 10:  # 避免误识别其他数字
                        max_number = max(max_number, num)
                    i += 1
                else:
                    i += 1
            else:
                i += 1
    
    return max_number


# ==================== LLM API调用接口 ====================

def call_gpt(prompt, temperature=0.7, max_tokens=None, timeout=None):
    """调用GPT模型"""
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout
    )
    return llm.invoke(prompt).content


def call_claude(prompt, temperature=0.7, max_tokens=None, timeout=None):
    """调用Claude模型"""
    llm = ChatAnthropic(
        model="claude-3-5-sonnet-20241022",
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout
    )
    return llm.invoke(prompt).content


def call_gemini(prompt, temperature=0.7, max_tokens=None, timeout=None):
    """调用Gemini模型"""
    llm = GoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout
    )
    return llm.invoke(prompt)


def call_ollama(prompt, model='deepseek-r1', temperature=0.7):
    """调用本地Ollama模型，自动处理DeepSeek-R1的思维链标记"""
    response = ollama.generate(
        model=model,
        options={"temperature": temperature},
        prompt=prompt
    )
    
    result = response['response']
    
    # DeepSeek-R1模型会输出<think>...</think>标记，需要提取实际内容
    if '</think>' in result:
        return result.split("</think>")[1].strip()
    
    return result


# ==================== 批量推理调度 ====================

def call_api(api_type, prompt_file, output_file, model, 
             ollama_model='deepseek-r1', temperature=0.7):
    """
    批量调用LLM进行推理
    
    参数说明：
        api_type: 'workflow'或'code'，指定任务类型
        prompt_file: 提示词CSV文件路径
        output_file: 响应结果输出路径
        model: 'gpt4'/'claude'/'gemini'/'ollama'
        ollama_model: 使用ollama时的具体模型名
        temperature: 采样温度参数
    """
    prompts = pd.read_csv(prompt_file)
    
    # 初始化输出文件
    with open(output_file, "w", newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'task_id', 'response_id', 'prompt_type', 'response_type',
            'Arcpy', 'llm_model', 'response_content', 'task_length'
        ])
    
    # 逐个处理提示词
    total = len(prompts)
    for idx, row in prompts.iterrows():
        if idx > 0:
            print('\r' + ' ' * 60 + '\r', end='')
        print(f'{idx + 1}/{total}', end='', flush=True)
        
        prompt = row['prompt_content']
        
        # 每个提示词请求3次以评估稳定性
        responses = []
        for _ in range(3):
            if model == 'gpt4':
                response = call_gpt(prompt, temperature)
            elif model == 'claude':
                response = call_claude(prompt, temperature)
            elif model == 'gemini':
                response = call_gemini(prompt, temperature)
            elif model == 'ollama':
                response = call_ollama(prompt, ollama_model, temperature)
            else:
                raise ValueError(f"不支持的模型类型：{model}")
            
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
        with open(output_file, "a", newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            for i, response in enumerate(responses):
                if api_type == 'workflow':
                    task_length = find_task_length(response)
                else:
                    task_length = 'none'
                
                writer.writerow([
                    row['task_id'],
                    f"{row['task_id']}{api_type}{i}",
                    prompt_type,
                    api_type,
                    row['Arcpy'],
                    model,
                    response,
                    task_length
                ])
    
    print()  # 换行


# ==================== 评估指标计算 ====================

def calculate_workflow_length_loss(annotations, responses):
    """
    计算工作流长度的平均绝对误差
    
    参数：
        annotations: 包含专家标注的DataFrame
        responses: 包含模型响应的DataFrame
    
    返回：
        平均长度偏差值
    """
    total_loss = 0
    
    for _, annotation in annotations.iterrows():
        task_responses = responses[responses["task_id"] == annotation["id"]]
        
        for _, response in task_responses.iterrows():
            total_loss += abs(
                response["task_length"] - annotation["task_length"]
            )
    
    return total_loss / len(annotations)