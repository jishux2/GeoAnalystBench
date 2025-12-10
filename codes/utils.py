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