"""
模型推理脚本
批量调用LLM完成工作流设计和代码编写任务
当前配置：优先使用本地ollama模型
"""

import os
from utils import call_api

# ==================== API密钥配置 ====================
# 如需使用商业模型，请取消注释并填入对应API密钥

# OPENAI_API_KEY = "your_openai_key_here"
# CLAUDE_API_KEY = "your_claude_key_here"
# GEMINI_API_KEY = "your_gemini_key_here"

# os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
# os.environ["ANTHROPIC_API_KEY"] = CLAUDE_API_KEY
# os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY


# ==================== Ollama本地模型推理 ====================
def run_ollama_inference(model_name='deepseek-r1', temperature=0.7):
    """使用本地ollama模型进行推理"""
    print(f"\n开始使用Ollama模型进行推理：{model_name}")
    print("=" * 60)
    
    print("\n[1/2] 执行代码生成任务...")
    call_api(
        api_type='code',
        prompt_file='codes/code_prompts.csv',
        output_file=f'codes/code_responses_ollama_{model_name.replace(":", "_")}.csv',
        model='ollama',
        ollama_model=model_name,
        temperature=temperature
    )
    
    print("\n[2/2] 执行工作流生成任务...")
    call_api(
        api_type='workflow',
        prompt_file='codes/workflow_prompts.csv',
        output_file=f'codes/workflow_responses_ollama_{model_name.replace(":", "_")}.csv',
        model='ollama',
        ollama_model=model_name,
        temperature=temperature
    )
    
    print("\n" + "=" * 60)
    print("Ollama推理完成！")


# ==================== 商业模型推理（暂时注释） ====================
"""
def run_gpt_inference(temperature=0.7):
    使用GPT模型进行推理
    print("\n开始使用GPT-4模型进行推理...")
    print("=" * 60)
    
    print("\n[1/2] 执行代码生成任务...")
    call_api(
        api_type='code',
        prompt_file='codes/code_prompts.csv',
        output_file='codes/code_responses_gpt.csv',
        model='gpt4',
        temperature=temperature
    )
    
    print("\n[2/2] 执行工作流生成任务...")
    call_api(
        api_type='workflow',
        prompt_file='codes/workflow_prompts.csv',
        output_file='codes/workflow_responses_gpt.csv',
        model='gpt4',
        temperature=temperature
    )
    
    print("\n" + "=" * 60)
    print("GPT推理完成！")


def run_claude_inference(temperature=0.7):
    使用Claude模型进行推理
    print("\n开始使用Claude模型进行推理...")
    print("=" * 60)
    
    print("\n[1/2] 执行代码生成任务...")
    call_api(
        api_type='code',
        prompt_file='codes/code_prompts.csv',
        output_file='codes/code_responses_claude.csv',
        model='claude',
        temperature=temperature
    )
    
    print("\n[2/2] 执行工作流生成任务...")
    call_api(
        api_type='workflow',
        prompt_file='codes/workflow_prompts.csv',
        output_file='codes/workflow_responses_claude.csv',
        model='claude',
        temperature=temperature
    )
    
    print("\n" + "=" * 60)
    print("Claude推理完成！")


def run_gemini_inference(temperature=0.7):
    使用Gemini模型进行推理
    print("\n开始使用Gemini模型进行推理...")
    print("=" * 60)
    
    print("\n[1/2] 执行代码生成任务...")
    call_api(
        api_type='code',
        prompt_file='codes/code_prompts.csv',
        output_file='codes/code_responses_gemini.csv',
        model='gemini',
        temperature=temperature
    )
    
    print("\n[2/2] 执行工作流生成任务...")
    call_api(
        api_type='workflow',
        prompt_file='codes/workflow_prompts.csv',
        output_file='codes/workflow_responses_gemini.csv',
        model='gemini',
        temperature=temperature
    )
    
    print("\n" + "=" * 60)
    print("Gemini推理完成！")
"""


# ==================== 主函数 ====================
def main():
    """主执行流程"""
    print("GeoAnalystBench 模型推理系统")
    print("=" * 60)
    
    # 当前配置：仅使用本地Ollama模型
    run_ollama_inference(model_name='deepseek-r1', temperature=0.7)
    
    # 如需使用其他模型，请取消下方注释并配置对应API密钥
    # run_gpt_inference(temperature=0.7)
    # run_claude_inference(temperature=0.7)
    # run_gemini_inference(temperature=0.7)
    
    print("\n所有推理任务执行完毕！")


if __name__ == "__main__":
    main()