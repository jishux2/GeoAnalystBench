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
        'model_name': 'deepseek-r1',
        'temperature': 0.7,
        'output_suffix': 'ollama_deepseek-r1'
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