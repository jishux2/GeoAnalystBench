# codes/run_deepseek_inference.py
"""
DeepSeek API推理执行入口
支持增量推理、并发控制和错误恢复
"""

import asyncio
import os
import sys
from pathlib import Path

# 添加deepseek模块到路径
sys.path.insert(0, str(Path(__file__).parent))

from deepseek.async_inference import AsyncInferenceEngine


def main():
    """主执行函数"""
    # 从环境变量获取API密钥
    api_key = os.getenv("DEEPSEEK_API_KEY")
    
    if not api_key:
        print("错误：未设置DEEPSEEK_API_KEY环境变量")
        print("\n请根据操作系统选择对应命令：")
        print("  Linux/macOS: export DEEPSEEK_API_KEY='your_api_key_here'")
        print("  Windows CMD: set DEEPSEEK_API_KEY=your_api_key_here")
        print("  PowerShell:  $env:DEEPSEEK_API_KEY='your_api_key_here'")
        return
    
    print("="*60)
    print("GeoAnalystBench - DeepSeek API 推理系统")
    print("="*60)
    print(f"API密钥：{'*' * (len(api_key) - 8)}{api_key[-8:]}")
    print("="*60)
    
    # 配置推理引擎
    engine = AsyncInferenceEngine(
        api_key=api_key,
        max_concurrent=30,  # 可根据实际情况调整
        temperature=0.7
    )
    
    # 执行推理
    try:
        asyncio.run(engine.run_all())
        print("\n所有推理任务执行完毕！")
    
    except KeyboardInterrupt:
        print("\n\n推理被用户中断")
        print("已完成的结果已保存，下次运行将自动跳过这些任务")
    
    except Exception as e:
        print(f"\n执行过程中发生错误：{e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()