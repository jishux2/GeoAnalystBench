# codes/setup_evaluation_env.py
"""
评估环境配置脚本
自动创建虚拟环境并安装地理空间处理库
"""

import subprocess
import sys
from pathlib import Path


def setup_opensource_env():
    """配置开源库环境"""
    env_path = Path("venv_gis_opensource")
    
    print("="*60)
    print("配置开源地理空间处理环境")
    print("="*60)
    
    # 创建虚拟环境
    if not env_path.exists():
        print("\n创建虚拟环境...")
        subprocess.run([sys.executable, "-m", "venv", str(env_path)], check=True)
        print("✓ 虚拟环境创建成功")
    else:
        print("\n虚拟环境已存在，跳过创建")
    
    # 确定pip路径
    if sys.platform == "win32":
        pip_path = env_path / "Scripts" / "pip.exe"
        python_path = env_path / "Scripts" / "python.exe"
    else:
        pip_path = env_path / "bin" / "pip"
        python_path = env_path / "bin" / "python"
    
    # 升级pip
    print("\n升级pip...")
    subprocess.run([str(python_path), "-m", "pip", "install", "--upgrade", "pip"], check=True)
    
    # 安装核心库
    print("\n安装地理空间处理库...")
    packages = [
        "geopandas>=0.14.0",
        "rasterio>=1.3.0",
        "shapely>=2.0.0",
        "fiona>=1.9.0",
        "pyproj>=3.6.0",
        "rtree>=1.0.0",
        "scikit-learn>=1.3.0",
        "scipy>=1.11.0",
        "matplotlib>=3.7.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "pykrige>=1.7.0",
        "contextily>=1.4.0",
        "pysal>=24.0",
        "osmnx>=1.8.0",
    ]
    
    for package in packages:
        print(f"  安装 {package}...")
        subprocess.run([str(pip_path), "install", package], check=True)
    
    print("\n" + "="*60)
    print("环境配置完成！")
    print("="*60)
    print(f"\n虚拟环境路径：{env_path.absolute()}")
    print(f"Python解释器：{python_path.absolute()}")
    print("\n使用方法：")
    
    if sys.platform == "win32":
        print(f"  激活环境：{env_path}\\Scripts\\activate")
    else:
        print(f"  激活环境：source {env_path}/bin/activate")
    
    print(f"  执行评测：python codes/run_evaluation.py")
    
    return str(python_path.absolute())


if __name__ == "__main__":
    interpreter_path = setup_opensource_env()
    
    # 将解释器路径写入配置文件
    config_file = Path("codes/evaluator_config.json")
    import json
    
    config = {
        "opensource_interpreter": interpreter_path,
        "arcgis_interpreter": None  # 后续配置ArcGIS Pro环境时填入
    }
    
    config_file.parent.mkdir(parents=True, exist_ok=True)
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2)
    
    print(f"\n解释器路径已保存至：{config_file}")