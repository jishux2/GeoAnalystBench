# codes/setup_evaluation_env.py
"""
评估环境配置脚本

创建独立的Python虚拟环境并安装地理空间分析所需的全部依赖。
该环境与项目主环境隔离，避免版本冲突，专用于执行模型生成的代码。

运行方式：
    python codes/setup_evaluation_env.py

生成产物：
    - venv_gis_opensource/          虚拟环境目录
    - codes/evaluator_config.json   解释器路径配置

后续操作：
    配置完成后，可直接运行 python codes/run_evaluation.py
    系统会自动从配置文件读取解释器路径并执行代码验证
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
    
    # 依赖包按功能分组组织，便于后续维护与扩展
    packages = [
        # === 基础科学计算栈 ===
        "numpy>=1.24.0",           # 数值计算基础
        "pandas>=2.0.0",           # 数据框操作
        "scipy>=1.11.0",           # 科学计算算法
        "scikit-learn>=1.3.0",     # 机器学习工具
        
        # === 矢量数据处理 ===
        "geopandas>=0.14.0",       # 地理数据框，基于pandas扩展
        "shapely>=2.0.0",          # 几何对象操作
        "fiona>=1.9.0",            # 矢量格式I/O（Shapefile/GeoJSON等）
        "pyproj>=3.6.0",           # 坐标系转换
        "rtree>=1.0.0",            # 空间索引加速
        
        # === 栅格数据处理 ===
        "rasterio>=1.3.0",         # 栅格格式I/O与操作
        "scikit-image>=0.21.0",    # 图像处理算法
        
        # === 空间分析专用库 ===
        "pykrige>=1.7.0",          # 克里金插值（地统计学）
        "pysal>=24.0",             # 空间分析工具集（包含libpysal等子包）
        "osmnx>=1.8.0",            # OpenStreetMap网络分析
        
        # === 可视化工具 ===
        "matplotlib>=3.7.0",       # 基础绘图
        "seaborn>=0.12.0",         # 统计可视化
        "contextily>=1.4.0",       # 底图服务集成
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
    
    # 将解释器路径持久化到配置文件
    # CodeExecutor会读取此配置，根据任务类型选择对应环境
    config_file = Path("codes/evaluator_config.json")
    import json
    
    config = {
        "opensource_interpreter": interpreter_path,
        "arcgis_interpreter": None  # 闭源任务环境需手动配置ArcGIS Pro的Python路径
    }
    
    config_file.parent.mkdir(parents=True, exist_ok=True)
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2)
    
    print(f"\n解释器路径已保存至：{config_file}")