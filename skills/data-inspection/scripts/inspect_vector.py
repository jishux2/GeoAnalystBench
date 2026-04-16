# skills/data-inspection/scripts/inspect_vector.py
"""
矢量数据探查工具
支持 Shapefile、GeoJSON、GeoPackage 等 OGR 兼容格式
对缺少 .cpg 声明的 Shapefile 自动尝试 GBK 回退
"""

import sys
from pathlib import Path
import geopandas as gpd


def detect_shapefile_encoding(filepath: str) -> str:
    """
    推断 Shapefile 的属性编码。

    优先读取 .cpg 伴生文件中的声明；若不存在，则以 UTF-8
    试读并在失败时回退至 GBK（覆盖中文 Windows 环境下
    创建的绝大多数 Shapefile）。
    """
    shp_path = Path(filepath)
    cpg_path = shp_path.with_suffix(".cpg")

    if cpg_path.exists():
        declared = cpg_path.read_text(encoding="ascii", errors="ignore").strip()
        if declared:
            return declared

    # 尝试 UTF-8，若字段名解码出现替换字符则判定为非 UTF-8
    try:
        test_gdf = gpd.read_file(filepath, rows=1, encoding="utf-8")
        for col in test_gdf.columns:
            if "\ufffd" in col:
                return "gbk"
        return "utf-8"
    except Exception:
        return "gbk"


def inspect(filepath: str):
    suffix = Path(filepath).suffix.lower()

    if suffix == ".shp":
        encoding = detect_shapefile_encoding(filepath)
        gdf = gpd.read_file(filepath, encoding=encoding)
        encoding_note = f"Encoding: {encoding}"
    else:
        gdf = gpd.read_file(filepath)
        encoding_note = None

    print(f"File: {filepath}")
    print(f"Records: {len(gdf)}")
    print(f"Geometry type(s): {gdf.geometry.geom_type.value_counts().to_dict()}")

    if gdf.crs:
        print(f"CRS: {gdf.crs}")
    else:
        print("CRS: not defined")

    if encoding_note:
        print(encoding_note)

    bounds = gdf.total_bounds
    print(f"\nBounding box: [{bounds[0]:.6f}, {bounds[1]:.6f}, {bounds[2]:.6f}, {bounds[3]:.6f}]")

    print(f"\nFields ({len(gdf.columns)} total):")
    for col in gdf.columns:
        null_count = gdf[col].isna().sum()
        dtype = gdf[col].dtype
        null_info = f"  ({null_count} null)" if null_count > 0 else ""
        print(f"  {col}: {dtype}{null_info}")

    display_cols = [c for c in gdf.columns if c != 'geometry']
    if display_cols:
        sample = gdf[display_cols].head(3)
        if len(display_cols) <= 8:
            print(f"\nSample (first {len(sample)} records):")
            print(sample.to_string(index=False))
        else:
            print(f"\nSample (first {len(sample)} records, vertical layout):")
            for idx, row in sample.iterrows():
                print(f"\n  Record {idx}:")
                for col in display_cols:
                    val = row[col]
                    val_str = str(val)
                    if len(val_str) > 80:
                        val_str = val_str[:77] + "..."
                    print(f"    {col}: {val_str}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python inspect_vector.py <file_path>")
        sys.exit(1)

    inspect(sys.argv[1])