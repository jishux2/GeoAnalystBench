"""
矢量数据探查工具
支持 Shapefile、GeoJSON、GeoPackage 等 OGR 兼容格式
"""

import sys
import geopandas as gpd


def inspect(filepath: str):
    gdf = gpd.read_file(filepath)

    print(f"File: {filepath}")
    print(f"Records: {len(gdf)}")
    print(f"Geometry type(s): {gdf.geometry.geom_type.value_counts().to_dict()}")

    if gdf.crs:
        print(f"CRS: {gdf.crs}")
    else:
        print("CRS: not defined")

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