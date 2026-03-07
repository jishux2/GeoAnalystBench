"""
栅格数据探查工具
支持 GeoTIFF、IMG 等 GDAL 兼容格式
"""

import sys
import numpy as np
import rasterio


def inspect(filepath: str):
    with rasterio.open(filepath) as src:
        print(f"File: {filepath}")
        print(f"Dimensions: {src.width} x {src.height} pixels")
        print(f"Band count: {src.count}")

        if src.crs:
            print(f"CRS: {src.crs}")
        else:
            print("CRS: not defined")

        print(f"Transform: {src.transform}")
        print(f"Pixel size: {abs(src.transform.a):.6f} x {abs(src.transform.e):.6f}")
        
        bounds = src.bounds
        print(f"Bounding box: [{bounds.left:.6f}, {bounds.bottom:.6f}, {bounds.right:.6f}, {bounds.top:.6f}]")

        for i in range(1, src.count + 1):
            band = src.read(i)
            nodata = src.nodata

            desc = src.descriptions[i - 1]
            band_label = f"Band {i}" + (f" ({desc})" if desc else "")

            if nodata is not None:
                valid = band[band != nodata]
                print(f"\n{band_label}:")
                print(f"  NoData value: {nodata}")
                print(f"  Valid pixels: {len(valid)} / {band.size}")
            else:
                valid = band.flatten()
                print(f"\n{band_label}:")
                print(f"  NoData value: not set")

            if len(valid) > 0:
                print(f"  Value range: [{np.nanmin(valid)}, {np.nanmax(valid)}]")
                print(f"  Mean: {np.nanmean(valid):.4f}")
                print(f"  Dtype: {band.dtype}")
            else:
                print("  No valid pixels")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python inspect_raster.py <file_path>")
        sys.exit(1)

    inspect(sys.argv[1])