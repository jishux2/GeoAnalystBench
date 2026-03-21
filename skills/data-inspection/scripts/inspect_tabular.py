"""
表格数据探查工具
支持 CSV 和 Excel 格式
"""

import sys
from pathlib import Path
import pandas as pd


def inspect(filepath: str):
    path = Path(filepath)
    suffix = path.suffix.lower()

    if suffix == '.csv':
        df = pd.read_csv(filepath)
    elif suffix in ('.xls', '.xlsx'):
        df = pd.read_excel(filepath)
    else:
        print(f"Unsupported format: {suffix}")
        sys.exit(1)

    print(f"File: {filepath}")
    print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")

    print(f"\nColumns ({len(df.columns)} total):")
    for col in df.columns:
        dtype = df[col].dtype
        null_count = df[col].isna().sum()
        unique_count = df[col].nunique()
        null_info = f"  ({null_count} null)" if null_count > 0 else ""
        print(f"  {col}: {dtype}, {unique_count} unique{null_info}")

    display_cols = list(df.columns)
    if display_cols:
        sample = df[display_cols].head(5)
        if len(display_cols) <= 8 and sample.to_string(index=False).find('\n') != -1:
            max_line = max(len(line) for line in sample.to_string(index=False).split('\n'))
            if max_line <= 120:
                print(f"\nSample (first {len(sample)} rows):")
                print(sample.to_string(index=False))
            else:
                _print_vertical_sample(sample, display_cols)
        else:
            _print_vertical_sample(sample, display_cols)

    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    if numeric_cols:
        print(f"\nNumeric summary:")
        pd.set_option('display.float_format', lambda x: f'{x:.4f}')
        print(df[numeric_cols].describe().to_string())
        pd.reset_option('display.float_format')


def _print_vertical_sample(sample, columns):
    """纵向逐记录展示样本数据"""
    print(f"\nSample (first {len(sample)} records, vertical layout):")
    for i, (idx, row) in enumerate(sample.iterrows()):
        print(f"\n  Record {i + 1}:")
        for col in columns:
            val = str(row[col]).replace('\n', '\\n').replace('\r', '\\r')
            if len(val) > 80:
                val = val[:77] + "..."
            print(f"    {col}: {val}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python inspect_tabular.py <file_path>")
        sys.exit(1)

    inspect(sys.argv[1])