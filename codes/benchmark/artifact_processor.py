# codes/benchmark/artifact_processor.py
"""
执行产物的收集与分流处理

扫描指定输出目录下的文件，按后缀分流至对应的
处理管线，生成可嵌入评估prompt的结构化描述。
矢量/栅格/表格走预置诊断脚本，图片转Data URL，
一般文本按阈值截取，未知类型仅列文件名。
"""

from __future__ import annotations

import base64
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


class ArtifactProcessor:
    """
    产物分流处理器

    将输出目录下的文件按类型转化为评估模型可消费的形态：
    结构化摘要文本或Data URL编码的图片引用。
    """

    # 后缀分类表
    VECTOR_SUFFIXES = {".shp", ".geojson", ".gpkg", ".json"}
    RASTER_SUFFIXES = {".tif", ".tiff", ".img"}
    TABULAR_SUFFIXES = {".csv", ".xls", ".xlsx"}
    IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".svg", ".tiff"}
    TEXT_SUFFIXES = {".txt", ".json", ".xml", ".md", ".html", ".htm", ".geojson"}

    # 文本截取阈值（字符数）
    TEXT_TRUNCATION_LIMIT = 8000

    # 图片的MIME映射
    IMAGE_MIME = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".bmp": "image/bmp",
        ".svg": "image/svg+xml",
        ".tiff": "image/tiff",
    }

    def __init__(
        self,
        interpreter: str,
        skills_dir: Path,
    ):
        """
        Args:
            interpreter: Python解释器路径，用于执行诊断脚本
            skills_dir: 技能目录根路径，用于定位预置诊断脚本
        """
        self.interpreter = interpreter
        self.scripts_dir = skills_dir / "data-inspection" / "scripts"

    def process_directory(self, output_dir: Path) -> List[Dict[str, Any]]:
        """
        扫描输出目录并处理全部产物。

        Returns:
            每个文件的处理结果列表，每项包含:
            - name: 文件名
            - type: 分类标识 (vector/raster/tabular/image/text/unknown)
            - content: 文本描述或None（图片通过image_url字段传递）
            - image_url: Data URL字符串或None
            - size: 文件大小（字节）
        """
        if not output_dir.exists():
            return []

        results = []
        for item in sorted(output_dir.iterdir()):
            if not item.is_file():
                continue
            results.append(self._process_single(item))

        return results

    def _process_single(self, file_path: Path) -> Dict[str, Any]:
        """按后缀分流处理单个文件。"""
        suffix = file_path.suffix.lower()
        size = file_path.stat().st_size

        base = {
            "name": file_path.name,
            "size": size,
            "image_url": None,
            "content": None,
        }

        # 图片优先判定（.tiff同时出现在栅格和图片中，按大小区分）
        if suffix in self.IMAGE_SUFFIXES:
            if suffix == ".tiff" and size > 1024 * 1024:
                # 大型TIFF更可能是栅格数据
                base["type"] = "raster"
                base["content"] = self._run_diagnostic("inspect_raster.py", file_path)
                return base
            base["type"] = "image"
            base["image_url"] = self._to_data_url(file_path, suffix)
            return base

        if suffix in self.VECTOR_SUFFIXES:
            base["type"] = "vector"
            base["content"] = self._run_diagnostic("inspect_vector.py", file_path)
            return base

        if suffix in self.RASTER_SUFFIXES:
            base["type"] = "raster"
            base["content"] = self._run_diagnostic("inspect_raster.py", file_path)
            return base

        if suffix in self.TABULAR_SUFFIXES:
            base["type"] = "tabular"
            base["content"] = self._run_diagnostic("inspect_tabular.py", file_path)
            return base

        if suffix in self.TEXT_SUFFIXES:
            base["type"] = "text"
            base["content"] = self._read_text_truncated(file_path)
            return base

        base["type"] = "unknown"
        base["content"] = f"[Binary or unrecognized format: {file_path.name}, {size:,} bytes]"
        return base

    def _run_diagnostic(self, script_name: str, file_path: Path) -> str:
        """执行预置诊断脚本并返回其stdout。"""
        script = self.scripts_dir / script_name
        if not script.exists():
            return f"[Diagnostic script not found: {script_name}]"

        try:
            result = subprocess.run(
                [self.interpreter, str(script), str(file_path.resolve())],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=60,
            )
            if result.returncode == 0:
                return result.stdout.strip()
            else:
                return (
                    f"[Diagnostic script failed for {file_path.name}]\n"
                    f"stderr: {result.stderr.strip()[:500]}"
                )
        except subprocess.TimeoutExpired:
            return f"[Diagnostic script timed out for {file_path.name}]"
        except Exception as e:
            return f"[Diagnostic script error for {file_path.name}: {e}]"

    def _to_data_url(self, file_path: Path, suffix: str) -> str:
        """将图片文件编码为Data URL。"""
        mime = self.IMAGE_MIME.get(suffix, "application/octet-stream")
        data = file_path.read_bytes()
        encoded = base64.b64encode(data).decode("ascii")
        return f"data:{mime};base64,{encoded}"

    def _read_text_truncated(self, file_path: Path) -> str:
        """读取文本文件，超出阈值时截取头部。"""
        try:
            text = file_path.read_text(encoding="utf-8")
            if len(text) <= self.TEXT_TRUNCATION_LIMIT:
                return text
            return (
                text[:self.TEXT_TRUNCATION_LIMIT]
                + f"\n\n[Truncated: {len(text):,} chars total, showing first {self.TEXT_TRUNCATION_LIMIT:,}]"
            )
        except UnicodeDecodeError:
            return f"[Unable to decode as text: {file_path.name}]"
        except Exception as e:
            return f"[Read error for {file_path.name}: {e}]"