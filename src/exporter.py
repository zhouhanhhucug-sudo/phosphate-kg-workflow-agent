from __future__ import annotations

import zipfile
from pathlib import Path

import pandas as pd


def save_dataframe(df: pd.DataFrame, path: str | Path) -> None:
    """保存表格。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".json":
        df.to_json(path, orient="records", force_ascii=False, indent=2)
    else:
        df.to_csv(path, index=False, encoding="utf-8-sig")


def save_text(text: str, path: str | Path) -> None:
    """保存文本文件。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text or "", encoding="utf-8")


def export_project_zip(output_dir: str | Path, zip_path: str | Path) -> str:
    """打包导出项目结果。"""
    output_dir = Path(output_dir)
    zip_path = Path(zip_path)
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file in output_dir.rglob("*"):
            if file.is_file() and file.resolve() != zip_path.resolve():
                zf.write(file, file.relative_to(output_dir))
    return str(zip_path)

