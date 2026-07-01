from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "config"
OUTPUT_DIR = PROJECT_ROOT / "data" / "output"


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def load_config(name: str) -> dict[str, Any]:
    return load_json(CONFIG_DIR / name)


def ensure_output_dir() -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR


def normalize_spaces(text: str) -> str:
    text = re.sub(r"[ \t\u3000]+", " ", str(text or ""))
    return re.sub(r"\s*\n\s*", "\n", text).strip()


def split_sentences(text: str) -> list[str]:
    cleaned = normalize_spaces(text).replace("\n", "")
    parts = re.split(r"(?<=[。！？；;])", cleaned)
    return [part.strip() for part in parts if part.strip()]


def evidence_sentence(text: str, keyword: str) -> str:
    for sentence in split_sentences(text):
        if keyword and keyword in sentence:
            return sentence
    return split_sentences(text)[0] if split_sentences(text) else normalize_spaces(text)


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")


def dataframe_to_json_bytes(df: pd.DataFrame) -> bytes:
    return df.to_json(orient="records", force_ascii=False, indent=2).encode("utf-8")


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y", "是", "需要"}


def safe_filename(name: str) -> str:
    safe = re.sub(r"[^\w\u4e00-\u9fff.-]+", "_", str(name).strip())
    return safe.strip("._") or "export"

