from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


def resolve_path(root: Path, value: str) -> Path:
    path = Path(value.replace("\\", "/"))
    return path if path.is_absolute() else root / path


OFFICIAL_COLUMNS = {
    "task1": ("label", "task1_input_wav", "wav_path"),
    "task2": ("segment_id", "organized_source_path", "fhr_path"),
    "task3": ("index", "organized_signal_path", "fhr_path"),
    "task4": ("label", "task4_input_wav", "wav_path"),
}


def read_manifest(path: Path, input_root: Path, task: str) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Manifest is empty: {path}")
    id_column, source_column, target_column = OFFICIAL_COLUMNS[task]
    for row in rows:
        row["record_id"] = row.get("record_id") or row.get(id_column)
        row[target_column] = row.get(target_column) or row.get(source_column)
    if any(not row.get("record_id") or not row.get(target_column) for row in rows):
        raise ValueError(
            f"{task} manifest requires record_id/{target_column}, or official columns "
            f"{id_column}/{source_column}: {path}"
        )
    if len({row["record_id"] for row in rows}) != len(rows):
        raise ValueError("Manifest record_id values must be unique")
    return [{**row, "_input_root": input_root} for row in rows]


def load_fhr(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        values = np.load(path)
    elif suffix == ".npz":
        with np.load(path) as item:
            key = next((name for name in ("fhr", "predicted_fhr") if name in item), None)
            if key is None:
                raise ValueError(f"NPZ has no fhr array: {path}")
            values = item[key]
    elif suffix == ".csv":
        values = np.genfromtxt(path, delimiter=",", names=True, dtype=np.float32)
        if values.dtype.names:
            key = next((name for name in ("fhr", "FHR") if name in values.dtype.names), values.dtype.names[0])
            values = values[key]
    else:
        raise ValueError(f"Unsupported FHR file: {path}")
    return np.asarray(values, dtype=np.float32).reshape(-1)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def append_jsonl(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")
