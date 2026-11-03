from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


def resolve_path(root: Path, value: str) -> Path:
    path = Path(value.replace("\\", "/"))
    return path if path.is_absolute() else root / path


def read_manifest(path: Path, input_root: Path, task: str) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Manifest is empty: {path}")
    target_column = "wav_path" if task in ("task1", "task4") else "fhr_path"
    if any(not row.get("record_id") or not row.get(target_column) for row in rows):
        raise ValueError(f"{task} manifest requires record_id and {target_column}: {path}")
    if len({row["record_id"] for row in rows}) != len(rows):
        raise ValueError("Manifest record_id values must be unique")
    return [{**row, "_input_root": input_root} for row in rows]


def load_fetal_fhr(path: str | Path) -> np.ndarray:
    """Read first-channel reference FHR (float32 bpm, 4 Hz) from .fetal.

    Supports the released v1 and legacy v0 layouts. Values outside
    [50, 210] become NaN without removing time points or interpolating.
    Malformed or incomplete records raise ValueError. See data/FETAL_FORMAT.md.
    """
    path = Path(path)
    raw = path.read_bytes()
    if len(raw) < 8:
        raise ValueError(f"Fetal file is shorter than the 8-byte header: {path}")
    version, fetal_count = raw[0], raw[1]
    if fetal_count < 1:
        raise ValueError(f"Invalid fetal count {fetal_count}: {path}")
    if version == 1:
        stride = 8 + (fetal_count - 1) * 4
        count, remainder = divmod(len(raw) - 8, stride)
        if remainder:
            raise ValueError(f"Incomplete fetal record: {path}")
    elif version == 0:
        stride = fetal_count + 2
        count = int.from_bytes(raw[2:6], "little")
        if len(raw) != 8 + count * stride:
            raise ValueError(f"Fetal payload does not match declared record count: {path}")
    else:
        raise ValueError(f"Unsupported fetal version {version}: {path}")
    if count == 0:
        raise ValueError(f"Fetal file contains no samples: {path}")
    fhr = np.frombuffer(raw, dtype=np.uint8)[8::stride].astype(np.float32)
    fhr[(fhr < 50) | (fhr > 210)] = np.nan
    return fhr


def load_fhr(path: Path) -> np.ndarray:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".fetal":
        return load_fetal_fhr(path)
    elif suffix == ".npy":
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
