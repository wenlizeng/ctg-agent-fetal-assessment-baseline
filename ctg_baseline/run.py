from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from .agent import RuleAgent
from .config import CHECKPOINTS
from .io import append_jsonl, load_fhr, read_manifest, resolve_path, write_json
from .tools import Task1Tool, Task2Tool, Task3Tool


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFESTS = {
    "task1": "train.csv",
    "task2": "train_windows.csv",
    "task3": "train.csv",
    "task4": "train.csv",
}


def device_from_name(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    return torch.device(name)


def prepare_output(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for name in ("task1", "task2", "task3", "task4/reports", "task4/artifacts"):
        (root / name).mkdir(parents=True, exist_ok=True)


def run_task1(rows, output: Path, device: torch.device, batch_size: int) -> None:
    tool = Task1Tool(device)
    for row in rows:
        result = tool.run(resolve_path(row["_input_root"], row["wav_path"]), batch_size)
        np.savez_compressed(output / "task1" / f"{row['record_id']}.npz", **result)


def run_task2(rows, output: Path, device: torch.device) -> None:
    tool = Task2Tool(device)
    for row in rows:
        fhr = load_fhr(resolve_path(row["_input_root"], row["fhr_path"]))
        if row.get("window_start_sample") and row.get("window_end_sample"):
            fhr = fhr[int(row["window_start_sample"]) : int(row["window_end_sample"])]
        result = tool.run(fhr)
        np.savez_compressed(output / "task2" / f"{row['record_id']}.npz", **result)


def run_task3(rows, output: Path, device: torch.device) -> None:
    task2 = Task2Tool(device)
    task3 = Task3Tool(device, CHECKPOINTS["task3_native"])
    prediction_file = output / "task3" / "classification.jsonl"
    prediction_file.unlink(missing_ok=True)
    for row in rows:
        fhr = load_fhr(resolve_path(row["_input_root"], row["fhr_path"]))
        segmentation = task2.run(fhr)
        result = task3.run(fhr, segmentation["probabilities"])
        append_jsonl(prediction_file, {"record_id": row["record_id"], **result})


def run_task4(rows, output: Path, device: torch.device, batch_size: int) -> None:
    agent = RuleAgent(device)
    prediction_file = output / "task4" / "predictions.jsonl"
    trace_file = output / "task4" / "agent_trace.jsonl"
    prediction_file.unlink(missing_ok=True)
    trace_file.unlink(missing_ok=True)
    for row in rows:
        result = agent.run(row["record_id"], resolve_path(row["_input_root"], row["wav_path"]), batch_size)
        np.savez_compressed(
            output / "task4" / "artifacts" / f"{row['record_id']}.npz",
            predicted_fhr=result["fhr"],
            segmentation_probabilities=result["task2_probabilities"].astype(np.float16),
            seg_labels=result["task2_labels"],
        )
        write_json(output / "task4" / "reports" / f"{row['record_id']}.json", result["report"])
        (output / "task4" / "reports" / f"{row['record_id']}.txt").write_text(result["report_text"] + "\n", encoding="utf-8")
        append_jsonl(
            prediction_file,
            {
                "record_id": result["record_id"],
                "risk_probability": result["risk_probability"],
                "risk_class": result["risk_class"],
                "abstain": result["report"]["abstain"],
            },
        )
        append_jsonl(trace_file, {"record_id": result["record_id"], "steps": result["agent_trace"]})


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline CTG challenge baseline")
    parser.add_argument("task", choices=("task1", "task2", "task3", "task4"))
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--input-root", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "output")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--task1-batch-size", type=int, default=32)
    args = parser.parse_args()
    if args.manifest is None:
        args.manifest = args.input_root / args.task / "manifests" / DEFAULT_MANIFESTS[args.task]
    if not args.manifest.is_file():
        parser.error(f"manifest not found: {args.manifest}")
    prepare_output(args.output_root)
    rows = read_manifest(args.manifest, args.input_root, args.task)
    device = device_from_name(args.device)
    if args.task == "task1":
        run_task1(rows, args.output_root, device, args.task1_batch_size)
    elif args.task == "task2":
        run_task2(rows, args.output_root, device)
    elif args.task == "task3":
        run_task3(rows, args.output_root, device)
    else:
        run_task4(rows, args.output_root, device, args.task1_batch_size)


if __name__ == "__main__":
    main()
