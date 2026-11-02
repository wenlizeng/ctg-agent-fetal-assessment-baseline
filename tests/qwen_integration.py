"""Opt-in real-model integration check using one training record per modality.

Run directly from any working directory with Python. Cases are read from the
released training manifests; generated results are written under output/.
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ctg_baseline.io import load_fhr, write_json
from ctg_baseline.llm_agent import QwenAgent


def main():
    root = ROOT
    output = root / "output/qwen_integration"
    output.mkdir(parents=True, exist_ok=True)
    agent = QwenAgent(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    fhr_row = next(csv.DictReader((root / "data/task3/manifests/train.csv").open(encoding="utf-8-sig")))
    agent.reset(fhr_row["record_id"], fhr=load_fhr(root / "data" / fhr_row["fhr_path"]))
    segmentation = agent.request("Segment baseline, acceleration, and deceleration events.")
    assert "risk_probability" not in segmentation
    assert segmentation["task2_probabilities"].shape == (3, 4800)
    risk = agent.request("Assess fetal risk for this recording.")
    write_json(output / "routing_trace.json", agent.trace)
    assert 0 <= risk["risk_probability"] <= 1
    assert "task3_native" in agent.tools and "task3_flow" not in agent.tools
    result = agent.request("Generate a complete assessment report.")
    np.testing.assert_array_equal(segmentation["task2_labels"], result["task2_labels"])
    assert [x.get("action") for x in agent.trace if x["kind"] == "tool"].count("segment") == 1
    write_json(output / "native_trace.json", agent.trace)
    write_json(output / "native_report.json", result["report"])
    (output / "native_report.txt").write_text(result["report_text"], encoding="utf-8")
    print("PASS: FHR segmentation -> risk -> report; Task2 reused; native checkpoint", flush=True)
    row = next(csv.DictReader((root / "data/task4/manifests/train.csv").open(encoding="utf-8-sig")))
    agent.reset(row["record_id"], wav_path=root / "data" / row["wav_path"])
    reconstruction = agent.request("Reconstruct FHR from this ultrasound recording.")
    assert "task2_labels" not in reconstruction and "risk_probability" not in reconstruction
    print("PASS: DUS reconstruction stops before segmentation", flush=True)
    result = agent.request("Generate a complete assessment report.")
    assert "task3_flow" in agent.tools
    assert [x.get("action") for x in agent.trace if x["kind"] == "tool"].count("reconstruct") == 1
    assert result["fhr"].shape == (4800,)
    write_json(output / "flow_trace.json", agent.trace)
    write_json(output / "flow_report.json", result["report"])
    (output / "flow_report.txt").write_text(result["report_text"], encoding="utf-8")
    np.savez_compressed(output / "flow.npz", fhr=result["fhr"], labels=result["task2_labels"],
                        probabilities=result["task2_probabilities"])
    print("PASS: full DUS workflow; Task1 reused; flow checkpoint", flush=True)


if __name__ == "__main__":
    main()
