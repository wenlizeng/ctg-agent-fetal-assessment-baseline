from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ctg_baseline.agent import Action, ALLOWED_TRANSITIONS, RuleAgent
from ctg_baseline.config import CHECKPOINTS
from ctg_baseline.tools import Task2Tool, Task3Tool


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fhr = (140 + 8 * np.sin(np.linspace(0, 20, 4800))).astype(np.float32)
    task2 = Task2Tool(device).run(fhr)
    task3 = Task3Tool(device, CHECKPOINTS["task3_native"]).run(fhr, task2["probabilities"])
    assert task2["probabilities"].shape == (3, 4800)
    assert task2["labels"].shape == (4800,)
    assert 0.0 <= task3["risk_probability"] <= 1.0
    state = {"stage": "start"}
    action = RuleAgent.choose_action(state)
    assert action == Action.RUN_TASK1 and action in ALLOWED_TRANSITIONS["start"]
    print("smoke test passed", {"device": str(device), "risk": task3["risk_probability"]})


if __name__ == "__main__":
    main()
