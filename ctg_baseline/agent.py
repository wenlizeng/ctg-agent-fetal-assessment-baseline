from __future__ import annotations

from enum import Enum
from pathlib import Path

import numpy as np
import torch

from .config import CHECKPOINTS, TASK1_MIN_COVERAGE
from .report import build_report
from .tools import Task1Tool, Task2Tool, Task3Tool


class Action(str, Enum):
    RUN_TASK1 = "RUN_TASK1"
    CHECK_FHR_QUALITY = "CHECK_FHR_QUALITY"
    RUN_TASK2 = "RUN_TASK2"
    RUN_TASK3 = "RUN_TASK3"
    GENERATE_REPORT = "GENERATE_REPORT"
    FINISH = "FINISH"


ALLOWED_TRANSITIONS = {
    "start": {Action.RUN_TASK1},
    "task1_done": {Action.CHECK_FHR_QUALITY},
    "quality_checked": {Action.RUN_TASK2},
    "task2_done": {Action.RUN_TASK3},
    "task3_done": {Action.GENERATE_REPORT},
    "report_done": {Action.FINISH},
}


class RuleAgent:
    """Finite, auditable controller; task models are exposed only as registered tools."""

    def __init__(self, device: torch.device):
        self.task1 = Task1Tool(device)
        self.task2 = Task2Tool(device)
        self.task3 = Task3Tool(device, CHECKPOINTS["task3_flow"])

    @staticmethod
    def choose_action(state: dict[str, object]) -> Action:
        return {
            "start": Action.RUN_TASK1,
            "task1_done": Action.CHECK_FHR_QUALITY,
            "quality_checked": Action.RUN_TASK2,
            "task2_done": Action.RUN_TASK3,
            "task3_done": Action.GENERATE_REPORT,
            "report_done": Action.FINISH,
        }[str(state["stage"])]

    def run(self, record_id: str, wav_path: Path, batch_size: int = 32) -> dict[str, object]:
        state: dict[str, object] = {"stage": "start", "record_id": record_id, "abstain": False}
        trace: list[dict[str, object]] = []
        while True:
            action = self.choose_action(state)
            if action not in ALLOWED_TRANSITIONS[str(state["stage"])]:
                raise RuntimeError(f"Illegal transition: {state['stage']} -> {action}")
            trace.append({"step": len(trace) + 1, "action": action.value})
            if action == Action.RUN_TASK1:
                state["task1"] = self.task1.run(wav_path, batch_size)
                state["stage"] = "task1_done"
            elif action == Action.CHECK_FHR_QUALITY:
                coverage = float(state["task1"]["coverage"])
                state["abstain"] = coverage < TASK1_MIN_COVERAGE
                trace[-1].update({"coverage": coverage, "threshold": TASK1_MIN_COVERAGE, "passed": not state["abstain"]})
                state["stage"] = "quality_checked"
            elif action == Action.RUN_TASK2:
                state["task2"] = self.task2.run(state["task1"]["fhr"])
                state["stage"] = "task2_done"
            elif action == Action.RUN_TASK3:
                state["task3"] = self.task3.run(state["task1"]["fhr"], state["task2"]["probabilities"])
                state["stage"] = "task3_done"
            elif action == Action.GENERATE_REPORT:
                structured, text = build_report(
                    record_id,
                    state["task1"]["fhr"],
                    state["task2"]["labels"],
                    float(state["task3"]["risk_probability"]),
                    float(state["task1"]["missing_rate"]),
                    bool(state["abstain"]),
                )
                state["report"] = structured
                state["report_text"] = text
                state["stage"] = "report_done"
            else:
                break
        return {
            "record_id": record_id,
            "fhr": np.asarray(state["task1"]["fhr"], dtype=np.float32),
            "missing_rate": float(state["task1"]["missing_rate"]),
            "task2_probabilities": np.asarray(state["task2"]["probabilities"], dtype=np.float32),
            "task2_labels": np.asarray(state["task2"]["labels"], dtype=np.int8),
            "risk_probability": float(state["task3"]["risk_probability"]),
            "risk_class": int(state["task3"]["risk_class"]),
            "report": state["report"],
            "report_text": state["report_text"],
            "agent_trace": trace,
        }
