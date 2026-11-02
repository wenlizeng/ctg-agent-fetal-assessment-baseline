"""Local Qwen controller. Only registered tools can execute; no generated code."""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

import numpy as np
import torch

from .config import CHECKPOINTS, FHR_POINTS, TASK1_MIN_COVERAGE
from .report import build_report
from .tools import Task1Tool, Task2Tool, Task3Tool, fill_fhr

GOAL_ROUTER_SYSTEM = (
    "You route English CTG requests. Choose the requested objective, not an intermediate tool. "
    'Return exactly one JSON object {"goal":"reconstruct|segment|risk|report|clarify"}. '
    "Reconstruction requests map to reconstruct; event-label requests to segment; fetal status "
    "questions to risk; complete assessment/report requests to report. A risk request alone "
    "does not request a full report."
)
ACTION_ROUTER_SYSTEM = (
    "Choose the next action needed to satisfy the requested CTG goal using the observations. "
    "Select only an item from available_actions. Return one JSON object with the action field."
)


def default_model_path():
    return Path(os.environ.get("CTG_LLM_MODEL", Path(__file__).resolve().parents[1] / "models/qwen"))


def default_adapter_path():
    path = Path(os.environ.get("CTG_LLM_ADAPTER", Path(__file__).resolve().parents[1] / "models/qwen_lora"))
    return path if (path / "adapter_config.json").is_file() else None


class LocalQwen:
    def __init__(self, path=None, device="cpu", adapter_path=None):
        os.environ.setdefault("USE_TF", "0")
        os.environ.setdefault("HF_HOME", str(Path(__file__).resolve().parents[1] / "output/hf_cache"))
        from transformers import AutoModelForCausalLM, AutoTokenizer
        path = Path(path or default_model_path())
        self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            path, local_files_only=True, torch_dtype=torch.float32 if str(device) == "cpu" else torch.float16,
        ).to(device).eval()
        if adapter_path:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter_path, is_trainable=False).eval()

    def ask(self, instruction, payload):
        messages = [{"role": "system", "content": instruction + " Return exactly one JSON object, no markdown."},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]
        tokens = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                     return_tensors="pt").to(self.model.device)
        options = payload.get("available_actions")
        field = "action"
        if "allowed_summaries" in payload:
            options = payload["allowed_summaries"]
            field = "summary"
        if options is None and set(payload) == {"request"}:
            options = ["reconstruct", "segment", "risk", "report", "clarify"]
            field = "goal"
        kwargs = {}
        if options:
            sequences = [self.tokenizer.encode(json.dumps({field: value}), add_special_tokens=False)
                         for value in options]
            prefix_length = tokens.shape[1]
            def allowed_tokens(batch_id, ids):
                prefix = ids[prefix_length:].tolist()
                choices = set()
                for sequence in sequences:
                    if sequence[:len(prefix)] == prefix:
                        choices.add(sequence[len(prefix)] if len(prefix) < len(sequence)
                                    else self.tokenizer.eos_token_id)
                return list(choices) or [self.tokenizer.eos_token_id]
            kwargs["prefix_allowed_tokens_fn"] = allowed_tokens
        with torch.inference_mode():
            output = self.model.generate(tokens, attention_mask=torch.ones_like(tokens),
                                         max_new_tokens=256, do_sample=False,
                                         temperature=1.0, top_p=1.0, top_k=50,
                                         pad_token_id=self.tokenizer.eos_token_id, **kwargs)
        raw = self.tokenizer.decode(output[0, tokens.shape[1]:], skip_special_tokens=True).strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        result = json.loads(raw)
        if not isinstance(result, dict):
            raise ValueError("Model must return a JSON object")
        return result


class QwenAgent:
    def __init__(self, device, model_path=None, llm_device=None, policy=None, adapter_path=None):
        self.device = device
        self.policy = policy or LocalQwen(model_path, llm_device or device, adapter_path or default_adapter_path())
        self.tools = {}
        self.reset()

    def reset(self, record_id=None, wav_path=None, fhr=None):
        if wav_path is not None and fhr is not None:
            raise ValueError("Provide DUS or FHR, not both, to preserve source provenance")
        self.record_id = record_id
        self.wav_path = Path(wav_path) if wav_path else None
        self.state = {}
        self.trace = []
        if fhr is not None:
            values = np.asarray(fhr, dtype=np.float32)
            if values.shape != (FHR_POINTS,):
                raise ValueError("FHR input must contain exactly 4800 samples on a 4 Hz axis")
            values, missing = fill_fhr(values)
            self.state["task1"] = {"fhr": values, "missing_rate": missing, "coverage": 1-missing}

    def _ask(self, instruction, payload):
        for attempt in range(3):
            try:
                answer = self.policy.ask(instruction, payload)
                self.trace.append({"kind": "llm", "observation": payload, "response": answer})
                return answer
            except (ValueError, KeyError) as exc:
                self.trace.append({"kind": "llm_error", "error": str(exc)})
                payload = {**payload, "correction": str(exc)}
        raise RuntimeError("Qwen failed to produce valid JSON after three attempts")

    def request(self, text, batch_size=32, goal=None):
        if goal is None:
            answer = self._ask(GOAL_ROUTER_SYSTEM, {"request": text})
            goal = answer.get("goal")
        if goal not in {"reconstruct", "segment", "risk", "report"}:
            return {"message": "Please specify whether you want FHR reconstruction, event segmentation, risk assessment, or a complete report."}
        if not self.record_id or (not self.wav_path and "task1" not in self.state):
            return {"message": "Please provide a record identifier and either a DUS recording or a 4 Hz FHR sequence."}
        if goal == "reconstruct" and not self.wav_path:
            return {"message": "FHR reconstruction requires the original DUS recording; the current input is already an FHR sequence."}
        target = {"reconstruct": "quality", "segment": "task2", "risk": "task3", "report": "report"}[goal]
        error = None
        for _ in range(14):
            available = []
            if "task1" not in self.state:
                available.append("reconstruct")
            else:
                if "quality" not in self.state:
                    available.append("check_quality")
                else:
                    if goal != "reconstruct" and "task2" not in self.state:
                        available.append("segment")
                    if goal in {"risk", "report"} and "task2" in self.state and "task3" not in self.state:
                        available.append("classify")
                    if goal == "report" and "task3" in self.state and "report" not in self.state:
                        available.append("report")
            if target in self.state:
                available.append("finish")
            observation = {
                "task1": ({"coverage": self.state["task1"]["coverage"]} if "task1" in self.state else None),
                "quality": self.state.get("quality"),
                "task2": ({"mean_confidence": self.state["task2"]["mean_confidence"],
                           "acceleration_count": int(np.sum(self.state["task2"]["labels"] == 1)),
                           "deceleration_count": int(np.sum(self.state["task2"]["labels"] == -1))}
                          if "task2" in self.state else None),
                "task3": self.state.get("task3"),
            }
            choice = self._ask(ACTION_ROUTER_SYSTEM,
                               {"request": text, "goal": goal, "completed": list(self.state),
                                "observations": observation,
                                "available_actions": available, "last_error": error})
            action = choice.get("action")
            if action not in available:
                error = "Action not available; use available_actions"
                self.trace.append({"kind": "rejected", "action": action})
                continue
            if action == "finish":
                return self.result()
            try:
                self.execute(action, batch_size)
                self.trace.append({"kind": "tool", "action": action, "status": "ok"})
                error = None
            except (ValueError, OSError, RuntimeError) as exc:
                error = str(exc)
                self.trace.append({"kind": "tool", "action": action, "status": "error", "error": error})
        raise RuntimeError(f"Agent step budget exceeded: {error}")

    def execute(self, action, batch_size):
        if action == "reconstruct":
            if "task1" not in self.tools:
                self.tools["task1"] = Task1Tool(self.device)
            self.state["task1"] = self.tools["task1"].run(self.wav_path, batch_size)
        elif action == "check_quality":
            coverage = float(self.state["task1"]["coverage"])
            self.state["quality"] = {"coverage": coverage, "passed": coverage >= TASK1_MIN_COVERAGE}
        elif action == "segment":
            if "task2" not in self.tools:
                self.tools["task2"] = Task2Tool(self.device)
            self.state["task2"] = self.tools["task2"].run(self.state["task1"]["fhr"])
        elif action == "classify":
            key = "task3_flow" if self.wav_path else "task3_native"
            if key not in self.tools:
                self.tools[key] = Task3Tool(self.device, CHECKPOINTS[key])
            self.state["task3"] = self.tools[key].run(self.state["task1"]["fhr"], self.state["task2"]["probabilities"])
        elif action == "report":
            report, facts = build_report(self.record_id, self.state["task1"]["fhr"],
                                        self.state["task2"]["labels"], self.state["task3"]["risk_probability"],
                                        self.state["task1"]["missing_rate"], not self.state["quality"]["passed"])
            required = [f"{report['acceleration_count']} acceleration event" + ("" if report['acceleration_count'] == 1 else "s"),
                        f"{report['deceleration_count']} deceleration event" + ("" if report['deceleration_count'] == 1 else "s"),
                        f"Assessment: {report['conclusion']}",
                        f"abnormality probability {report['risk_probability']:.4f}"]
            event_sentence = f"The segmentation tool detected {required[0]} and {required[1]}."
            risk_sentence = f"{required[2]}; model {required[3]}."
            candidates = [event_sentence + " " + risk_sentence, risk_sentence + " " + event_sentence]
            answer = self._ask('Select the most appropriate English summary from allowed_summaries. '
                               'Preserve numerical evidence exactly. Return {"summary":"..."}.',
                               {"allowed_summaries": candidates})
            summary = answer.get("summary")
            if not isinstance(summary, str) or any(p not in summary for p in required):
                raise ValueError("Summary must preserve all required factual phrases verbatim")
            numbers = set(re.findall(r"\d+(?:\.\d+)?", summary))
            allowed_numbers = set(re.findall(r"\d+(?:\.\d+)?", " ".join(required)))
            if not numbers <= allowed_numbers:
                raise ValueError("Summary introduced unsupported numerical values")
            self.state["report"] = report
            self.state["report_text"] = facts + "\n\nAssessment summary:\n" + summary

    def result(self):
        result = {"record_id": self.record_id, "agent_trace": list(self.trace)}
        if "task1" in self.state:
            result.update(self.state["task1"])
        if "task2" in self.state:
            result.update(task2_labels=self.state["task2"]["labels"], task2_probabilities=self.state["task2"]["probabilities"])
        if "task3" in self.state:
            result.update(self.state["task3"])
        for key in ("report", "report_text"):
            if key in self.state:
                result[key] = self.state[key]
        return result

    def run(self, record_id, wav_path, batch_size=32):
        self.reset(record_id, wav_path=wav_path)
        return self.request("Complete end-to-end assessment and English report", batch_size, goal="report")
