"""One-case conversational session; results are reused within this process."""
import argparse
from pathlib import Path

import numpy as np
import torch

from .io import load_fhr, write_json
from .llm_agent import QwenAgent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--input-kind", choices=("dus", "fhr"), required=True)
    parser.add_argument("--record-id", required=True)
    parser.add_argument("--request", help="Omit for interactive conversation")
    parser.add_argument("--llm-model", type=Path)
    parser.add_argument("--lora-adapter", type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--llm-device", choices=("cpu", "cuda"))
    parser.add_argument("--output", type=Path, default=Path("output/chat"))
    args = parser.parse_args()
    if not args.input.is_file():
        parser.error("Input file does not exist")
    agent = QwenAgent(torch.device(args.device), args.llm_model, args.llm_device, adapter_path=args.lora_adapter)
    agent.reset(args.record_id, wav_path=args.input if args.input_kind == "dus" else None,
                fhr=load_fhr(args.input) if args.input_kind == "fhr" else None)
    args.output.mkdir(parents=True, exist_ok=True)
    turn = 0
    while True:
        try:
            request = args.request or input("CTG request (exit to quit)> ").strip()
            if request.lower() in {"exit", "quit"}:
                break
            turn += 1
            result = agent.request(request)
            arrays = {k: v for k, v in result.items() if isinstance(v, np.ndarray)}
            details = {k: v for k, v in result.items() if k not in arrays}
            if arrays:
                np.savez_compressed(args.output / f"turn_{turn}.npz", **arrays)
            write_json(args.output / f"turn_{turn}.json", details)
            print(details.get("message") or details.get("report_text") or
                  {k: v for k, v in details.items() if k != "agent_trace"})
            print("Available outputs:", ", ".join(arrays), "Saved to", args.output)
        except (RuntimeError, ValueError, OSError) as exc:
            write_json(args.output / f"turn_{turn}_error.json", {"error": str(exc), "trace": agent.trace})
            if args.request:
                raise
            print("Agent error:", exc)
        except (EOFError, KeyboardInterrupt):
            break
        if args.request:
            break


if __name__ == "__main__":
    main()
