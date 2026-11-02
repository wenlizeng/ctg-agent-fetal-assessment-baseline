# Model Card

## Intended use

This package is the executable reference baseline for four CTG challenge tracks. It includes three signal-processing tools and a local Qwen agent that routes English requests, calls the tools, and prepares structured reports. The Task4 rule controller is retained as a comparison option.

## Components

| Component | Input | Output | Checkpoint |
|---|---|---|---|
| Task1 tool | DUS waveform | reconstructed 4 Hz FHR | `task1_autofhr.pt` |
| Task2 tool | 20-minute 4 Hz FHR | baseline/acceleration/deceleration probabilities and labels | `task2_light_unet.pt` |
| Task3 dataset-input tool | FHR supplied by the Task3 dataset + Task2 probabilities | abnormality probability and class | `task3_fused_native.pt` |
| Task4 reconstructed-input tool | Task1-reconstructed FHR + Task2 probabilities | abnormality probability and class | `task3_fused_flow.pt` |
| Rule agent | tool outputs and quality state | structured JSON, English report and execution trace | deterministic rules |
| Qwen agent (default Task4) | English request and tool observations | tool calls and evidence-based English report | Qwen2.5-1.5B-Instruct with CTG LoRA adapter |

Task2 uses a three-channel softmax output. Task3 uses a single-logit sigmoid classifier; the exported `risk_probability` is the probability of class 1, and `risk_class` uses a fixed threshold of 0.5.

## Validation results

On the 495-record Task4 validation set, the Qwen LoRA baseline achieved Task4 Score 0.697321, E2E Score 0.713333, and Report Score 0.633273 using the organizer's Task4 scoring implementation. Task1 and Task2 validation outputs were reused after checkpoint-hash verification; Task3 was recomputed using the flow checkpoint.

## Agent behavior

The Qwen controller selects a requested goal and chooses only from registered actions. The executor checks tool dependencies, reuses completed outputs, and records an action trace. Structured report fields and event intervals are generated from tool results. A rule-based controller remains available with `--agent rule`.

## Important limitations

- Task1–Task3 use separately trained models. Task3 dataset-input and Task4 reconstructed-input workflows use different checkpoints.
- Qwen base weights are downloaded separately from the official Qwen2.5-1.5B-Instruct model repository; the CTG LoRA adapter is included under `models/qwen_lora/`.
