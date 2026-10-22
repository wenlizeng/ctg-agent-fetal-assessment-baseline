# Model Card

## Intended use

This package is the organizer's executable reference baseline for four CTG challenge tracks. It is intended to demonstrate the required file interfaces and a reproducible offline rule-agent workflow. It is not a medical device and must not be used for clinical decision-making.

## Components

| Component | Input | Output | Checkpoint |
|---|---|---|---|
| Task1 tool | DUS waveform | reconstructed 4 Hz FHR | `task1_autofhr.pt` |
| Task2 tool | 20-minute 4 Hz FHR | baseline/acceleration/deceleration probabilities and labels | `task2_light_unet.pt` |
| Task3 native tool | real FHR + Task2 probabilities | abnormality probability and class | `task3_fused_native.pt` |
| Task3 flow tool | reconstructed FHR + Task2 probabilities | abnormality probability and class | `task3_fused_flow.pt` |
| Rule agent | tool outputs and quality state | structured JSON, English report and execution trace | deterministic rules |

Task2 uses a three-channel softmax output. Task3 uses a single-logit sigmoid classifier; the exported `risk_probability` is the probability of class 1, and `risk_class` uses a fixed threshold of 0.5.

## Current validation results

The following values are development/validation results from the organizer's current split. They are not hidden-test results and must not be presented as final leaderboard scores.

| Track | Primary score | Supporting results |
|---|---:|---|
| Task1 | 0.16299 | MAE 4.0401 bpm; RMSE 6.8385 bpm |
| Task2 | 0.72230 | challenge Task2 score |
| Task3 native | 0.77458 | AUROC 0.83033; F1 0.74235; normalized MCC derived from MCC 0.50215 |
| Task4 flow | 0.69646 | flow Task2 0.53285; flow Task3 0.83591; report 0.62353; E2E 0.71469 |

The recorded development overall score was 0.61446 under the then-current overall formula. Recompute all scores with the final published scoring package before release if the competition rules change.

## Agent behavior

The controller is a finite-state rule agent with an explicit action allow-list. It orchestrates the three learned tools, checks reconstruction coverage, selects the flow-specific Task3 checkpoint, generates event-grounded reports, and records an action trace. It does not call an external service or execute model-generated code.

## Important limitations

- Results are split-dependent and have not yet been confirmed on the hidden competition test set.
- The package includes inference only; training data and private labels are not included.
- The organizer must finalize the repository license, Docker resource limits, schemas and scoring definitions before public release.
