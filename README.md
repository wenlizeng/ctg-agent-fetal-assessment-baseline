# CTG Challenge Official Rule-Agent Baseline

This repository is the offline reference baseline for the CTG challenge. It contains three frozen task tools and a deterministic rule-based Task4 controller. It does not use an external API or an LLM.

## Architecture

```text
Task1Tool: DUS WAV -> reconstructed 4 Hz FHR
Task2Tool: FHR -> three-class probabilities and labels
Task3Tool: FHR + Task2 probabilities -> abnormality probability

RuleAgent:
RUN_TASK1 -> CHECK_FHR_QUALITY -> RUN_TASK2 -> RUN_TASK3
          -> GENERATE_REPORT -> FINISH
```

The controller can execute only the actions declared in `ctg_baseline/agent.py`. A reconstruction coverage below 95% does not suppress required outputs, but marks the final report as `abstain=true`.

## Model checkpoints

- `task1_autofhr.pt`: Task1 reconstruction model.
- `task2_light_unet.pt`: Task2 FHR-only three-class segmentation model.
- `task3_fused_native.pt`: Task3 independent-track model evaluated on real FHR plus Task2 probabilities.
- `task3_fused_flow.pt`: the same Task3 architecture adapted to Task1-reconstructed FHR for Task4.

Model weights are intentionally not included in this public repository. Place locally trained or organizer-provided weights under `checkpoints/` using the filenames above before running inference. Training caches, optimizer states, validation predictions and other experimental outputs are also excluded.

## Repository layout and dataset placement

Keep the baseline and released dataset under one project root:

```text
ctg_agent_baseline_release/
├── checkpoints/
├── ctg_baseline/
├── data/                 # released training inputs and labels
│   ├── task1/
│   ├── task2/
│   ├── task3/
│   └── task4/
└── output/               # created automatically
```

The released training inputs, labels and manifests are stored directly under `data/`. Detailed placement rules are in `data/README.md`. Validation and test data are not included in this public repository.

## Input manifests

The runner directly accepts the released training manifests. Paths in those manifests are resolved relative to `data/` by default.

- Task1 and Task4: `record_id,wav_path`
- Task2 and Task3: `record_id,fhr_path`

FHR files may be `.npy`, `.npz` (`fhr` or `predicted_fhr`) or CSV with an `fhr`/`FHR` column. Task2 and Task3 expect a 20-minute, 4 Hz signal; invalid lengths are normalized by the published preprocessing code.

## Run locally

Install a PyTorch build appropriate for the local CPU/CUDA environment first, then install the remaining dependencies:

```bash
pip install -r requirements.txt
```

```bash
python -m ctg_baseline.run task1
python -m ctg_baseline.run task2
python -m ctg_baseline.run task3
python -m ctg_baseline.run task4
```

The public baseline loads only the training manifest for each task. The defaults are `--input-root data` and `--output-root output`. A custom manifest remains available for organizer-side checks:

```bash
python -m ctg_baseline.run task2 --manifest examples/task2_task3_manifest.csv --input-root /custom/data
```

Smoke test without challenge data:

```bash
python tests/smoke_test.py
```

## Docker

```bash
docker build -t ctg-rule-agent-baseline:1.0 .
docker run --rm --gpus all --network none \
  -v /host/data:/app/data:ro \
  -v /host/output:/app/output \
  ctg-rule-agent-baseline:1.0 \
  task4
```

The scoring program and reference labels must remain outside the participant container.

## Output

- Task1: `task1/<record_id>.npz`
- Task2: `task2/<record_id>.npz`
- Task3: `task3/classification.jsonl`
- Task4: predictions, reports, artifacts and `agent_trace.jsonl`

## Reproducibility and scope

The official baseline is deliberately small and auditable. Participants may replace the rule controller with a local LLM, a learned policy or another agent implementation, provided that the official task outputs, offline Docker constraints and resource limits are respected.

The model architecture, preprocessing, checkpoint roles and current validation-only results are documented in `MODEL_CARD.md`.

## License

No reuse license is asserted in this staging directory. The organizer must add the approved code and model-weight license before publishing the GitHub repository.
