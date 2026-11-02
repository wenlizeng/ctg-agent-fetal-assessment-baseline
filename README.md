# CTG Agent Challenge Baseline

## Update Notes

**November 2, 2026** — Updated the baseline release with the Qwen2.5-1.5B-Instruct LoRA Agent, English request routing, Task1–Task4 workflow instructions, checkpoint descriptions (including the Task3 dataset-input and Task4 reconstructed-input weights), model download guidance, and reproduction steps.

Official baseline for **Agent-Based Model for Fetal Well-being Assessment through Cardiotocography**.

The baseline models are trained using the challenge training data. A local **Qwen2.5-1.5B-Instruct model with a CTG LoRA adapter** interprets English requests, calls the task models, and produces a structured JSON report and an English assessment report.

```text
Task1: DUS -> FHR reconstruction
Task2: FHR -> baseline, acceleration, and deceleration segmentation
Task3: FHR + Task2 probabilities -> fetal risk probability
Task4: DUS -> Task1 -> Task2 -> Task3 -> assessment report
```

## 1. Installation

```bash
git clone https://github.com/wenlizeng/ctg-agent-fetal-assessment-baseline.git
cd ctg-agent-fetal-assessment-baseline
```

Use Python 3.10 and install [PyTorch](https://pytorch.org/get-started/locally/) for your CUDA environment, followed by the baseline dependencies:

```bash
python -m pip install -r requirements.txt accelerate==1.1.1
```

Run the following commands from the repository root.

## 2. Data and model files

Place the challenge training dataset under `data/`, following the [dataset directory structure](data/README.md). File paths in the manifests are relative to `data/`.

The trained task checkpoints and CTG LoRA adapter are included in the repository under `checkpoints/` and `models/qwen_lora/`:

```text
repository-root/
|-- ctg_baseline/
|-- checkpoints/
|   |-- task1_autofhr.pt
|   |-- task2_light_unet.pt
|   |-- task3_fused_native.pt
|   `-- task3_fused_flow.pt
|-- models/
|   |-- qwen/
|   `-- qwen_lora/
|       |-- adapter_config.json
|       `-- adapter_model.safetensors
|-- data/
|   |-- task1/manifests/train.csv
|   |-- task2/manifests/train_windows.csv
|   |-- task3/manifests/train.csv
|   `-- task4/manifests/train.csv
`-- output/
```

| Checkpoint | Purpose |
|---|---|
| `task1_autofhr.pt` | Reconstruct FHR from DUS |
| `task2_light_unet.pt` | Segment baseline, acceleration, and deceleration |
| `task3_fused_native.pt` | Task3 classification using FHR supplied by the Task3 dataset and Task2 probabilities |
| `task3_fused_flow.pt` | Task4 classification using Task1-reconstructed FHR and Task2 probabilities |
| `adapter_model.safetensors` | CTG-specific Qwen LoRA adapter |

The two Task3 checkpoints share the same model architecture but use different trained weights. The Task3 checkpoint is trained on FHR supplied with the Task3 dataset; the Task4 checkpoint is adapted to FHR reconstructed by Task1. The runner automatically selects the checkpoint for the requested workflow.

## 3. Download Qwen

Download the base model from the official [Qwen2.5-1.5B-Instruct repository](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct):

```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='Qwen/Qwen2.5-1.5B-Instruct', revision='989aa7980e4cf806f80c7fef2b1adb7bc71aa306', local_dir='models/qwen')"
```

Keep the downloaded model weights, configuration, and tokenizer files together in `models/qwen/`. Inference loads Qwen and the CTG adapter locally.

## 4. Run the baseline

Run an individual task or the complete Task4 workflow:

```bash
python -m ctg_baseline.run task1 --device cuda --output-root output/task1
python -m ctg_baseline.run task2 --device cuda --output-root output/task2
python -m ctg_baseline.run task3 --device cuda --output-root output/task3
python -m ctg_baseline.run task4 --device cuda --llm-model models/qwen --lora-adapter models/qwen_lora --output-root output/task4
```

Each command processes its training manifest. To use another manifest, supply `--manifest PATH --input-root DATA_DIRECTORY`. Use `--device cpu` for CPU inference.

For a single-case example, run the following in PowerShell:

```powershell
New-Item -ItemType Directory -Force output | Out-Null
Import-Csv data/task4/manifests/train.csv | Select-Object -First 1 | Export-Csv output/task4_one.csv -NoTypeInformation -Encoding UTF8
python -m ctg_baseline.run task4 --manifest output/task4_one.csv --input-root data --output-root output/example --device cuda --llm-model models/qwen --lora-adapter models/qwen_lora
```

Task4 results are saved under the selected output directory:

```text
task4/
|-- artifacts/<record_id>.npz    # FHR, segmentation probabilities, and labels
|-- reports/<record_id>.json    # structured report
|-- reports/<record_id>.txt     # English report
|-- predictions.jsonl          # risk predictions
`-- agent_trace.jsonl           # agent decisions and tool calls
```

## 5. Use English instructions

Start an interactive session with a training DUS recording (PowerShell):

```powershell
$case = Import-Csv data/task4/manifests/train.csv | Select-Object -First 1
$dus = Join-Path (Resolve-Path data).Path $case.wav_path
python -m ctg_baseline.chat --input $dus --input-kind dus --record-id $case.record_id --device cuda --llm-model models/qwen --lora-adapter models/qwen_lora --output output/chat
```

Example requests:

```text
Reconstruct FHR from this Doppler ultrasound recording.
Segment the FHR into baseline, acceleration, and deceleration.
Assess fetal risk for this recording.
Generate a complete CTG assessment report.
exit
```

The Agent reuses intermediate outputs within the session. For FHR input, use `--input-kind fhr` and a 4,800-sample, 4 Hz FHR file in NPY, NPZ, or CSV format. Add `--request "Assess fetal risk for this recording."` to run one request and exit.

## Contact

[wenlizeng7461@gmail.com](mailto:wenlizeng7461@gmail.com)
