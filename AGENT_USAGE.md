# Qwen Agent Usage

The Task4 baseline uses Qwen2.5-1.5B-Instruct with the included CTG LoRA adapter. The base Qwen model is downloaded separately; see the [README](README.md) for setup and reproduction commands.

The agent accepts English requests for FHR reconstruction, CTG event segmentation, fetal risk assessment, or a complete report. It routes each request to the registered Task1, Task2, and Task3 tools as needed. Results and intermediate tool outputs are reused during an interactive session.

For a complete Task4 analysis, the workflow is:

```text
DUS -> Task1 FHR reconstruction -> Task2 event segmentation
    -> Task3 risk classification -> structured JSON and English report
```

Task3 classification uses `task3_fused_native.pt` for FHR supplied with the Task3 dataset and `task3_fused_flow.pt` for Task1-reconstructed FHR in the Task4 workflow. The agent selects the appropriate checkpoint based on the input source.

Start an interactive session with:

```bash
python -m ctg_baseline.chat --input PATH --input-kind dus --record-id RECORD_ID
```

For an FHR file, set `--input-kind fhr`. Use `--llm-model PATH` and `--lora-adapter PATH` when model files are outside their default directories.
