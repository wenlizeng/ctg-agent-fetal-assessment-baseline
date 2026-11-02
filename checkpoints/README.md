# Baseline Checkpoints

The four trained task checkpoints are included in this repository:

- `task1_autofhr.pt`: DUS-to-FHR reconstruction.
- `task2_light_unet.pt`: three-class FHR event segmentation.
- `task3_fused_native.pt`: Task3 dataset-input FHR with Task2 probabilities.
- `task3_fused_flow.pt`: Task1-reconstructed FHR with Task2 probabilities for Task4.

`SHA256SUMS` contains hashes for these checkpoint files and the Task4 LoRA adapter. From this directory, verify with `sha256sum -c SHA256SUMS` on Linux/macOS. The Qwen base model is downloaded separately as described in the repository README.
