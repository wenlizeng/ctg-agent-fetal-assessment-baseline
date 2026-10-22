from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_ROOT = PACKAGE_ROOT / "checkpoints"

TASK1_CONFIG = {
    "target_sr": 1000,
    "fhr_sr": 4,
    "segment_duration": 3.75,
    "segment_stride": 0.75,
    "time_bins": 1000,
    "freq_bins": 40,
    "filters": 40,
    "kernel_size": 20,
    "num_heads": 8,
    "key_dim": 40,
    "dropout_rate": 0.25,
    "min_peak_distance": 60,
}

CHECKPOINTS = {
    "task1": CHECKPOINT_ROOT / "task1_autofhr.pt",
    "task2": CHECKPOINT_ROOT / "task2_light_unet.pt",
    "task3_native": CHECKPOINT_ROOT / "task3_fused_native.pt",
    "task3_flow": CHECKPOINT_ROOT / "task3_fused_flow.pt",
}

FHR_POINTS = 4800
FHR_MIN = 50.0
FHR_MAX = 210.0
TASK1_MIN_COVERAGE = 0.95
