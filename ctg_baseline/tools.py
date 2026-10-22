from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
from scipy.io import wavfile
from scipy.signal import find_peaks

from .config import CHECKPOINTS, FHR_MAX, FHR_MIN, FHR_POINTS, TASK1_CONFIG
from .models import build_task1_model, build_task2_model, build_task3_fused_model
from .signal_processing import DUS_filtering, create_scalogram_from_signal, signal_resample


INDEX_TO_LABEL = np.asarray((0, 1, -1), dtype=np.int8)


def load_weights(model: torch.nn.Module, path: Path, device: torch.device) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    state = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state.get("model", state) if isinstance(state, dict) else state)
    model.eval()


def fill_fhr(values: np.ndarray) -> tuple[np.ndarray, float]:
    fhr = np.asarray(values, dtype=np.float32).copy()
    valid = np.isfinite(fhr) & (fhr >= FHR_MIN) & (fhr <= FHR_MAX)
    missing_rate = 1.0 - float(valid.mean()) if len(fhr) else 1.0
    if not valid.any():
        return np.full(FHR_POINTS, 140.0, dtype=np.float32), 1.0
    index = np.arange(len(fhr))
    fhr[~valid] = np.interp(index[~valid], index[valid], fhr[valid])
    if len(fhr) != FHR_POINTS:
        fhr = np.interp(
            np.linspace(0, 1, FHR_POINTS), np.linspace(0, 1, len(fhr)), fhr
        ).astype(np.float32)
    return fhr, missing_rate


def clean_events(labels: np.ndarray, minimum_points: int = 60, merge_gap: int = 20) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int8).copy()
    for event_label in (1, -1):
        indices = np.flatnonzero(labels == event_label)
        if not len(indices):
            continue
        groups = np.split(indices, np.where(np.diff(indices) > merge_gap + 1)[0] + 1)
        for group in groups:
            start, end = int(group[0]), int(group[-1]) + 1
            if end - start >= minimum_points:
                labels[start:end] = event_label
            else:
                labels[group] = 0
    return labels


class Task1Tool:
    def __init__(self, device: torch.device, checkpoint: Path = CHECKPOINTS["task1"]):
        self.device = device
        self.model = build_task1_model(
            (TASK1_CONFIG["time_bins"], TASK1_CONFIG["freq_bins"]), TASK1_CONFIG
        ).to(device)
        load_weights(self.model, checkpoint, device)

    @staticmethod
    def _scalogram(signal: np.ndarray, sampling_rate: int) -> np.ndarray:
        signal = DUS_filtering(signal, fs=sampling_rate)
        if sampling_rate != TASK1_CONFIG["target_sr"]:
            signal = signal_resample(signal, sampling_rate, TASK1_CONFIG["target_sr"])
        target = int(TASK1_CONFIG["segment_duration"] * TASK1_CONFIG["target_sr"])
        signal = np.pad(signal[:target], (0, max(0, target - len(signal))))
        image = create_scalogram_from_signal(
            signal,
            TASK1_CONFIG["target_sr"],
            TASK1_CONFIG["time_bins"],
            TASK1_CONFIG["freq_bins"],
        ).T
        low, high = float(image.min()), float(image.max())
        if high > low:
            image = (image - low) / (high - low)
        return image.astype(np.float32)

    @staticmethod
    def _fhr_from_mask(mask: np.ndarray) -> float:
        peaks, _ = find_peaks(
            np.asarray(mask).squeeze(), height=0.5, distance=TASK1_CONFIG["min_peak_distance"]
        )
        if len(peaks) < 2:
            return float("nan")
        intervals = np.diff(peaks).astype(np.float64)
        median = np.median(intervals)
        keep = (intervals >= 0.7 * median) & (intervals <= 1.3 * median)
        mean_interval = np.mean(intervals[keep if keep.any() else np.ones_like(keep, bool)])
        bpm = 60.0 / (mean_interval * TASK1_CONFIG["segment_duration"] / TASK1_CONFIG["time_bins"])
        return float(bpm) if FHR_MIN <= bpm <= FHR_MAX else float("nan")

    def run(self, wav_path: Path, batch_size: int = 32) -> dict[str, object]:
        sampling_rate, signal = wavfile.read(wav_path)
        if signal.ndim > 1:
            signal = signal[:, 0]
        signal = signal.astype(np.float32)
        window = int(TASK1_CONFIG["segment_duration"] * sampling_rate)
        stride = int(TASK1_CONFIG["segment_stride"] * sampling_rate)
        starts = list(range(0, max(1, len(signal) - window + 1), stride))
        output_length = int(math.ceil(len(signal) / sampling_rate * TASK1_CONFIG["fhr_sr"]))
        total = np.zeros(output_length, dtype=np.float64)
        count = np.zeros(output_length, dtype=np.int32)
        with torch.no_grad():
            for offset in range(0, len(starts), batch_size):
                selected = starts[offset : offset + batch_size]
                features = [self._scalogram(signal[start : start + window], sampling_rate) for start in selected]
                masks = torch.sigmoid(self.model(torch.from_numpy(np.asarray(features)).to(self.device))).cpu().numpy()
                for start, mask in zip(selected, masks):
                    bpm = self._fhr_from_mask(mask)
                    if not np.isfinite(bpm):
                        continue
                    first = int(start / sampling_rate * TASK1_CONFIG["fhr_sr"])
                    last = min(first + int(TASK1_CONFIG["segment_duration"] * TASK1_CONFIG["fhr_sr"]), output_length)
                    total[first:last] += bpm
                    count[first:last] += 1
        raw = np.divide(total, count, out=np.full(output_length, np.nan), where=count > 0)
        fhr, missing_rate = fill_fhr(raw)
        return {"fhr": fhr, "missing_rate": missing_rate, "coverage": 1.0 - missing_rate}


class Task2Tool:
    def __init__(self, device: torch.device, checkpoint: Path = CHECKPOINTS["task2"]):
        self.device = device
        self.model = build_task2_model().to(device)
        load_weights(self.model, checkpoint, device)

    def run(self, fhr_values: np.ndarray) -> dict[str, object]:
        fhr, missing_rate = fill_fhr(fhr_values)
        model_input = np.clip((fhr - 110.0) / 50.0, -1.2, 2.0)
        with torch.no_grad():
            probabilities = self.model(
                torch.from_numpy(model_input[None, None].astype(np.float32)).to(self.device)
            )[0].cpu().numpy()
        labels = clean_events(INDEX_TO_LABEL[probabilities.argmax(0)])
        confidence = float(probabilities.max(axis=0).mean())
        return {
            "probabilities": probabilities.astype(np.float32),
            "labels": labels,
            "mean_confidence": confidence,
            "input_missing_rate": missing_rate,
        }


class Task3Tool:
    def __init__(self, device: torch.device, checkpoint: Path):
        self.device = device
        self.model = build_task3_fused_model(
            input_channels_wave=1, input_channels_pattern=3, input_length=FHR_POINTS, dropout=0.5
        ).to(device)
        load_weights(self.model, checkpoint, device)

    def run(self, fhr_values: np.ndarray, task2_probabilities: np.ndarray) -> dict[str, object]:
        fhr, _ = fill_fhr(fhr_values)
        probabilities = np.asarray(task2_probabilities, dtype=np.float32)
        if probabilities.shape != (3, FHR_POINTS):
            raise ValueError(f"Expected Task2 probabilities (3, {FHR_POINTS}), got {probabilities.shape}")
        if not np.isfinite(probabilities).all() or not np.allclose(probabilities.sum(0), 1.0, atol=2e-3):
            raise ValueError("Invalid Task2 probabilities")
        standardized = (fhr - fhr.mean()) / (fhr.std() + 1e-8)
        with torch.no_grad():
            risk = float(
                self.model(
                    torch.from_numpy(standardized[None, None].astype(np.float32)).to(self.device),
                    torch.from_numpy(probabilities[None]).to(self.device),
                ).item()
            )
        return {"risk_probability": risk, "risk_class": int(risk >= 0.5)}
