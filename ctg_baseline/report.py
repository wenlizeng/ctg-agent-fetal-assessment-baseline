from __future__ import annotations

import numpy as np


def event_intervals(labels: np.ndarray, value: int, fs: int = 4) -> list[dict[str, float]]:
    indices = np.flatnonzero(np.asarray(labels) == value)
    groups = [] if not len(indices) else np.split(indices, np.where(np.diff(indices) > 1)[0] + 1)
    return [
        {"start_sec": round(float(group[0]) / fs, 2), "end_sec": round(float(group[-1] + 1) / fs, 2)}
        for group in groups
    ]


def build_report(
    record_id: str,
    fhr: np.ndarray,
    labels: np.ndarray,
    risk_probability: float,
    missing_rate: float,
    abstain: bool,
) -> tuple[dict[str, object], str]:
    accelerations = event_intervals(labels, 1)
    decelerations = event_intervals(labels, -1)
    risk_class = int(risk_probability >= 0.5)
    quality = "poor" if missing_rate > 0.05 else "good"
    structured = {
        "record_id": record_id,
        "signal_quality": quality,
        "baseline_fhr": round(float(np.median(fhr)), 2),
        "mean_fhr_bpm": round(float(np.mean(fhr)), 2),
        "fhr_std_bpm": round(float(np.std(fhr)), 2),
        "acceleration_count": len(accelerations),
        "acceleration_intervals": accelerations,
        "deceleration_count": len(decelerations),
        "deceleration_intervals": decelerations,
        "risk_probability": round(float(risk_probability), 8),
        "risk_class": risk_class,
        "conclusion": "indeterminate" if abstain else ("abnormal" if risk_class else "normal"),
        "evidence": [
            *[f"Acceleration detected at {x['start_sec']:.2f}-{x['end_sec']:.2f} s" for x in accelerations],
            *[f"Deceleration detected at {x['start_sec']:.2f}-{x['end_sec']:.2f} s" for x in decelerations],
        ],
        "abstain": abstain,
    }
    text = (
        "CTG Agent Diagnostic Report\n"
        f"Record ID: {record_id}\n"
        f"Signal quality: {quality}; pre-fill missing FHR: {missing_rate:.2%}\n"
        f"FHR: median {structured['baseline_fhr']:.1f} bpm, mean {structured['mean_fhr_bpm']:.1f} bpm, "
        f"standard deviation {structured['fhr_std_bpm']:.1f} bpm\n"
        f"Accelerations: {len(accelerations)}; intervals: {accelerations or 'none'}\n"
        f"Decelerations: {len(decelerations)}; intervals: {decelerations or 'none'}\n"
        f"Assessment: {structured['conclusion']}; abnormality probability {risk_probability:.2%}\n"
        "This challenge report is not a clinical diagnosis or management recommendation."
    )
    return structured, text
