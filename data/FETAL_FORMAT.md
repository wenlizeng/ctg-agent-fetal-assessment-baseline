# Task1 reference FHR files (.fetal)

Task1 pairs a single-channel DUS `.wav` recording with a device-reference FHR
trace in a binary `.fetal` file. The `.fetal` file is a training target, not raw
ultrasound, beat timestamps, or a text file. Task1 inference takes DUS only.
Task4 training provides the same type of FHR reference.

## File pairing and time axis

Use `data/task1/manifests/train.csv`: `record_id` identifies the released
segment, `wav_path` locates its input, and `fetal_path` locates its reference.
Both paths are relative to `data/`. Use the manifest pairing rather than
inferring a source file by stripping suffixes from a filename.

Each released segment spans 20 minutes: 1,200,000 DUS samples at 1 kHz and
4,800 FHR samples at 4 Hz. FHR sample `i` is on the segment-relative time axis
`i / 4` seconds (0 to 1199.75 seconds). No additional source-window offset
is needed for these already extracted segments.

## Binary layout used by the loader

The following fields describe the layouts used by the project reader. The
reader extracts the first fetal FHR channel; other device fields are not decoded.

| Field | Version 1 | Version 0 (legacy) |
|---|---|---|
| Header size | 8 bytes | 8 bytes |
| Byte 0 | Version = 1 | Version = 0 |
| Byte 1 | Fetal channel count `k` | Fetal channel count `k` |
| Bytes 2–5 | Not used by the reader | Sample count, unsigned 32-bit little-endian |
| Bytes 6–7 | Not used by the reader | Not used by the reader |
| Bytes per sample record | `8 + (k - 1) * 4` | `k + 2` |
| Number of samples | `(file_size - 8) / bytes_per_record` | Header sample count |
| First-channel FHR | First byte of each sample record, unsigned 8-bit bpm | Same |

Sample `i` is read at byte offset `8 + i * bytes_per_record`. The sampling
rate is the challenge's 4 Hz convention; the reader does not infer it from
the header. It rejects unsupported versions, invalid channel counts, empty
payloads, partial records, and legacy payload lengths inconsistent with the header.

## Python interface

```python
from pathlib import Path
import numpy as np
from ctg_baseline.io import load_fetal_fhr, read_manifest, resolve_path

root = Path("data")
row = read_manifest(root / "task1/manifests/train.csv", root, "task1")[0]
reference_fhr = load_fetal_fhr(resolve_path(root, row["fetal_path"]))
time_sec = np.arange(reference_fhr.size) / 4.0
valid_mask = np.isfinite(reference_fhr)
assert reference_fhr.shape == (4800,)
```

The result is a one-dimensional `float32` array in bpm. Values outside
**[50, 210] bpm**, including codes 0 and 255, become `NaN`. Missing points
remain on the time axis: the reader does not drop, interpolate, normalize,
resample, or pad them. The general `load_fhr(path)` interface also accepts
`.fetal` files.

When using these values as training targets, retain the validity mask and
compute supervised loss only at valid reference positions. Do not turn missing
reference values into zero-bpm targets. If a training window has no valid
targets, skip its supervised loss. Prediction validity and official scoring
are handled separately by the evaluation rules.

## Runnable example

From the repository root, inspect the first training pair:

```bash
python -m examples.read_task1_label
```

Optionally export its label to CSV (`time_sec,fhr,valid`):

```bash
python -m examples.read_task1_label --output output/task1_reference.csv
```

Use `--record-id ID` to select another training record. Export preserves missing
values as `nan` and writes `valid=0` at those positions.
