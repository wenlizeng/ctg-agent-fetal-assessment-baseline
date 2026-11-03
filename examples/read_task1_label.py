"""Read one released Task1 training pair and optionally export its FHR label."""
import argparse
from pathlib import Path

import numpy as np
from scipy.io import wavfile

from ctg_baseline.io import load_fetal_fhr, read_manifest, resolve_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--record-id", help="Defaults to the first training record")
    parser.add_argument("--output", type=Path, help="Optional CSV export path")
    args = parser.parse_args()
    rows = read_manifest(args.data_root / "task1/manifests/train.csv", args.data_root, "task1")
    row = next((r for r in rows if args.record_id is None or r["record_id"] == args.record_id), None)
    if row is None:
        parser.error("record-id is not in the Task1 training manifest")
    fhr = load_fetal_fhr(resolve_path(args.data_root, row["fetal_path"]))
    sample_rate, dus = wavfile.read(resolve_path(args.data_root, row["wav_path"]))
    if sample_rate != 1000 or dus.ndim != 1 or len(dus) != 1200000 or len(fhr) != 4800:
        raise ValueError("Expected paired 20-minute mono 1 kHz DUS and 4 Hz FHR")
    valid = np.isfinite(fhr)
    print(f"record_id={row['record_id']} DUS={dus.shape} FHR={fhr.shape} "
          f"valid_reference_fraction={valid.mean():.4f}")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(args.output, np.column_stack((np.arange(len(fhr)) / 4, fhr, valid)),
                   delimiter=",", header="time_sec,fhr,valid", comments="", fmt=["%.2f", "%.1f", "%d"])
        print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
