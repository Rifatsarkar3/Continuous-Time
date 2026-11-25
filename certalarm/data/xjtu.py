"""XJTU-SY audit: processed windows vs raw 1-min files (numeric file order, consecutive windows)."""
from pathlib import Path

import numpy as np

COND_DIR = {1: "35Hz12kN", 2: "37.5Hz11kN", 3: "40Hz10kN"}


def raw_files(raw_root, name):
    d = Path(raw_root) / COND_DIR[int(name[7])] / name
    return sorted(d.glob("*.csv"), key=lambda p: int(p.stem))


def read_horizontal(path, max_rows=None):
    return np.loadtxt(path, delimiter=",", skiprows=1, usecols=0, dtype=np.float64,
                      max_rows=max_rows).astype(np.float32)


def audit_processed(proc_dir, raw_root, per_file=12, win=2560, atol=1e-4):
    """Compare each processed <split>_<nn>_<Bearing>.npz with the raw 1-min files.

    Checks: window count == per_file * number of files, and that the first two windows of file 1,
    file 2, the middle file and the last file (numeric order) equal the consecutive `win`-sample
    slices of that raw file. A raw file shorter than required counts as a mismatch.
    Verdicts: ok | missing_raw | mismatch.
    """
    out = {}
    for f in sorted(Path(proc_dir).glob("*_Bearing*.npz")):
        name = f.stem.split("_", 2)[2]
        w = np.load(f)["vibration"]
        files = raw_files(raw_root, name)
        n = len(files)
        if n == 0:
            out[name] = {"files": 0, "windows": int(len(w)), "verdict": "missing_raw"}
            continue

        def match(f_idx):
            base = f_idx * per_file
            raw = read_horizontal(files[f_idx], max_rows=2 * win)
            return all(base + k < len(w) and raw.size >= (k + 1) * win
                       and np.allclose(w[base + k], raw[k * win:(k + 1) * win], atol=atol)
                       for k in range(min(2, per_file)))

        checks = {"first": match(0), "second_file": match(1) if n > 1 else True,
                  "middle_file": match(n // 2), "last_file": match(n - 1)}
        count_ok = len(w) == per_file * n
        out[name] = {"files": n, "windows": int(len(w)), "count_ok": count_ok, **checks,
                     "verdict": "ok" if count_ok and all(checks.values()) else "mismatch"}
    return out
