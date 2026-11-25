"""PRONOSTIA (FEMTO / IEEE PHM 2012) rebuilt from the raw CSVs.

Raw layout: <root>/<Learning_set|Full_Test_Set>/<BearingX_Y>/acc_NNNNN.csv, one 0.1 s snapshot
(2560 samples at 25.6 kHz) every 10 s. Columns: hour, minute, second, microsecond,
horizontal acc, vertical acc. Bearing1_4 uses ';' as the separator.
"""
import json
from pathlib import Path

import numpy as np

H_COL = 4
BEARINGS = {
    "Bearing1_1": ("Learning_set", 1), "Bearing1_2": ("Learning_set", 1),
    "Bearing2_1": ("Learning_set", 2), "Bearing2_2": ("Learning_set", 2),
    "Bearing3_1": ("Learning_set", 3), "Bearing3_2": ("Learning_set", 3),
    **{f"Bearing1_{i}": ("Full_Test_Set", 1) for i in range(3, 8)},
    **{f"Bearing2_{i}": ("Full_Test_Set", 2) for i in range(3, 8)},
    "Bearing3_3": ("Full_Test_Set", 3),
}
CONDITIONS = {1: (1800.0, 4000.0), 2: (1650.0, 4200.0), 3: (1500.0, 5000.0)}  # (rpm, N)


def acc_files(bearing_dir):
    return sorted(Path(bearing_dir).glob("acc_*.csv"))


def read_acc(path):
    with open(path) as f:
        first = f.readline()
    sep = ";" if ";" in first else ","
    arr = np.loadtxt(path, delimiter=sep, dtype=np.float64, ndmin=2)
    return arr[:, H_COL].astype(np.float32)


def build_bearing(root, name):
    sub, cond = BEARINGS[name]
    files = acc_files(Path(root) / sub / name)
    if not files:
        raise FileNotFoundError(f"no acc_*.csv for {name} under {root}/{sub}")
    W = np.stack([read_acc(f) for f in files])
    n = len(W)
    rul = np.linspace(1.0, 0.0, n).astype(np.float32)
    return {"windows": W, "rul": rul, "condition": cond, "n": n}


def build_all(root, out_dir, names=None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for name in (BEARINGS if names is None else names):
        b = build_bearing(root, name)
        np.savez(out_dir / f"{name}.npz", windows=b["windows"], rul=b["rul"], condition=b["condition"])
        manifest[name] = {"n": b["n"], "condition": b["condition"], "source": BEARINGS[name][0]}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def _rms_rows(x):
    x = np.atleast_2d(np.asarray(x, np.float64))
    return np.sqrt((x ** 2).mean(axis=1))


def audit_processed(root, proc_dir, names=None, atol=1e-3):
    """Compare <proc_dir>/<name>/windows_vib.npy with the raw record, bearing by bearing.

    Checked windows: the first, up to three interior quartile positions, and the last raw
    snapshot (processed index n_raw - 1). Bearings without a processed file are skipped: the
    processed set covers only part of the catalogue. Verdicts:
      ok                           equal counts and every checked window matches
      extra_windows_after_raw_end  every checked window matches; processed has more windows
      missing_raw                  no raw snapshot files found for the bearing
      mismatch                     anything else
    """
    out = {}
    for name in (BEARINGS if names is None else names):
        f = Path(proc_dir) / name / "windows_vib.npy"
        if not f.exists():
            continue
        w = np.load(f, mmap_mode="r")
        files = acc_files(Path(root) / BEARINGS[name][0] / name)
        n_raw, n_proc = len(files), len(w)
        if n_raw == 0:
            out[name] = {"n_raw": 0, "n_processed": n_proc, "verdict": "missing_raw"}
            continue

        def same(k):
            return bool(k < n_proc and np.allclose(w[k], read_acc(files[k]), atol=atol))

        interior = sorted({n_raw // 4, n_raw // 2, (3 * n_raw) // 4} - {0, n_raw - 1})
        first_ok = same(0)
        interior_ok = all(same(k) for k in interior)
        last_raw = read_acc(files[-1])
        end_ok = bool(n_proc >= n_raw and np.allclose(w[n_raw - 1], last_raw, atol=atol))
        checked_ok = first_ok and interior_ok and end_ok
        if checked_ok and n_proc == n_raw:
            verdict = "ok"
        elif checked_ok and n_proc > n_raw:
            verdict = "extra_windows_after_raw_end"
        else:
            verdict = "mismatch"
        rec = {"n_raw": n_raw, "n_processed": n_proc, "extra_windows": n_proc - n_raw,
               "first_window_matches": first_ok, "interior_checked": interior,
               "interior_matches": interior_ok,
               "raw_end_found_at_index": (n_raw - 1) if end_ok else None,
               "rms_last_raw": float(_rms_rows(last_raw)[0]), "verdict": verdict}
        if n_proc > n_raw:
            rms_extra = _rms_rows(w[n_raw:])
            rec["rms_extra_first"] = float(rms_extra[0])
            rec["rms_extra_median"] = float(np.median(rms_extra))
            rec["rms_extra_last"] = float(rms_extra[-1])
        out[name] = rec
    return out
