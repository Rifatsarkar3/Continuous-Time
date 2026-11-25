"""E3: which bearings are late, and what distinguishes them.

A pooled late rate hides whether the risk is spread evenly or concentrated in a few bearings.
This breaks it down per bearing and sets it against two properties a maintainer could see in
advance: how long the bearing lived, and how abruptly its vibration energy rose at the end.

Run E2 (or its smoke) first, then:
    python experiments/e3_per_asset.py
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from certalarm import assets, paths  # noqa: E402


def onset_sharpness(asset):
    """Last-decile RMS over the median RMS: how abruptly the failure announced itself."""
    rms = np.sqrt((asset.windows.astype(np.float64) ** 2).mean(1))
    n = len(rms) // asset.per_min
    grid = rms[:n * asset.per_min].reshape(n, asset.per_min).mean(1)
    return float(grid[int(0.9 * n):].mean() / (np.median(grid) + 1e-12))


def _correlate(rows, key):
    late = np.array([r["late"] for r in rows], float)
    v = np.array([r[key] for r in rows], float)
    if late.std() == 0 or v.std() == 0:
        return None
    return float(np.corrcoef(late, v)[0, 1])


def main():
    src = paths.RESULTS / "e2_main_alarms.json"
    if not src.exists():
        src = paths.RESULTS / "e2_main_alarms_smoke.json"
    if not src.exists():
        raise SystemExit("run experiments/e2_main_alarms.py first")
    payload = json.loads(src.read_text())
    rows = payload["rows"]
    print(f"reading {src.name} (smoke={payload['provenance']['smoke']})")

    props = {}
    for ds in sorted({r["dataset"] for r in rows}):
        for a in assets.load(ds):
            props[(ds, a.name)] = {"minutes": a.n_min, "condition": a.condition,
                                   "onset_sharpness": onset_sharpness(a)}

    out = []
    for ds in sorted({r["dataset"] for r in rows}):
        for rule in sorted({r["rule"] for r in rows}):
            sel = [r for r in rows if r["dataset"] == ds and r["rule"] == rule]
            by_asset = {}
            for r in sel:
                by_asset.setdefault(r["asset"], []).append(r)
            for name, rs in sorted(by_asset.items()):
                out.append({"dataset": ds, "rule": rule, "asset": name,
                            "late": float(np.mean([x["late"] for x in rs])),
                            "waste": float(np.mean([x["waste"] for x in rs])),
                            "n": len(rs), **props[(ds, name)]})

    (paths.RESULTS / "e3_per_asset.json").write_text(json.dumps(out, indent=2))

    print(f"\nCRC, per bearing\n{'dataset':<11}{'bearing':<14}{'minutes':>8}{'sharp':>8}"
          f"{'late':>8}{'waste':>8}")
    crc = [r for r in out if r["rule"] == "crc"]
    for row in sorted(crc, key=lambda r: -r["late"]):
        print(f"{row['dataset']:<11}{row['asset']:<14}{row['minutes']:>8}"
              f"{row['onset_sharpness']:>8.2f}{row['late']:>8.3f}{row['waste']:>8.3f}")

    if len(crc) >= 4:
        print()
        for key in ("minutes", "onset_sharpness"):
            c = _correlate(crc, key)
            print(f"correlation of late rate with {key}: "
                  + ("undefined (no variation)" if c is None else f"{c:+.3f}"))
        share = sum(1 for r in crc if r["late"] > 0)
        print(f"{share} of {len(crc)} bearings account for all of the late alarms")
    print("\nwritten to results/e3_per_asset.json")


if __name__ == "__main__":
    main()
