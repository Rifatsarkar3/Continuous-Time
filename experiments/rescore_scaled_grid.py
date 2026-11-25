"""Condition-scaled CRC on XJTU-SY with the corrected threshold grid (descriptive, 2026-11-19).

The second defect found in the condition-scaled rule (fixed at 77ded6d): its grid stopped at 1.0
while mu / s exceeds 1 for scales below 1, so seven CT-GRU folds recorded "declined". This refits
plain and condition-scaled CRC on the trajectories the pre-declared re-score saved
(data/rescore_xjtu/, written by rescore_crc_scaled_xjtu.py; trusted local files) with the fixed
evaluate.fit_rules logic, and compares the two rules per bearing on identical trajectories.

    python experiments/rescore_scaled_grid.py      # writes results/rescore_scaled_grid.json
"""
import collections
import json
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from certalarm import certify, evaluate, paths  # noqa: E402
from experiments.c1_leadpoint_xjtu import paired_summary  # noqa: E402

ALPHA, LEAD = 0.10, 0.05


def load(path):
    # Trusted input only: written locally by rescore_crc_scaled_xjtu.py, never downloaded.
    d = pickle.loads(path.read_bytes())
    to = lambda reps: [SimpleNamespace(mu=m.astype(float), rul=r.astype(float), can_fire=c, scale=s)
                       for m, r, c, s in reps]
    return {k: {a: to(reps) for a, reps in d[k].items()} for k in ("calibration", "test")}


def threshold(cal, scaled):
    """The fit_rules logic for crc / crc_scaled, including the corrected grid for scaled signals."""
    sig = evaluate._scaled_signal if scaled else (lambda t: t.mu)
    ts = evaluate.THETAS
    if scaled:
        top = max(float(np.max(sig(t))) for reps in cal.values() for t in reps)
        if top >= ts[-1]:
            ts = np.concatenate([ts, np.arange(ts[-1] + 0.005, np.ceil(top / 0.005) * 0.005 + 0.0101, 0.005)])
    return certify.crc_threshold(evaluate.loss_matrix(cal, ts, LEAD, signal_of=sig), ts, ALPHA), sig


def main():
    out = {}
    for arm_dir in sorted(p for p in (paths.DATA / "rescore_xjtu").iterdir() if p.is_dir()):
        rows = []
        for f in sorted(arm_dir.glob("s*_f*.pkl")):
            u = load(f)
            for scaled in (False, True):
                th, sig = threshold(u["calibration"], scaled)
                for a, reps in u["test"].items():
                    if th is None:
                        late, waste = 1.0, 0.0
                    else:
                        l, w = evaluate._asset_curves(reps, np.array([th]), LEAD, sig)
                        late, waste = float(l[0]), float(w[0])
                    rows.append({"unit": f.stem, "asset": a, "rule": "crc_scaled" if scaled else "crc",
                                 "theta": th, "late": late, "waste": waste})
        res = {}
        for rule in ("crc", "crc_scaled"):
            sel = [r for r in rows if r["rule"] == rule]
            # the bounds of Tables 6-7: asset-clustered bootstrap and conservative Clopper-Pearson
            s = evaluate.summarize({f"{r['unit']}/{r['asset']}": r for r in sel},
                                   cluster_key={f"{r['unit']}/{r['asset']}": r["asset"] for r in sel})
            res[rule] = {"late": float(np.mean([r["late"] for r in sel])),
                         "waste": float(np.mean([r["waste"] for r in sel])),
                         "boot_upper": s["late_upper_bootstrap"], "cp_upper": s["late_upper_binomial"],
                         "certified_folds": len({r["unit"] for r in sel if r["theta"] is not None}),
                         "max_theta": max(r["theta"] for r in sel if r["theta"] is not None)}
        per = collections.defaultdict(lambda: collections.defaultdict(list))
        for r in rows:
            per[r["asset"]][r["rule"]].append(r)
        names = sorted(per)
        for f in ("waste", "late"):
            res[f"scaled_minus_crc_{f}"] = paired_summary(
                [np.mean([r[f] for r in per[n]["crc_scaled"]]) - np.mean([r[f] for r in per[n]["crc"]])
                 for n in names])
        out[arm_dir.name] = res
        d = res["scaled_minus_crc_waste"]
        print(f"{arm_dir.name}: crc late {res['crc']['late']:.3f} waste {res['crc']['waste']:.3f} | "
              f"scaled late {res['crc_scaled']['late']:.3f} waste {res['crc_scaled']['waste']:.3f} "
              f"certified {res['crc_scaled']['certified_folds']}/15 max theta {res['crc_scaled']['max_theta']:.3f} | "
              f"scaled-crc waste {d['mean']:+.3f} [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}] p {d['perm_p_two_sided']:.3f}")
    (paths.RESULTS / "rescore_scaled_grid.json").write_text(json.dumps(out, indent=2, default=float),
                                                          encoding="utf-8")


if __name__ == "__main__":
    main()
