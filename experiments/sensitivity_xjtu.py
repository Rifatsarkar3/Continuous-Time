"""Descriptive sensitivity of XJTU-SY conformal risk control to alpha and lead.

Reads the trajectories saved by experiments/rescore_crc_scaled_xjtu.py (data/rescore_xjtu/,
git-ignored) and refits only the conformal threshold per fold; the test bearings were read before,
so this is descriptive. With 12 calibration bearings the certifiability floor 1/13 = 0.077 exceeds
alpha = 0.05, so no fold can certify there; an uncertified fold scores late 1 and waste 0, as
everywhere in this study.

    python experiments/sensitivity_xjtu.py        # writes results/sensitivity_xjtu.json
"""
import json
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from certalarm import certify, evaluate, paths  # noqa: E402

ALPHAS, LEADS = (0.05, 0.08, 0.10, 0.12, 0.15, 0.16, 0.20, 0.23, 0.24, 0.30), (0.02, 0.05, 0.10)
TRAJ = paths.DATA / "rescore_xjtu"


def load(path):
    # Trusted input only: written locally by rescore_crc_scaled_xjtu.py, never downloaded.
    d = pickle.loads(path.read_bytes())
    to = lambda reps: [SimpleNamespace(mu=m.astype(float), rul=r.astype(float), can_fire=c, scale=s)
                       for m, r, c, s in reps]
    return {k: {a: to(reps) for a, reps in d[k].items()} for k in ("calibration", "test")}


def main():
    out = {}
    for arm_dir in sorted(p for p in TRAJ.iterdir() if p.is_dir()):
        units = [load(p) for p in sorted(arm_dir.glob("s*_f*.pkl"))]
        if len(units) != 15:
            print(f"{arm_dir.name}: {len(units)}/15 units, skipped")
            continue
        res = {}
        for lead in LEADS:
            for alpha in ALPHAS:
                late, waste, cert = [], [], 0
                for u in units:
                    th = certify.crc_threshold(evaluate.loss_matrix(u["calibration"], evaluate.THETAS,
                                                                    lead), evaluate.THETAS, alpha)
                    cert += th is not None
                    for reps in u["test"].values():
                        if th is None:
                            late.append(1.0), waste.append(0.0)
                            continue
                        l, w = evaluate._asset_curves(reps, np.array([th]), lead, lambda t: t.mu)
                        late.append(float(l[0])), waste.append(float(w[0]))
                res[f"alpha={alpha}/lead={lead}"] = {"late": float(np.mean(late)),
                                                     "waste": float(np.mean(waste)),
                                                     "certified_folds": [cert, len(units)]}
        out[arm_dir.name] = res
        for k, v in res.items():
            print(f"{arm_dir.name:<9} {k:<22} late {v['late']:.3f} waste {v['waste']:.3f} "
                  f"cert {v['certified_folds'][0]}/{v['certified_folds'][1]}")
    (paths.RESULTS / "sensitivity_xjtu.json").write_text(json.dumps(
        {"note": "descriptive; XJTU-SY test bearings were read before",
         "floor_n12": certify.certifiable_floor(12), "arms": out}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
