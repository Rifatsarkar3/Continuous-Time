"""Price of exactness: leave-one-out CV-CRC against the 4-fold cross-fitted rule, PRONOSTIA dev.

PRE-DECLARED 2026-11-19, committed before the full run (a smoke run with 20 training iterations
on outer fold 0 checks the path only and writes nothing to results/).

WHY. The bearing arms certify with 4-fold cross-fitted CRC: each calibration bearing is scored by
the inner model that did not see it, the threshold uses the asset-level term 1/(n+1), and a test
bearing's outcome is averaged over the four inner models. No finite-sample theorem covers that
construction. Cross-validation CRC (Cohen et al., ISIT 2024, Theorem 1) does guarantee expected
loss <= alpha, but needs K >= B/alpha - 1 = 9 folds and deploys the union of the K fold models'
decisions, which for a running-minimum alarm is the EARLIEST alarm across the K models. With one
asset per fold (K = n) its risk estimate is exactly the asset-level formula.

VARIANTS, CT-GRU, seed 42, `full` observation process, the five outer folds of E2 seed 42:
  V1  stored E2 rule: 4 inner folds, outcome averaged over the 4 models     (no theorem)
  V2  leave-one-out, one model per calibration bearing, outcome averaged    (no theorem)
  V3  leave-one-out, earliest alarm across the n models = exact CV-CRC      (Theorem 1, n >= 9)
V2 and V3 share their threshold, so V2 -> V3 isolates the deployment rule and V1 -> V2 the fold
count. V1 is read from results/checkpoints/pron_ctgru_pronostia_ctgru.jsonl, never recomputed.

REPORTED, descriptive: per variant the pooled realized late rate and wasted life over the 17
bearings; paired per-bearing differences V2 - V1 and V3 - V1 (mean, 95% t-interval, sign-flip p);
thresholds per fold; and the lead-point check that each leave-one-out threshold is the first
grid value above the largest calibration lead-point value (k_n = 0 at n = 13, 14).

STATUS. PRONOSTIA test folds are spent; this is development evidence on one seed, not a test.

FORECAST, recorded so it can be checked: V3 wastes more than V1 by more than 0.05; V3's late rate
is at most V1's; V2 lies within 0.05 of V1 in wasted life.

    python experiments/dev_exact_cvcrc.py --smoke
    python experiments/dev_exact_cvcrc.py
"""
import argparse
import collections
import datetime
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from certalarm import assets, certify, checkpoint, crossfit, evaluate, paths  # noqa: E402
from experiments.c1_leadpoint_xjtu import paired_summary  # noqa: E402
from experiments.e2_main_alarms import ALPHA, LEAD, CTGRUPredictor  # noqa: E402

SEED, PROCESS = 42, "full"
STORED = "pron_ctgru_pronostia_ctgru"


def trajectory(pred, model, asset):
    return crossfit._trajectory(pred, model, asset, {"M": np.ones(asset.n_min, bool)})


def lead_point(t):
    """Running minimum of mu up to the last minute with rul >= LEAD, the lead-point value."""
    last = np.flatnonzero(t.rul >= LEAD)
    return float(np.min(t.mu[:last[-1] + 1])) if last.size else float("inf")


def run_fold(fleet, spec, pred, i):
    by = {a.name: a for a in fleet}
    cal_names = spec["calibration"]
    models = {c: pred.fit([by[n] for n in cal_names if n != c], seed=SEED * 1000 + 100 * i + k,
                          process=PROCESS)
              for k, c in enumerate(cal_names)}
    cal = {c: [trajectory(pred, models[c], by[c])] for c in cal_names}
    theta = certify.crc_threshold(evaluate.loss_matrix(cal, evaluate.THETAS, LEAD),
                                  evaluate.THETAS, ALPHA)
    tstar = max(lead_point(reps[0]) for reps in cal.values())
    grid_next = float(evaluate.THETAS[np.searchsorted(evaluate.THETAS, tstar, side="right")])
    out = {"fold": i, "n_cal": len(cal_names), "theta": theta, "max_theta_star": tstar,
           "prop1_holds": theta is not None and abs(theta - grid_next) < 1e-12, "test": {}}
    for t in spec["test"]:
        reps = [trajectory(pred, models[c], by[t]) for c in cal_names]
        th = np.array([theta])
        l2, w2 = evaluate._asset_curves(reps, th, LEAD, lambda r: r.mu)
        earliest = crossfit.Trajectory(np.min([r.mu for r in reps], axis=0), reps[0].sigma,
                                       reps[0].rul, reps[0].can_fire)
        l3, w3 = evaluate._asset_curves([earliest], th, LEAD, lambda r: r.mu)
        out["test"][t] = {"V2": {"late": float(l2[0]), "waste": float(w2[0])},
                          "V3": {"late": float(l3[0]), "waste": float(w3[0])}}
    return out


def stored_v1():
    rows = [r for rs in checkpoint.completed(STORED).values() for r in rs
            if r["seed"] == SEED and r["process"] == PROCESS and r["rule"] == "crc"]
    return {r["asset"]: {"late": r["late"], "waste": r["waste"], "theta": r["theta"],
                         "fold": r["fold"]} for r in rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="outer fold 0, 20 iterations, no output")
    a = ap.parse_args()
    fleet = assets.load("pronostia")
    specs = crossfit.build_fold(fleet, seed=SEED)
    pred = CTGRUPredictor("pronostia", iters=20 if a.smoke else None)
    folds = [run_fold(fleet, s, pred, i) for i, s in enumerate(specs[:1] if a.smoke else specs)]
    v1 = stored_v1()
    per = {t: {"V1": {k: v1[t][k] for k in ("late", "waste")}, **f["test"][t]}
           for f in folds for t in f["test"]}
    names = sorted(per)
    summary = {v: {"late": float(np.mean([per[t][v]["late"] for t in names])),
                   "waste": float(np.mean([per[t][v]["waste"] for t in names]))}
               for v in ("V1", "V2", "V3")}
    paired = {f"{v}-V1": {f: paired_summary([per[t][v][f] - per[t]["V1"][f] for t in names])
                          for f in ("waste", "late")} for v in ("V2", "V3")}
    for f in folds:
        print(f"fold {f['fold']}: n_cal {f['n_cal']} theta {f['theta']} max theta* "
              f"{f['max_theta_star']:.4f} prop1 {f['prop1_holds']}  stored V1 theta "
              f"{sorted({v1[t]['theta'] for t in f['test']})}")
    for v, s in summary.items():
        print(f"{v}: late {s['late']:.3f} waste {s['waste']:.3f}  ({len(names)} bearings)")
    for k, d in paired.items():
        w = d["waste"]
        print(f"{k} waste: mean {w['mean']:+.4f} CI [{w['ci95'][0]:+.4f}, {w['ci95'][1]:+.4f}] "
              f"p {w['perm_p_two_sided']:.3f}")
    if a.smoke:
        print("smoke: path checked, nothing written")
        return
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=paths.ROOT,
                            capture_output=True, text=True).stdout.strip()
    (paths.RESULTS / "dev_exact_cvcrc.json").write_text(json.dumps({
        "provenance": {"generated_at_utc": datetime.datetime.now(datetime.timezone.utc)
                       .isoformat(timespec="seconds"), "git_commit": commit,
                       "seed": SEED, "process": PROCESS, "alpha": ALPHA, "lead": LEAD},
        "folds": folds, "per_bearing": per, "summary": summary, "paired": paired},
        indent=2, default=float), encoding="utf-8")
    print("written to results/dev_exact_cvcrc.json")


if __name__ == "__main__":
    main()
