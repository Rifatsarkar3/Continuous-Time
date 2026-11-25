"""Defect-corrected re-score of condition-scaled CRC (crc_scaled) on the two XJTU-SY arms.

Pre-declared on 2026-11-19, committed
before this script was first run.

THE DEFECT, fixed at 5665fb1. evaluate.apply_rules chose each rule's test-time signal from its
`shift` alone. crc_scaled is calibrated on mu / s_k but carries shift 0.0, so its threshold was
applied to the raw mu of the test bearings. Its theta and certification decision are
calibration-side and were unaffected; its test-side late and wasted life in the XJTU-SY arms were
not those of the pre-registered rule mu_t < theta * s_k.

WHAT THIS DOES. Rebuilds every (seed, outer fold) unit of the XJTU-SY CT-GRU and DSAFLite arms as
experiments/e2_main_alarms.py did - same seeds, nested folds, observation masks and cached inner
models - refits all seven rules and applies them with the corrected apply_rules. It never trains:
a model-cache miss, or a cache file that fails to load, aborts. It never reads or writes
results/registry.jsonl: the registry already holds these splits, and this corrects how one rule's
outcome was computed; it evaluates no new model, threshold or rule. Trajectories are saved under
data/rescore_xjtu/ (git-ignored) for descriptive analyses, which are reported as such.

REPRODUCTION CHECKS, fixed before running.
  CT-GRU    deterministic predictor: every stored theta and certification flag (all seven rules)
            and every stored per-asset late and waste of the six unaffected rules must reproduce
            to 1e-9, or the arm is NOT REPRODUCED and its corrected crc_scaled numbers are unused.
  DSAFLite  the original run drew its 30 Monte Carlo dropout passes from an unseeded generator,
            so it cannot be reproduced bit for bit. Every predict call here is seeded from
            (seed, fold, call index). The arm is REPRODUCED if every certification flag matches
            and pooled crc late and wasted life each lie within 0.02 of the stored values;
            crc_scaled is then compared with plain crc from this same re-run, paired on identical
            trajectories.

Run with the interpreter that produced the stored arms (PYTHON in run_xjtu.sh):

    python experiments/rescore_crc_scaled_xjtu.py --arm ctgru
    python experiments/rescore_crc_scaled_xjtu.py --arm dsaflite
    python experiments/rescore_crc_scaled_xjtu.py --analyse
"""
import argparse
import collections
import datetime
import itertools
import json
import pickle
import platform
import subprocess
import sys
import zlib
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from certalarm import assets, checkpoint, crossfit, evaluate, paths  # noqa: E402
from certalarm.predictors import dsaf_train, modelcache  # noqa: E402
from experiments.c1_leadpoint_xjtu import paired_summary  # noqa: E402
from experiments.e2_main_alarms import ALPHA, LEAD, CTGRUPredictor, DSAFLitePredictor  # noqa: E402

SEEDS, PROCESS, N_REP = (42, 123, 777), "full", 8
STORED = {"ctgru": "xjtu_ctgru_xjtu_ctgru", "dsaflite": "xjtu_dsaflite_xjtu_dsaflite"}
TOL_EXACT, TOL_POOLED = 1e-9, 0.02
UNAFFECTED = tuple(r for r in evaluate.RULES if r != "crc_scaled")
TRAJ = paths.DATA / "rescore_xjtu"
OUT = paths.RESULTS / "rescore_crc_scaled_xjtu.json"


def cached_only(cache_key, make_empty, train_fn, enabled=True):
    """modelcache.cached without its fallback to training: a miss or a bad file aborts."""
    path = modelcache.DIR / f"{cache_key}.pt"
    if not enabled or not path.exists():
        raise RuntimeError(f"model cache miss {cache_key}: refusing to train in a re-score")
    model = make_empty()
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    model.eval()
    return model


class SeededMC:
    """Seeds every DSAFLite predict call from (seed, fold, call index); call order is fixed."""

    def __init__(self):
        self.orig, self.base, self.calls = dsaf_train.predict_minutes, 0, itertools.count()

    def start(self, seed, fold):
        self.base, self.calls = zlib.crc32(f"xjtu/{seed}/{fold}".encode()), itertools.count()

    def __call__(self, model, asset, *a, **k):
        torch.manual_seed((self.base + next(self.calls)) % (2 ** 31))
        return self.orig(model, asset, *a, **k)


def compare(old, new):
    """-> mismatch counts between the stored and the re-scored rows of one unit."""
    o = {(r["rule"], r["asset"]): r for r in old}
    n = {(r["rule"], r["asset"]): r for r in new}
    if set(o) != set(n):
        raise RuntimeError(f"row sets differ: {sorted(set(o) ^ set(n))}")
    rep = {"theta_mismatch": 0, "max_dtheta": 0.0, "flag_mismatch": 0,
           "outcome_mismatch": 0, "max_doutcome": 0.0}
    for k, a in o.items():
        b = n[k]
        if (a["theta"] is None) != (b["theta"] is None):
            rep["theta_mismatch"] += 1
            rep["max_dtheta"] = float("inf")
        elif a["theta"] is not None:
            d = abs(a["theta"] - b["theta"])
            rep["max_dtheta"] = max(rep["max_dtheta"], d)
            rep["theta_mismatch"] += int(d > TOL_EXACT)
        rep["flag_mismatch"] += int(bool(a["certifiable"]) != bool(b["certifiable"]))
        if k[0] in UNAFFECTED:
            d = max(abs(a["late"] - b["late"]), abs(a["waste"] - b["waste"]))
            rep["max_doutcome"] = max(rep["max_doutcome"], d)
            rep["outcome_mismatch"] += int(d > TOL_EXACT)
    return rep


def _dump(arm, seed, fold, res):
    pack = lambda d: {a: [(t.mu.astype(np.float32), t.rul.astype(np.float32), t.can_fire, t.scale)
                          for t in reps] for a, reps in d.items()}
    (TRAJ / arm).mkdir(parents=True, exist_ok=True)
    with open(TRAJ / arm / f"s{seed}_f{fold}.pkl", "wb") as f:
        pickle.dump({"calibration": pack(res.calibration), "test": pack(res.test)}, f)


def run(arm):
    modelcache.cached = cached_only                       # no training, ever
    pred = CTGRUPredictor("xjtu") if arm == "ctgru" else DSAFLitePredictor(dann=False)
    mc = None
    if arm == "dsaflite":
        mc = SeededMC()
        dsaf_train.predict_minutes = mc
    fleet = assets.load("xjtu")
    stored = checkpoint.completed(STORED[arm])
    if len(stored) != 15:
        raise SystemExit(f"{arm}: expected 15 stored units, found {len(stored)}")
    tag = f"rescore_xjtu_{arm}"
    done = checkpoint.completed(tag)
    for seed in SEEDS:
        for i, spec in enumerate(crossfit.build_fold(fleet, seed=seed)):
            unit = {"dataset": "xjtu", "predictor": pred.name, "process": PROCESS,
                    "seed": seed, "fold": i}
            ukey = checkpoint.unit_key(unit)
            if ukey in done:
                continue
            if mc:
                mc.start(seed, i)
            fold = crossfit.run_fold(fleet, spec, pred, PROCESS, N_REP, seed)
            rules = evaluate.fit_rules(fold.calibration, ALPHA, LEAD)
            got = evaluate.apply_rules(rules, fold.test, LEAD)
            mae = {n: float(np.mean([np.abs(t.mu - t.rul).mean() for t in reps]))
                   for n, reps in fold.test.items()}
            rows = [{"dataset": "xjtu", "predictor": pred.name, "process": PROCESS,
                     "seed": seed, "fold": i, "rule": rule, "asset": a,
                     "certifiable": rules[rule].get("certifiable", False),
                     "theta": rules[rule]["theta"], "predictor_mae": mae[a], **row}
                    for rule, per_asset in got.items() for a, row in per_asset.items()]
            rep = compare(stored[ukey], rows)
            _dump(arm, seed, i, fold)
            print(f"  [done] {ukey}  {json.dumps(rep)}", flush=True)
            with (checkpoint.DIR / f"{tag}_reproduction.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps({"key": ukey, **rep}) + "\n")
            checkpoint.append(tag, unit, rows)


def summaries(rows):
    """Per rule, as Tables 4-5 report them: clustered on the bearing, n_boot 2000, seed 0."""
    out = {}
    for rule in evaluate.RULES:
        sel = [r for r in rows if r["rule"] == rule]
        keyed = {f"{r['seed']}/{r['fold']}/{r['asset']}": r for r in sel}
        s = evaluate.summarize(keyed, cluster_key={k: r["asset"] for k, r in keyed.items()})
        folds = {(r["seed"], r["fold"]): r["certifiable"] for r in sel}
        out[rule] = {"late": s["late_rate"], "boot_upper": s["late_upper_bootstrap"],
                     "cp_upper": s["late_upper_binomial"], "waste": s["waste_mean"],
                     "certified_folds": f"{sum(folds.values())}/{len(folds)}"}
    return out


def per_bearing(rows, rule, field):
    by = collections.defaultdict(list)
    for r in rows:
        if r["rule"] == rule:
            by[r["asset"]].append(r[field])
    return {a: float(np.mean(v)) for a, v in sorted(by.items())}


def analyse():
    report = {"provenance": {
        "generated_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=paths.ROOT,
                                     capture_output=True, text=True).stdout.strip(),
        "python": platform.python_version(), "torch": torch.__version__},
        "arms": {}}
    for arm in STORED:
        new = [r for rows in checkpoint.completed(f"rescore_xjtu_{arm}").values() for r in rows]
        old = [r for rows in checkpoint.completed(STORED[arm]).values() for r in rows]
        if len({(r["seed"], r["fold"]) for r in new}) != 15:
            report["arms"][arm] = {"status": "INCOMPLETE"}
            continue
        reps = [json.loads(l) for l in (checkpoint.DIR / f"rescore_xjtu_{arm}_reproduction.jsonl")
                .read_text(encoding="utf-8").splitlines() if l.strip()]
        s_new, s_old = summaries(new), summaries(old)
        flags_ok = all(r["flag_mismatch"] == 0 for r in reps)
        if arm == "ctgru":
            ok = flags_ok and all(r["theta_mismatch"] == 0 and r["outcome_mismatch"] == 0
                                  for r in reps)
        else:
            ok = flags_ok and all(abs(s_new["crc"][f] - s_old["crc"][f]) <= TOL_POOLED
                                  for f in ("late", "waste"))
        d = {f: paired_summary(np.subtract(list(per_bearing(new, "crc_scaled", f).values()),
                                           list(per_bearing(new, "crc", f).values())))
             for f in ("waste", "late")}
        report["arms"][arm] = {
            "status": "REPRODUCED" if ok else "NOT REPRODUCED",
            "reproduction_units": reps,
            "corrected": s_new, "stored_defective": s_old,
            "crc_scaled_minus_crc_per_bearing": d}
    OUT.write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
    for arm, a in report["arms"].items():
        print(f"\n=== {arm}: {a['status']}")
        if "corrected" in a:
            for rule in ("crc", "crc_scaled"):
                c, o = a["corrected"][rule], a["stored_defective"][rule]
                print(f"  {rule:<11} corrected late {c['late']:.3f} up {c['boot_upper']:.3f}/"
                      f"{c['cp_upper']:.3f} waste {c['waste']:.3f} cert {c['certified_folds']}"
                      f"   | stored late {o['late']:.3f} waste {o['waste']:.3f}")
            for f, s in a["crc_scaled_minus_crc_per_bearing"].items():
                print(f"  scaled - crc {f}: mean {s['mean']:+.4f} CI [{s['ci95'][0]:+.4f}, "
                      f"{s['ci95'][1]:+.4f}] sign-flip p {s['perm_p_two_sided']:.3f} "
                      f"lower {s['lp_lower']} higher {s['lp_higher']}")
    print(f"\nwritten to {OUT.relative_to(paths.ROOT)}")


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--arm", choices=sorted(STORED))
    g.add_argument("--analyse", action="store_true")
    a = ap.parse_args()
    if a.arm:
        print(f"python {platform.python_version()} torch {torch.__version__} "
              f"cuda {torch.cuda.is_available()}", flush=True)
        run(a.arm)
    else:
        analyse()


if __name__ == "__main__":
    main()
