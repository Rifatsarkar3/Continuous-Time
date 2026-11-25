"""C1: confirmatory run of the LeadPoint ridge alarm on XJTU-SY. Its test bearings are spent once.

PRE-REGISTERED, 2026-09-23, before LeadPoint has scored any XJTU-SY bearing.

DISCLOSURE. This confirmation was decided AFTER Gate 0 (experiments/dev_leadpoint_gate0.py,
results committed at 32d7372) failed its pre-declared waste criterion: the winner, lp_ridge, had
dev LOO waste 0.690 against DSAFLite's 0.661. Gate 0's speed criterion passed (59,422x). The
candidate is Gate 0's winner, unchanged. The framing below was chosen before any XJTU look, after
a power calculation on dev data showed a non-inferiority test at margin 0.05 over 15 bearings has
roughly 10-20% power: the primary result is therefore an estimate with its interval, and the
tests are secondary.

CANDIDATE, frozen. The 18 minute-grid features plus the nine envelope-spectrum features, each
bearing baseline-corrected on its first 10 minutes, scaled on the inner-fit bearings only; ridge
regression on normalized remaining life, lambda 1.0, unpenalised intercept. 28 parameters. The
fit is closed form, so the seed does not enter. The predictive scale handed to the uncertified
baselines (lbc, supband) is the fit-set residual SD; it plays no part in the certified rule.

PROTOCOL, identical to the two XJTU arms it is compared with (run_xjtu.sh): seeds 42, 123, 777;
the `full` observation process; n_rep 8; 5 outer x 4 inner nested, condition-stratified folds;
CV-CRC with test outcomes averaged over the inner models; all seven rules fitted in one pass; the
write-once registry claims each (config, weights, split) once. Certification is the grid CRC used
by the reference arms - not Gate 0's exact threshold - so the three arms differ only in the
predictor.

PRIMARY ESTIMANDS. For each of the 15 bearings b, W_arm(b) = mean over the three seeds of that
bearing's `crc` wasted life. Paired differences d_DS(b) = W_LP(b) - W_DSAFLite(b) and
d_CT(b) = W_LP(b) - W_CTGRU(b). Reported for each: mean, median, 95% two-sided t-interval
(df 14), exact two-sided sign-flip permutation p over all 2^15 sign patterns, and the counts of
bearings where LeadPoint is lower / higher.

SECONDARY TESTS, fixed sequence at one-sided 5%; testing stops at the first failure:
  S1 non-inferiority vs DSAFLite: one-sided 95% upper bound of mean d_DS < 0.05
  S2 superiority vs CT-GRU:       one-sided 95% upper bound of mean d_CT < 0
  S3 superiority vs DSAFLite:     one-sided 95% upper bound of mean d_DS < 0
Power computed beforehand from dev (per-bearing SD of differences 0.21-0.24): S1 ~10-20%,
S2 ~28%, S3 under 10%.

VALIDITY. LeadPoint's pooled realized late rate with its upper bound, as E2 reports it
(evaluate.summarize, clustered on the bearing). 15 bearings cannot verify alpha (floor 0.218), so
this is a sanity check: a late rate above 0.25 is the pre-declared U2 signal and is investigated
as a defect before anything is claimed. Every fold must certify (n_cal = 12, floor 1/13); an
uncertified fold scores waste 0 and would flatter the arm, so it is reported and the analysis
stops.

SPEED, measured with nothing else running, on seed 42, outer fold 0, inner split 0:
  training   LeadPoint median of 5 fits from stored features; CT-GRU and DSAFLite one fresh fit
             each, cache disabled, weights discarded; devices recorded
  inference  outer fold 0's 12 calibration bearings, raw windows to minute signal, median of 3
A speed claim is made for training only. Inference is reported as measured and is expected to
be close to parity (Gate 0: 5.86 s against 5.97 s); "faster inference" is claimed only at >= 10x.

WORDING, committed now:
  S1 passes            "non-inferior to DSAFLite within 0.05 of normalized life (one-sided 95%)"
  S1 fails, CI has 0   "mean difference X (95% CI [a, b]); compatible with LeadPoint being up to b
                        worse or up to -a better; non-inferiority at 0.05 was not established"
                        - never "equivalent", "as accurate", or "matches"
  CI entirely above 0  "LeadPoint wasted more life than DSAFLite by X (95% CI [a, b])"
  S2 passes            "lower certified waste than CT-GRU (one-sided 95%)"

FORECAST, recorded so it can be checked: mean d_DS between -0.05 and +0.10 with a CI spanning
0; S1 fails; mean d_CT negative with a CI spanning 0; training speedup above 10,000x; inference
near parity.

    python experiments/c1_leadpoint_xjtu.py --smoke     # PRONOSTIA, no registry, path check only
    python experiments/c1_leadpoint_xjtu.py --run       # the one XJTU evaluation
    python experiments/c1_leadpoint_xjtu.py --timing    # after the DSAFLite arm has finished
    python experiments/c1_leadpoint_xjtu.py --analyse   # needs all three XJTU result files
"""
import argparse
import collections
import datetime
import itertools
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from scipy import stats as sps

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from certalarm import assets, crossfit, evaluate, features, leadpoint, paths  # noqa: E402
from certalarm.predictors import ctgru_train, dsaf_train  # noqa: E402
from experiments.e2_main_alarms import (ALPHA, LEAD, CTGRUPredictor, DSAFLitePredictor,  # noqa: E402
                                        report, run)

SEEDS = (42, 123, 777)
N_REP = 8
RIDGE_LAM = 1.0
NI_MARGIN = 0.05
U2_LATE = 0.25
ENV_CACHE = paths.DATA / "features_env"
OUT = paths.RESULTS / "c1_leadpoint_xjtu.json"
TIMING_OUT = paths.RESULTS / "c1_leadpoint_xjtu_timing.json"
ANALYSIS_OUT = paths.RESULTS / "c1_leadpoint_xjtu_analysis.json"
REFS = {"dsaflite": paths.RESULTS / "e2_main_alarms_xjtu_dsaflite.json",
        "ctgru": paths.RESULTS / "e2_main_alarms_xjtu_ctgru.json"}


class LeadPointRidgePredictor:
    """Gate 0's lp_ridge, in the interface e2_main_alarms.run expects."""

    name = "leadpoint_ridge"

    def __init__(self):
        self.grids = {}

    def grid(self, a):
        if a.name not in self.grids:
            f18 = features.asset_grid(a.dataset, a.name, a.windows, a.per_min,
                                      cache_dir=paths.DATA / "features")
            env = leadpoint.env_grid(a.dataset, a.name, a.windows, a.per_min, ENV_CACHE)
            assert len(f18) == len(env) == a.n_min, a.name
            self.grids[a.name] = np.concatenate([f18, env], 1)
        return self.grids[a.name]

    def fit(self, fit_assets, seed, process):
        G = [self.grid(a) for a in fit_assets]
        scale = features.fit_scale(G)
        X = [features.normalize(g, scale) for g in G]
        m = leadpoint.fit_ridge(X, [a.rul for a in fit_assets], lam=RIDGE_LAM)
        resid = np.concatenate([leadpoint.score(m, x) - a.rul for x, a in zip(X, fit_assets)])
        return {"model": leadpoint.ridge_module(m, scale, float(resid.std()))}

    def predict(self, model, asset, M):
        mod = model["model"]
        mu = leadpoint.module_score(mod, features.normalize(self.grid(asset), mod.scale.numpy()))
        return mu, np.full(len(mu), float(mod.resid_sd))


# ---------------------------------------------------------------- analysis (pre-registered)

def signflip_p(d):
    """Exact two-sided sign-flip permutation p-value for the mean of paired differences."""
    d = np.asarray(d, float)
    obs = abs(d.mean())
    signs = np.array(list(itertools.product((-1.0, 1.0), repeat=len(d))))
    return float(np.mean(np.abs(signs @ d / len(d)) >= obs - 1e-12))


def paired_summary(d):
    d = np.asarray(d, float)
    n = len(d)
    se = d.std(ddof=1) / np.sqrt(n)
    t2, t1 = sps.t.ppf(0.975, n - 1), sps.t.ppf(0.95, n - 1)
    return {"n": n, "mean": float(d.mean()), "median": float(np.median(d)),
            "sd": float(d.std(ddof=1)),
            "ci95": [float(d.mean() - t2 * se), float(d.mean() + t2 * se)],
            "upper_one_sided_95": float(d.mean() + t1 * se),
            "perm_p_two_sided": signflip_p(d),
            "lp_lower": int((d < -1e-12).sum()), "lp_higher": int((d > 1e-12).sum())}


def per_bearing(rows, predictor):
    sel = [r for r in rows if r["predictor"] == predictor and r["rule"] == "crc"
           and r["process"] == "full" and r["seed"] in SEEDS]
    by = collections.defaultdict(list)
    for r in sel:
        by[r["asset"]].append(r)
    bad = {a: len(v) for a, v in by.items() if len(v) != len(SEEDS)}
    if bad:
        raise SystemExit(f"{predictor}: bearings without exactly one row per seed: {bad}")
    return {a: {"waste": float(np.mean([r["waste"] for r in v])),
                "late": float(np.mean([r["late"] for r in v])),
                "certified": all(r["certifiable"] for r in v),
                "mae": float(np.mean([r["predictor_mae"] for r in v]))} for a, v in by.items()}


def fixed_sequence(ds, ct):
    steps = [("S1 non-inferiority vs DSAFLite", ds["upper_one_sided_95"] < NI_MARGIN),
             ("S2 superiority vs CT-GRU", ct["upper_one_sided_95"] < 0.0),
             ("S3 superiority vs DSAFLite", ds["upper_one_sided_95"] < 0.0)]
    out, going = [], True
    for name, ok in steps:
        out.append({"test": name, "result": ("PASS" if ok else "FAIL") if going else "NOT TESTED"})
        going = going and ok
    return out


def analyse():
    missing = [str(p) for p in [OUT, *REFS.values()] if not p.exists()]
    if missing:
        raise SystemExit(f"cannot analyse yet, missing: {missing}")
    rows = {"leadpoint_ridge": json.loads(OUT.read_text())["rows"]}
    for k, p in REFS.items():
        rows[k] = json.loads(p.read_text())["rows"]
    pb = {k: per_bearing(r, k) for k, r in rows.items()}
    names = sorted(pb["leadpoint_ridge"])
    if not all(sorted(v) == names for v in pb.values()) or len(names) != 15:
        raise SystemExit("the three arms do not cover the same 15 bearings")
    uncert = {k: [a for a in names if not v[a]["certified"]] for k, v in pb.items()}
    if any(uncert.values()):
        raise SystemExit(f"uncertified folds present, analysis stops: {uncert}")

    lp = pb["leadpoint_ridge"]
    d_ds = [lp[a]["waste"] - pb["dsaflite"][a]["waste"] for a in names]
    d_ct = [lp[a]["waste"] - pb["ctgru"][a]["waste"] for a in names]
    ds, ct = paired_summary(d_ds), paired_summary(d_ct)
    validity = {}
    for k, r in rows.items():
        sel = [x for x in r if x["predictor"] == k and x["rule"] == "crc" and x["seed"] in SEEDS]
        keyed = {f"{x['seed']}/{x['fold']}/{x['asset']}": x for x in sel}
        validity[k] = evaluate.summarize(keyed, cluster_key={q: x["asset"] for q, x in keyed.items()})
    u2 = validity["leadpoint_ridge"]["late_rate"] > U2_LATE
    seq = fixed_sequence(ds, ct)

    arms = {k: {"waste_mean": float(np.mean([v[a]["waste"] for a in names])),
                "late_rate": validity[k]["late_rate"], "late_upper": validity[k]["late_upper"],
                "mae": float(np.mean([v[a]["mae"] for a in names]))} for k, v in pb.items()}
    print(f"{'arm':<17}{'waste':>8}{'late':>8}{'upper':>8}{'MAE':>8}")
    for k, v in arms.items():
        print(f"{k:<17}{v['waste_mean']:>8.3f}{v['late_rate']:>8.3f}{v['late_upper']:>8.3f}"
              f"{v['mae']:>8.3f}")
    for lab, s in (("LP - DSAFLite", ds), ("LP - CT-GRU", ct)):
        print(f"\n{lab}: mean {s['mean']:+.3f}  median {s['median']:+.3f}  95% CI "
              f"[{s['ci95'][0]:+.3f}, {s['ci95'][1]:+.3f}]  one-sided upper "
              f"{s['upper_one_sided_95']:+.3f}  perm p {s['perm_p_two_sided']:.3f}  "
              f"LP lower {s['lp_lower']} / higher {s['lp_higher']} of {s['n']}")
    print()
    for s in seq:
        print(f"{s['test']:<34}{s['result']}")
    if u2:
        print(f"\nU2 SIGNAL: LeadPoint late rate {validity['leadpoint_ridge']['late_rate']:.3f} > "
              f"{U2_LATE}. Investigate as a defect before claiming anything.")
    timing = json.loads(TIMING_OUT.read_text()) if TIMING_OUT.exists() else None
    ANALYSIS_OUT.write_text(json.dumps({"arms": arms, "per_bearing": pb,
                                        "lp_minus_dsaflite": ds, "lp_minus_ctgru": ct,
                                        "fixed_sequence": seq, "u2_signal": u2,
                                        "timing": timing}, indent=2))
    print(f"\nwritten to {ANALYSIS_OUT.relative_to(paths.ROOT)}")


# ---------------------------------------------------------------- run, timing

def _median_time(fn, reps):
    t = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        t.append(time.perf_counter() - t0)
    return float(np.median(t))


def timing():
    fleet = assets.load("xjtu")
    by = {a.name: a for a in fleet}
    spec = crossfit.build_fold(fleet, seed=SEEDS[0])[0]
    fit_assets = [by[n] for n in spec["inner"][0]["fit"]]
    cal = [by[n] for n in spec["calibration"]]
    seed = SEEDS[0] * 100
    lp = LeadPointRidgePredictor()
    tm = {"device": {"leadpoint": "cpu", "ctgru": str(ctgru_train.DEVICE),
                     "dsaflite": str(dsaf_train.DEVICE)},
          "fit_seconds": {"leadpoint": _median_time(lambda: lp.fit(fit_assets, seed, "full"), 5)}}
    ct = CTGRUPredictor("xjtu", cache=False)
    tm["fit_seconds"]["ctgru"] = _median_time(lambda: ct.fit(fit_assets, seed, "full"), 1)
    ds = DSAFLitePredictor(cache=False)
    tm["fit_seconds"]["dsaflite"] = _median_time(lambda: ds.fit(fit_assets, seed, "full"), 1)

    lpm = lp.fit(fit_assets, seed, "full")["model"]
    ctm = CTGRUPredictor("xjtu").fit(fit_assets, seed, "full")
    dsm = DSAFLitePredictor().fit(fit_assets, seed, "full")["model"]

    def lp_infer():
        for a in cal:
            g = np.concatenate([features.minute_features(a.windows, a.per_min),
                                leadpoint.minute_env_features(a.windows, a.per_min)], 1)
            leadpoint.module_score(lpm, features.normalize(g, lpm.scale.numpy()))

    def ct_infer():
        for a in cal:
            X = features.normalize(features.minute_features(a.windows, a.per_min), ctm["scale"])
            ctgru_train.predict(ctm["model"], X, np.ones(a.n_min, np.float32))

    def ds_infer():
        for a in cal:
            dsaf_train.predict_minutes(dsm, a)

    tm["inference_seconds"] = {"leadpoint": _median_time(lp_infer, 3),
                               "ctgru": _median_time(ct_infer, 3),
                               "dsaflite": _median_time(ds_infer, 3), "n_bearings": len(cal)}
    tm["n_params"] = {"leadpoint": int(lpm.lin.weight.numel() + lpm.lin.bias.numel()),
                      "ctgru": int(sum(p.numel() for p in ctm["model"].parameters())),
                      "dsaflite": int(sum(p.numel() for p in dsm.parameters()))}
    tm["train_speedup_vs_dsaflite"] = tm["fit_seconds"]["dsaflite"] / tm["fit_seconds"]["leadpoint"]
    tm["inference_speedup_vs_dsaflite"] = (tm["inference_seconds"]["dsaflite"]
                                           / tm["inference_seconds"]["leadpoint"])
    TIMING_OUT.write_text(json.dumps(tm, indent=2))
    print(json.dumps(tm, indent=2))


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--smoke", action="store_true")
    g.add_argument("--run", action="store_true")
    g.add_argument("--timing", action="store_true")
    g.add_argument("--analyse", action="store_true")
    a = ap.parse_args()

    if a.analyse:
        return analyse()
    if a.timing:
        return timing()
    if a.smoke:
        rows = run("pronostia", LeadPointRidgePredictor(), [42], ("full",), 2, use_registry=False)
        crc = [r for r in rows if r["rule"] == "crc"]
        print(f"smoke: {len(rows)} rows, {len(crc)} crc rows, all certified "
              f"{all(r['certifiable'] for r in crc)}, rules {sorted({r['rule'] for r in rows})}")
        return
    rows = run("xjtu", LeadPointRidgePredictor(), list(SEEDS), ("full",), N_REP,
               use_registry=True, ckpt_tag="c1_xjtu_leadpoint_ridge")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=paths.ROOT,
                            capture_output=True, text=True).stdout.strip()
    OUT.write_text(json.dumps({"provenance": {
        "generated_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(
            timespec="seconds"),
        "git_commit": commit, "seeds": list(SEEDS), "n_rep": N_REP, "alpha": ALPHA,
        "lead": LEAD}, "rows": rows}, indent=2))
    report(rows)
    print(f"\nwritten to {OUT.relative_to(paths.ROOT)}")


if __name__ == "__main__":
    main()
