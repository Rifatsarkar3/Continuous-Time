"""The lead-point reduction under masked observation: per-replicate reduction checks and the pooled order statistic.

PRONOSTIA development calibration bearings, seed 42, all three observation processes, cached CT-GRU
inner models only; outer test bearings are never scored. Writes results/dev_reduction_masked.json.
"""
# Read-only check of the lead-point reduction on PRONOSTIA dev calibration trajectories (seed 42), all three
# observation processes, using cached CT-GRU models only. Outer test bearings are never scored.
# Guards: any attempt to write a model or feature cache raises instead of writing.
import sys, json, math
import numpy as np, torch
from pathlib import Path
REPO = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, REPO)
def _refuse(*a, **k): raise RuntimeError("refusing to write inside the repository")
np.save = _refuse; torch.save = _refuse
from certalarm import assets, certify, crossfit, evaluate
from certalarm.predictors import modelcache
modelcache._save = _refuse
from certalarm.decision import outcome_curves
from experiments.e2_main_alarms import ALPHA, LEAD, CTGRUPredictor

SEED, NREP = 42, 8
TH = np.asarray(evaluate.THETAS, float)
fleet = assets.load("pronostia"); by = {a.name: a for a in fleet}
pred = CTGRUPredictor("pronostia")
specs = crossfit.build_fold(fleet, seed=SEED)
e2 = json.load(open(REPO + "/results/e2_main_alarms_pron_ctgru.json"))["rows"]
e2th = {(r["process"], r["fold"]): r["theta"] for r in e2 if r["rule"] == "crc" and r["seed"] == SEED}

def tstar_masked(mu, rul, cf, lead):
    ok = np.flatnonzero(np.asarray(rul) >= lead)
    if ok.size == 0: return math.inf
    sel = np.asarray(cf, bool)[: ok[-1] + 1]
    return float(np.min(np.asarray(mu, float)[: ok[-1] + 1][sel])) if sel.any() else math.inf

def grid_first(pred_fn):
    for th in TH:
        if pred_fn(th): return float(th)
    return None

out = {}
for process in ("full", "mixture", "sparse_shift"):
    masks = crossfit._masks(fleet, process, NREP, SEED)
    checks = mism = 0; folds = []
    for fi, spec in enumerate(specs):
        own = set(spec["test"])
        Lrows, T = [], []          # per asset: mean late curve; per asset: list of replicate theta*
        for j, inner in enumerate(spec["inner"]):
            assert not set(inner["held"]) & own and not set(inner["fit"]) & own
            model = pred.fit([by[n] for n in inner["fit"]], seed=SEED * 100 + j, process=process)
            for name in inner["held"]:
                a = by[name]; rul = np.asarray(a.rul, float); lates, ts = [], []
                for m in masks[name]:
                    M = m["M"]; mu, _ = pred.predict(model, a, M); mu = np.asarray(mu, float)
                    late, _ = outcome_curves(mu, rul, TH, LEAD, M.astype(bool))
                    t = tstar_masked(mu, rul, M, LEAD)
                    mism += int((late != (TH <= t).astype(float)).sum()); checks += len(TH)
                    lates.append(late); ts.append(t)
                Lrows.append(np.mean(lates, 0)); T.append(ts)
        n = len(T); L = np.stack(Lrows); T = np.array(T)          # (n, R)
        crc = certify.crc_threshold(L, TH, ALPHA)
        pooled = grid_first(lambda th: (T >= th).sum() / NREP + 1 <= ALPHA * (n + 1) + 1e-12)
        m_rank = math.floor(round(NREP * (ALPHA * (n + 1) - 1), 12))
        conf_max = grid_first(lambda th: (T.max(1) >= th).sum() + 1 <= ALPHA * (n + 1) + 1e-12)
        conf_med = grid_first(lambda th: (np.median(T, 1) >= th).sum() + 1 <= ALPHA * (n + 1) + 1e-12)
        conf_rep0 = grid_first(lambda th: (T[:, 0] >= th).sum() + 1 <= ALPHA * (n + 1) + 1e-12)
        srt = np.sort(T.ravel())[::-1]
        top_owner = [int(np.argwhere(T == v)[0][0]) for v in srt[: m_rank + 1]]
        folds.append({"fold": fi, "n": n, "crc": crc, "e2_registered_crc": e2th[(process, fi)],
                      "pooled_order_stat": pooled, "pooled_rank_m_plus_1": m_rank + 1,
                      "split_conf_on_asset_max": conf_max, "split_conf_on_asset_median": conf_med,
                      "split_conf_on_replicate0": conf_rep0,
                      "max_tstar": float(T.max()), "median_tstar_assets": float(np.median(np.median(T, 1))),
                      "n_distinct_assets_in_top_m_plus_1": len(set(top_owner)),
                      "replicate_spread_max_minus_min_median": float(np.median(T.max(1) - T.min(1)))})
        print(process, folds[-1], flush=True)
    out[process] = {"R1_checks": checks, "R1_mismatches": mism, "folds": folds}
    print(process, "R1 checks", checks, "mismatches", mism, flush=True)
json.dump(out, open(REPO + "/results/dev_reduction_masked.json", "w"), indent=1)
