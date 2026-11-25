"""Gate 0 for the LeadPoint alarm. PRONOSTIA development data only.

PRE-DECLARED, 2026-09-23, before any LeadPoint candidate was trained or scored on bearing data.
Nothing below is tuned after results are seen. Bugs found by the smoke run may be fixed and are
listed in the results file; hyperparameters, candidates and criteria may not change.

QUESTION. Can a tiny scorer (a linear map or one hidden layer on minute-grid features), trained
with a loss derived from the lead-point reduction, match or beat the heavy DSAFLite predictor on
certified wasted life, while training at least 100x faster?

CANDIDATES - five, fixed now. Features are the existing 18 minute-grid features plus nine
envelope-spectrum features (leadpoint.envelope_band_features), baseline-corrected per bearing and
scaled on the inner-fit bearings only, exactly as CT-GRU's are.

    lp_ridge       linear ridge regression on normalized RUL (lambda 1.0)   - tiny model, old loss
    lp_rank        linear scorer, cross-asset lead-point ranking loss       - the mechanism
    lp_rank_mlp    one hidden layer (16 tanh), same loss                    - capacity
    lp_rank_ewma   lp_rank on causal EWMA-smoothed features (half-life 5 min) - smoothing
    lp_rank_noenv  lp_rank without the nine envelope features               - feature ablation

    ranking loss: leadpoint.rank_loss; Adam, full batch, 300 iterations, lr 0.05, L2 1e-2 on
    weights; seed = 42*100 + inner split, as for every other arm.

The set is a small factorial: lp_ridge vs lp_rank isolates the loss, lp_rank vs lp_rank_noenv the
envelope features, lp_rank vs lp_rank_mlp the capacity.

REFERENCE ARMS. CT-GRU (original loss) and DSAFLite, loaded from the model cache: the same folds,
fit sets and seeds as E2 and the predictor search, so no reference model is retrained.

PROTOCOL. Seed-42 nested folds (5 outer x 4 inner). Every calibration bearing is predicted by the
inner model that did not train on it. The outer test bearings of each fold are asserted absent
from every fit and every scored trajectory. The evaluation registry is not touched.

METRIC. Leave-one-bearing-out certified waste over each outer fold's calibration bearings: for
bearing i, certify on the others and score i. Certification is exact for every arm, candidates
and references alike: `leadpoint.exact_threshold` on the lead-point values, which by the verified
reduction is the CRC threshold without grid rounding. For the two references the grid-CRC LOO
numbers are also computed, and CT-GRU's must reproduce results/dev_predictor_search.json
(loo_waste 0.7713) or the run stops - an integrity check that the harness is the same.

SELECTION RULE. Winner = the candidate with the lowest exact-LOO waste whose LOO late rate is at
or below alpha = 0.10. If none qualifies, the gate fails.

GATE PASSES iff both hold:
  (W) winner's exact-LOO waste <= DSAFLite's exact-LOO waste
  (S) DSAFLite training time / winner training time >= 100

TIMING PROTOCOL, fixed now. Same machine; the device each arm uses is recorded (DSAFLite on the
GPU, the others on the CPU - which can only favour DSAFLite).
  training   fold 0, inner split 0: wall-clock of one fit from stored inputs (minute-grid
             features for LeadPoint and CT-GRU, raw windows for DSAFLite). LeadPoint: median of
             5 fits. CT-GRU and DSAFLite: one fresh fit each, cache disabled, weights discarded.
             Criterion (S) reads this number.
  end-to-end training for the winner, including feature extraction from raw windows: reported,
             no criterion.
  inference  fold 0's calibration bearings, raw windows -> minute signal, median of 3 passes.
             Reported, no criterion here (H1 of the confirmatory stage carries one).

FORECAST, recorded so it can be checked: (S) passes by a wide margin. (W) is about even odds. If
(W) fails, the expected cause is the binding bearing - Bearing2_7 in 4 of 5 dev folds - whose
lead-point value no loss setting has moved so far.

NEXT. Pass: pre-register the XJTU-SY confirmation (H1 speed, H2 non-inferiority, H3
superiority, fixed sequence) as its own commit before any XJTU bearing is scored. Fail: report,
stop; any follow-up is a disclosed second attempt, never a silent retry.

    python experiments/dev_leadpoint_gate0.py --smoke    # fold 0, short training, path check
    python experiments/dev_leadpoint_gate0.py            # the gate, run once
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from certalarm import assets, crossfit, evaluate, features, leadpoint, paths  # noqa: E402
from certalarm.decision import alarm_outcome, outcome_curves  # noqa: E402
from certalarm.predictors import ctgru_train, dsaf_train  # noqa: E402
from experiments.dev_predictor_search import loo_crc  # noqa: E402
from experiments.e2_main_alarms import ALPHA, LEAD, CTGRUPredictor, DSAFLitePredictor  # noqa: E402

SEED = 42
RIDGE_LAM = 1.0
RANK = {"iters": 300, "lr": 0.05, "weight_decay": 1e-2}
CANDIDATES = {
    "lp_ridge": {"loss": "ridge", "hidden": 0, "env": True, "ewma": None},
    "lp_rank": {"loss": "rank", "hidden": 0, "env": True, "ewma": None},
    "lp_rank_mlp": {"loss": "rank", "hidden": 16, "env": True, "ewma": None},
    "lp_rank_ewma": {"loss": "rank", "hidden": 0, "env": True, "ewma": 5.0},
    "lp_rank_noenv": {"loss": "rank", "hidden": 0, "env": False, "ewma": None},
}
PREDICTOR_SEARCH_CTGRU_LOO_WASTE = 0.7713372427422334   # results/dev_predictor_search.json
ENV_CACHE = paths.DATA / "features_env"
SPEED_CRITERION = 100.0


class LeadPointPredictor:
    def __init__(self, cfg, rank_iters):
        self.cfg, self.rank_iters = cfg, rank_iters

    @staticmethod
    def raw(a):
        f18 = features.asset_grid(a.dataset, a.name, a.windows, a.per_min,
                                  cache_dir=paths.DATA / "features")
        env = leadpoint.env_grid(a.dataset, a.name, a.windows, a.per_min, ENV_CACHE)
        assert len(f18) == len(env) == a.n_min, a.name
        return f18, env

    def _grid(self, a):
        f18, env = self.raw(a)
        return np.concatenate([f18, env], 1) if self.cfg["env"] else f18

    def _prep(self, g, scale):
        x = features.normalize(g, scale)
        return leadpoint.ewma(x, self.cfg["ewma"]) if self.cfg["ewma"] else x

    def fit(self, fit_assets, seed):
        G = [self._grid(a) for a in fit_assets]
        scale = features.fit_scale(G)
        X = [self._prep(g, scale) for g in G]
        r = [a.rul for a in fit_assets]
        if self.cfg["loss"] == "ridge":
            m = leadpoint.fit_ridge(X, r, lam=RIDGE_LAM)
        else:
            m = leadpoint.fit_rank(X, r, LEAD, hidden=self.cfg["hidden"], iters=self.rank_iters,
                                   lr=RANK["lr"], weight_decay=RANK["weight_decay"], seed=seed)
        return {"model": m, "scale": scale}

    def predict(self, model, a):
        return leadpoint.score(model["model"], self._prep(self._grid(a), model["scale"]))


def arms(rank_iters):
    out = {n: (lambda p: (lambda fa, s: p.fit(fa, s), p.predict))(LeadPointPredictor(c, rank_iters))
           for n, c in CANDIDATES.items()}
    ct = CTGRUPredictor("pronostia")
    ds = DSAFLitePredictor()
    out["ref_ctgru"] = (lambda fa, s: ct.fit(fa, s, "full"),
                        lambda m, a: ct.predict(m, a, np.ones(a.n_min, bool))[0])
    out["ref_dsaflite"] = (lambda fa, s: ds.fit(fa, s, "full"),
                           lambda m, a: ds.predict(m, a, None)[0])
    return out


def collect(fit, predict, by, specs):
    """Per outer fold, the calibration trajectories, each from the inner model blind to it."""
    folds_out = []
    for spec in specs:
        own_test = set(spec["test"])
        rows = []
        for j, inner in enumerate(spec["inner"]):
            assert not set(inner["held"]) & own_test, "an outer test bearing reached scoring"
            assert not set(inner["fit"]) & own_test, "an outer test bearing reached training"
            model = fit([by[n] for n in inner["fit"]], SEED * 100 + j)
            for n in inner["held"]:
                rows.append({"name": n, "s": np.asarray(predict(model, by[n]), float),
                             "rul": by[n].rul.astype(float)})
        folds_out.append(rows)
    return folds_out


def loo_exact(rows):
    ts = np.array([leadpoint.lead_point(r["s"], r["rul"], LEAD) for r in rows])
    out = []
    for i, r in enumerate(rows):
        th = leadpoint.exact_threshold(np.delete(ts, i), ALPHA)
        if th is None:
            out.append(None)
            continue
        late, rem = alarm_outcome(r["s"], r["rul"], th, LEAD)
        assert bool(late) == bool(th <= ts[i]), "reduction violated in LOO scoring"
        out.append((float(late), float(rem)))
    return ts, out


def summarise(folds_rows, grid=False):
    late, waste, declined, per_bearing, binding = [], [], 0, {}, []
    g_late, g_waste = [], []
    for fi, rows in enumerate(folds_rows):
        ts, res = loo_exact(rows)
        binding.append(rows[int(np.argmax(ts))]["name"])
        for r, o in zip(rows, res):
            if o is None:
                declined += 1
                continue
            late.append(o[0])
            waste.append(o[1])
            per_bearing[f"{fi}/{r['name']}"] = o[1]
        if grid:
            L, W = zip(*[outcome_curves(r["s"], r["rul"], evaluate.THETAS, LEAD,
                                        np.ones(len(r["s"]), bool)) for r in rows])
            la, wa, _ = loo_crc(np.stack(L), np.stack(W))
            g_late += la
            g_waste += wa
    out = {"loo_late": float(np.mean(late)), "loo_waste": float(np.mean(waste)),
           "declined": declined, "n": len(late), "binding_bearing_per_fold": binding,
           "per_bearing_waste": per_bearing}
    if grid:
        out["grid_loo_late"] = float(np.mean(g_late))
        out["grid_loo_waste"] = float(np.mean(g_waste))
    return out


def median_time(fn, reps):
    t = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        t.append(time.perf_counter() - t0)
    return float(np.median(t))


def timing(by, spec, winner, rank_iters, smoke):
    fit_assets = [by[n] for n in spec["inner"][0]["fit"]]
    cal = [by[n] for n in spec["calibration"]]
    seed = SEED * 100
    tm = {"device": {"leadpoint": "cpu", "ctgru": str(ctgru_train.DEVICE),
                     "dsaflite": str(dsaf_train.DEVICE)}}

    fit_s = {}
    for n, c in CANDIDATES.items():
        p = LeadPointPredictor(c, rank_iters)
        fit_s[n] = median_time(lambda: p.fit(fit_assets, seed), 5)
    tm["fit_seconds"] = fit_s
    ct = CTGRUPredictor("pronostia", cache=False, iters=5 if smoke else None)
    tm["ctgru_fit_seconds"] = median_time(lambda: ct.fit(fit_assets, seed, "full"), 1)
    ds = DSAFLitePredictor(cache=False, epochs=1 if smoke else None)
    tm["dsaflite_fit_seconds"] = median_time(lambda: ds.fit(fit_assets, seed, "full"), 1)

    wp = LeadPointPredictor(CANDIDATES[winner], rank_iters)

    def end_to_end_fit():
        G = []
        for a in fit_assets:
            f = features.minute_features(a.windows, a.per_min)
            G.append(np.concatenate([f, leadpoint.minute_env_features(a.windows, a.per_min)], 1)
                     if wp.cfg["env"] else f)
        scale = features.fit_scale(G)
        X = [wp._prep(g, scale) for g in G]
        if wp.cfg["loss"] == "ridge":
            leadpoint.fit_ridge(X, [a.rul for a in fit_assets], lam=RIDGE_LAM)
        else:
            leadpoint.fit_rank(X, [a.rul for a in fit_assets], LEAD, hidden=wp.cfg["hidden"],
                               iters=rank_iters, lr=RANK["lr"],
                               weight_decay=RANK["weight_decay"], seed=seed)
    tm["winner_end_to_end_fit_seconds"] = median_time(end_to_end_fit, 1)

    wm = wp.fit(fit_assets, seed)
    ctm = CTGRUPredictor("pronostia").fit(fit_assets, seed, "full")
    dsm = DSAFLitePredictor().fit(fit_assets, seed, "full")

    def lp_infer():
        for a in cal:
            f = features.minute_features(a.windows, a.per_min)
            g = np.concatenate([f, leadpoint.minute_env_features(a.windows, a.per_min)], 1) \
                if wp.cfg["env"] else f
            leadpoint.score(wm["model"], wp._prep(g, wm["scale"]))

    def ct_infer():
        for a in cal:
            X = features.normalize(features.minute_features(a.windows, a.per_min), ctm["scale"])
            ctgru_train.predict(ctm["model"], X, np.ones(a.n_min, np.float32))

    def ds_infer():
        for a in cal:
            dsaf_train.predict_minutes(dsm["model"], a)

    tm["inference_seconds"] = {"winner": median_time(lp_infer, 3), "ctgru": median_time(ct_infer, 3),
                               "dsaflite": median_time(ds_infer, 3), "n_bearings": len(cal)}
    tm["n_params"] = {"winner": leadpoint.n_params(wm["model"]),
                      "ctgru": int(sum(p.numel() for p in ctm["model"].parameters())),
                      "dsaflite": int(sum(p.numel() for p in dsm["model"].parameters()))}
    tm["train_speedup_vs_dsaflite"] = tm["dsaflite_fit_seconds"] / fit_s[winner]
    return tm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    rank_iters = 20 if a.smoke else RANK["iters"]

    fleet = assets.load("pronostia")
    by = {x.name: x for x in fleet}
    specs = crossfit.build_fold(fleet, seed=SEED)
    if a.smoke:
        specs = specs[:1]
    for x in fleet:                      # build the envelope cache once, up front
        LeadPointPredictor.raw(x)

    results = {}
    for name, (fit, predict) in arms(rank_iters).items():
        t0 = time.perf_counter()
        results[name] = summarise(collect(fit, predict, by, specs), grid=name.startswith("ref_"))
        r = results[name]
        print(f"{name:<15} LOO late {r['loo_late']:.3f}  LOO waste {r['loo_waste']:.3f}  "
              f"declined {r['declined']}  n {r['n']}  binding {r['binding_bearing_per_fold']}  "
              f"({time.perf_counter() - t0:.0f}s)", flush=True)

    ref_ct = results["ref_ctgru"]
    print(f"\nintegrity: CT-GRU grid LOO waste {ref_ct['grid_loo_waste']:.4f} "
          f"(predictor search {PREDICTOR_SEARCH_CTGRU_LOO_WASTE:.4f})")
    if not a.smoke and abs(ref_ct["grid_loo_waste"] - PREDICTOR_SEARCH_CTGRU_LOO_WASTE) > 1e-9:
        sys.exit("INTEGRITY FAILED: the CT-GRU reference does not reproduce the predictor search")

    ok = {n: results[n] for n in CANDIDATES if results[n]["loo_late"] <= ALPHA + 1e-12}
    winner = min(ok, key=lambda n: ok[n]["loo_waste"]) if ok else None
    tm = timing(by, specs[0], winner or "lp_rank", rank_iters, a.smoke)
    ds_w = results["ref_dsaflite"]["loo_waste"]
    crit_w = winner is not None and results[winner]["loo_waste"] <= ds_w
    crit_s = tm["train_speedup_vs_dsaflite"] >= SPEED_CRITERION

    paired = None
    if winner:
        wb, db = results[winner]["per_bearing_waste"], results["ref_dsaflite"]["per_bearing_waste"]
        d = np.array([wb[k] - db[k] for k in sorted(set(wb) & set(db))])
        paired = {"n": int(d.size), "winner_lower": int((d < -1e-12).sum()),
                  "winner_higher": int((d > 1e-12).sum()), "mean_diff": float(d.mean())}

    print(f"\n{'arm':<15}{'LOO late':>10}{'LOO waste':>11}")
    for n, r in results.items():
        print(f"{n:<15}{r['loo_late']:>10.3f}{r['loo_waste']:>11.3f}")
    print(f"\nqualifying (late <= {ALPHA}): {sorted(ok)}   WINNER: {winner}")
    print(f"fit seconds: {json.dumps({k: round(v, 3) for k, v in tm['fit_seconds'].items()})}")
    print(f"CT-GRU fit {tm['ctgru_fit_seconds']:.1f}s   DSAFLite fit {tm['dsaflite_fit_seconds']:.1f}s"
          f"   ({tm['device']})")
    print(f"inference, {tm['inference_seconds']['n_bearings']} bearings: "
          f"{json.dumps({k: round(v, 2) for k, v in tm['inference_seconds'].items()})}")
    print(f"params: {tm['n_params']}")
    print(f"paired vs DSAFLite, per bearing: {paired}")
    print(f"\n(W) winner waste <= DSAFLite waste {ds_w:.3f}: {'PASS' if crit_w else 'FAIL'}")
    print(f"(S) training speedup {tm['train_speedup_vs_dsaflite']:.0f}x >= {SPEED_CRITERION:.0f}x: "
          f"{'PASS' if crit_s else 'FAIL'}")
    verdict = "PASS" if crit_w and crit_s else "FAIL"
    print(f"GATE 0: {verdict}{'   [SMOKE - not the gate]' if a.smoke else ''}")

    out = paths.RESULTS / f"dev_leadpoint_gate0{'_smoke' if a.smoke else ''}.json"
    out.write_text(json.dumps({"pre_declared": True, "smoke": a.smoke, "seed": SEED,
                               "alpha": ALPHA, "lead": LEAD, "candidates": CANDIDATES,
                               "rank": {**RANK, "iters": rank_iters}, "ridge_lambda": RIDGE_LAM,
                               "results": results, "winner": winner, "timing": tm,
                               "paired_vs_dsaflite": paired,
                               "criteria": {"W": crit_w, "S": crit_s}, "verdict": verdict,
                               "bug_fixes_after_smoke": []}, indent=2))
    print(f"written to {out.relative_to(paths.ROOT)}")


if __name__ == "__main__":
    main()
