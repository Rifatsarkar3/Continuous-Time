"""Predictor search for the alarm task. PRONOSTIA development data only.

PRE-DECLARED, 2026-09-21, before any candidate below was trained or scored.

Why. The diagnosis (experiments/dev_predictor_diagnosis.py) found that both predictors
over-estimate remaining life near failure, that late-life error ranks predictors correctly where
overall MAE ranks them backwards, and that CRC's threshold is set by the single most optimistic
calibration bearing. The training loss - uniform, symmetric MSE - produces exactly that.

Candidates. A 2x2 factorial on the two components the diagnosis implicates, plus one stronger
setting. Five in total, fixed now; nothing is added after the results are seen.

    baseline   late_weight 0  asym 1    the original loss
    late       late_weight 4  asym 1    weight toward failure only
    asym       late_weight 0  asym 4    penalise optimism only
    both       late_weight 4  asym 4    both components
    strong     late_weight 9  asym 9    both, harder

The factorial is deliberate: it says *which* component matters, not just which setting won.

Selection rule, fixed now. The development metric is leave-one-bearing-out CRC on the inner
calibration bearings of each outer fold: for each calibration bearing, fit theta on the others and
score it. That is the quantity this study cares about, estimated without the optimism of scoring
theta on the bearings it was fitted to. The winner is the candidate with the **lowest LOO waste
whose LOO late rate is at or below alpha**. If none qualifies, the baseline is kept.

Protocol. Outer test bearings are never read: each outer fold's own test set is asserted absent
from every trajectory used here. PRONOSTIA is development-only for this search because its test
results have already been seen. The winner gets exactly one confirmatory evaluation, on XJTU,
alongside the existing arms, in a single pass.

ATTEMPT 2, pre-declared 2026-09-21 AFTER attempt 1 was run and reported. This is disclosed as a
second attempt, not presented as the original plan.

Attempt 1 produced a poor result (-10% waste) for a diagnosable reason: its asymmetry term
applied at every minute, so penalising optimism near failure dragged the whole trajectory down.
`strong` ended up under-reading remaining life by 0.41 in early life, and since the alarm fires at
the first crossing of a running minimum, a predictor biased low everywhere crosses the threshold
far too early. Observed waste 0.694 is close to theta + 0.41, which is that effect arithmetically.

Attempt 2 gates the asymmetry to late life, as the weight already was. Three candidates, fixed
now; the selection rule is unchanged. Because the penalty is now localised it can be sharper, so
one candidate pushes it further than attempt 1 did.

    both_g     late_weight 4  asym 4   gated
    strong_g   late_weight 9  asym 9   gated
    sharp_g    late_weight 9  asym 20  gated

    python experiments/dev_predictor_search.py            # attempt 1
    python experiments/dev_predictor_search.py --attempt 2
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from certalarm import assets, certify, crossfit, evaluate, paths, pooling  # noqa: E402
from certalarm.decision import outcome_curves  # noqa: E402
from experiments.e2_main_alarms import ALPHA, LEAD, CTGRUPredictor  # noqa: E402

CANDIDATES = {
    "baseline": (0.0, 1.0),
    "late": (4.0, 1.0),
    "asym": (0.0, 4.0),
    "both": (4.0, 4.0),
    "strong": (9.0, 9.0),
}
CANDIDATES_2 = {
    "baseline": (0.0, 1.0, False),
    "both_g": (4.0, 4.0, True),
    "strong_g": (9.0, 9.0, True),
    "sharp_g": (9.0, 20.0, True),
}
LATE_LIFE = 0.2
SEED = 42


class _T:
    def __init__(self, mu, rul):
        self.mu, self.rul = np.asarray(mu, float), np.asarray(rul, float)


def loo_crc(L, W):
    """Leave-one-bearing-out CRC: fit theta without bearing i, score bearing i with it."""
    late, waste, declined = [], [], 0
    for i in range(len(L)):
        keep = np.arange(len(L)) != i
        th = certify.crc_threshold(L[keep], evaluate.THETAS, ALPHA)
        if th is None:
            declined += 1
            continue
        k = int(np.searchsorted(evaluate.THETAS, th))
        late.append(L[i, k])
        waste.append(W[i, k])
    return late, waste, declined


def score(predictor, fleet):
    by = {a.name: a for a in fleet}
    diag, late_all, waste_all, declined = [], [], [], 0
    for spec in crossfit.build_fold(fleet, seed=SEED):
        own_test = set(spec["test"])
        L, W = [], []
        for j, inner in enumerate(spec["inner"]):
            assert not set(inner["held"]) & own_test, "an outer test bearing reached the search"
            assert not set(inner["fit"]) & own_test, "an outer test bearing reached training"
            model = predictor.fit([by[n] for n in inner["fit"]], seed=SEED * 100 + j,
                                  process="full")
            for n in inner["held"]:
                a = by[n]
                mu, _ = predictor.predict(model, a, np.ones(a.n_min, dtype=bool))
                mu, rul = np.asarray(mu, float), np.asarray(a.rul, float)
                ll = rul <= LATE_LIFE
                diag.append({"mae": float(np.abs(mu - rul).mean()),
                             "ll_mae": float(np.abs(mu - rul)[ll].mean()),
                             "ll_bias": float((mu - rul)[ll].mean()),
                             "mu_at_lead": pooling.scale_at_lead(_T(mu, rul), LEAD)})
                l, w = outcome_curves(mu, rul, evaluate.THETAS, LEAD, np.ones(a.n_min, bool))
                L.append(l)
                W.append(w)
        la, wa, d = loo_crc(np.stack(L), np.stack(W))
        late_all += la
        waste_all += wa
        declined += d
    m = lambda k: float(np.mean([r[k] for r in diag]))
    return {"mae": m("mae"), "ll_mae": m("ll_mae"), "ll_bias": m("ll_bias"),
            "mal_max": float(np.max([r["mu_at_lead"] for r in diag])),
            "mal_median": float(np.median([r["mu_at_lead"] for r in diag])),
            "loo_late": float(np.mean(late_all)), "loo_waste": float(np.mean(waste_all)),
            "declined": declined, "n": len(late_all)}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--attempt", type=int, default=1, choices=(1, 2))
    a = ap.parse_args()
    cands = CANDIDATES if a.attempt == 1 else CANDIDATES_2

    fleet = assets.load("pronostia")
    results = {}
    for name, cfg in cands.items():
        lw, asym, gate = (cfg + (False,))[:3] if len(cfg) == 2 else cfg
        pred = CTGRUPredictor("pronostia", late_weight=lw, asym=asym, gate_asym=gate,
                              name="ctgru" if name == "baseline" else f"ctgru_{name}")
        print(f"--- {name}: late_weight {lw}, asym {asym}, gated {gate} ...", flush=True)
        results[name] = score(pred, fleet)
        r = results[name]
        print(f"    LL-MAE {r['ll_mae']:.4f}  LL-bias {r['ll_bias']:+.4f}  "
              f"LOO late {r['loo_late']:.3f}  LOO waste {r['loo_waste']:.3f}", flush=True)

    print(f"\n{'candidate':<10}{'MAE':>8}{'LL-MAE':>9}{'LL-bias':>10}{'mu@lead med':>13}"
          f"{'mu@lead max':>13}{'LOO late':>10}{'LOO waste':>11}")
    for name, r in results.items():
        print(f"{name:<10}{r['mae']:>8.4f}{r['ll_mae']:>9.4f}{r['ll_bias']:>+10.4f}"
              f"{r['mal_median']:>13.3f}{r['mal_max']:>13.3f}{r['loo_late']:>10.3f}"
              f"{r['loo_waste']:>11.3f}")

    ok = {n: r for n, r in results.items() if r["loo_late"] <= ALPHA}
    winner = min(ok, key=lambda n: ok[n]["loo_waste"]) if ok else "baseline"
    base = results["baseline"]["loo_waste"]
    print(f"\npre-declared rule: lowest LOO waste with LOO late <= {ALPHA}")
    print(f"qualifying: {sorted(ok)}")
    print(f"WINNER: {winner}   LOO waste {results[winner]['loo_waste']:.3f} "
          f"vs baseline {base:.3f} ({(results[winner]['loo_waste'] - base) / base:+.1%})")

    (paths.RESULTS / f"dev_predictor_search{'' if a.attempt == 1 else '_2'}.json").write_text(json.dumps(
        {"pre_declared": True, "attempt": a.attempt,
         "disclosure": ("attempt 2 was declared after attempt 1 was run and reported; it fixes a "
                        "diagnosed flaw (ungated asymmetry), it is not the original plan")
                       if a.attempt == 2 else None,
         "seed": SEED, "alpha": ALPHA, "candidates": {k: list(v) for k, v in cands.items()},
         "rule": "lowest LOO waste with LOO late <= alpha; baseline if none qualifies",
         "results": results, "winner": winner}, indent=2))
    print("written to results/dev_predictor_search.json")


if __name__ == "__main__":
    main()
