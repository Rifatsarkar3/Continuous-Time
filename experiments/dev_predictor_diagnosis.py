"""What property of a predictor actually drives CRC's wasted life? Development data only.

MAE mis-ranked our predictors: DSAFLite has the worse MAE (0.199 vs 0.176) yet the better alarms
(point-rule lateness 0.348 vs 0.604). So average error is not what the alarm consumes, and a
strong predictor cannot be chosen by it. This script tests a mechanistic alternative.

The chain: a bearing is late if the predictor still reads high when true remaining life reaches
`lead`. CRC must place theta high enough to catch the bearings that read highest at that moment,
and every other bearing then alarms early - that is the waste. So theta should sit near the upper
tail of `mu @ lead` across calibration bearings, and a predictor is good *for alarms* exactly when
that reading is low and tightly concentrated, whatever its average error.

Everything here is computed on inner-fold HELD-OUT bearings - calibration data, never an outer
test bearing - using models already in the cache, so it trains nothing and spends nothing.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from certalarm import assets, certify, crossfit, evaluate, pooling  # noqa: E402
from certalarm.decision import outcome_curves  # noqa: E402
from experiments.e2_main_alarms import ALPHA, LEAD, CTGRUPredictor, DSAFLitePredictor  # noqa: E402

LATE_LIFE = 0.2      # "late life" = the last 20% of true remaining life


class _T:
    def __init__(self, mu, rul):
        self.mu, self.rul = np.asarray(mu, float), np.asarray(rul, float)


def diagnose(predictor, fleet, seed=42):
    by = {a.name: a for a in fleet}
    rows, fold_rows = [], []
    for i, spec in enumerate(crossfit.build_fold(fleet, seed=seed)):
        own_test = set(spec["test"])
        L, mu_at_lead = [], []
        for j, inner in enumerate(spec["inner"]):
            assert not set(inner["held"]) & own_test, "a test bearing reached the diagnosis"
            model = predictor.fit([by[n] for n in inner["fit"]], seed=seed * 100 + j,
                                  process="full")
            for n in inner["held"]:
                a = by[n]
                mu, _ = predictor.predict(model, a, np.ones(a.n_min, dtype=bool))
                mu, rul = np.asarray(mu, float), np.asarray(a.rul, float)
                late_mask = rul <= LATE_LIFE
                m_lead = pooling.scale_at_lead(_T(mu, rul), LEAD)
                rows.append({"asset": n, "mae": float(np.abs(mu - rul).mean()),
                             "ll_mae": float(np.abs(mu - rul)[late_mask].mean()),
                             "ll_bias": float((mu - rul)[late_mask].mean()),
                             "mu_at_lead": m_lead})
                l, _ = outcome_curves(mu, rul, evaluate.THETAS, LEAD, np.ones(a.n_min, bool))
                L.append(l)
                mu_at_lead.append(m_lead)
        theta = certify.crc_threshold(np.stack(L), evaluate.THETAS, ALPHA)
        fold_rows.append({"fold": i, "theta": theta, "n": len(mu_at_lead),
                          "q90": float(np.quantile(mu_at_lead, 0.90)),
                          "max": float(np.max(mu_at_lead))})
    return rows, fold_rows


def main():
    fleet = assets.load("pronostia")
    arms = {"ctgru": CTGRUPredictor("pronostia"), "dsaflite": DSAFLitePredictor()}
    summary = {}
    for name, pred in arms.items():
        rows, folds = diagnose(pred, fleet)
        m = lambda k: float(np.mean([r[k] for r in rows]))
        mal = np.array([r["mu_at_lead"] for r in rows])
        summary[name] = {"mae": m("mae"), "ll_mae": m("ll_mae"), "ll_bias": m("ll_bias"),
                         "mal_median": float(np.median(mal)),
                         "mal_q90": float(np.quantile(mal, 0.9)), "folds": folds}
        print(f"\n=== {name}: {len(rows)} calibration-bearing trajectories ===")
        print(f"{'fold':>5}{'n_cal':>7}{'CRC theta':>11}{'q90 mu@lead':>13}{'max mu@lead':>13}")
        for f in folds:
            th = f"{f['theta']:.3f}" if f["theta"] is not None else "declined"
            print(f"{f['fold']:>5}{f['n']:>7}{th:>11}{f['q90']:>13.3f}{f['max']:>13.3f}")

    print("\n=== which predictor property tracks the alarm? ===")
    print(f"{'arm':<10}{'MAE':>8}{'late-life MAE':>15}{'late-life bias':>16}"
          f"{'mu@lead median':>16}{'mu@lead q90':>13}")
    for name, s in summary.items():
        print(f"{name:<10}{s['mae']:>8.4f}{s['ll_mae']:>15.4f}{s['ll_bias']:>+16.4f}"
              f"{s['mal_median']:>16.3f}{s['mal_q90']:>13.3f}")

    th = [f["theta"] for s in summary.values() for f in s["folds"] if f["theta"] is not None]
    q9 = [f["q90"] for s in summary.values() for f in s["folds"] if f["theta"] is not None]
    mx = [f["max"] for s in summary.values() for f in s["folds"] if f["theta"] is not None]
    if len(th) > 2:
        print(f"\ncorrelation of CRC theta with q90(mu@lead) across {len(th)} folds: "
              f"{np.corrcoef(th, q9)[0, 1]:+.3f}")
        print(f"correlation of CRC theta with max(mu@lead):                  "
              f"{np.corrcoef(th, mx)[0, 1]:+.3f}")


if __name__ == "__main__":
    main()
