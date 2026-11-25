"""Development pre-check for condition-scaled CRC. Training assets only - no test bearing read.

This is prediction (2) of the pre-registration: if the spread of the scale readings
*within* a condition is not smaller than the pooled spread, the condition grouping explains none
of the heterogeneity, the construction cannot help, and no confirmatory evaluation should be
spent on it.

Everything here runs on the `fit` assets of the inner splits, which are disjoint from both the
calibration and the test bearings. The test set is asserted untouched rather than merely avoided,
because "I did not mean to read it" is not a protocol.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from certalarm import assets, crossfit, pooling  # noqa: E402
from experiments.e2_main_alarms import ALPHA, CTGRUPredictor, LEAD  # noqa: E402


class _Traj:
    def __init__(self, mu, rul):
        self.mu, self.rul = np.asarray(mu, float), np.asarray(rul, float)


def forecast(fleet, by_name, predictor, specs, seed):
    """Development-side estimate of the waste change, on calibration assets only.

    This is NOT a result. Theta is fitted on these same assets, so the absolute waste is
    optimistic for both arms. Only the difference between them is read, and only to decide
    whether spending a confirmatory dataset's single evaluation is worth it. Reporting this
    number as an outcome would be exactly the self-evaluation the protocol exists to prevent.
    """
    from certalarm import certify, evaluate

    rows = []
    for i, spec in enumerate(specs):
        # Calibration is pooled across the four inner splits, as CV-CRC requires: each held-out
        # asset contributes its own outcome, scored by the model that did not see it. Running CRC
        # per inner split instead would face a floor of 1/(3+1) = 0.25 and certify nothing.
        plain_l, plain_w, scaled_l, scaled_w = [], [], [], []
        for j, inner in enumerate(spec["inner"]):
            fit_names = list(inner["fit"])
            model = predictor.fit([by_name[n] for n in fit_names], seed=seed * 100 + j,
                                  process="full")
            fit = {}
            for n in fit_names:
                a = by_name[n]
                mu, _ = predictor.predict(model, a, np.ones(a.n_min, dtype=bool))
                fit[n] = {"condition": a.condition, "reps": [_Traj(mu, a.rul)]}
            scales = pooling.condition_scales(fit, LEAD, fit_names=fit_names)

            for n in inner["held"]:
                a = by_name[n]
                mu, sg = predictor.predict(model, a, np.ones(a.n_min, dtype=bool))
                t = _Traj(mu, a.rul)
                t.sigma, t.can_fire = sg, np.ones(a.n_min, bool)
                l, w = evaluate._asset_curves([t], evaluate.THETAS, LEAD, lambda x: x.mu)
                plain_l.append(l), plain_w.append(w)
                sig = pooling.signal_for(scales, a.condition)
                l2, w2 = evaluate._asset_curves([t], evaluate.THETAS, LEAD, sig)
                scaled_l.append(l2), scaled_w.append(w2)

        for name, L, W in (("pooled", plain_l, plain_w), ("scaled", scaled_l, scaled_w)):
            th = certify.crc_threshold(np.stack(L), evaluate.THETAS, ALPHA)
            if th is None:
                rows.append((name, None, None))
                continue
            k = int(np.searchsorted(evaluate.THETAS, th))
            rows.append((name, float(np.stack(L)[:, k].mean()),
                         float(np.stack(W)[:, k].mean())))

    for name in ("pooled", "scaled"):
        sel = [r for r in rows if r[0] == name and r[1] is not None]
        if sel:
            print(f"  [dev] {name:<7} calibration-side late {np.mean([r[1] for r in sel]):.3f} "
                  f"waste {np.mean([r[2] for r in sel]):.3f}   ({len(sel)} inner splits)")


def main(dataset="pronostia", seed=42):
    fleet = assets.load(dataset)
    by_name = {a.name: a for a in fleet}
    predictor = CTGRUPredictor(dataset)
    specs = crossfit.build_fold(fleet, seed=seed)

    touched, pooled_all, within_all = set(), [], []
    for i, spec in enumerate(specs):
        # Per fold, not pooled across folds: in nested cross-fitting every bearing is a test
        # asset in exactly one outer fold and a training asset in the others, so the union of
        # all test sets is the whole fleet and comparing against it proves nothing.
        own_test = set(spec["test"])
        for j, inner in enumerate(spec["inner"]):
            fit_names = list(inner["fit"])
            assert not (set(fit_names) & own_test), \
                f"outer fold {i} inner {j}: a bearing from this fold's test set is in its fit set"
            touched.update(fit_names)
            # Cached from stage 1: this is a disk read, not a training run.
            model = predictor.fit([by_name[n] for n in fit_names], seed=seed * 100 + j,
                                  process="full")
            fit = {}
            for n in fit_names:
                a = by_name[n]
                mu, _ = predictor.predict(model, a, np.ones(a.n_min, dtype=bool))
                fit[n] = {"condition": a.condition, "reps": [_Traj(mu, a.rul)]}
            p, w = pooling.spread_within_and_pooled(fit, LEAD)
            pooled_all.append(p)
            within_all.append(w)
            scales = pooling.condition_scales(fit, LEAD, fit_names=fit_names)
            if i == 0 and j == 0:
                print("example scales (outer 0, inner 0):",
                      {k: round(v, 3) for k, v in sorted(scales.items())})

    print(f"\ntraining assets read: {len(touched)} of {len(fleet)}; "
          f"every fold's own test bearings stayed out of its own fit sets")

    forecast(fleet, by_name, predictor, specs, seed)

    p, w = float(np.mean(pooled_all)), float(np.mean(within_all))
    print(f"\npooled spread of scale readings   {p:.4f}")
    print(f"within-condition spread           {w:.4f}")
    print(f"variance explained by condition   {1 - (w / p) ** 2:+.1%}")
    verdict = "SUPPORTED" if w < p else "NOT SUPPORTED"
    print(f"\nprediction (2) {verdict}: "
          f"{'condition explains part of the heterogeneity' if w < p else 'the grouping explains nothing; do not spend a confirmatory run'}")


if __name__ == "__main__":
    main()
