"""Verify the lead-point reduction exactly, on PRONOSTIA development trajectories only.

Claim. For the running-minimum alarm in certalarm.decision, with t* the last step at which the
true remaining life is still >= lead, and theta* = min(signal[:t*+1]):

    (R1) late(theta) == 1[theta <= theta*]      for every theta
    (R2) the CRC threshold over n calibration assets equals the split-conformal threshold on the
         scalars {theta*_i}: the smallest grid theta with (#{theta*_i >= theta} + 1)/(n+1) <= alpha

If both hold, certifying a trajectory-level monitoring alarm is exactly split-conformal inference
on one scalar per asset. This script checks both on every inner held-out trajectory, and fails
loudly on the first mismatch.

Protocol. Development data only: the outer test bearings of each fold are asserted absent from
every trajectory used. Models come from the on-disk cache. The evaluation registry is never
touched, so no test evaluation is spent.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from certalarm import assets, certify, crossfit, evaluate  # noqa: E402
from certalarm.decision import outcome_curves  # noqa: E402
from experiments.e2_main_alarms import ALPHA, LEAD, CTGRUPredictor  # noqa: E402

SEED = 42


def lead_point(mu, rul, lead):
    """theta* = running minimum of the signal at the last step with rul >= lead."""
    ok = np.flatnonzero(np.asarray(rul) >= lead)
    if ok.size == 0:
        return -np.inf          # no acceptable alarm time exists: late for every theta
    return float(np.min(np.asarray(mu, float)[: ok[-1] + 1]))


def conformal_on_scalars(tstar, thetas, alpha):
    tstar = np.asarray(tstar, float)
    n = len(tstar)
    for th in thetas:
        if (np.sum(tstar >= th) + 1) / (n + 1) <= alpha + 1e-12:
            return float(th)
    return None


def main():
    fleet = assets.load("pronostia")
    by = {a.name: a for a in fleet}
    pred = CTGRUPredictor("pronostia")
    thetas = np.asarray(evaluate.THETAS, float)
    n_traj = n_theta_checks = 0
    folds = []
    for spec in crossfit.build_fold(fleet, seed=SEED):
        own_test = set(spec["test"])
        L, tstar = [], []
        for j, inner in enumerate(spec["inner"]):
            assert not set(inner["held"]) & own_test and not set(inner["fit"]) & own_test
            model = pred.fit([by[n] for n in inner["fit"]], seed=SEED * 100 + j, process="full")
            for name in inner["held"]:
                a = by[name]
                mu, _ = pred.predict(model, a, np.ones(a.n_min, dtype=bool))
                mu, rul = np.asarray(mu, float), np.asarray(a.rul, float)
                late, _ = outcome_curves(mu, rul, thetas, LEAD, np.ones(a.n_min, bool))
                ts = lead_point(mu, rul, LEAD)
                predicted = (thetas <= ts).astype(float)
                if not np.array_equal(late, predicted):
                    bad = np.flatnonzero(late != predicted)
                    sys.exit(f"R1 FAILED on {name}: theta*={ts:.6f}, first mismatch at "
                             f"theta={thetas[bad[0]]:.4f} (late={late[bad[0]]}, "
                             f"predicted={predicted[bad[0]]})")
                L.append(late)
                tstar.append(ts)
                n_traj += 1
                n_theta_checks += len(thetas)
        crc = certify.crc_threshold(np.stack(L), thetas, ALPHA)
        conf = conformal_on_scalars(tstar, thetas, ALPHA)
        if crc != conf:
            sys.exit(f"R2 FAILED: CRC threshold {crc} != conformal-on-theta* {conf} "
                     f"(n={len(tstar)})")
        folds.append((len(tstar), crc, float(np.max(tstar)), float(np.median(tstar))))

    print(f"R1 exact: late(theta) == 1[theta <= theta*] on all {n_traj} trajectories x "
          f"{len(thetas)} thresholds = {n_theta_checks} checks, 0 mismatches")
    print(f"R2 exact: CRC threshold == split-conformal on theta* in all {len(folds)} folds")
    print("\n  fold  n_cal  CRC theta  max theta*  median theta*")
    for i, (n, th, mx, md) in enumerate(folds):
        print(f"  {i:>4}  {n:>5}  {th:>9.4f}  {mx:>10.4f}  {md:>13.4f}")


if __name__ == "__main__":
    main()
