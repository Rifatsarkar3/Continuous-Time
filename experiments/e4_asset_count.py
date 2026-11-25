"""E4: how certification behaves as the fleet grows. C-MAPSS, 709 run-to-failure units.

The bearing fleets cannot answer this question. At 17 assets the verification floor is 0.195, so
every realized risk below 20% looks the same. C-MAPSS has enough units that the two floors
separate and can be watched separately as n grows:

  certifiable floor  1/(n+1)            - below it CRC must decline to issue a threshold at all
  verification floor 1-(ci/2)^(1/n)     - below it no experiment can show the risk was achieved

Design, and why:

* **One predictor, fixed.** It is trained once on a disjoint training partition, and every
  calibration draw reuses it. The guarantee is about exchangeability between calibration and test
  units, which a fixed predictor does not disturb, and it makes the sweep cheap enough to run
  many replicates. It also isolates what is being measured: this experiment is about the
  certification machinery, not about predictor quality.

* **The pool is reused across n, deliberately.** E4 characterizes a function of n, so it must
  evaluate the same pool at many values of n. That makes it descriptive, not confirmatory, and
  **E4 is therefore never used to pass a gate.** A reserved slice of units is held back untouched
  so a future one-shot confirmatory evaluation still has a clean test set to spend.

Smoke:  python experiments/e4_asset_count.py --smoke
Full:   python experiments/e4_asset_count.py
"""
import argparse
import datetime
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from certalarm import certify, cmapss, evaluate, features, paths, stats  # noqa: E402
from certalarm.decision import outcome_curves  # noqa: E402
from certalarm.predictors import ctgru_train  # noqa: E402
from certalarm.predictors.ctgru import CTGRU  # noqa: E402

ALPHA, LEAD = 0.10, 0.05
N_GRID = (2, 4, 6, 8, 9, 10, 12, 15, 20, 30, 50, 80, 120, 180)
N_TEST = 150          # fixed, so the verification floor is the same at every n on the x axis
N_RESERVED = 100      # never read here; kept clean for a future one-shot confirmatory gate
ITERS = 300           # fixed a priori by budget, not selected on any outcome


def partition(fleet, seed):
    """-> (train, calibration pool, test, reserved), disjoint, stratified by subset."""
    rng = np.random.default_rng(seed)
    by_subset = {}
    for u in fleet:
        by_subset.setdefault(u.condition, []).append(u)
    train, pool, test, reserved = [], [], [], []
    for s in sorted(by_subset):
        units = sorted(by_subset[s], key=lambda u: u.name)
        rng.shuffle(units)
        n = len(units)
        n_res = max(1, round(N_RESERVED * n / len(fleet)))
        n_test = max(1, round(N_TEST * n / len(fleet)))
        n_train = (n - n_res - n_test) // 2
        reserved += units[:n_res]
        test += units[n_res:n_res + n_test]
        train += units[n_res + n_test:n_res + n_test + n_train]
        pool += units[n_res + n_test + n_train:]
    return train, pool, test, reserved


def make_dummy(train):
    """A predictor carrying no information about the individual unit.

    It reads the median training life and assumes every unit follows it, so its error is entirely
    the spread of lifetimes in the fleet. It exists as a control: the CRC guarantee is a property
    of the calibration machinery, not of the model, so validity must survive a predictor this bad.
    If it does not, the law E4 reports is really a fact about CT-GRU.
    """
    median_life = float(np.median([u.n_min for u in train]))

    def predict(u):
        t = np.arange(u.n_min, dtype=float)
        return np.clip(1.0 - t / max(median_life, 1.0), 0.0, 1.0)
    return predict


def make_ctgru(model, scale):
    def predict(u):
        X = features.normalize(u.grid, scale)
        mu, _ = ctgru_train.predict(model, X, np.ones(u.n_min, dtype=np.float32))
        return np.asarray(mu, float)
    return predict


def curves(predict, units, thetas):
    """-> (late, waste) matrices, one row per unit. Full observation: E6 owns the mask question."""
    late, waste = [], []
    for u in units:
        can_fire = np.ones(u.n_min, bool)
        l, r = outcome_curves(predict(u), np.asarray(u.rul, float), thetas, LEAD, can_fire)
        late.append(l)
        waste.append(r)
    return np.stack(late), np.stack(waste)


def sweep(cal_late, cal_waste, test_late, test_waste, thetas, n_grid, n_rep, seed):
    rng = np.random.default_rng(seed)
    rows = []
    for n in n_grid:
        if n > len(cal_late):
            continue
        for rep in range(n_rep):
            idx = rng.choice(len(cal_late), size=n, replace=False)
            th = certify.crc_threshold(cal_late[idx], thetas, ALPHA)
            if th is None:
                # Below the certifiable floor CRC declines. That is the correct behaviour and it
                # is recorded as an outcome, never silently replaced by a fallback threshold.
                rows.append({"n": int(n), "rep": rep, "certified": False,
                             "late": None, "waste": None})
                continue
            k = int(np.searchsorted(thetas, th))
            rows.append({"n": int(n), "rep": rep, "certified": True,
                         "theta": float(th),
                         "late": float(test_late[:, k].mean()),
                         "waste": float(test_waste[:, k].mean()),
                         "late_per_unit": test_late[:, k].tolist()})
    return rows


def report(rows, n_test):
    print(f"\n{'n':>5}{'certified':>11}{'late':>9}{'boot_hi':>9}{'waste':>9}"
          f"{'cert_floor':>12}{'ver_floor':>11}")
    for n in sorted({r["n"] for r in rows}):
        sel = [r for r in rows if r["n"] == n]
        ok = [r for r in sel if r["certified"]]
        rate = len(ok) / len(sel)
        if not ok:
            print(f"{n:>5}{rate:>11.0%}{'--':>9}{'--':>9}{'--':>9}"
                  f"{certify.certifiable_floor(n):>12.3f}{certify.verification_floor(n_test):>11.3f}")
            continue
        late = float(np.mean([r["late"] for r in ok]))
        waste = float(np.mean([r["waste"] for r in ok]))
        # Cluster on the test unit: replicates share the same test fleet, so pooling their rows
        # as independent draws would understate the interval.
        per_unit = np.mean([r["late_per_unit"] for r in ok], axis=0)
        _, hi = stats.cluster_bootstrap_ci(per_unit, np.arange(len(per_unit)), n_boot=2000,
                                           seed=0, alpha=0.05)
        print(f"{n:>5}{rate:>11.0%}{late:>9.3f}{hi:>9.3f}{waste:>9.3f}"
              f"{certify.certifiable_floor(n):>12.3f}{certify.verification_floor(n_test):>11.3f}")


def one_run(fleet, seed, arm, iters, n_grid, n_rep, smoke):
    """One (seed, predictor arm) cell of the robustness matrix."""
    train, pool, test, reserved = partition(fleet, seed)
    if smoke:
        train, pool, test = train[:20], pool[:40], test[:20]

    overlap = set(u.name for u in train) & set(u.name for u in test)
    assert not overlap, f"train and test overlap: {sorted(overlap)[:3]}"
    reserved_names = {u.name for u in reserved}
    assert not reserved_names & {u.name for u in train + pool + test}, "reserved units leaked"

    print(f"\n=== seed {seed} | arm {arm} ===", flush=True)
    print(f"train {len(train)}  pool {len(pool)}  test {len(test)}  "
          f"reserved (untouched) {len(reserved)}", flush=True)

    if arm == "dummy":
        predict = make_dummy(train)
    else:
        scale = features.fit_scale([u.grid for u in train])
        X = [features.normalize(u.grid, scale) for u in train]
        r = [u.rul for u in train]
        print(f"training CT-GRU on {len(train)} units, {iters} iters ...", flush=True)
        predict = make_ctgru(ctgru_train.train(X, r, "full", iters, seed), scale)

    thetas = evaluate.THETAS
    cal_late, cal_waste = curves(predict, pool, thetas)
    test_late, test_waste = curves(predict, test, thetas)
    mae = float(np.mean([np.abs(predict(u) - u.rul).mean() for u in test]))
    rows = sweep(cal_late, cal_waste, test_late, test_waste, thetas, n_grid, n_rep, seed)
    for row in rows:
        row["seed"], row["arm"], row["predictor_mae"] = seed, arm, mae
    print(f"predictor MAE on test units: {mae:.4f}")
    report(rows, len(test))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--arms", nargs="+", default=["ctgru"],
                    help="ctgru, weak (few iterations), dummy (no unit-level information)")
    ap.add_argument("--n-rep", type=int, default=40)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    fleet = cmapss.load()
    n_grid, n_rep = N_GRID, a.n_rep
    if a.smoke:
        n_grid, n_rep = (2, 6, 9, 20), 4

    rows = []
    for seed in a.seeds:
        for arm in a.arms:
            iters = 30 if a.smoke else (25 if arm == "weak" else ITERS)
            rows += one_run(fleet, seed, arm, iters, n_grid, n_rep, a.smoke)

    if len(a.seeds) * len(a.arms) > 1:
        invariance(rows)

    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=paths.ROOT,
                            capture_output=True, text=True).stdout.strip()
    out = {"provenance": {"generated_at_utc": datetime.datetime.now(datetime.timezone.utc)
                          .isoformat(timespec="seconds"), "git_commit": commit,
                          "smoke": a.smoke, "seeds": a.seeds, "arms": a.arms,
                          "alpha": ALPHA, "lead": LEAD, "n_rep": n_rep,
                          "descriptive_not_confirmatory": True},
           "rows": rows}
    paths.RESULTS.mkdir(exist_ok=True)
    name = (f"e4_asset_count{'_' + a.tag if a.tag else ''}"
            f"{'_smoke' if a.smoke else ''}.json")
    (paths.RESULTS / name).write_text(json.dumps(out, indent=2))
    print(f"\nwritten to results/{name}")


def invariance(rows):
    """Does the law hold across seeds and predictors, or is it a fact about one model?"""
    print("\n=== predictor and seed invariance ===")
    print(f"{'arm':<8}{'MAE':>8}{'switch_on':>11}{'max_late':>10}{'waste@9':>10}{'waste@180':>11}")
    for arm in sorted({r["arm"] for r in rows}):
        sel = [r for r in rows if r["arm"] == arm]
        ok = [r for r in sel if r["certified"]]
        switch = min((r["n"] for r in ok), default=None)
        mae = float(np.mean([r["predictor_mae"] for r in sel]))
        worst = max((r["late"] for r in ok), default=float("nan"))
        w9 = [r["waste"] for r in ok if r["n"] == 9]
        w180 = [r["waste"] for r in ok if r["n"] == max(r2["n"] for r2 in ok)]
        print(f"{arm:<8}{mae:>8.4f}{str(switch):>11}{worst:>10.3f}"
              f"{np.mean(w9) if w9 else float('nan'):>10.3f}"
              f"{np.mean(w180) if w180 else float('nan'):>11.3f}")

    # CRC bounds the EXPECTATION of the loss, not every draw of it. Testing each replicate
    # against alpha asks for something the theorem never promised and fails almost surely;
    # the quantity to compare is the mean, and its spread across seeds is the spread across
    # test-set draws, which is the only part of the expectation a fixed test set cannot average.
    print(f"\n{'arm':<8}{'mean late':>11}{'per-seed spread':>34}   consistent with alpha")
    for arm in sorted({r["arm"] for r in rows}):
        ok = [r for r in rows if r["arm"] == arm and r["certified"]]
        per_seed = [float(np.mean([r["late"] for r in ok if r["seed"] == s]))
                    for s in sorted({r["seed"] for r in ok})]
        mean = float(np.mean(per_seed))
        se = float(np.std(per_seed, ddof=1) / np.sqrt(len(per_seed))) if len(per_seed) > 1 else 0.0
        print(f"{arm:<8}{mean:>11.4f}{str([round(x, 3) for x in per_seed]):>34}   "
              f"{mean - 2 * se <= ALPHA}")


if __name__ == "__main__":
    main()
