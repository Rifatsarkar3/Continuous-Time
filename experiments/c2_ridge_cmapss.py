"""C2: the tiny ridge alarm on C-MAPSS. DESCRIPTIVE - informed by E4, not a blind test.

Written 2026-09-24, before the ridge has scored any C-MAPSS unit.

WHY. On the bearing fleets a tiny ridge certifies about as well as CT-GRU (PRONOSTIA dev LOO
0.690 vs 0.750; XJTU 0.734 vs 0.739). The claim this was first going to test - "fleet size, not
model size, governs certified efficiency" - is already contradicted by E4: on C-MAPSS the waste
gap between predictors is flat in n (dummy minus CT-GRU 0.10 at n = 9 and at n = 180). The
corrected claim is that efficiency is set by the fleet's least predictable asset: where one exists
(the bearing fleets) capacity cannot help, and in a homogeneous fleet predictor quality pays at
every n. This run adds the ridge as a fourth arm on C-MAPSS to see whether that holds for it too.

DISCLOSURE. The prediction below was formed after reading E4's CT-GRU, weak and dummy arms on
these same units. Nothing here is confirmatory, and no gate reads it.

DESIGN, identical to E4's robustness run so every calibration draw is paired with E4's arms:
partitions of seeds 42, 123, 777 (train / pool / 150 test units; the reserved units are never
read); n in E4's grid; 40 draws per n from the same rng stream; grid CRC; waste and late on the
test units. The ridge is Gate 0's lp_ridge minus the envelope features, which do not exist for
turbofan channels: the 24 C-MAPSS channels, baseline-corrected per unit, scaled on the training
units, ridge on normalized RUL, lambda 1.0. 25 parameters.

PREDICTION: ridge waste is above CT-GRU's at every n >= 9 (mean over the paired draws), by a gap
that does not shrink as n falls - operationally, the gap at n = 9 is at least half the largest
gap on the grid. No prediction is made about ridge against the dummy or the weak arm.

    python experiments/c2_ridge_cmapss.py
"""
import collections
import datetime
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from certalarm import cmapss, evaluate, features, leadpoint, paths  # noqa: E402
from experiments.e4_asset_count import ALPHA, LEAD, N_GRID, curves, partition, sweep  # noqa: E402

SEEDS = (42, 123, 777)
N_REP = 40
RIDGE_LAM = 1.0
E4_ROBUST = paths.RESULTS / "e4_asset_count_robust.json"
OUT = paths.RESULTS / "c2_ridge_cmapss.json"


def make_ridge(train):
    scale = features.fit_scale([u.grid for u in train])
    m = leadpoint.fit_ridge([features.normalize(u.grid, scale) for u in train],
                            [u.rul for u in train], lam=RIDGE_LAM)
    return (lambda u: leadpoint.score(m, features.normalize(u.grid, scale))), \
        leadpoint.n_params(m)


def main():
    fleet = cmapss.load()
    rows = []
    for seed in SEEDS:
        train, pool, test, reserved = partition(fleet, seed)
        assert not {u.name for u in reserved} & {u.name for u in train + pool + test}
        predict, n_params = make_ridge(train)
        cal_late, cal_waste = curves(predict, pool, evaluate.THETAS)
        test_late, test_waste = curves(predict, test, evaluate.THETAS)
        mae = float(np.mean([np.abs(predict(u) - u.rul).mean() for u in test]))
        r = sweep(cal_late, cal_waste, test_late, test_waste, evaluate.THETAS, N_GRID, N_REP, seed)
        for x in r:
            x.update(seed=seed, arm="ridge", predictor_mae=mae)
        rows += r
        print(f"seed {seed}: train {len(train)} pool {len(pool)} test {len(test)}  "
              f"ridge params {n_params}  test MAE {mae:.4f}", flush=True)

    e4 = [x for x in json.loads(E4_ROBUST.read_text())["rows"] if x["seed"] in SEEDS]
    key = lambda x: (x["seed"], x["n"], x["rep"])
    by_arm = collections.defaultdict(dict)
    for x in e4 + rows:
        by_arm[x["arm"]][key(x)] = x
    assert set(by_arm["ridge"]) == set(by_arm["ctgru"]), "draws do not pair with E4"

    arms = ("ctgru", "ridge", "dummy", "weak")
    table = {}
    print(f"\n{'n':>5}" + "".join(f"{a:>9}" for a in arms)
          + f"{'ridge-ct':>10}{'P(ridge>ct)':>13}{'ridge late':>12}")
    for n in sorted({x["n"] for x in rows}):
        ks = [k for k in by_arm["ridge"] if k[1] == n
              and all(by_arm[a][k]["certified"] for a in arms)]
        if not ks:
            continue
        w = {a: np.array([by_arm[a][k]["waste"] for k in ks]) for a in arms}
        d = w["ridge"] - w["ctgru"]
        late = float(np.mean([by_arm["ridge"][k]["late"] for k in ks]))
        table[n] = {**{a: float(w[a].mean()) for a in arms}, "ridge_minus_ctgru": float(d.mean()),
                    "p_ridge_worse": float((d > 1e-12).mean()), "ridge_late": late,
                    "draws": len(ks)}
        print(f"{n:>5}" + "".join(f"{w[a].mean():>9.3f}" for a in arms)
              + f"{d.mean():>+10.3f}{(d > 1e-12).mean():>13.2f}{late:>12.3f}")

    held = all(v["ridge_minus_ctgru"] > 0 for n, v in table.items() if n >= 9)
    gaps = [table[n]["ridge_minus_ctgru"] for n in sorted(table) if n >= 9]
    no_shrink = gaps[0] >= 0.5 * max(gaps) if gaps else False
    print(f"\nprediction - ridge worse than CT-GRU at every n >= 9: {'HELD' if held else 'FAILED'}")
    print(f"prediction - gap does not shrink as n falls (gap at n=9 >= half the largest): "
          f"{'HELD' if no_shrink else 'FAILED'}")

    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=paths.ROOT,
                            capture_output=True, text=True).stdout.strip()
    OUT.write_text(json.dumps({"provenance": {
        "generated_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(
            timespec="seconds"), "git_commit": commit, "seeds": list(SEEDS), "n_rep": N_REP,
        "alpha": ALPHA, "lead": LEAD, "descriptive_not_confirmatory": True,
        "informed_by": "e4_asset_count_robust.json"},
        "table": {str(k): v for k, v in table.items()},
        "prediction": {"ridge_worse_every_n": held, "gap_not_shrinking": no_shrink},
        "rows": rows}, indent=2))
    print(f"written to {OUT.relative_to(paths.ROOT)}")


if __name__ == "__main__":
    main()
