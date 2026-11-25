"""C2 defect check. The ridge beating CT-GRU on C-MAPSS was a surprise, so before trusting it:

1. retrain seed 42's CT-GRU in the current code and require E4's thresholds draw for draw;
2. compare the lead-point distributions (theta*) of the two models on the calibration pool,
   which by the reduction is all the certificate depends on.

Result at commit time: 560/560 draws identical.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from certalarm import cmapss, evaluate, features, leadpoint, paths
from certalarm.predictors import ctgru_train
from experiments.e4_asset_count import ITERS, LEAD, N_GRID, curves, make_ctgru, partition, sweep
from experiments.c2_ridge_cmapss import make_ridge

fleet = cmapss.load()
train, pool, test, reserved = partition(fleet, 42)
t0 = time.perf_counter()
scale = features.fit_scale([u.grid for u in train])
ct = make_ctgru(ctgru_train.train([features.normalize(u.grid, scale) for u in train],
                                  [u.rul for u in train], "full", ITERS, 42), scale)
print(f"CT-GRU retrained in {time.perf_counter()-t0:.0f}s")
cl, cw = curves(ct, pool, evaluate.THETAS); tl, tw = curves(ct, test, evaluate.THETAS)
mine = sweep(cl, cw, tl, tw, evaluate.THETAS, N_GRID, 40, 42)
e4 = [r for r in json.load(open(paths.RESULTS / "e4_asset_count_robust.json"))["rows"]
      if r["seed"] == 42 and r["arm"] == "ctgru"]
same = sum(abs((a.get("theta") or -1) - (b.get("theta") or -1)) < 1e-12 and a["n"] == b["n"] and a["rep"] == b["rep"]
           for a, b in zip(mine, e4))
print(f"E4 reproduction: {same}/{len(e4)} draws identical theta; "
      f"waste@180 mine {np.mean([r['waste'] for r in mine if r['n']==180]):.4f} "
      f"E4 {np.mean([r['waste'] for r in e4 if r['n']==180]):.4f}")

rg, _ = make_ridge(train)
for name, f in (("ctgru", ct), ("ridge", rg)):
    ts = np.array([leadpoint.lead_point(f(u), u.rul, LEAD) for u in pool])
    mu_all = np.concatenate([f(u) for u in pool])
    early = np.array([np.min(f(u)[u.rul >= 0.5]) for u in pool])     # running min over first half
    q = lambda p: np.quantile(ts, p)
    print(f"{name}: theta* median {q(.5):.3f} q90 {q(.9):.3f} max {ts.max():.3f} | "
          f"min over first half of life: median {np.median(early):.3f} | "
          f"signal range [{mu_all.min():.2f}, {mu_all.max():.2f}] | "
          f"units with first-half min below theta*_q90: {(early < q(.9)).mean():.2f}")
    for s in (1, 2, 3, 4):
        sel = [i for i, u in enumerate(test) if u.condition == s]
        th = evaluate.THETAS[np.searchsorted(evaluate.THETAS, q(.9))]
        k = int(np.searchsorted(evaluate.THETAS, th))
        print(f"   FD00{s}: test waste at theta=q90 {tw[sel, k].mean() if name=='ctgru' else curves(f, [test[i] for i in sel], evaluate.THETAS)[1][:, k].mean():.3f}")
