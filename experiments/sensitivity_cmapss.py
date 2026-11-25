"""Descriptive sensitivity of the C-MAPSS asset-count law to alpha and lead.

Reuses E4 unchanged - partitions, a CT-GRU of 300 iterations per seed, outcome curves and the
40-draw calibration sweep - with E4's module-level ALPHA and LEAD set per cell. Each seed's CT-GRU
reproduces E4's thresholds exactly (c2_defect_check.py: 560/560 draws). E4 already read these
test partitions, so the results are descriptive; reserved units are not touched.

Theory check: certification switches on at n = ceil(1/alpha - 1), i.e. 19 at
alpha = 0.05 (first grid value 20), 9 at 0.10, 4 at 0.20, whatever the lead.

    python experiments/sensitivity_cmapss.py      # writes results/sensitivity_cmapss.json
"""
import collections
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import experiments.e4_asset_count as e4  # noqa: E402
from certalarm import certify, cmapss, evaluate, features, paths  # noqa: E402
from certalarm.predictors import ctgru_train  # noqa: E402

ALPHAS, LEADS, SEEDS = (0.05, 0.08, 0.10, 0.15, 0.20, 0.30), (0.02, 0.05, 0.10), (42, 123, 777)


def main():
    fleet = cmapss.load()
    rows = []
    for seed in SEEDS:
        train, pool, test, _ = e4.partition(fleet, seed)
        scale = features.fit_scale([u.grid for u in train])
        model = ctgru_train.train([features.normalize(u.grid, scale) for u in train],
                                  [u.rul for u in train], "full", e4.ITERS, seed)
        raw, memo = e4.make_ctgru(model, scale), {}

        def predict(u):
            if u.name not in memo:
                memo[u.name] = raw(u)
            return memo[u.name]

        for lead in LEADS:
            e4.LEAD = lead
            cl, cw = e4.curves(predict, pool, evaluate.THETAS)
            tl, tw = e4.curves(predict, test, evaluate.THETAS)
            for alpha in ALPHAS:
                e4.ALPHA = alpha
                for r in e4.sweep(cl, cw, tl, tw, evaluate.THETAS, e4.N_GRID, 40, seed):
                    r.pop("late_per_unit", None)
                    rows.append({"seed": seed, "lead": lead, "alpha": alpha, **r})
        print(f"seed {seed} done", flush=True)

    cells = collections.defaultdict(list)
    for r in rows:
        cells[(r["alpha"], r["lead"], r["n"])].append(r)
    summary = []
    for (a, ld, n), rs in sorted(cells.items()):
        cert = [r for r in rs if r["certified"]]
        summary.append({"alpha": a, "lead": ld, "n": n, "certified": len(cert) / len(rs),
                        "late": float(np.mean([r["late"] for r in cert])) if cert else None,
                        "waste": float(np.mean([r["waste"] for r in cert])) if cert else None})
    switch = {f"{a}/{ld}": {"observed": min((s["n"] for s in summary if s["alpha"] == a
                                             and s["lead"] == ld and s["certified"] == 1.0),
                                            default=None),
                            "predicted": certify.min_assets(a)}
              for a in ALPHAS for ld in LEADS}
    (paths.RESULTS / "sensitivity_cmapss.json").write_text(json.dumps(
        {"note": "descriptive; E4 test partitions reused; reserved units untouched",
         "switch_on": switch, "summary": summary, "rows": rows}, indent=2, default=float),
        encoding="utf-8")
    print(json.dumps(switch))
    for s in summary:
        if s["n"] in (9, 20, 50, 180) or (s["alpha"] == 0.2 and s["n"] == 4):
            print(f"alpha {s['alpha']:.2f} lead {s['lead']:.2f} n {s['n']:>3} cert {s['certified']:.2f} "
                  f"late {s['late'] if s['late'] is None else round(s['late'], 3)} "
                  f"waste {s['waste'] if s['waste'] is None else round(s['waste'], 3)}")


if __name__ == "__main__":
    main()
