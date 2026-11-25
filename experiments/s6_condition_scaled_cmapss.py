"""S6: does condition-scaled CRC actually work? Tested on C-MAPSS reserved units.

Pre-registered on 2026-09-20.

Why here rather than XJTU. S6 asks whether the new rule moves the late/waste frontier. Answering
that needs the power to see a modest shift: 100 reserved units have a verification floor of 0.036,
where XJTU's 15 bearings sit at 0.218 and could not resolve the ~6% effect the development
forecast predicted. XJTU is the right place to ask whether the small-fleet pathology replicates;
this is the right place to ask whether the method works.

**One-shot.** The reserved units of each partition were held out of E4 and have never been
scored. This run spends them, and the registry refuses a second evaluation of the same models.

**Disclosure.** Seed 42's reserved units were read once during development, by a smoke run with a
20-iteration predictor and the registry disabled. No threshold, gate or analysis was changed as a
result - all of them were fixed in the pre-registration beforehand - but the outcome was seen, so
seed 42 no longer qualifies as a blind partition. **Seed 123 is therefore the primary partition
and 777 the replicate; seed 42 is reported as exploratory and labelled as such.** Recording this
costs a partition and is cheaper than a result nobody can trust.

    python experiments/s6_condition_scaled_cmapss.py
"""
import argparse
import datetime
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from certalarm import certify, cmapss, evaluate, features, paths, pooling, registry, stats  # noqa: E402
from certalarm.decision import outcome_curves  # noqa: E402
from certalarm.predictors import ctgru_train  # noqa: E402
from experiments.e4_asset_count import ALPHA, ITERS, LEAD, partition  # noqa: E402

S6A_MARGIN = 0.05      # S6a: boot_hi <= ALPHA + this, the same instrument and margin as S1


class _Traj:
    def __init__(self, mu, rul):
        self.mu, self.rul = np.asarray(mu, float), np.asarray(rul, float)


def _curves(units, predict, scales, thetas, scaled):
    late, waste = [], []
    for u in units:
        mu = predict(u)
        if scaled:
            mu = mu / max(scales.get(u.condition, 1.0), 1e-9)
        l, r = outcome_curves(mu, np.asarray(u.rul, float), thetas, LEAD,
                              np.ones(u.n_min, bool))
        late.append(l)
        waste.append(r)
    return np.stack(late), np.stack(waste)


def one_partition(fleet, seed, iters, use_registry=True):
    train, pool, test, reserved = partition(fleet, seed)
    names = {"train": {u.name for u in train}, "pool": {u.name for u in pool},
             "reserved": {u.name for u in reserved}}
    assert not names["train"] & names["reserved"], "reserved leaked into training"
    assert not names["pool"] & names["reserved"], "reserved leaked into calibration"

    scale = features.fit_scale([u.grid for u in train])
    X = [features.normalize(u.grid, scale) for u in train]
    print(f"seed {seed}: training on {len(train)}, calibrating on {len(pool)}, "
          f"testing on {len(reserved)} reserved units ({iters} iters) ...", flush=True)
    model = ctgru_train.train(X, [u.rul for u in train], "full", iters, seed)

    def predict(u):
        mu, _ = ctgru_train.predict(model, features.normalize(u.grid, scale),
                                    np.ones(u.n_min, dtype=np.float32))
        return np.asarray(mu, float)

    # Scales from the TRAINING units only - the condition CRC's exchangeability rests on.
    scales = pooling.condition_scales(
        {u.name: {"condition": u.condition, "reps": [_Traj(predict(u), u.rul)]} for u in train},
        LEAD, fit_names=[u.name for u in train])
    print(f"  condition scales: {({k: round(v, 3) for k, v in sorted(scales.items())})}")

    thetas = evaluate.THETAS
    out = {}
    for arm, scaled in (("crc", False), ("crc_scaled", True)):
        cal_l, _ = _curves(pool, predict, scales, thetas, scaled)
        th = certify.crc_threshold(cal_l, thetas, ALPHA)
        if th is None:
            out[arm] = None
            continue
        k = int(np.searchsorted(thetas, th))
        tl, tw = _curves(reserved, predict, scales, thetas, scaled)
        out[arm] = {"theta": float(th), "late": tl[:, k], "waste": tw[:, k],
                    "n_calibration": int(cal_l.shape[0])}

    if use_registry:
        cfg = {"tag": f"s6/cmapss/seed{seed}", "alpha": ALPHA, "lead": LEAD,
               "reserved": sorted(names["reserved"]),
               "thetas": {a: (v["theta"] if v else None) for a, v in out.items()}}
        registry.register(cfg, model)
        registry.check(cfg, model, f"s6/cmapss/seed{seed}")
    return out, len(reserved)


def report(out, n_test, seed):
    print(f"\n--- seed {seed}: {n_test} reserved units, "
          f"verification floor {certify.verification_floor(n_test):.3f} ---")
    print(f"{'arm':<12}{'theta':>8}{'late':>9}{'boot_hi':>9}{'waste':>9}")
    summary = {}
    for arm in ("crc", "crc_scaled"):
        v = out[arm]
        if v is None:
            print(f"{arm:<12}{'declined':>8}")
            continue
        _, hi = stats.cluster_bootstrap_ci(v["late"], np.arange(len(v["late"])),
                                           n_boot=2000, seed=0, alpha=0.05)
        summary[arm] = {"late": float(v["late"].mean()), "hi": float(hi),
                        "waste": float(v["waste"].mean()), "theta": v["theta"]}
        print(f"{arm:<12}{v['theta']:>8.3f}{v['late'].mean():>9.3f}{hi:>9.3f}"
              f"{v['waste'].mean():>9.3f}")

    if len(summary) == 2:
        s6a = summary["crc_scaled"]["hi"] <= ALPHA + S6A_MARGIN
        w = stats.paired_wilcoxon(out["crc"]["waste"], out["crc_scaled"]["waste"])
        lower = summary["crc_scaled"]["waste"] < summary["crc"]["waste"]
        s6b = lower and w["p"] < 0.05
        print(f"  S6a validity preserved : boot_hi {summary['crc_scaled']['hi']:.3f} <= "
              f"{ALPHA + S6A_MARGIN:.2f}   {'PASS' if s6a else 'FAIL'}")
        print(f"  S6b efficiency recovered: waste {summary['crc_scaled']['waste']:.3f} < "
              f"{summary['crc']['waste']:.3f}, p = {w['p']:.4f}   {'PASS' if s6b else 'FAIL'}")
        summary["gates"] = {"S6a": bool(s6a), "S6b": bool(s6b), "p": float(w["p"]),
                            "units_improved": int((out["crc_scaled"]["waste"]
                                                   < out["crc"]["waste"]).sum())}
        print(f"  units where the scaled rule wastes less: "
              f"{summary['gates']['units_improved']} of {n_test}")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[123, 777, 42],
                    help="primary first; 42 is exploratory, see the module docstring")
    ap.add_argument("--iters", type=int, default=ITERS)
    ap.add_argument("--no-registry", action="store_true")
    a = ap.parse_args()

    fleet = cmapss.load()
    all_summary = {}
    for seed in a.seeds:
        out, n_test = one_partition(fleet, seed, a.iters, use_registry=not a.no_registry)
        all_summary[seed] = report(out, n_test, seed)

    gates = [s["gates"] for s in all_summary.values() if "gates" in s]
    if gates:
        print(f"\n=== S6 across {len(gates)} partitions ===")
        print(f"  S6a passes: {sum(g['S6a'] for g in gates)}/{len(gates)}")
        print(f"  S6b passes: {sum(g['S6b'] for g in gates)}/{len(gates)}")
        print(f"  Holm-adjusted S6b p-values: "
              f"{[round(p, 4) for p in stats.holm([g['p'] for g in gates])]}")

    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=paths.ROOT,
                            capture_output=True, text=True).stdout.strip()
    (paths.RESULTS / "s6_condition_scaled_cmapss.json").write_text(json.dumps(
        {"provenance": {"generated_at_utc": datetime.datetime.now(datetime.timezone.utc)
                        .isoformat(timespec="seconds"), "git_commit": commit,
                        "alpha": ALPHA, "lead": LEAD, "iters": a.iters, "seeds": a.seeds,
                        "confirmatory": True, "primary_seed": a.seeds[0],
                        "exploratory_seeds": [42],
                        "disclosure": "seed 42 reserved units were read during a development smoke with a 20-iteration predictor; gates were fixed beforehand"},
         "summary": all_summary}, indent=2, default=float))
    print("\nwritten to results/s6_condition_scaled_cmapss.json")


if __name__ == "__main__":
    main()
