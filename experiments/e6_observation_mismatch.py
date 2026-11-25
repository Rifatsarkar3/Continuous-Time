"""E6: calibrate under one observation process, deploy under another.

The certification assumes the deployment a threshold was calibrated for is the deployment it
meets. This asks what happens when it is not: a threshold calibrated on densely monitored assets,
then deployed on a fleet whose telemetry drops out and whose sensors fail faster as the bearing
approaches failure.

    python experiments/e6_observation_mismatch.py --seeds 42 123 777 999 7
"""
import argparse
import datetime
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from certalarm import assets, crossfit, evaluate, paths  # noqa: E402
from experiments.e2_main_alarms import ALPHA, LEAD, CTGRUPredictor  # noqa: E402

# (calibrated under, deployed under). The matched pair is the control.
PAIRS = (("sparse_shift", "sparse_shift"), ("full", "sparse_shift"),
         ("mixture", "sparse_shift"), ("full", "mixture"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--datasets", nargs="+", default=["pronostia"])
    ap.add_argument("--n-rep", type=int, default=8)
    a = ap.parse_args()

    rows = []
    for dataset in a.datasets:
        fleet = assets.load(dataset)
        predictor = CTGRUPredictor(dataset)
        for seed in a.seeds:
            for i, spec in enumerate(crossfit.build_fold(fleet, seed=seed)):
                for cal_proc, dep_proc in PAIRS:
                    cal = crossfit.run_fold(fleet, spec, predictor, cal_proc, a.n_rep, seed)
                    dep = crossfit.run_fold(fleet, spec, predictor, dep_proc, a.n_rep, seed + 1)
                    rules = evaluate.fit_rules(cal.calibration, ALPHA, LEAD)
                    got = evaluate.apply_rules(rules, dep.test, LEAD)
                    s = evaluate.summarize(got["crc"])
                    rows.append({"dataset": dataset, "seed": seed, "fold": i,
                                 "calibrated_under": cal_proc, "deployed_under": dep_proc,
                                 "matched": cal_proc == dep_proc,
                                 "certifiable": rules["crc"]["certifiable"], **s})

    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=paths.ROOT,
                            capture_output=True, text=True).stdout.strip()
    payload = {"provenance": {"generated_at_utc": datetime.datetime.now(datetime.timezone.utc)
                              .isoformat(timespec="seconds"),
                              "git_commit": commit, "seeds": a.seeds, "n_rep": a.n_rep,
                              "alpha": ALPHA, "lead": LEAD},
               "rows": rows}
    paths.RESULTS.mkdir(exist_ok=True)
    (paths.RESULTS / "e6_observation_mismatch.json").write_text(json.dumps(payload, indent=2))

    print(f"{'calibrated':<14}{'deployed':<14}{'late':>8}{'upper':>8}{'waste':>8}   target "
          f"alpha={ALPHA}")
    for cal_proc, dep_proc in PAIRS:
        sel = [r for r in rows if r["calibrated_under"] == cal_proc
               and r["deployed_under"] == dep_proc]
        if not sel:
            continue
        late = sum(r["late_rate"] for r in sel) / len(sel)
        upper = max(r["late_upper"] for r in sel)
        waste = sum(r["waste_mean"] for r in sel) / len(sel)
        mark = "  (matched)" if cal_proc == dep_proc else ""
        print(f"{cal_proc:<14}{dep_proc:<14}{late:>8.3f}{upper:>8.3f}{waste:>8.3f}{mark}")
    print("\nwritten to results/e6_observation_mismatch.json")
    print("E6 is descriptive. It becomes a claim only if it survives 5 seeds (criterion S5).")


if __name__ == "__main__":
    main()
