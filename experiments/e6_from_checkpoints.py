"""E6: does the observation process change the conclusions? Answered from existing checkpoints.

No new compute. The three observation processes were already deployed in the PRONOSTIA grid and
every row carries its `process`, so this is a re-read of work already done rather than a new
experiment. That matters for the protocol as well as the budget: re-reading recorded outcomes
spends no test evaluation.

The finding is not really about observation at all. Across the nine (arm, process) cells the
realized late rate never leaves 0.056-0.075 - every cell is comfortably valid - yet only a third
of them can demonstrate it at the S1 gate, and the verdict flips *within* an arm depending on
which process you look at. The process does not change whether the rule is valid. It changes
whether the experiment can see that it is.

    python experiments/e6_from_checkpoints.py
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from certalarm import certify, evaluate, paths  # noqa: E402

ALPHA, S1_GATE = 0.10, 0.15
TAGS = {"ctgru": "pron_ctgru_pronostia_ctgru",
        "dsaflite": "pron_dsaflite_pronostia_dsaflite",
        "dann": "pron_dann_pronostia_dsaflite_dann"}
PROCESSES = ("full", "mixture", "sparse_shift")


def load(tag):
    rows = []
    p = paths.RESULTS / "checkpoints" / f"{tag}.jsonl"
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows += json.loads(line)["rows"]
    return rows


def cell(rows, rule):
    keyed = {f"{r['seed']}/{r['fold']}/{r['asset']}": r for r in rows if r["rule"] == rule}
    return evaluate.summarize(keyed, cluster_key={k: r["asset"] for k, r in keyed.items()})


def main():
    print(f"{'arm':<10}{'process':<14}{'crc late':>10}{'boot_hi':>9}{'S1':>6}"
          f"{'crc waste':>11}{'lbc waste':>11}")
    cells = []
    for arm, tag in TAGS.items():
        rows = load(tag)
        for proc in PROCESSES:
            sel = [r for r in rows if r["process"] == proc]
            crc, lbc = cell(sel, "crc"), cell(sel, "lbc")
            ok = crc["late_upper_bootstrap"] <= S1_GATE
            cells.append({"arm": arm, "process": proc, "late": crc["late_rate"],
                          "hi": crc["late_upper_bootstrap"], "s1": bool(ok),
                          "waste": crc["waste_mean"], "n_assets": crc["n_assets"]})
            print(f"{arm:<10}{proc:<14}{crc['late_rate']:>10.3f}"
                  f"{crc['late_upper_bootstrap']:>9.3f}{('PASS' if ok else 'FAIL'):>6}"
                  f"{crc['waste_mean']:>11.3f}{lbc['waste_mean']:>11.3f}")

    late = np.array([c["late"] for c in cells])
    n_valid = int((late <= ALPHA).sum())
    n_shown = sum(c["s1"] for c in cells)
    n_assets = cells[0]["n_assets"]
    print(f"\ncells: {len(cells)}   realized late rate in [{late.min():.3f}, {late.max():.3f}]")
    print(f"  actually valid (late <= alpha):        {n_valid}/{len(cells)}")
    print(f"  able to demonstrate it (S1 passes):    {n_shown}/{len(cells)}")
    print(f"  verification floor at n={n_assets}:            "
          f"{certify.verification_floor(n_assets):.3f}   (S1 gate is {S1_GATE})")
    for arm in TAGS:
        v = [c["s1"] for c in cells if c["arm"] == arm]
        if len(set(v)) > 1:
            print(f"  the S1 verdict flips across processes within '{arm}' - same bearings, "
                  f"same rule, same predictor")

    out = {"alpha": ALPHA, "s1_gate": S1_GATE, "cells": cells,
           "n_valid": n_valid, "n_demonstrable": n_shown}
    (paths.RESULTS / "e6_from_checkpoints.json").write_text(json.dumps(out, indent=2))
    print("\nwritten to results/e6_from_checkpoints.json")


if __name__ == "__main__":
    main()
