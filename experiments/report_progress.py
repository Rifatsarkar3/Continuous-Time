"""Read-only progress report over whatever the running grid has checkpointed so far.

Safe to run at any time against a live run: it only reads the checkpoint journals, never the
registry, so it cannot spend a test evaluation or disturb the run it is reporting on.

A (predictor, process, seed) block is the smallest unit that covers the whole fleet, so gates are
only meaningful once a block is complete. Partial blocks are counted but reported separately,
because a half-finished block is a biased sample of bearings, not a small one.

    python experiments/report_progress.py
    python experiments/report_progress.py --tag pron_ctgru
"""
import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from certalarm import checkpoint, evaluate  # noqa: E402

ALPHA, LEAD = 0.10, 0.05
S1_MARGIN = 0.05          # S1 gate: crc late_upper_bootstrap <= ALPHA + S1_MARGIN
S3_RATIO = 0.7            # S3 gate: crc waste <= S3_RATIO * lbc waste
FOLDS_PER_BLOCK = 5


def load(tag=None):
    """-> (rows, blocks) where blocks maps (predictor, process, seed) -> set of finished folds."""
    files = sorted(checkpoint.DIR.glob(f"{tag}*.jsonl" if tag else "*.jsonl"))
    rows, blocks = [], collections.defaultdict(set)
    for f in files:
        for line in f.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue          # a truncated final line from a crash; the unit is unfinished
            u = rec["unit"]
            blocks[(u["predictor"], u["process"], u["seed"])].add(u["fold"])
            rows += rec["rows"]
    return rows, blocks


def table(rows, title):
    """Per-rule table for one set of rows, clustered on the bearing."""
    if not rows:
        return {}
    mae = float(np.mean([r["predictor_mae"] for r in rows]))
    print(f"\n{title}  |  predictor MAE {mae:.4f}")
    print(f"{'rule':<14}{'late':>8}{'boot_hi':>9}{'cp_hi':>8}{'waste':>8}{'assets':>8}   "
          f"certified folds")
    out = {}
    for rule in evaluate.RULES:
        sel = [r for r in rows if r["rule"] == rule]
        if not sel:
            continue
        # The key must name every field that distinguishes a row, or rows collide in this dict
        # and are silently dropped. Pooling across observation processes without `process` here
        # keeps only the last process seen.
        keyed = {f"{r['process']}/{r['seed']}/{r['fold']}/{r['asset']}": r for r in sel}
        s = evaluate.summarize(keyed, cluster_key={k: r["asset"] for k, r in keyed.items()})
        out[rule] = s
        # Per-fold, not pooled. A rule that declines to certify scores every asset late, so a
        # refusing rule and a rule whose alarms fire too late are indistinguishable in `late`
        # alone. Reporting how many folds actually issued a certificate separates them.
        by_fold = {(r["process"], r["seed"], r["fold"]): r["certifiable"] for r in sel}
        cert = f"{sum(by_fold.values())}/{len(by_fold)}"
        print(f"{rule:<14}{s['late_rate']:>8.3f}{s['late_upper_bootstrap']:>9.3f}"
              f"{s['late_upper_binomial']:>8.3f}{s['waste_mean']:>8.3f}{s['n_assets']:>8}   {cert}")
    return out


def gates(s):
    """Report the two pre-registered gates. Reported as measured - never re-tuned to pass."""
    if "crc" not in s or "lbc" not in s:
        return
    crc, lbc = s["crc"], s["lbc"]
    s1 = crc["late_upper_bootstrap"] <= ALPHA + S1_MARGIN
    budget = S3_RATIO * lbc["waste_mean"]
    s3 = crc["waste_mean"] <= budget
    print(f"  S1 validity   crc boot_hi {crc['late_upper_bootstrap']:.3f} <= "
          f"{ALPHA + S1_MARGIN:.2f}   {'PASS' if s1 else 'FAIL'}")
    print(f"  S3 efficiency crc waste   {crc['waste_mean']:.3f} <= {budget:.3f}"
          f"   {'PASS' if s3 else 'FAIL'}")
    if "cost_optimal" in s:
        co = s["cost_optimal"]
        # Whether the certificate buys anything over an uncertified cost-optimal threshold is the
        # question that decides the framing, so the gap is reported as a magnitude on
        # every run. A bare "identical / not identical" hides a difference of 1e-4, which reads
        # as a broken tie when it is nothing of the sort.
        d_late = abs(crc["late_rate"] - co["late_rate"])
        d_waste = abs(crc["waste_mean"] - co["waste_mean"])
        print(f"  crc - cost_optimal gap: late {d_late:.2e}  waste {d_waste:.2e}")


def frontier(s):
    """Rank rules in (waste, late) space and mark the ones nothing dominates.

    S3 reduces the comparison to a single ratio against LBC, which stage 1 showed is
    near-degenerate under a weak predictor - it buys a zero late rate by burning 98% of the
    remaining life, so any ratio against it says more about LBC than about CRC. The Pareto view
    asks the question that actually matters: is there a rule that is better on both axes at once?
    """
    pts = {k: (v["waste_mean"], v["late_rate"]) for k, v in s.items()}
    print(f"\n{'rule':<14}{'waste':>8}{'late':>8}   dominated by")
    for name in sorted(pts, key=lambda n: pts[n][0]):
        w, l = pts[name]
        # Strictly better on one axis and no worse on the other.
        over = [o for o, (ow, ol) in pts.items()
                if o != name and ow <= w and ol <= l and (ow < w or ol < l)]
        print(f"{name:<14}{w:>8.3f}{l:>8.3f}   {', '.join(sorted(over)) if over else '-- on front'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default=None, help="only checkpoints whose file name starts with this")
    ap.add_argument("--blocks", action="store_true", help="also print each complete block on its own")
    ap.add_argument("--frontier", action="store_true", help="show the late/waste Pareto structure")
    a = ap.parse_args()

    rows, blocks = load(a.tag)
    if not rows:
        print("no checkpointed units yet")
        return

    complete = {k for k, v in blocks.items() if len(v) == FOLDS_PER_BLOCK}
    partial = {k: v for k, v in blocks.items() if len(v) < FOLDS_PER_BLOCK}
    units = sum(len(v) for v in blocks.values())
    print(f"units finished: {units}   complete blocks: {len(complete)}   partial: {len(partial)}")
    for k, v in sorted(partial.items()):
        print(f"  in progress: {k[0]}/{k[1]}/seed{k[2]} - {len(v)}/{FOLDS_PER_BLOCK} folds")

    if a.blocks:
        for k in sorted(complete):
            sel = [r for r in rows if (r["predictor"], r["process"], r["seed"]) == k]
            gates(table(sel, f"block {k[0]}/{k[1]}/seed{k[2]}"))

    for pred in sorted({k[0] for k in complete}):
        sel = [r for r in rows if (r["predictor"], r["process"], r["seed"]) in complete
               and r["predictor"] == pred]
        n_blocks = len({k for k in complete if k[0] == pred})
        s = table(sel, f"=== {pred}: {n_blocks} complete block(s) pooled ===")
        gates(s)
        if a.frontier:
            frontier(s)

    if partial:
        print("\nPartial blocks are excluded from every table above: a half-finished block "
              "covers a biased subset of bearings, not a smaller sample of them.")


if __name__ == "__main__":
    main()
