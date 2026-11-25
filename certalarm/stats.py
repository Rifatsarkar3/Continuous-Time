"""Statistics for the pre-registered gates.

Validity uses an exact Clopper-Pearson bound: at these counts the normal approximation is
noticeably anti-conservative, and a gate that passes only because of the approximation has not
passed. Efficiency compares rules on paired assets with Holm correction. The bootstrap
resamples assets, not asset-seed rows, because the seeds of one bearing are not independent.
"""
import numpy as np
from scipy import stats as sps


def clopper_pearson(k, n, alpha=0.05):
    """Exact two-sided binomial confidence interval for k successes in n trials.

    `k` must be a whole number of events. Replicate-averaged outcomes are fractional, and a
    binomial interval is the wrong instrument for them: use `cluster_bootstrap_ci` over the
    per-asset rates instead. Refusing here stops a fractional total being quietly truncated.
    """
    if float(k) != int(k) or float(n) != int(n):
        raise ValueError(f"k and n must be whole counts, got k={k}, n={n}")
    k, n = int(k), int(n)
    if n <= 0:
        raise ValueError("n must be positive")
    if not 0 <= k <= n:
        raise ValueError(f"k={k} outside [0, {n}]")
    lo = 0.0 if k == 0 else float(sps.beta.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(sps.beta.ppf(1 - alpha / 2, k + 1, n - k))
    return lo, hi


def holm(p_values):
    """Holm-Bonferroni adjusted p-values, returned in the input order.

    A NaN means the comparison carried no test at all (`paired_wilcoxon` reports NaN when every
    pair ties). Such entries are excluded from the family and returned as NaN. Leaving them in
    would be wrong twice over: `max(x, nan)` returns `x`, so the NaN would come back as a
    finite, apparently meaningful p-value, and it would also consume a rank and inflate the
    genuine comparisons beside it.
    """
    p = np.asarray(p_values, dtype=float)
    adj = np.full(len(p), np.nan)
    tested = np.flatnonzero(~np.isnan(p))
    m = len(tested)
    running = 0.0
    for rank, idx in enumerate(tested[np.argsort(p[tested])]):
        running = max(running, (m - rank) * p[idx])
        adj[idx] = min(1.0, running)
    return adj.tolist()


def paired_wilcoxon(a, b):
    """Wilcoxon signed-rank on paired asset values. Positive median_delta means b is lower."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"paired inputs must match: {a.shape} vs {b.shape}")
    d = a - b
    try:
        p = float(sps.wilcoxon(a, b).pvalue) if np.any(d != 0) else float("nan")
    except ValueError:
        p = float("nan")
    # scipy's default zero_method="wilcox" discards tied pairs, so the test rests on fewer
    # assets than were compared. Report both: n alone would overstate the evidence.
    return {"median_delta": float(np.median(d)), "mean_delta": float(d.mean()),
            "p": p, "n": int(a.size), "n_effective": int(np.count_nonzero(d))}


def cluster_bootstrap_mean(values, asset, n_boot=2000, seed=0):
    """Bootstrap means resampling whole assets with replacement.

    Each asset is averaged first and the asset means are then averaged, so every asset carries
    the same weight. Pooling the rows instead would let an asset that happens to contribute more
    rows - more seeds, more replicates, a longer run - pull a claim that is about assets.
    """
    values = np.asarray(values, dtype=float)
    asset = np.asarray(asset)
    if values.shape != asset.shape:
        raise ValueError("values and asset labels must have the same length")
    per_asset = np.array([values[asset == a].mean() for a in np.unique(asset)])
    rng = np.random.default_rng(seed)
    return per_asset[rng.integers(len(per_asset), size=(n_boot, len(per_asset)))].mean(axis=1)


def cluster_bootstrap_ci(values, asset, n_boot=2000, seed=0, alpha=0.05):
    b = cluster_bootstrap_mean(values, asset, n_boot=n_boot, seed=seed)
    return float(np.percentile(b, 100 * alpha / 2)), float(np.percentile(b, 100 * (1 - alpha / 2)))
