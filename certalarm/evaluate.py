"""From trajectories to thresholds to outcomes.

Six alarm rules are fitted on the calibration assets of a fold and applied to its test assets.
Only `crc` carries a guarantee; the rest are comparisons. When `crc` cannot certify at the
requested alpha - which happens below the certifiability floor of 1/(n+1) - the threshold is
None and that is reported as an outcome, never replaced by another rule.
"""
import numpy as np

from . import baselines, certify, stats
from .decision import outcome_curves

THETAS = np.linspace(0.0, 1.0, 201)
POINT_THETA = 0.15
COST_LATE, COST_WASTE = 10.0, 1.0
RULES = ("point", "lbc", "supband", "cost_optimal", "empirical", "crc", "crc_scaled")


def _asset_curves(reps, thetas, lead, signal_of):
    late, rem = [], []
    for t in reps:
        l, r = outcome_curves(signal_of(t), t.rul, thetas, lead, t.can_fire)
        late.append(l)
        rem.append(r)
    return np.mean(late, axis=0), np.mean(rem, axis=0)


def loss_matrix(calibration, thetas, lead, signal_of=lambda t: t.mu):
    """(n_assets, n_thetas) mean late indicator per calibration asset, non-increasing in theta."""
    return np.stack([_asset_curves(reps, thetas, lead, signal_of)[0]
                     for _, reps in sorted(calibration.items())])


def waste_matrix(calibration, thetas, lead, signal_of=lambda t: t.mu):
    return np.stack([_asset_curves(reps, thetas, lead, signal_of)[1]
                     for _, reps in sorted(calibration.items())])


def _lbc_shift(calibration, alpha):
    mu = [t.mu for reps in calibration.values() for t in reps]
    sg = [t.sigma for reps in calibration.values() for t in reps]
    y = [t.rul for reps in calibration.values() for t in reps]
    return baselines.lbc_quantile(mu, sg, y, alpha)


def _supband_shift(calibration, alpha):
    """Conformal quantile over per-asset sup scores.

    `sup_score` already maximises over time, which is what makes the band simultaneous. The
    replicates are averaged, not maximised: they are repeated draws of which minutes happen to be
    observed, and a test asset's outcome is itself an average over its own replicates. Taking a
    second maximum here would calibrate against each asset's worst draw, inflate the band, and
    make this baseline waste more life than it should - flattering the rule it is compared with.
    """
    scores = [float(np.mean([baselines.sup_score(t.mu, t.sigma, t.rul, t.can_fire) for t in reps]))
              for reps in calibration.values()]
    return baselines.supband_quantile(scores, alpha)


def _scaled_signal(t):
    """mu divided by the trajectory's own condition scale - the rule `mu < theta * s_k`.

    The scale rides on the trajectory because in CV-CRC each trajectory belongs to one inner
    model, and must use the scales estimated from that model's fit assets. A single fold-level
    scale would mix models and reintroduce the dependence the construction exists to avoid.
    """
    return np.asarray(t.mu, float) / (t.scale if t.scale > 0 else 1.0)


def fit_rules(calibration, alpha, lead, thetas=THETAS):
    """Fit every rule on the calibration assets. Returns {rule: {theta, shift, certifiable}}."""
    L = loss_matrix(calibration, thetas, lead)
    W = waste_matrix(calibration, thetas, lead)
    q_lbc = _lbc_shift(calibration, alpha)
    q_sup = _supband_shift(calibration, alpha)
    crc = certify.crc_threshold(L, thetas, alpha)
    # Condition-scaled CRC calibrates the same single scalar over the same calibration assets,
    # so it keeps the pooled floor 1/(n+1) rather than the stratified 1/(n_k+1).
    # mu / s exceeds 1 when s < 1, so the grid must reach above the largest scaled signal or the
    # certificate that exists for n >= 1/alpha - 1 cannot be found; values up to 1 keep the original grid.
    top = max(float(np.max(_scaled_signal(t))) for reps in calibration.values() for t in reps)
    ts = thetas if top < thetas[-1] else np.concatenate(
        [thetas, np.arange(thetas[-1] + 0.005, np.ceil(top / 0.005) * 0.005 + 0.0101, 0.005)])
    Ls = loss_matrix(calibration, ts, lead, signal_of=_scaled_signal)
    crc_scaled = certify.crc_threshold(Ls, ts, alpha)
    return {
        "point": {"theta": POINT_THETA, "shift": 0.0, "certifiable": False},
        "lbc": {"theta": POINT_THETA, "shift": float(q_lbc), "certifiable": False},
        "supband": {"theta": POINT_THETA, "shift": float(q_sup), "certifiable": False},
        "cost_optimal": {"theta": baselines.cost_optimal_threshold(L, W, thetas, COST_LATE,
                                                                  COST_WASTE),
                         "shift": 0.0, "certifiable": False},
        "empirical": {"theta": certify.empirical_threshold(L, thetas, alpha), "shift": 0.0,
                      "certifiable": False},
        "crc": {"theta": crc, "shift": 0.0, "certifiable": crc is not None,
                "n_calibration": int(L.shape[0]),
                "floor": float(certify.certifiable_floor(L.shape[0]))},
        "crc_scaled": {"theta": crc_scaled, "shift": 0.0, "scaled": True,
                       "certifiable": crc_scaled is not None,
                       "n_calibration": int(Ls.shape[0]),
                       "floor": float(certify.certifiable_floor(Ls.shape[0]))},
    }


def apply_rules(rules, test, lead):
    """-> {rule: {asset: {late, waste}}}. An unfitted threshold counts as never alarming."""
    out = {}
    for name, spec in rules.items():
        theta, shift = spec["theta"], spec["shift"]
        per_asset = {}
        for asset, reps in sorted(test.items()):
            if theta is None:
                per_asset[asset] = {"late": 1.0, "waste": 0.0}
                continue
            if spec.get("scaled"):
                # Calibrated on mu / s_k, so scored on mu / s_k: the rule is mu < theta * s_k.
                signal_of = _scaled_signal
            elif shift == 0.0:
                signal_of = lambda t: t.mu
            else:
                signal_of = lambda t, s=shift: baselines.lower_signal(t.mu, t.sigma, s)
            late, rem = _asset_curves(reps, np.array([theta]), lead, signal_of)
            per_asset[asset] = {"late": float(late[0]), "waste": float(rem[0])}
        out[name] = per_asset
    return out


def summarize(per_asset, cluster_key=None, alpha_ci=0.05, n_boot=2000, seed=0):
    """Pool assets: realized late rate with the more conservative of two 95% upper bounds.

    Each asset's late value is a mean over observation replicates and so is fractional. The
    asset-clustered bootstrap takes the fractions as they are; Clopper-Pearson needs whole
    counts and is computed on the rounded per-asset outcomes. Both are reported and the gate
    reads `late_upper`, the larger of the two, so the choice of instrument cannot flatter the
    result. Counting each replicate as an independent trial is not an option: the replicates of
    one bearing share a bearing and a model, and the interval would come out far too narrow.

    `cluster_key` maps each row to the asset it belongs to, and the bootstrap resamples those
    assets. Pass it whenever the rows are finer than one row per asset - pooling several seeds or
    folds of the same bearing, say - because the rows of one bearing are not independent draws.
    Leaving it out clusters on the row keys themselves, which is correct only when each key
    already is a distinct asset.
    """
    names = sorted(per_asset)
    late = np.array([per_asset[n]["late"] for n in names])
    waste = np.array([per_asset[n]["waste"] for n in names])
    clusters = np.array([(cluster_key or {}).get(n, n) for n in names])
    # The asset is the unit for both instruments, so rows are collapsed to one value per asset
    # before either bound is computed. Leaving several rows of one bearing in place would let a
    # bearing that appears under more seeds count as several independent observations.
    uniq = np.unique(clusters)
    by_asset = np.array([late[clusters == c].mean() for c in uniq])
    waste_by_asset = np.array([waste[clusters == c].mean() for c in uniq])
    _, boot_hi = stats.cluster_bootstrap_ci(late, clusters, n_boot=n_boot, seed=seed,
                                            alpha=alpha_ci)
    # An asset counts as late if it is late in any replicate: rounding the fraction would let a fleet
    # late 40% of the time per asset report zero late assets. Conservative, hence a valid upper bound.
    cp_lo, cp_hi = stats.clopper_pearson(int((by_asset > 1e-12).sum()), len(uniq), alpha=alpha_ci)
    return {"n_assets": int(len(uniq)), "n_rows": len(names),
            "late_rate": float(by_asset.mean()),
            "late_upper": float(max(boot_hi, cp_hi)),
            "late_upper_bootstrap": float(boot_hi), "late_upper_binomial": float(cp_hi),
            "late_lower_binomial": float(cp_lo),
            "waste_mean": float(waste_by_asset.mean()),
            "waste_median": float(np.median(waste_by_asset))}
