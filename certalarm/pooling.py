"""Condition-scaled CRC: keep one global scalar, move condition structure into the rule.

Pre-registered on 2026-09-20.

The alarm becomes `mu_t < theta * s_k` for an asset in condition k, so CRC still calibrates a
single scalar over the whole calibration set and keeps the floor `1/(n+1)`. Running CRC per
condition instead would raise the floor to `1/(n_k+1)` - 0.143 at six bearings per condition,
above the alpha = 0.10 we need - which is why the scale lives in the rule and not in the split.

The scales are estimated on the *training* assets only. Estimating them on the calibration assets
would make each calibration loss depend on the whole calibration set, break the exchangeability
CRC rests on, and yield a threshold that is too loose. `condition_scales` therefore refuses any
asset that is not in the declared fit set rather than trusting the caller to pass the right one.

Nothing in the running experiment imports this module. It is wired in only after the PRONOSTIA
grid finishes, so that stage 3 runs the same code stages 1 and 2 did.
"""
import numpy as np

DEFAULT_SCALE = 1.0


def scale_at_lead(traj, lead):
    """The predicted remaining life at the minute true remaining life first reaches `lead`.

    This is the reading the threshold is trying to catch, so a condition where the predictor sits
    systematically high produces a proportionally higher scale. If the asset never reaches `lead`
    (a truncated record), the last minute is used.
    """
    below = np.flatnonzero(np.asarray(traj.rul, float) <= lead)
    i = int(below[0]) if below.size else len(traj.rul) - 1
    return float(np.asarray(traj.mu, float)[i])


def condition_scales(fit, lead, fit_names=None):
    """-> {condition: scale}, normalized so the fleet median scale is 1.

    `fit` maps an asset name to {"condition": c, "reps": [trajectory, ...]}. `fit_names`, when
    given, is the authoritative training set: anything outside it is refused, because a scale
    estimated from a calibration or test asset would silently void the guarantee.
    """
    if fit_names is not None:
        stray = sorted(set(fit) - set(fit_names))
        if stray:
            raise ValueError(f"scales may only be estimated on training assets; got {stray}")
    per_condition = {}
    for name, rec in sorted(fit.items()):
        vals = [scale_at_lead(t, lead) for t in rec["reps"]]
        per_condition.setdefault(rec["condition"], []).append(float(np.mean(vals)))
    if not per_condition:
        return {}
    raw = {c: float(np.median(v)) for c, v in per_condition.items()}
    mid = float(np.median(list(raw.values())))
    if not np.isfinite(mid) or mid <= 0:
        return {c: DEFAULT_SCALE for c in raw}
    # A non-positive scale would flip the inequality and destroy monotonicity in theta, so it is
    # replaced rather than allowed through to certify._check as a confusing failure.
    return {c: (v / mid if np.isfinite(v) and v > 0 else DEFAULT_SCALE) for c, v in raw.items()}


def signal_for(scales, condition):
    """-> signal_of(traj) dividing mu by the condition's scale.

    `mu / s_k < theta` is the same rule as `mu < theta * s_k`, and in this form it drops straight
    into the existing curve machinery, which already takes a `signal_of`.
    """
    s = scales.get(condition, DEFAULT_SCALE)
    if not np.isfinite(s) or s <= 0:
        s = DEFAULT_SCALE
    return lambda t, s=s: np.asarray(t.mu, float) / s


def spread_within_and_pooled(fit, lead):
    """-> (pooled_sd, within_condition_sd) of the per-asset scale readings.

    This is prediction (2) of the pre-registration, and it is checkable on
    calibration folds alone. If the within-condition spread is not smaller than the pooled
    spread, the condition grouping explains none of the heterogeneity, the construction cannot
    help, and no test evaluation should be spent on it.
    """
    by_condition, all_vals = {}, []
    for name, rec in sorted(fit.items()):
        v = float(np.mean([scale_at_lead(t, lead) for t in rec["reps"]]))
        by_condition.setdefault(rec["condition"], []).append(v)
        all_vals.append(v)
    if len(all_vals) < 2:
        return 0.0, 0.0
    pooled = float(np.std(all_vals, ddof=1))
    # Residual spread after removing each condition's own mean - the part a per-condition scale
    # cannot explain. Conditions with a single asset contribute no residual and no degrees of
    # freedom, so they are dropped rather than counted as a perfect fit.
    resid, dof = [], 0
    for vals in by_condition.values():
        if len(vals) < 2:
            continue
        resid += list(np.asarray(vals, float) - np.mean(vals))
        dof += len(vals) - 1
    within = float(np.sqrt(np.sum(np.square(resid)) / dof)) if dof else pooled
    return pooled, within
