"""Asset-level conformal risk control (CRC) for alarm thresholds.

loss_matrix[i, j] = late-alarm loss of calibration asset i at thetas[j] (in [0, 1],
non-increasing in theta; thetas ascending). With exchangeable assets, the CRC threshold gives
E[late of a new asset] <= alpha (Angelopoulos et al., Thm 1). Below the floor
B/(n+1) nothing is certifiable.
"""
import math

import numpy as np


def certifiable_floor(n, B=1.0):
    """Smallest risk that can be *certified* from n calibration assets."""
    return B / (n + 1)


def min_assets(alpha, B=1.0):
    return max(1, math.ceil(round(B / alpha - 1, 12)))


def verification_floor(n, ci=0.05):
    """Smallest risk that can be *demonstrated* on n test assets, at confidence 1 - ci.

    Even with no late alarm at all, an exact binomial upper bound on n assets cannot fall below
    1 - (ci/2)^(1/n). This is a separate limit from `certifiable_floor` and it binds later: with
    17 bearings a 10% risk is certifiable (floor 0.056) but can never be shown to have been
    achieved (floor 0.195). Reporting a fleet's certified risk without this number invites the
    reader to assume the experiment could have detected a violation when it could not.
    """
    return 1.0 - (ci / 2.0) ** (1.0 / n)


def min_assets_to_verify(alpha, ci=0.05):
    """Fewest test assets whose zero-event upper bound falls at or below `alpha`."""
    return max(1, math.ceil(round(math.log(ci / 2.0) / math.log1p(-alpha), 12)))


def _check(loss_matrix, thetas=None, tol=1e-12):
    L = np.asarray(loss_matrix, float)
    if L.ndim != 2:
        raise ValueError("loss_matrix must be (n_assets, n_thetas)")
    if np.any(np.diff(L, axis=1) > tol):
        raise ValueError("each asset's loss must be non-increasing in theta")
    if thetas is not None:
        t = np.asarray(thetas, float)
        if t.size != L.shape[1]:
            raise ValueError("thetas length must match the loss matrix columns")
        if np.any(np.diff(t) < 0):
            raise ValueError("thetas must be ascending")
    return L


def crc_threshold(loss_matrix, thetas, alpha, B=1.0):
    L = _check(loss_matrix, thetas)
    n = L.shape[0]
    risk = (n * L.mean(axis=0) + B) / (n + 1)
    ok = np.flatnonzero(risk <= alpha + 1e-12)
    return float(thetas[ok[0]]) if ok.size else None


def empirical_threshold(loss_matrix, thetas, alpha):
    """Uncorrected plug-in rule (no finite-sample term); a baseline, not certified."""
    L = _check(loss_matrix, thetas)
    ok = np.flatnonzero(L.mean(axis=0) <= alpha + 1e-12)
    return float(thetas[ok[0]]) if ok.size else None


def conformal_quantile(scores, alpha):
    s = np.sort(np.asarray(scores, float).ravel())
    if s.size and not np.all(np.isfinite(s)):
        raise ValueError("scores must be finite")
    k = math.ceil(round((len(s) + 1) * (1 - alpha), 12))
    return float(s[k - 1]) if k <= len(s) else math.inf
