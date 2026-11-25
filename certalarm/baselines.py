"""Comparison alarm rules. None carries a decision-level guarantee."""
import numpy as np

from .certify import conformal_quantile


def _checked_sigma(sigma):
    s = np.asarray(sigma, float)
    if s.size and np.any(~(s > 0)):
        raise ValueError("sigma must be strictly positive (zero or NaN scale hides alarms)")
    return s


def lbc_quantile(mu_list, sigma_list, y_list, alpha):
    """Per-time one-sided conformal quantile of (mu - y)/sigma pooled over calibration time points."""
    s = np.concatenate([(np.asarray(m, float) - np.asarray(y, float)) / _checked_sigma(sg)
                        for m, sg, y in zip(mu_list, sigma_list, y_list)])
    return conformal_quantile(s, alpha)


def sup_score(mu, sigma, y, can_fire=None):
    z = (np.asarray(mu, float) - np.asarray(y, float)) / _checked_sigma(sigma)
    if can_fire is not None:
        z = z[np.asarray(can_fire, bool)]
    return float(z.max()) if z.size else -np.inf


def supband_quantile(asset_sup_scores, alpha):
    """Conformal quantile over per-asset sup scores: one lower band for the whole trajectory."""
    return conformal_quantile(asset_sup_scores, alpha)


def lower_signal(mu, sigma, q):
    return np.asarray(mu, float) - q * np.asarray(sigma, float)


def cost_optimal_threshold(late_matrix, remaining_matrix, thetas, c_late, c_waste):
    cost = c_late * np.asarray(late_matrix, float).mean(0) + c_waste * np.asarray(remaining_matrix, float).mean(0)
    return float(thetas[int(np.argmin(cost))])
