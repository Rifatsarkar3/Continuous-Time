"""Laws of a certified alarm that need no predictor: the odds of a demonstration, and certification in service.

An asset's lead-point value theta* is the running minimum of its signal up to the last inspection at which an alarm is
still on time. The rank-k threshold lies just above the (k+1)-th largest theta* of the assets seen so far, and an asset
is late iff its theta* is at or above the threshold. In service an asset is kept until it fails or until a lead tau
after its alarm, so the theta* of an on-time asset is never known; `run_in_service` uses only what the operator sees.
"""
import heapq
from dataclasses import dataclass
from math import comb

import numpy as np
from scipy import stats as st
from scipy.special import gammaln

binom_cdf = st.binom.cdf


@dataclass(frozen=True)
class Asset:
    name: str
    group: int
    t: np.ndarray      # inspection times, ascending, up to failure
    s: np.ndarray      # signal at those times; the alarm is the first value strictly below the threshold
    T: float           # failure time
    tau: float         # lead: an alarm at time A is on time iff T - A >= tau

    @property
    def lead_value(self):
        ok = self.T - self.t >= self.tau
        return float(np.min(self.s[ok])) if ok.any() else np.inf

    def crossing(self, theta):
        """Index of the first inspection whose signal is below theta, or None."""
        hit = np.flatnonzero(self.s < theta)
        return int(hit[0]) if hit.size else None


def threshold(values, k):
    """Just above the (k+1)-th largest value."""
    return float(np.nextafter(sorted(values, reverse=True)[k], np.inf))


def level_rank(n, alpha):
    """Rank of the level-alpha rule on n assets; negative below the certifiable fleet size."""
    return int(np.floor(round(alpha * (n + 1), 9))) - 1


# --- demonstration ---------------------------------------------------------------------------------------------------

def late_count(n, m, k=0):
    """Law of the number of late assets among m, for the rank-k threshold of n exchangeable assets (no ties)."""
    return st.nhypergeom(n + m, m, k + 1)


def late_tail(n, m, x, k=0):
    """P(at least x late assets among m), exact; scipy's survival function is 1 - cdf and loses small tails."""
    return sum(comb(j + k, k) * comb(n + m - j - k - 1, n - k - 1) for j in range(x, m + 1)) / comb(n + m, n)


def pass_odds(n, m, c=0, k=0):
    """Probability that a demonstration accepting at most c late assets among m passes."""
    return float(late_count(n, m, k).cdf(c))


def demo_size(alpha, delta, c=0):
    """Fewest test assets for which a true risk above alpha passes with probability at most delta."""
    m = c + 1
    while binom_cdf(c, m, alpha) > delta:
        m += 1
    return m


def smallest_fleet(alpha, delta, odds, c_max=30):
    """-> (N, n, m, c): fewest assets in total that certify at rank 0 and pass the plan with the given odds."""
    best = None
    for c in range(c_max + 1):
        m = demo_size(alpha, delta, c)
        n = 1
        while pass_odds(n, m, c) < odds - 1e-9:
            n += 1
            if best and n + m >= best[0]:
                break
        else:
            if best is None or n + m < best[0]:
                best = (n + m, n, m, c)
    return best


# --- in service ------------------------------------------------------------------------------------------------------

def record_probs(N, k=0):
    """P(asset t is late), t = 1..N, for the rank-k rule in service: independent across t."""
    return np.minimum(1.0, (k + 1) / np.arange(1, N + 1))


def count_pmf(p):
    """Law of a sum of independent Bernoulli(p_t)."""
    pmf = np.ones(1)
    for q in p:
        pmf = np.convolve(pmf, [1.0 - q, q])
    return pmf


def run_in_service(fleet, k=0):
    """Operate the rank-k rule on assets in order. Only the failed assets' values are ever used."""
    known, out = [], []
    for n, a in enumerate(fleet):
        theta = threshold(known, k) if n > k else -np.inf
        i = a.crossing(theta)
        late = i is None or a.T - a.t[i] < a.tau
        if late:
            known.append(a.lead_value)               # the asset ran to failure, so its record is complete
        out.append({"asset": a.name, "theta": theta, "late": late, "index": i, **_outcome(a, i, late)})
    return out


def run_level_in_service(fleet, alpha):
    """The level-alpha rule in service; a replaced asset enters at the value it had at its alarm, an upper bound."""
    seen, out = [], []
    for n, a in enumerate(fleet):
        theta = threshold(seen, max(0, level_rank(n, alpha))) if n else -np.inf
        i = a.crossing(theta)
        late = i is None or a.T - a.t[i] < a.tau
        seen.append(a.lead_value if late else float(np.min(a.s[: i + 1])))
        out.append({"asset": a.name, "theta": theta, "late": late, "index": i, **_outcome(a, i, late)})
    return out


def run_overlapping(fleet, slots, k=0):
    """Rank-k rule with `slots` assets in service at a time, entering in order; threshold fixed at entry from the
    assets completed (failed or replaced) by then. Only failed assets' values are used."""
    busy, known, done, clock, out = [], [], 0, 0.0, []
    for n, a in enumerate(fleet):
        if len(busy) == slots:
            clock = busy[0][0]
        while busy and busy[0][0] <= clock:
            v = heapq.heappop(busy)[2]
            done += 1
            known += [] if v is None else [v]
        theta = threshold(known, k) if done > k else -np.inf
        i = a.crossing(theta)
        late = i is None or a.T - a.t[i] < a.tau
        o = _outcome(a, i, late)
        heapq.heappush(busy, (clock + o["time"], n, a.lead_value if late else None))
        out.append({"asset": a.name, "theta": theta, "late": late, "index": i, "completed": done, **o})
    return out


def _outcome(a, i, late):
    """Wasted life at the alarm (0 if none) and time in operation (to failure, or to the alarm plus the lead)."""
    return {"waste": float((a.T - a.t[i]) / a.T) if i is not None else 0.0,
            "time": float(a.T if late else a.t[i] + a.tau)}


def order_weights(N, n, k):
    """P(the (k+1)-th largest of a random n-subset is the j-th largest of N), j = 1..N."""
    j = np.arange(1, N + 1)
    lc = lambda a, b: np.where((b >= 0) & (b <= a), gammaln(np.maximum(a, 0) + 1) - gammaln(np.maximum(b, 0) + 1)
                               - gammaln(np.maximum(a - b, 0) + 1), -np.inf)
    return np.exp(lc(j - 1, np.full(N, k)) + lc(N - j, np.full(N, n - k - 1)) - lc(np.full(N, N), np.full(N, n)))
