"""Minute-grid vibration features for the CT-GRU predictor.

Eighteen features per 0.1 s window (time-domain shape, eight log band energies, spectral
centroid and entropy, envelope kurtosis), averaged over the windows of each minute. Normalization
is two-stage: subtract the bearing's own commissioning baseline, then divide by a scale fitted on
the inner-fit bearings of the current fold. The scale is always passed in, never global, so a
fold cannot borrow statistics from bearings it is not allowed to see.
"""
from pathlib import Path

import numpy as np

FS = 25600.0
BASELINE_MIN = 10

FEATURE_NAMES = (
    ["log_rms", "log_peak", "log_kurt", "skew", "log_crest", "log_impulse", "log_shape"]
    + [f"log_band{i}" for i in range(8)]
    + ["centroid", "spec_entropy", "log_env_kurt"]
)


def window_features(w):
    """w: (N, samples) raw windows -> (N, 18) float32 features."""
    x = np.asarray(w, dtype=np.float64)
    x = x - x.mean(axis=1, keepdims=True)
    ax = np.abs(x)
    rms = np.sqrt((x ** 2).mean(1)) + 1e-12
    peak = ax.max(1) + 1e-12
    mabs = ax.mean(1) + 1e-12
    kurt = (x ** 4).mean(1) / rms ** 4
    skew = (x ** 3).mean(1) / rms ** 3
    P = (np.abs(np.fft.rfft(x, axis=1)) ** 2)[:, 1:]
    freqs = np.fft.rfftfreq(x.shape[1], 1.0 / FS)[1:]
    band_e = np.stack([P[:, b].sum(1) for b in np.array_split(np.arange(P.shape[1]), 8)], 1)
    Pn = P / (P.sum(1, keepdims=True) + 1e-12)
    centroid = (Pn * freqs).sum(1) / (FS / 2)
    entropy = -(Pn * np.log(Pn + 1e-12)).sum(1) / np.log(P.shape[1])
    h = np.zeros(x.shape[1])
    h[0] = 1
    h[1:(x.shape[1] + 1) // 2] = 2
    if x.shape[1] % 2 == 0:
        h[x.shape[1] // 2] = 1
    env = np.abs(np.fft.ifft(np.fft.fft(x, axis=1) * h, axis=1))
    env = env - env.mean(1, keepdims=True)
    env_kurt = (env ** 4).mean(1) / ((env ** 2).mean(1) ** 2 + 1e-12)
    f = np.stack([np.log(rms), np.log(peak), np.log(kurt), np.clip(skew, -5, 5),
                  np.log(peak / rms), np.log(peak / mabs), np.log(rms / mabs)], 1)
    return np.concatenate([f, np.log(band_e + 1e-12), centroid[:, None], entropy[:, None],
                           np.log(env_kurt)[:, None]], 1).astype(np.float32)


def minute_features(windows, per_min, chunk=512):
    """(n_win, samples) raw windows -> (n_min, 18); a partial final minute is dropped."""
    feats = np.concatenate([window_features(windows[i:i + chunk])
                            for i in range(0, len(windows), chunk)])
    n = len(feats) // per_min
    return feats[:n * per_min].reshape(n, per_min, -1).mean(1).astype(np.float32)


def asset_grid(dataset, name, windows, per_min, cache_dir):
    """Cached minute grid for one asset. `windows` may be None when the cache exists."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{dataset}_{name}.npy"
    if path.exists():
        return np.load(path)
    if windows is None:
        raise FileNotFoundError(f"{path} not cached and no windows given")
    grid = minute_features(windows, per_min)
    np.save(path, grid)
    return grid


def baseline_correct(g):
    """Subtract the bearing's own first-BASELINE_MIN-minute mean."""
    g = np.asarray(g, dtype=np.float32)
    if len(g) < BASELINE_MIN:
        raise ValueError(f"need at least {BASELINE_MIN} minutes for the baseline, got {len(g)}")
    return g - g[:BASELINE_MIN].mean(0, keepdims=True)


def fit_scale(grids):
    """Per-feature scale from the given grids only. Pass inner-fit bearings, never calibration."""
    return (np.concatenate([baseline_correct(g) for g in grids]).std(0) + 1e-6).astype(np.float32)


def normalize(g, scale):
    return np.clip(baseline_correct(g) / scale, -10, 10).astype(np.float32)
