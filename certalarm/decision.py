"""Alarm decisions on remaining-life trajectories.

A replicate is (signal, rul, can_fire):
  signal   (L,) decision statistic, lower = closer to failure (e.g. predicted normalized RUL)
  rul      (L,) true remaining life per step, in the unit of `lead`; non-increasing in time
  can_fire (L,) bool, steps where the alarm may fire (acquired measurements); None = every step
The alarm fires at the first step k with can_fire[k] and signal[k] < theta.
Late = no alarm, or remaining life at the alarm < lead. Late is non-increasing in theta.
"""
import numpy as np


def first_crossing(signal, theta, can_fire=None):
    hit = np.asarray(signal) < theta
    if can_fire is not None:
        hit = hit & np.asarray(can_fire, bool)
    idx = np.flatnonzero(hit)
    return int(idx[0]) if idx.size else None


def alarm_outcome(signal, rul, theta, lead, can_fire=None):
    k = first_crossing(signal, theta, can_fire)
    if k is None:
        return True, 0.0
    rem = float(rul[k])
    return rem < lead, rem


def _crossing_index(signal, thetas, can_fire):
    s = np.asarray(signal, float)
    if can_fire is not None:
        s = np.where(np.asarray(can_fire, bool), s, np.inf)
    m = np.minimum.accumulate(s)
    return np.searchsorted(-m, -np.asarray(thetas, float), side="right")


def outcome_curves(signal, rul, thetas, lead, can_fire=None):
    """Vectorised over thetas -> (late (T,) in {0,1}, remaining life at alarm (T,))."""
    thetas = np.asarray(thetas, float)
    if len(signal) == 0:
        return np.ones(thetas.shape), np.zeros(thetas.shape)
    k = _crossing_index(signal, thetas, can_fire)
    L = len(signal)
    fired = k < L
    rem = np.where(fired, np.asarray(rul, float)[np.minimum(k, L - 1)], 0.0)
    late = ~fired | (rem < lead)
    return late.astype(float), rem


def asset_curves(replicates, thetas, lead):
    """Mean late fraction and mean remaining life over the replicates of one asset."""
    outs = [outcome_curves(s, r, thetas, lead, c) for s, r, c in replicates]
    return np.mean([o[0] for o in outs], axis=0), np.mean([o[1] for o in outs], axis=0)
