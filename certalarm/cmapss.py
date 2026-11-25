"""C-MAPSS turbofan units as run-to-failure assets.

The bearing fleets are too small to verify the risk they certify: at 17 assets the verification
floor is 0.195, and demonstrating alpha = 0.10 needs 36. C-MAPSS has 709 training units, so it is
the only fleet here where a certified risk level can also be shown to have been achieved. That is
what it is for - not a third benchmark, but the one that closes the gap between `certifiable_floor`
and `verification_floor`.

Only the *training* files are used. The test files are truncated part-way through life, so they
have no failure time, and an alarm's lateness is undefined without one.

A cycle plays the role a minute plays for the bearings, and the 24 recorded channels play the role
of the 18 computed vibration features, so `features.fit_scale` and `features.normalize` apply
unchanged: `baseline_correct` removes each unit's own offset and a constant channel collapses to
zero rather than dividing by zero.

Nothing in the running PRONOSTIA grid imports this module.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import paths
from .features import BASELINE_MIN

SUBSETS = (1, 2, 3, 4)
# A unit shorter than the baseline window cannot be baseline-corrected, so the loader refuses to
# emit one rather than handing the feature pipeline something it will reject later. The shortest
# real C-MAPSS training unit is well above this, so nothing is dropped in practice.
MIN_CYCLES = BASELINE_MIN + 1
N_SETTINGS, N_SENSORS = 3, 21
DIR = paths.DATA / "cmapss" / "CMAPSSData"


@dataclass(frozen=True)
class Unit:
    """One turbofan, commissioning to failure. Duck-typed to match `assets.Asset`."""

    name: str
    dataset: str
    condition: int
    grid: np.ndarray      # (n_cycles, 24) float32: 3 operating settings + 21 sensors

    @property
    def n_min(self):
        return len(self.grid)

    @property
    def rul(self):
        """Normalized remaining life, 1 at the first recorded cycle to 0 at failure."""
        return np.linspace(1.0, 0.0, self.n_min, dtype=np.float32)


def _read(path):
    """-> (n_rows, 26) array: unit, cycle, 3 settings, 21 sensors."""
    raw = np.loadtxt(path)
    expected = 2 + N_SETTINGS + N_SENSORS
    if raw.ndim != 2 or raw.shape[1] != expected:
        raise ValueError(f"{path.name}: expected {expected} columns, got {raw.shape}")
    return raw


def load_subset(subset, data_dir=None, min_cycles=MIN_CYCLES):
    """Every training unit of one FD subset, in unit order.

    `condition` is the subset index. The four subsets differ in fault mode and in how many
    operating regimes they mix, which is a real grouping of the fleet; within FD002 and FD004 the
    regime changes cycle to cycle, so no single regime label describes a unit.
    """
    d = Path(data_dir) if data_dir is not None else DIR
    path = d / f"train_FD{subset:03d}.txt"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found")
    raw = _read(path)
    out = []
    for uid in np.unique(raw[:, 0]).astype(int):
        rows = raw[raw[:, 0] == uid]
        rows = rows[np.argsort(rows[:, 1])]        # by cycle; the files are ordered but not promised to be
        if len(rows) < min_cycles:
            continue
        out.append(Unit(name=f"FD{subset:03d}_u{uid}", dataset="cmapss", condition=subset,
                        grid=np.ascontiguousarray(rows[:, 2:], dtype=np.float32)))
    return out


def load(subsets=SUBSETS, data_dir=None):
    """The pooled fleet across the requested subsets."""
    fleet = []
    for s in subsets:
        fleet += load_subset(s, data_dir=data_dir)
    if not fleet:
        raise ValueError("no units loaded")
    return fleet


def condition_of(name):
    """-> the subset index encoded in a unit name, for condition-stratified folds."""
    return int(name[2:5])
