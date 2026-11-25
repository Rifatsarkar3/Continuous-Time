"""Run-to-failure assets on a one-minute grid.

An Asset is one bearing: raw 0.1 s vibration snapshots, the operating condition, and normalized
remaining life. PRONOSTIA is read from the Phase 1 raw rebuild in data/pronostia, never from the
authors' earlier processed arrays, in which E1 found temperature records appended to Bearing1_4 and
Bearing1_5. XJTU-SY is
read from the processed arrays, which E1 verified against raw for all 15 bearings.
"""
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import paths

PER_MIN = {"pronostia": 6, "xjtu": 12}
CONDITIONS = {
    "pronostia": {1: (1800.0, 4000.0), 2: (1650.0, 4200.0), 3: (1500.0, 5000.0)},
    "xjtu": {1: (2100.0, 12000.0), 2: (2250.0, 11000.0), 3: (2400.0, 10000.0)},
}


@dataclass(frozen=True)
class Asset:
    name: str
    dataset: str
    condition: int
    windows: np.ndarray   # (n_min * per_min, samples) float32, whole minutes only
    per_min: int

    @property
    def n_min(self):
        return len(self.windows) // self.per_min

    @property
    def speed_load(self):
        return CONDITIONS[self.dataset][self.condition]

    @property
    def rul(self):
        """Normalized remaining life on the minute grid, 1 at commissioning to 0 at failure."""
        return np.linspace(1.0, 0.0, self.n_min, dtype=np.float32)


def _whole_minutes(windows, per_min, name):
    n = (len(windows) // per_min) * per_min
    if n < per_min:
        raise ValueError(f"{name}: {len(windows)} windows is not at least one whole minute "
                          f"of {per_min}")
    return np.asarray(windows[:n], dtype=np.float32)


def load_pronostia(names=None, data_dir=None):
    d = Path(data_dir) if data_dir is not None else paths.DATA / "pronostia"
    man_path = d / "manifest.json"
    if not man_path.exists():
        raise FileNotFoundError(f"{man_path} not found; run experiments/e1_data_audit.py --rebuild")
    manifest = json.loads(man_path.read_text())
    out = []
    for name in (sorted(manifest) if names is None else names):
        w = np.load(d / f"{name}.npz")["windows"]
        out.append(Asset(name, "pronostia", int(manifest[name]["condition"]),
                          _whole_minutes(w, PER_MIN["pronostia"], name), PER_MIN["pronostia"]))
    return out


def load_xjtu(names=None, proc_dir=None):
    d = Path(proc_dir) if proc_dir is not None else paths.XJTU_PROCESSED
    files = {f.stem.split("_", 2)[2]: f for f in sorted(d.glob("*_Bearing*.npz"))}
    out = []
    for name in (sorted(files) if names is None else names):
        w = np.load(files[name])["vibration"]
        out.append(Asset(name, "xjtu", int(name[7]),
                          _whole_minutes(w, PER_MIN["xjtu"], name), PER_MIN["xjtu"]))
    return out


def load(dataset, names=None):
    return {"pronostia": load_pronostia, "xjtu": load_xjtu}[dataset](names)
