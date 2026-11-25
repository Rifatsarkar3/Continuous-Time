"""Per-fold result checkpoints, so an interrupted run resumes instead of restarting.

A run is a sequence of independent units - one (dataset, predictor, process, seed, fold) each -
and every unit's rows are appended to a journal as soon as they exist. A later run of the same
tag reads the journal, reuses those rows and skips the work behind them.

Skipping matters for more than time. A completed unit has already claimed its one permitted test
evaluation in the registry, so re-running it would be refused; resuming by skipping keeps the
write-once guarantee intact instead of forcing anyone to clear the journal that enforces it.
"""
import json
import os
from pathlib import Path

from . import paths

DIR = paths.RESULTS / "checkpoints"


def path(tag):
    return DIR / f"{tag or 'run'}.jsonl"


def unit_key(unit):
    """Stable identity of one unit of work."""
    return "|".join(f"{k}={unit[k]}" for k in sorted(unit))


def completed(tag):
    """-> {unit_key: rows} for every unit already finished under this tag."""
    p = path(tag)
    if not p.exists():
        return {}
    done = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            # A crash mid-write can leave one truncated final line; the unit simply counts as
            # unfinished and is redone. Earlier lines stay usable.
            continue
        done[rec["key"]] = rec["rows"]
    return done


def append(tag, unit, rows):
    """Record one finished unit. Flushed and fsynced so a crash cannot lose it."""
    DIR.mkdir(parents=True, exist_ok=True)
    rec = {"key": unit_key(unit), "unit": unit, "rows": rows}
    with path(tag).open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, default=float) + "\n")
        f.flush()
        os.fsync(f.fileno())
