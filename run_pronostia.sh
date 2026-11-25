#!/usr/bin/env bash
# Full PRONOSTIA grid. Every stage checkpoints each fold and resumes by skipping finished ones,
# so re-running this script after any interruption picks up where it stopped. Retries are safe
# for the same reason: a finished fold is never re-evaluated, so the write-once registry stays
# intact.
P="${PYTHON:-python}"
S="42 123 777 999 7"
run_stage () {            # name, then the experiment's own arguments
  local name="$1"; shift
  for attempt in 1 2 3; do
    echo "=== $name (attempt $attempt) ==="
    if "$P" experiments/e2_main_alarms.py --seeds $S --datasets pronostia "$@"; then
      echo "=== $name COMPLETE ==="
      return 0
    fi
    echo "=== $name failed on attempt $attempt; resuming from the checkpoint ===" >&2
  done
  echo "=== $name FAILED after 3 attempts ===" >&2
  return 1
}
run_stage "1/3 CT-GRU"        --tag pron_ctgru                        || exit 1
run_stage "2/3 DSAFLite"      --no-ctgru --dsaflite --tag pron_dsaflite || exit 1
run_stage "3/3 DSAFLite+DANN" --no-ctgru --dann     --tag pron_dann     || exit 1
echo "=== ALL THREE STAGES COMPLETE ==="
