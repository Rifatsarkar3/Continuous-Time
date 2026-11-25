#!/usr/bin/env bash
# XJTU confirmatory run. Its test bearings are a one-shot resource and this spends them.
#
# Scope decisions, made before launch and recorded here so they are not re-chosen after seeing
# results:
#   * Two predictor arms (CT-GRU, DSAFLite). The claims XJTU must replicate - that validity gates
#     are uninformative at this fleet size, and that CRC's realized risk is stable under a
#     predictor change where the baselines' is not - both need at least two arms. Two is enough.
#   * DANN is NOT replicated. PRONOSTIA established it as a null (no benefit, Holm p = 0.24) and
#     replicating a null is not worth a third of the budget. Adding it later would be a second
#     touch of the same test bearings, so it is declined now rather than deferred.
#   * `full` observation process only. The other two feed E6, which is descriptive (S5) and is
#     already answered on PRONOSTIA.
#   * Three seeds. Paired tests are over the 15 bearings; seeds only reduce per-bearing noise.
#
# All seven rules are fitted in this one pass, including crc_scaled, so the test set is scored
# exactly once.
P="${PYTHON:-python}"
S="42 123 777"
run_stage () {
  local name="$1"; shift
  for attempt in 1 2 3; do
    echo "=== $name (attempt $attempt) ==="
    if "$P" experiments/e2_main_alarms.py --seeds $S --datasets xjtu --processes full "$@"; then
      echo "=== $name COMPLETE ==="
      return 0
    fi
    echo "=== $name failed on attempt $attempt; resuming from the checkpoint ===" >&2
  done
  echo "=== $name FAILED after 3 attempts ===" >&2
  return 1
}
run_stage "1/2 XJTU CT-GRU"   --tag xjtu_ctgru                        || exit 1
run_stage "2/2 XJTU DSAFLite" --no-ctgru --dsaflite --tag xjtu_dsaflite || exit 1
echo "=== XJTU BOTH STAGES COMPLETE ==="
