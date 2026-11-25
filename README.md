# Certified late-alarm risk of prognostic alarms: code, trained models and results

Code, trained models and result files for a study of maintenance alarms whose late-alarm risk is controlled at the asset
level by conformal risk control: how many assets, and how many failures, it costs to certify such a risk, to
hold it with confidence, to demonstrate it, and to earn it in service. Fleets: 68 fatigue-crack tests
(Virkler), 124 and 251 lithium-ion cells (MATR, ISU-ILCC), two bearing fleets (PRONOSTIA, XJTU-SY) and the
C-MAPSS turbofan corpus.

## Layout

| Path | Contents |
|---|---|
| `certalarm/` | Library: data loaders and audits, the running-minimum alarm, conformal risk control (`certify.py`), cross-fitting (`crossfit.py`), demonstration and certification in service (`service.py`), parametric tolerance limits (`tolerance.py`), comparison rules, statistics, the write-once evaluation registry (`registry.py`), predictors (`predictors/`) |
| `experiments/` | One script per experiment. Analyses planned in advance say so, with the date, in the module docstring. |
| `tests/` | 287 tests (`python -m pytest`); the three that read a dataset are skipped where it is absent |
| `results/` | Result files, the write-once registry (`registry.jsonl`) and the per-fold checkpoint journals (`checkpoints/`) |
| `data/model_cache/` | 847 trained models (584 CT-GRU, 263 DSAFLite with and without its domain-adversarial head), named by the key of `certalarm/predictors/modelcache.py`; plain weights that load with `torch.load(..., weights_only=True)` |
| `data/README.md` | Where to get the public datasets; no data are redistributed |

## Which script writes which result

| Result | Script |
|---|---|
| `in_service.json`, `in_service_matr.json`, `in_service_overlap.json` | `in_service_certification.py`, `in_service_matr.py`, `in_service_overlap.py` |
| `comparators.json` | `comparators.py` (parametric limits and cost-optimal thresholds) |
| `exact_split_published.json`, `second_fleet_split.json`, `second_fleet_split_final20.json` | `exact_split_published.py`, `second_fleet_split.py` (read saved predictions of an earlier study) |
| `age_baseline.json` | `age_baseline.py` |
| `e0_synthetic_law.json`, `e4_asset_count.json`, `e4_asset_count_robust.json` | `e0_synthetic_law.py`, `e4_asset_count.py` |
| `e2_main_alarms_*.json` | `e2_main_alarms.py` (`run_pronostia.sh`, `run_xjtu.sh`) |
| `c1_leadpoint_xjtu*.json`, `c2_ridge_cmapss.json` | `c1_leadpoint_xjtu.py`, `c2_ridge_cmapss.py` |
| `s6_condition_scaled_cmapss.json`, `rescore_crc_scaled_xjtu.json`, `rescore_scaled_grid.json` | the scripts of the same names |
| `sensitivity_cmapss.json`, `sensitivity_xjtu.json`, `e6_from_checkpoints.json` | the scripts of the same names |
| `e1_data_audit.json` | `e1_data_audit.py` |
| `dev_*.json` | the development scripts of the same names |

## Environment

Python 3.14 with `requirements.txt`. The XJTU-SY arms and their defect-corrected re-score ran with
torch 2.11.0+cu128 on a 12 GB GPU; CT-GRU trains on the CPU.

## Reproducing

```bash
python -m pytest                                     # unit tests
python -m certalarm.data.fleets                      # downloads Virkler, ISU-ILCC and MATR (about 2 min)
python experiments/in_service_matr.py                # 124 cells measured at every cycle; needs C-MAPSS (6 s)
python experiments/in_service_certification.py       # five fleets; needs C-MAPSS and both bearing sets (15 s)
python experiments/in_service_overlap.py             # overlapping service (post hoc)
python experiments/comparators.py                    # parametric limits, cost-optimal thresholds; needs C-MAPSS (15 s)
python experiments/e4_asset_count.py                 # C-MAPSS asset-count law
bash run_pronostia.sh                                # PRONOSTIA arms (about 30 h)
bash run_xjtu.sh                                     # XJTU-SY arms
python experiments/sensitivity_cmapss.py             # alpha and lead sensitivity
```

The models in `data/model_cache/` are keyed by predictor, dataset, seed, the exact set of fit assets and every
training setting, so a cached model is never reused for a fold in which it would have seen a forbidden asset. The registry permits one test evaluation per configuration, so a full re-run of the bearing
arms needs `results/registry.jsonl` and `results/checkpoints/` moved away first.

## Integrity mechanisms

- **Write-once registry.** `certalarm/registry.py` permits exactly one test evaluation per (configuration,
  trained weights, split) and refuses a second; `results/registry.jsonl` records every claim with its time.
- **Checkpoints.** Each finished fold is journaled in `results/checkpoints/` before the next starts.
- **Plans before runs.** Hypotheses, criteria and analyses were fixed before the runs they govern. Commit hashes in
  docstrings refer to the development history, which is not public.
- **Disclosed corrections.** A scoring defect in the condition-scaled rule was found, fixed and re-scored under a
  plan fixed before the re-score (`experiments/rescore_crc_scaled_xjtu.py`).

## License

No license has been granted yet; please contact the authors before reusing the code.
