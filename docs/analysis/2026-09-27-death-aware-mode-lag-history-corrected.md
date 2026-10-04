# Corrected death-aware mode history experiment

## Method and safeguards

This is an **experimental candidate**, not the production model. The experiment reads only the death-aware `train.npz` and `validation.npz`; `test_access=false`. It uses five-fold `GroupKFold` with complete games held together. For each horizon, it compares the same LightGBM setup using current causal features (`current_raw`) against current features plus corrected, causal history of `base_mode` within each `(game_id, side)` (`with_mode_history`). The report includes training OOF and validation results. Validation confidence intervals compare history directly with current raw, average red and blue within each game, and bootstrap the 122 paired games.

The history features include exact 3/6/9/15-second mode lags, run length based on adjacent three-second samples, change counts over observed consecutive intervals, same-mode ratios, elapsed time since the last transition, and a one-hot six-second lag. Missing timestamps break a run; lag values require the exact timestamp. No JSONL is parsed. The run-length fix is covered by tests for continuous samples, missing timestamps, shuffled input order, and causal independence from future samples (3 tests passed).

## Validation results

| Horizon | Accuracy raw → history (game equal) | Paired accuracy delta (95% CI) | Macro-F1 raw → history (game equal) | Paired Macro-F1 delta (95% CI) | Brier raw → history (game equal) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 10s | 72.24% → 72.50% | +0.27 pp [+0.02, +0.51] | 53.05% → 52.70% | −0.34 pp [−0.79, +0.05] | 0.3976 → 0.3879 |
| 30s | 60.83% → 61.94% | +1.11 pp [+0.40, +1.80] | 37.87% → 37.85% | −0.02 pp [−0.59, +0.53] | 0.5035 → 0.4901 |
| 60s | 57.90% → 58.93% | +1.02 pp [+0.35, +1.73] | 33.36% → 33.61% | +0.26 pp [−0.22, +0.75] | 0.5329 → 0.5247 |
| 120s | 52.71% → 53.52% | +0.81 pp [−0.08, +1.64] | 28.78% → 28.75% | −0.04 pp [−0.79, +0.68] | 0.5765 → 0.5687 |

`pp` means percentage points. These point metrics now use the same game-equal aggregation as their paired intervals. The earlier headline table mixed pooled-row Macro-F1 points with game-equal confidence intervals; its pooled values are retained only under the `validation` key in the machine-readable JSON. The accuracy interval is above zero at 10s, 30s, and 60s, though the 10s lift is small. Macro-F1 intervals include zero at every horizon. Brier improves at every horizon; detailed paired intervals are in the game-equal report.

## Decision

The corrected history features are worth retaining as an **experimental candidate for further evaluation**, especially for probability calibration and accuracy at 30/60s. They are **not ready to replace the current model**: Macro-F1 does not show a reliable gain, and the 10s accuracy gain is only about 0.27 percentage points. Next, inspect per-class precision/recall and calibration across folds before considering any promotion. No production default was changed.

Machine-readable original pooled-row report: [`report.json`](../../out_m2/death_aware_global_mode_experiments/mode_lag_history_corrected/report.json). Recomputed consistent game-equal metrics and paired intervals: [`report.json`](../../out_m2/death_aware_global_mode_experiments/mode_lag_history_game_equal/report.json). Source: [`experiment_death_aware_mode_lag_features.py`](../../eval/experiment_death_aware_mode_lag_features.py). Tests: [`test_mode_lag_history_features.py`](../../tests/test_mode_lag_history_features.py).
