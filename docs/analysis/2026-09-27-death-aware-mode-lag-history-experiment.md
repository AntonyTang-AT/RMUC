# Death-aware global mode: causal history feature experiment

## Scope and comparison

This is an **experimental candidate only**. It reads only `train.npz` and `validation.npz` from `out_m2/temporal_forecast_death_aware_candidate`; `test_access=false`. It does not modify the production model or defaults.

The comparison is the same compact LightGBM setup with current causal features (`current_raw`) versus those features plus same-`(game_id, side)` historical `base_mode` features (`with_mode_history`). History uses exact timestamps at 3/6/9/15 seconds before issue time, past change counts, same-mode ratios and time since transition. Five-fold `GroupKFold` keeps complete games together; both red and blue sides stay in a game's fold. Reported confidence intervals bootstrap paired game-level red/blue averaged metrics, comparing the history model directly with the current-raw model. They are **not** comparisons against persistence.

## Validation results

| Horizon | Current raw accuracy | + history accuracy | Paired accuracy delta (95% CI) | Current raw Macro-F1 | + history Macro-F1 | Paired Macro-F1 delta (95% CI) | Brier: raw → history |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10s | 72.23% | 72.42% | +0.19 pp [−0.05, +0.44] | 71.42% | 71.71% | −0.43 pp [−0.86, −0.03] | 0.3977 → 0.3883 |
| 30s | 60.82% | 62.01% | +1.19 pp [+0.52, +1.86] | 58.85% | 60.34% | +0.21 pp [−0.37, +0.78] | 0.5037 → 0.4900 |
| 60s | 57.89% | 58.97% | +1.07 pp [+0.43, +1.76] | 55.01% | 56.23% | +0.36 pp [−0.17, +0.88] | 0.5331 → 0.5247 |
| 120s | 52.71% | 53.20% | +0.49 pp [−0.31, +1.32] | 49.80% | 50.30% | −0.23 pp [−0.96, +0.47] | 0.5766 → 0.5701 |

`pp` means percentage points. Accuracy's paired game interval is positive at 30s and 60s, but Macro-F1's interval includes zero at every horizon; at 10s, its interval is below zero. The Brier score improves at each horizon. This supports a possible probability-quality benefit and a horizon-specific accuracy signal, but not a robust overall categorical prediction improvement.

## Implementation audit and decision

After this run, source review found a run-length implementation defect: its continuity check compared the current timestamp with each older timestamp, so the emitted value only represented whether the immediately preceding sample matched (1 or 2), rather than the full run. Exact lag values, change counts, same-mode ratios and elapsed time since transition are causal. The source has been corrected, but these metrics were not recomputed with the corrected run-length calculation. The report explicitly records this audit.

**Decision: do not promote.** Keep the candidate experimental. The next confirmatory run should regenerate the feature cache from the corrected implementation, compare current raw vs corrected-history on the same whole-game folds, and leave validation as a single confirmation. Do not interpret the present result as proving the corrected full run-length feature.

Machine-readable metrics: [`report.json`](../../out_m2/death_aware_global_mode_experiments/mode_lag_history/report.json). Reproduction script: [`experiment_death_aware_mode_lag_features.py`](../../eval/experiment_death_aware_mode_lag_features.py).
