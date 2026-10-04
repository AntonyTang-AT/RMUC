# Death-aware global-mode candidate check

## Scope and decision

Only `out_m2/temporal_forecast_death_aware_candidate/train.npz` and `validation.npz` were opened. The report contains `test_access: false`; no test split, production checkpoint, or default runtime setting was changed. Current decision: **do not promote**. Keep the LightGBM option experimental for the 60-second accuracy use case only; it does not establish a reliable Macro-F1 gain. The warm-start GRU is rejected.

## LightGBM candidate

The feature set is causal and compact: current 232-dim snapshot, trailing 3/6 snapshot means, current-minus-1/3-snapshot deltas, and the current mode. Five `GroupKFold` folds held out complete games; red and blue views of a game stayed in the same fold. Horizon temperature scaling and the persistence blend were selected only on train OOF predictions. The chosen model/persistence weights were 1.00 / 0.95 / 1.00 / 1.00 at 10/30/60/120 seconds. Validation used the frozen 122-game split once. All probability comparisons use a smoothed persistence distribution (0.98 current class; 0.01 for each other class).

| Horizon | Train OOF candidate acc / Macro-F1 (pooled rows) | Train OOF persistence acc / Macro-F1 (pooled rows) | Validation candidate acc / Macro-F1 (game equal) | Validation persistence acc / Macro-F1 (game equal) |
|---:|---:|---:|---:|---:|
| 10s | 69.15% / 69.09% | 68.63% / 68.69% | 72.30% / 52.45% | 72.25% / 53.16% |
| 30s | 58.36% / 57.80% | 56.96% / 56.96% | 61.14% / 37.98% | 60.39% / 38.59% |
| 60s | 54.64% / 53.39% | 52.90% / 52.93% | 57.84% / 33.23% | 55.89% / 33.73% |
| 120s | 51.74% / 50.20% | 47.73% / 47.84% | 51.43% / 28.25% | 50.23% / 28.72% |

Validation point metrics above are the mean of per-game scores after giving red/blue sides equal weight within each game. The previous pooled-row Macro-F1 values are not comparable to the paired-game confidence intervals and are superseded here. Pooled row-level metrics remain available in `out_m2/death_aware_global_mode_experiments/lightgbm_game_equal/report.json` under `validation`.

Paired bootstrap resampled whole games and averaged the two sides per game:

| Horizon | Validation accuracy delta, pp (95% CI) | Validation Macro-F1 delta, pp (95% CI) | Brier delta (95% CI) |
|---:|---:|---:|---:|
| 10s | +0.04 [−0.33, +0.43] | −0.72 [−1.25, −0.24] | −0.150 [−0.171, −0.131] |
| 30s | +0.75 [−0.56, +2.00] | −0.61 [−1.42, +0.19] | −0.272 [−0.302, −0.243] |
| 60s | **+1.95 [+0.16, +3.71]** | −0.50 [−1.63, +0.60] | −0.323 [−0.359, −0.288] |
| 120s | +1.20 [−1.50, +3.82] | −0.47 [−1.97, +0.96] | −0.387 [−0.434, −0.338] |

The candidate gives a confirmed validation accuracy improvement only at 60s. Its Macro-F1 difference there includes zero and is slightly negative; the same negative Macro-F1 direction appears in train OOF at 30/60s. Keep that 60s model experimental and do not replace the baseline. Other horizons do not show robust validation accuracy lift. The Brier/log-loss/ECE improvements show useful probability ranking/calibration versus a hard-to-calibrate persistence guess, but do not substitute for the missing Macro-F1 lift.

## Warm-start GRU

Warm-started the canonical GRU from its saved checkpoint, trained only the death-aware global-mode objective, and early-stopped on validation equal-side mean (0.6 accuracy + 0.4 Macro-F1). Best epoch was 1; the next three epochs degraded the selection score. Its validation accuracies were 61.24/55.62/52.73/49.63% versus persistence 72.24/60.37/55.87/50.21%. Game-paired accuracy deltas were −10.99pp [−13.20, −8.90], −4.75pp [−7.27, −2.48], −3.14pp [−5.86, −0.56], and −0.57pp [−3.41, +2.17]. This rejects “just train the random-start GRU longer” and “warm-start alone solves label drift.”

## What likely caused the GRU failure

The death-aware relabel changed 9.48/9.96/10.53/11.72% of the old targets at 10/30/60/120 seconds. The initial death-aware checkpoint was random initialized and trained only three epochs with a shared multi-task loss. The class weights are all close to 1, so class imbalance weighting is not the explanation. Warm-starting and isolating the global-mode objective still failed badly. The evidence points to **target-contract drift plus a weak input-to-target signal**, not simply insufficient epochs or the GRU size. The current labels are a thresholded, side-relative future score; most near-term labels persist, while the rarer transitions depend on small group composition/position/HP changes. The next useful step is audit and augment the causal inputs for those threshold crossings (including explicit alive-unit counts, push-margin/base-health score and recent score slope), then repeat game-group CV. Do not tune against the test set.

## Reproducibility artifacts and checks

- LightGBM code: `eval/experiment_death_aware_mode_lightgbm.py`
- Original pooled-row report: `out_m2/death_aware_global_mode_experiments/lightgbm/report.json`
- Recomputed game-equal report and paired intervals: `out_m2/death_aware_global_mode_experiments/lightgbm_game_equal/report.json`
- Warm-start code: `eval/finetune_death_aware_global_mode_only.py`
- Warm-start checkpoint/history/validation metrics: `out_m2/death_aware_global_mode_experiments/warmstart/`
- Both experiment scripts pass `py_compile` and their `--help` CLI smoke checks.
