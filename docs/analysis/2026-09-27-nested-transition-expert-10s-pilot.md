# Nested transition expert: 10-second low-cost pilot

## Method and limits

This is a fast method-direction pilot, not a final accuracy estimate. It reuses the existing train-only causal feature cache (101,410 issue rows, 363 matches) and runs 10-second forecasting only. Two outer `GroupKFold` folds hold out complete matches; each outer training partition uses two inner complete-match folds to select the transition-mixture weight. Red and blue views stay in the same fold. The state and transition experts use the same fold membership and identical 20-tree LightGBM settings. The transition model predicts a nine-way current-mode × future-mode class and marginalizes probabilities to three future modes. Each outer-fold mixture weight is selected from inner OOF game-equal accuracy, with Macro-F1 as tie-breaker.

Validation and test data were not opened. Outer held-out rows were not used to choose the fusion weight. The two-fold/20-tree setting is intentionally small; use this result only to decide whether the method merits a full nested evaluation.

## Outer held-out results

Metrics are averaged equally over matches and, within each match, red and blue. Deltas use paired 2,000-replicate whole-match bootstrap.

| Metric | Transition mixture | State multiclass baseline | Delta, 95% CI |
|---|---:|---:|---:|
| Accuracy | 68.65% | 68.66% | −0.012 pp [−0.050, +0.026] |
| Macro-F1 | 52.35% | 52.38% | −0.026 pp [−0.076, +0.018] |
| Balanced accuracy | 53.11% | 53.13% | −0.020 pp [−0.063, +0.020] |
| Brier | 0.44649 | 0.45426 | −0.00777 [−0.00838, −0.00715] |
| LogLoss | 0.77388 | 0.78753 | −0.01364 [−0.01462, −0.01263] |

The inner-selected transition weights were 0.25 and 0.50 in the two outer folds. By side, red Accuracy was 69.11% vs 69.11% and blue Accuracy was 68.18% vs 68.21% (mixture vs baseline). Brier and LogLoss improved for both sides. Classification accuracy and Macro-F1 show no reliable gain in this pilot; probability quality improves.

## Artifacts

- Script: `eval/experiment_death_aware_nested_transition_expert.py`
- Machine report: `out_m2/death_aware_global_mode_experiments/nested_transition_expert_light_10_30/report.json`
- Outer OOF vectors: `out_m2/death_aware_global_mode_experiments/nested_transition_expert_light_10_30/nested_transition_oof.npz`
- Reused cache: `out_m2/death_aware_global_mode_experiments/nested_transition_expert_light_10_30/causal_train_features.npz`

The saved run took 68.651 seconds and scored 99,230 valid issue rows across 363 paired matches. It does not establish expected performance under five-fold outer CV or larger LightGBM models.
