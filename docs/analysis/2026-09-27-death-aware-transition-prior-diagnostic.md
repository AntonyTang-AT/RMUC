# Death-aware mode transition-prior diagnostic

## Decision

An incremental five-fold `GroupKFold` experiment tested a current-mode → future-mode transition matrix. It used only the death-aware train and validation NPZs. Each held-out training fold's prior was estimated from the other training games; Dirichlet smoothing and the persistence blend were chosen from those OOF predictions. The validation split was then scored once. **There is no accuracy or Macro-F1 gain at any horizon; do not promote this prior.** It does improve probability scores (Brier/log loss/calibration), so it may inform uncertainty display if desired.

## Validation results

| Horizon | Transition prior accuracy / Macro-F1 | Persistence accuracy / Macro-F1 | Paired accuracy delta, pp (95% CI) | Paired Macro-F1 delta, pp (95% CI) | Brier delta (95% CI) |
|---:|---:|---:|---:|---:|---:|
| 10s | 72.24% / 71.48% | 72.24% / 71.48% | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | −0.120 [−0.139, −0.101] |
| 30s | 60.37% / 59.47% | 60.37% / 59.47% | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | −0.241 [−0.271, −0.210] |
| 60s | 55.87% / 55.10% | 55.87% / 55.10% | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | −0.295 [−0.330, −0.261] |
| 120s | 50.21% / 49.73% | 50.21% / 49.73% | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | −0.366 [−0.411, −0.321] |

The transition prior's argmax was the same as persistence for every scored row at all horizons. OOF train selected a 0% transition-prior mixture weight under the accuracy/Macro-F1 objective. The probability gains come from a better estimate of transition likelihood while leaving the class decision unchanged.

## Label support and why accuracy/F1 did not move

Validation targets are not sparse or strongly imbalanced. Class counts in `(进攻, 均衡, 防守)` are 11,710 / 9,954 / 11,714 at 10s; 11,187 / 9,296 / 11,191 at 30s; 10,329 / 8,572 / 10,333 at 60s; and 8,486 / 7,378 / 8,490 at 120s. Future labels change from the current mode in 27.8%, 39.6%, 44.1%, and 49.8% of rows respectively.

At 120s, the validation transition counts (current rows, future columns) are:

```text
进攻: 4805  2397  1185
均衡: 2540  2679  2562
防守: 1141  2302  4743
```

Even at 120s, the most likely destination conditioned only on the current class remains that same class; the `均衡` row is nearly tied across all three outcomes. The label distribution is well supported, but `P(future mode | current mode)` alone does not identify when or which class changes. This explains why a global transition prior cannot improve 10/30 Macro-F1 or resolve the later horizon's transitions. The previous LightGBM candidate showed a train-OOF 120s accuracy lift but no validation lift with a game-paired CI spanning zero, consistent with features that are not robust enough across matches.

## Summary-feature experiment status

A more ambitious raw+20-observable-summary five-fold tree run was stopped before it emitted a report because its full raw-observable reprocessing and training had no per-horizon checkpoint and took too long. The prior sidecar cache was not reused: its source contract is `relative_margin_v2`, whereas this candidate is `relative_margin_v2_death_aware_hp_le_zero`; its dead-unit push counts and `mode_score` are therefore not certified for the new targets. There is no summary-feature performance claim from that interrupted run.

## Artifacts

- Incremental code: `eval/experiment_death_aware_transition_prior_incremental.py`
- Full per-horizon report: `out_m2/death_aware_global_mode_experiments/transition_prior_incremental/report.json`
- Test access: false; no production files/defaults changed. The script compiles, writes each horizon before proceeding, and its output contract was verified.
