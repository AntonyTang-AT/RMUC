# Death-aware cumulative ordinal threshold OOF experiment

## Design and safeguards

This experiment tests the ordered labels `防守 < 均衡 < 进攻`. Each horizon uses two independent binary LightGBM heads for `P(Y ≥ 均衡)` and `P(Y ≥ 进攻)`. If their outputs violate the cumulative ordering, the pair is projected to the nearest point satisfying `P(Y ≥ 进攻) ≤ P(Y ≥ 均衡)`, then converted to defense/balanced/attack class probabilities.

- Data: death-aware **train** only, 101,410 rows from 363 matches. Train JSONL features were built only from issue-time observable state and causal history; NPZ/JSONL row parity and current-mode parity both passed.
- CV: five-fold `GroupKFold` by full `game_id`; red and blue views stay in the same held-out fold. The script asserts candidate fold IDs match the saved multiclass baseline fold IDs row-for-row.
- Comparator: the existing zone + observable-score multiclass model’s saved OOF predictions from those same folds.
- Fixed candidate config: 140 trees, learning rate .045, 15 leaves, min child 90, L2 4, column subsampling .8, row subsampling .85. No search, validation, test, production default changes, or threshold tuning.
- CIs: paired 2,000-replicate bootstrap over whole matches, red/blue metrics averaged within each match. Game-equal metrics are the primary numbers.

## Results

Candidate and baseline absolute metrics are equal-weighted by match, with red and blue averaged within each match. Delta CIs compare ordinal against the multiclass baseline. For Accuracy, Macro-F1, and Balanced Accuracy, positive is better; for Brier and LogLoss, negative is better.

| Horizon | Model | Accuracy | Macro-F1 | Balanced Acc. | Brier | LogLoss |
|---:|---|---:|---:|---:|---:|---:|
| 10s | Multiclass baseline | 69.33% | 51.63% | 52.18% | 0.4079 | 0.6826 |
|  | Cumulative ordinal | 69.08% | 51.65% | 52.22% | 0.4113 | 0.6902 |
|  | Delta, 95% CI | −0.24 pp [−0.41, −0.06] | +0.03 pp [−0.19, +0.25] | +0.04 pp [−0.16, +0.25] | +0.00338 [+0.00243, +0.00434] | +0.00765 [+0.00614, +0.00911] |
| 30s | Multiclass baseline | 59.19% | 37.78% | 39.45% | 0.5086 | 0.8456 |
|  | Cumulative ordinal | 58.85% | 37.96% | 39.41% | 0.5118 | 0.8532 |
|  | Delta, 95% CI | −0.34 pp [−0.66, −0.03] | +0.18 pp [−0.14, +0.47] | −0.03 pp [−0.35, +0.26] | +0.00321 [+0.00166, +0.00484] | +0.00761 [+0.00545, +0.00992] |
| 60s | Multiclass baseline | 55.48% | 33.01% | 35.41% | 0.5469 | 0.9098 |
|  | Cumulative ordinal | 54.93% | 33.04% | 35.25% | 0.5498 | 0.9163 |
|  | Delta, 95% CI | −0.56 pp [−0.93, −0.18] | +0.03 pp [−0.30, +0.37] | −0.16 pp [−0.48, +0.16] | +0.00297 [+0.00103, +0.00489] | +0.00647 [+0.00349, +0.00938] |
| 120s | Multiclass baseline | 52.26% | 29.37% | 33.21% | 0.5749 | 0.9563 |
|  | Cumulative ordinal | 51.56% | 29.25% | 32.81% | 0.5785 | 0.9634 |
|  | Delta, 95% CI | −0.70 pp [−1.23, −0.16] | −0.13 pp [−0.54, +0.32] | −0.40 pp [−0.86, +0.07] | +0.00352 [+0.00069, +0.00626] | +0.00716 [+0.00271, +0.01147] |

The ordinal heads produced only four raw order violations across all OOF rows, all at 120 seconds; projection enforced monotonicity. However, order consistency did not improve decision quality: accuracy fell significantly at every horizon, and Brier/LogLoss also worsened significantly at every horizon. Macro-F1 gains at 10/30/60s were tiny and their CIs include zero. Do not promote this candidate.

## Interpretation and next direction

This rejects the simplest cumulative ordinal decomposition with independent binary heads. It suggests that fitting the two thresholds separately disrupts the useful shared structure learned by the multiclass model; monotonicity alone is not enough. Since motion features, temporal smoothing, and threshold decomposition have now failed to yield robust 10/30-second gains, the next model experiment should target actual transition forecasting: retain the current-state multiclass expert for persistence, and train a fold-contained transition expert specifically on future changes, then calibrate its switch probability only from inner game-group OOF predictions. Report transition recall/lead time alongside overall metrics, since the current error audit shows most remaining losses around mode-boundary crossings. Do not choose weights or thresholds on validation/test until this new candidate has a fully nested train-only evaluation.

## Reproduction

- Script: `eval/experiment_death_aware_cumulative_ordinal_oof.py`
- Report JSON and row-level OOF predictions: `out_m2/death_aware_global_mode_experiments/cumulative_ordinal_oof/report.json`, `ordinal_oof_probabilities.npz`
