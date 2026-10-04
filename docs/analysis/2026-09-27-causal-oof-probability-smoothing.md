# Death-aware causal OOF probability smoothing

## Scope

This low-cost experiment tests whether the current model's previous issue-time forecast helps stabilize its next forecast. It reads only the saved train OOF archive `motion_candidate_oof.npz` (101,410 rows, 363 complete matches). That archive contains zone/observable-score candidate predictions from five-fold `GroupKFold` by whole `game_id`; red and blue views are kept together. The script verifies one fold per game and both sides per game. Validation and test data are not opened, no model is retrained, and no production setting is changed.

For each horizon and each `(game_id, side)`, a candidate uses only the exact preceding 3-second forecast:

`q(t) = (1 - w) p(t) + w p(t - 3s)`, with `w ∈ {0.25, 0.50, 0.75}`.

Missing intervals or missing prior predictions leave the current probability vector unchanged. The target label is used only for scoring. Metrics are equal-weighted by game with red and blue averaged within a game; paired 95% CIs bootstrap the 363 whole games 2,000 times.

## Results

The comparator is the unsmoothed zone + observable-score OOF prediction. Accuracy/Macro-F1 are percentage points; Brier and LogLoss are absolute differences. Negative probability-score differences are improvements.

| Horizon | w | Accuracy | Δ Accuracy (95% CI) | Δ Macro-F1 (95% CI) | Δ Brier | Δ LogLoss |
|---:|---:|---:|---:|---:|---:|---:|
| 10s | raw | 69.33% | — | — | — | — |
| 10s | 0.25 | 69.08% | −0.25 [−0.37, −0.13] pp | −0.67 [−0.85, −0.49] pp | +0.00172 | +0.00286 |
| 10s | 0.50 | 68.43% | −0.89 [−1.07, −0.72] pp | −1.43 [−1.67, −1.19] pp | +0.00693 | +0.01091 |
| 10s | 0.75 | 67.59% | −1.74 [−1.93, −1.54] pp | −2.17 [−2.45, −1.90] pp | +0.01565 | +0.02453 |
| 30s | raw | 59.19% | — | — | — | — |
| 30s | 0.25 | 59.22% | +0.03 [−0.07, +0.13] pp | −0.10 [−0.22, +0.02] pp | +0.00013 | +0.00019 |
| 30s | 0.50 | 59.09% | −0.11 [−0.24, +0.03] pp | −0.21 [−0.37, −0.03] pp | +0.00127 | +0.00197 |
| 30s | 0.75 | 58.76% | −0.44 [−0.58, −0.30] pp | −0.43 [−0.64, −0.24] pp | +0.00342 | +0.00535 |
| 60s | raw | 55.48% | — | — | — | — |
| 60s | 0.25 | 55.50% | +0.02 [−0.07, +0.12] pp | −0.16 [−0.31, −0.03] pp | −0.00024 | −0.00035 |
| 60s | 0.50 | 55.44% | −0.04 [−0.16, +0.08] pp | −0.23 [−0.41, −0.07] pp | +0.00020 | +0.00039 |
| 60s | 0.75 | 55.26% | −0.22 [−0.34, −0.09] pp | −0.37 [−0.56, −0.19] pp | +0.00134 | +0.00220 |
| 120s | raw | 52.26% | — | — | — | — |
| 120s | 0.25 | 52.23% | −0.03 [−0.12, +0.07] pp | −0.14 [−0.24, −0.04] pp | −0.00040 | −0.00060 |
| 120s | 0.50 | 52.27% | +0.01 [−0.12, +0.14] pp | −0.11 [−0.25, +0.03] pp | −0.00026 | −0.00033 |
| 120s | 0.75 | 52.17% | −0.09 [−0.23, +0.06] pp | −0.07 [−0.24, +0.09] pp | +0.00043 | +0.00079 |

## Decision and next experiment

Do not promote temporal probability smoothing. It consistently harms the 10-second classifier, gives only a negligible and statistically uncertain 30-second accuracy change, and does not improve 60/120-second accuracy or Macro-F1. A few Brier/LogLoss changes at 60/120s are tiny and mixed; they do not compensate for the lost Macro-F1. This candidate does not close the requested 10/30s accuracy gaps.

Existing OOF error decomposition identifies the more promising next direction: labels are easiest far from the score proxy's `±1` thresholds and hardest near them; 30s+ errors also over-predict the extreme classes and miss the middle “balanced” state. The next candidate should therefore model the three ordered states via two train-fold-only cumulative threshold classifiers (`future_mode ≥ balanced`, `future_mode ≥ attack`) and construct a monotone three-class distribution. Keep features and LightGBM settings fixed, generate outer five-fold whole-game OOF predictions with red/blue paired, and report per-class recall plus game-bootstrap Accuracy, Macro-F1, Brier, and LogLoss. That tests the label's ordinal structure directly without post-hoc decision thresholds or validation/test tuning.

## Reproduction

- Script: `eval/experiment_death_aware_oof_causal_probability_smoothing.py`
- Machine-readable report: `out_m2/death_aware_global_mode_experiments/causal_probability_smoothing_oof/report.json`
- Input: `out_m2/death_aware_global_mode_experiments/motion_zone_score_oof/motion_candidate_oof.npz`
