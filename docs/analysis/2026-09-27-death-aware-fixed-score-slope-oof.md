# Death-aware fixed score-slope OOF experiment

Train-only 363-match experiment. Lambda=1 was fixed; no horizon-wise threshold, coefficient, or calibration was selected. Current score is reconstructed exclusively from each issue-time observable using the death-aware mode formula. Both red and blue sides from each full match remain in one GroupKFold fold.

| Horizon | Model | Accuracy (game-equal) | Macro-F1 | Balanced accuracy | Attack recall | Balance recall | Defense recall |
|---:|---|---:|---:|---:|---:|---:|---:|
| 10s | score_slope_lambda_1 | 62.39% | 48.42% | 50.55% | 51.63% | 49.22% | 50.80% |
| 10s | persistence | 68.64% | 52.38% | 53.14% | 53.69% | 53.95% | 51.78% |
| 10s | motion_zone_score_oof | 69.56% | 51.66% | 52.17% | 50.40% | 55.08% | 51.04% |
| 30s | score_slope_lambda_1 | 47.88% | 34.76% | 36.80% | 43.55% | 24.12% | 42.73% |
| 30s | persistence | 56.99% | 38.77% | 39.28% | 40.06% | 39.31% | 38.47% |
| 30s | motion_zone_score_oof | 59.32% | 37.84% | 39.58% | 42.52% | 32.73% | 43.48% |
| 60s | score_slope_lambda_1 | 44.04% | 31.27% | 33.33% | 40.49% | 19.75% | 39.75% |
| 60s | persistence | 52.93% | 34.37% | 35.25% | 35.11% | 36.72% | 33.93% |
| 60s | motion_zone_score_oof | 55.37% | 32.67% | 35.18% | 39.28% | 26.37% | 39.91% |
| 120s | score_slope_lambda_1 | 40.60% | 27.96% | 29.73% | 35.78% | 17.77% | 35.64% |
| 120s | persistence | 47.77% | 29.29% | 30.69% | 29.35% | 33.73% | 28.99% |
| 120s | motion_zone_score_oof | 52.29% | 28.98% | 33.01% | 38.74% | 22.32% | 37.96% |

## Paired whole-game bootstrap deltas

Each delta is fixed-slope minus reference; 95% percentile intervals resample complete matches (both side views together), 2,000 replicates. A CI spanning zero is inconclusive.

### 10s

| Comparator | Metric | Delta | 95% CI |
|---|---|---:|---:|
| persistence | accuracy | -6.25 pp | [-6.73, -5.79] pp |
| persistence | macro_f1 | -3.96 pp | [-4.43, -3.51] pp |
| persistence | balanced_accuracy | -2.59 pp | [-3.17, -2.03] pp |
| persistence | recall_进攻 | -2.06 pp | [-2.79, -1.33] pp |
| persistence | recall_均衡 | -4.74 pp | [-5.62, -3.85] pp |
| persistence | recall_防守 | -0.97 pp | [-1.75, -0.18] pp |
| motion_zone_score_oof | accuracy | -7.17 pp | [-7.74, -6.62] pp |
| motion_zone_score_oof | macro_f1 | -3.24 pp | [-3.80, -2.67] pp |
| motion_zone_score_oof | balanced_accuracy | -1.62 pp | [-2.25, -0.95] pp |
| motion_zone_score_oof | recall_进攻 | +1.24 pp | [+0.17, +2.38] pp |
| motion_zone_score_oof | recall_均衡 | -5.86 pp | [-7.26, -4.40] pp |
| motion_zone_score_oof | recall_防守 | -0.23 pp | [-1.22, +0.80] pp |
### 30s

| Comparator | Metric | Delta | 95% CI |
|---|---|---:|---:|
| persistence | accuracy | -9.11 pp | [-9.76, -8.42] pp |
| persistence | macro_f1 | -4.01 pp | [-4.50, -3.50] pp |
| persistence | balanced_accuracy | -2.48 pp | [-3.21, -1.70] pp |
| persistence | recall_进攻 | +3.49 pp | [+2.40, +4.67] pp |
| persistence | recall_均衡 | -15.19 pp | [-16.47, -13.99] pp |
| persistence | recall_防守 | +4.27 pp | [+3.12, +5.50] pp |
| motion_zone_score_oof | accuracy | -11.44 pp | [-12.33, -10.49] pp |
| motion_zone_score_oof | macro_f1 | -3.08 pp | [-3.79, -2.39] pp |
| motion_zone_score_oof | balanced_accuracy | -2.78 pp | [-3.65, -1.90] pp |
| motion_zone_score_oof | recall_进攻 | +1.03 pp | [-0.49, +2.59] pp |
| motion_zone_score_oof | recall_均衡 | -8.61 pp | [-10.95, -6.37] pp |
| motion_zone_score_oof | recall_防守 | -0.75 pp | [-2.17, +0.70] pp |
### 60s

| Comparator | Metric | Delta | 95% CI |
|---|---|---:|---:|
| persistence | accuracy | -8.89 pp | [-9.75, -8.10] pp |
| persistence | macro_f1 | -3.10 pp | [-3.66, -2.58] pp |
| persistence | balanced_accuracy | -1.93 pp | [-2.73, -1.10] pp |
| persistence | recall_进攻 | +5.38 pp | [+4.15, +6.68] pp |
| persistence | recall_均衡 | -16.98 pp | [-18.23, -15.71] pp |
| persistence | recall_防守 | +5.82 pp | [+4.59, +7.10] pp |
| motion_zone_score_oof | accuracy | -11.32 pp | [-12.58, -10.12] pp |
| motion_zone_score_oof | macro_f1 | -1.41 pp | [-2.16, -0.66] pp |
| motion_zone_score_oof | balanced_accuracy | -1.86 pp | [-2.81, -0.92] pp |
| motion_zone_score_oof | recall_进攻 | +1.21 pp | [-0.59, +3.16] pp |
| motion_zone_score_oof | recall_均衡 | -6.62 pp | [-9.22, -4.15] pp |
| motion_zone_score_oof | recall_防守 | -0.16 pp | [-1.93, +1.58] pp |
### 120s

| Comparator | Metric | Delta | 95% CI |
|---|---|---:|---:|
| persistence | accuracy | -7.17 pp | [-8.17, -6.20] pp |
| persistence | macro_f1 | -1.33 pp | [-1.90, -0.73] pp |
| persistence | balanced_accuracy | -0.96 pp | [-1.81, -0.03] pp |
| persistence | recall_进攻 | +6.43 pp | [+4.99, +7.93] pp |
| persistence | recall_均衡 | -15.95 pp | [-17.15, -14.75] pp |
| persistence | recall_防守 | +6.65 pp | [+5.15, +8.10] pp |
| motion_zone_score_oof | accuracy | -11.69 pp | [-13.29, -10.08] pp |
| motion_zone_score_oof | macro_f1 | -1.02 pp | [-1.95, -0.18] pp |
| motion_zone_score_oof | balanced_accuracy | -3.27 pp | [-4.44, -2.16] pp |
| motion_zone_score_oof | recall_进攻 | -2.96 pp | [-4.99, -0.93] pp |
| motion_zone_score_oof | recall_均衡 | -4.54 pp | [-7.29, -1.74] pp |
| motion_zone_score_oof | recall_防守 | -2.32 pp | [-4.28, -0.30] pp |

## Data and parity audit

- Rows: 101,410; full games: 363; label contract: `relative_margin_v2_death_aware_hp_le_zero`.
- Observable score reproduces current `base_mode` exactly: True (0 mismatches).
- Future-observation violations: 0.
- Persistence and motion+zone+score predictions are read from the existing full-game GroupKFold OOF artifact; its fold IDs are checked against a fresh deterministic GroupKFold reconstruction.
- No validation/test data was opened. No runtime defaults or production model changed.

## Reproduction

```powershell
python eval/experiment_death_aware_score_slope_fixed_oof.py
```
