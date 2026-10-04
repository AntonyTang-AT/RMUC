# Death-aware train OOF probability quality

This audit evaluates the probability vectors from cross-fitted train predictions. It opens only saved train OOF predictions and labels; validation and test were not accessed. Each score first averages red and blue within a match, then averages matches. Paired 95% intervals resample whole matches 2,000 times.

`Brier(sum3)` is the mean sum of squared errors over three class probabilities, `LogLoss` penalizes low probability assigned to the true class, and `top-label ECE` is a descriptive 10-bin estimate of whether the winning class confidence matches its empirical accuracy. Lower is better for all three. ECE depends on binning and should not be used alone. These OOF candidates were selected after prior OOF inspection, so this is exploratory evidence, not blind confirmation.

## Zone/score candidate vs direct LightGBM

| Horizon | Brier direct → candidate (Δ, 95% CI) | LogLoss direct → candidate (Δ, 95% CI) | Top-label ECE direct → candidate (Δ, 95% CI) |
|---|---|---|---|
| 10s | 0.4219 → 0.4079 (−0.0140, −0.0160 to −0.0120) | 0.7099 → 0.6826 (−0.0273, −0.0307 to −0.0238) | 0.1151 → 0.1004 (−0.0147, −0.0179 to −0.0114) |
| 30s | 0.5211 → 0.5086 (−0.0125, −0.0150 to −0.0101) | 0.8671 → 0.8456 (−0.0216, −0.0255 to −0.0178) | 0.1436 → 0.1361 (−0.0075, −0.0115 to −0.0036) |
| 60s | 0.5539 → 0.5469 (−0.0070, −0.0095 to −0.0047) | 0.9213 → 0.9098 (−0.0115, −0.0154 to −0.0078) | 0.1562 → 0.1542 (−0.0021, −0.0057 to +0.0016) |
| 120s | 0.5803 → 0.5749 (−0.0053, −0.0079 to −0.0027) | 0.9646 → 0.9563 (−0.0083, −0.0121 to −0.0043) | 0.1798 → 0.1811 (+0.0013, −0.0029 to +0.0051) |

The zone/score model improves Brier and LogLoss at every horizon. Its top-label ECE improvement is supported at 10/30 seconds; the 60/120 second intervals include zero. This is stronger evidence for the probability distribution than accuracy alone, though it does not fix weak balanced-class recall at long horizons.

## Motion candidate vs zone/score candidate

Motion-derived features yield small Brier/LogLoss improvements at 10 seconds; the paired intervals exclude zero there. At 30/60/120 seconds, Brier and LogLoss intervals include zero. ECE is lower with motion at 60/120 seconds, but remains bin-sensitive. This does not outweigh its weaker 60/120-second Macro-F1 and the lack of stable accuracy gains; do not promote it as the default class predictor.

## Reproduction

- Script: `eval/audit_death_aware_oof_probability_quality.py`
- Machine results: `out_m2/death_aware_global_mode_experiments/oof_probability_quality/report.json`
- Only train OOF predictions and labels are opened; all comparisons are game-paired.
