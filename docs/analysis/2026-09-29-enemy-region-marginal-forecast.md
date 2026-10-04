# Enemy-Region Marginal Forecast Audit

The exact enemy-region head predicts a 12-cell `x0y0` through `x3y2`
distribution. Because sparse edge cells make exact-cell Macro-F1 unstable, the
same frozen probabilities were also collapsed into horizontal x-band (4
classes) and vertical y-band (3 classes) marginals. This is a deterministic
aggregation of the existing probabilities; it does not train or promote a new
model. Scores and intervals use the same 128-game locked test and 2,000 paired
whole-game bootstrap draws as the production forecast audit. Red and blue are
kept paired within each resampled game and equally weighted.
Intervals are pointwise nominal 95% intervals with no multiple-comparison
adjustment; these marginals are diagnostic and do not change deployment gates.

## Horizontal Band

| Horizon | Accuracy / persistence | Accuracy delta 95% CI | Macro-F1 / persistence | Macro-F1 delta 95% CI | Brier delta 95% CI | Triple gate |
| ---: | ---: | --- | ---: | --- | --- | --- |
| 10s | 0.8703 / 0.8653 | [+0.274, +0.737] pp | 0.6778 / 0.6765 | [-0.111, +0.363] pp | [-0.0522, -0.0334] | no |
| 30s | 0.7957 / 0.7795 | [+1.315, +1.922] pp | 0.5438 / 0.5386 | [+0.169, +0.855] pp | [-0.1339, -0.1049] | yes |
| 60s | 0.7645 / 0.7311 | [+2.503, +4.300] pp | 0.4395 / 0.4294 | [-0.194, +2.179] pp | [-0.1872, -0.1359] | no |
| 120s | 0.7079 / 0.6842 | [+1.715, +3.283] pp | 0.3737 / 0.3677 | [+0.096, +1.195] pp | [-0.2161, -0.1745] | yes |

## Vertical Band

| Horizon | Accuracy / persistence | Accuracy delta 95% CI | Macro-F1 / persistence | Macro-F1 delta 95% CI | Brier delta 95% CI | Triple gate |
| ---: | ---: | --- | ---: | --- | --- | --- |
| 10s | 0.8575 / 0.8575 | [-0.049, +0.049] pp | 0.6149 / 0.6204 | [-1.700, +0.065] pp | [-0.0570, -0.0407] | no |
| 30s | 0.7929 / 0.7891 | [+0.054, +0.730] pp | 0.4778 / 0.4855 | [-1.403, -0.227] pp | [-0.1316, -0.1062] | no |
| 60s | 0.7620 / 0.7572 | [-0.094, +1.067] pp | 0.4282 / 0.4259 | [-1.339, +1.728] pp | [-0.1196, -0.0748] | no |
| 120s | 0.7701 / 0.7384 | [+2.590, +3.791] pp | 0.4095 / 0.4169 | [-2.421, +0.577] pp | [-0.1947, -0.1632] | no |

## Interpretation

- Horizontal band probabilities have the strongest evidence at 30s and 120s:
  Accuracy, fixed-class Macro-F1, and Brier intervals all favor production.
- At 10s and 60s horizontal accuracy and Brier improve, but Macro-F1 is not
  confirmed; do not present a confident horizontal class as established there.
- Vertical-band Brier improves at all horizons, but categorical discrimination
  is not jointly established. At 30s, Macro-F1 is significantly worse despite
  slightly higher Accuracy.
- This is evidence for a broad horizontal pressure signal only at 30s and
  120s on the canonical-side locked replay test. It is not causal policy-value
  evidence, a real radar observability test, or evidence of actual robot
  execution outcomes.
