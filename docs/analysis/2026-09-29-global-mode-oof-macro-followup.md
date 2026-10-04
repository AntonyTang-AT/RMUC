# Global Mode OOF-Macro Follow-Up

This CPU-only experiment selected the persistence blend and class-logit bias
with five-fold whole-game OOF predictions from the training split. The locked
test split was read only after each configuration had been frozen. The purpose
was to test whether the current global-mode uncertainty could be removed by
changing selection protocol alone.

| Horizon | OOF-selected alpha | Test Accuracy | Persistence Accuracy | Test Macro-F1 | Persistence Macro-F1 | Directional gate |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 10s | 1.00 | 0.7548 | 0.7525 | 0.7264 | 0.7251 | pass |
| 30s | 0.90 | 0.6433 | 0.6417 | 0.6080 | 0.6079 | pass |
| 60s | 0.90 | 0.5950 | 0.5789 | 0.5538 | 0.5460 | pass |
| 120s | 0.85 | 0.5642 | 0.5432 | 0.5246 | 0.5170 | pass |

The 10s and 30s results are effectively the current production choices. The
60s and 120s point estimates are directionally stronger, but this experiment
does not replace the production route. The candidate model files remain
isolated under `out_m2/forecast_confidence_followup/canonical_mode_oof_macro/`.

The candidate was then scored with the paired 2,000-draw bootstrap on all 128
locked test games. Its Accuracy delta intervals were [-0.114, +0.667] pp,
[-0.476, +0.737] pp, [+0.715, +2.571] pp, and [+0.991, +3.249] pp at 10/30/
60/120 seconds. Macro-F1 intervals were [-0.290, +0.709] pp,
[-0.632, +0.655] pp, [-0.061, +1.678] pp, and [-0.377, +1.947] pp. Brier
improvement was confirmed at all four horizons. Thus even the 60s/120s
candidate does not establish a Macro-F1 improvement; its interval crosses
zero, so no horizon passes the three-metric confidence-confirmed gate.

Conclusion: selection leakage control is useful and now documented, but the
remaining global-mode limitation requires better features or a different
label/model formulation rather than another sweep over the same blend.
