# Production Forecast Confidence Audit

## Purpose

The frozen production-ensemble report previously required only directional
improvement over persistence on mean Accuracy, Macro-F1, and Brier score. That
check is necessary but not sufficient when a small point gain can arise from
the composition of a limited number of matches. The evaluator now adds a
paired, complete-game bootstrap: a draw resamples whole `game_id` values while
keeping red and blue views together, scores each side separately, and gives
the two sides equal weight.

The resulting evidence is stored in
`out_m2/production_ensemble_eval/report_current_transition10.json` under each
global head's `paired_game_bootstrap` field. The compact production summary
records both `double_gate` (directional) and `confidence_confirmed_gate`
(all three paired 95% intervals favor production).

## Locked-Test Result

All intervals below use 2,000 deterministic bootstrap draws over the same 128
complete locked-test games. Accuracy and Macro-F1 deltas are production minus
persistence; Brier deltas are also production minus persistence, so negative
values favor production. These are nominal pointwise 95% percentile intervals;
no family-wise or false-discovery adjustment has been applied across the 12
task/horizon heads. Treat `confirmed` as per-head evidence, not a project-wide
multiple-testing guarantee.

| Head | Accuracy 95% CI | Macro-F1 95% CI | Brier 95% CI | Interpretation |
| --- | --- | --- | --- | --- |
| Global mode, 10s | [-0.057, +0.684] pp | [-0.158, +0.806] pp | [-0.146, -0.106] | Probability quality confirmed; discrete-class gain not confirmed. |
| Global mode, 30s | [-0.504, +0.766] pp | [-0.662, +0.622] pp | [-0.261, -0.195] | Probability quality confirmed; discrete-class gain not confirmed. |
| Global mode, 60s | [+0.701, +2.992] pp | [-0.559, +1.675] pp | [-0.341, -0.254] | Accuracy and probability improvement confirmed; Macro-F1 not confirmed. |
| Global mode, 120s | [+1.150, +3.474] pp | [-0.776, +1.542] pp | [-0.383, -0.292] | Accuracy and probability improvement confirmed; Macro-F1 not confirmed. |
| Opponent trend, 10s | [+6.930, +9.383] pp | [+24.793, +28.248] pp | [-0.405, -0.353] | All three metrics confirmed. |
| Opponent trend, 30s | [+13.670, +17.813] pp | [+31.822, +34.484] pp | [-0.710, -0.646] | All three metrics confirmed. |
| Opponent trend, 60s | [+22.775, +27.507] pp | [+33.214, +36.092] pp | [-0.859, -0.799] | All three metrics confirmed. |
| Opponent trend, 120s | [+27.801, +33.085] pp | [+31.655, +35.216] pp | [-0.950, -0.887] | All three metrics confirmed. |
| Enemy region, 10s | [+0.158, +0.594] pp | [-0.060, +0.107] pp | [-0.121, -0.098] | Accuracy and probability improvement confirmed; Macro-F1 not confirmed. |
| Enemy region, 30s | [+1.025, +1.722] pp | [-0.017, +0.246] pp | [-0.253, -0.221] | Accuracy and probability improvement confirmed; Macro-F1 not confirmed. |
| Enemy region, 60s | [+1.247, +2.663] pp | [-0.163, +1.444] pp | [-0.301, -0.246] | Accuracy and probability improvement confirmed; Macro-F1 not confirmed. |
| Enemy region, 120s | [+0.678, +1.662] pp | [-0.248, +0.286] pp | [-0.350, -0.304] | Accuracy and probability improvement confirmed; Macro-F1 not confirmed. |

## Runtime Interpretation

- `opponent_trend` has stable evidence for use as a probability-bearing
  forecast context at every evaluated horizon.
- `global_mode` and `enemy_region` retain their useful calibrated probability
  signal, but their top-1 discrete labels are not promoted as independently
  confirmed improvements over persistence until Macro-F1 is also confirmed.
- The deployment audit now labels such heads `directional_only` rather than
  representing them as equally strong evidence. This is descriptive evidence,
  not a causal controller-value claim.

## Follow-Up Experiment

`out_m2/forecast_confidence_followup/canonical_mode_oof_macro/` contains a
pre-registered CPU-only alternate for global mode. It uses GroupKFold on the
training games to select the persistence blend and class-logit bias by
Macro-F1, then freezes that choice before inspecting validation and locked
test output. It does not replace the current runtime manifest unless its
validation-selected configuration later improves the paired frozen-test
evidence without reducing calibrated probability quality.
