# Nested transition expert: 30-second five-outer-fold pilot

## Design

This train-only pilot reuses the saved causal train feature cache (no match JSONL reparsing). It uses five outer and two inner `GroupKFold` splits by complete match, 20 LightGBM trees, one thread, and 30-second labels only. Red and blue views of a match stay in the same fold. The multiclass state expert and nine-way current-mode × future-mode transition expert use identical splits and model settings. Each outer fold's mixture weight is selected from inner OOF accuracy (Macro-F1 tie-breaker) only, then scored on outer-held matches. Validation and test were not accessed.

This remains a low-capacity pilot. The same-fold state comparator is retrained with 20 trees and must not be confused with the full 260-tree candidate's earlier 59.2% train-OOF accuracy.

## Results

Values are game-equal with red/blue averaged within each match. Deltas are paired 2,000-replicate whole-game bootstrap CIs.

| Metric | Nested transition mixture | Same-fold state baseline | Delta, 95% CI |
|---|---:|---:|---:|
| Accuracy | 58.13% | 57.93% | +0.200 pp [−0.014, +0.418] |
| Macro-F1 | 38.28% | 38.08% | +0.198 pp [−0.013, +0.428] |
| Balanced accuracy | 39.13% | 38.96% | +0.164 pp [−0.048, +0.385] |
| Brier | 0.53704 | 0.54457 | −0.00753 [−0.00852, −0.00660] |
| LogLoss | 0.90661 | 0.91906 | −0.01245 [−0.01395, −0.01106] |

Inner-fold selected transition weights were **0.25, 0.50, 0.75, 0.75, 0.25**, showing fold-to-fold variability. By side, red Accuracy was 58.37% vs 58.30%, and blue Accuracy was 57.90% vs 57.57% (mixture vs state baseline). Both sides have better Brier/LogLoss with the transition mixture.

## Assessment

The transition expert gives a small, directionally positive accuracy difference against its matched 20-tree state baseline, but the whole-game CI crosses zero. This does **not** establish a stable Accuracy improvement. The 58.13% pilot score remains about 1.1 percentage points below the approximately 59.3% full-candidate OOF reference; the comparator is also deliberately low-capacity, so those absolute scores are not directly interchangeable. Probability scores improve robustly, but a full-capacity nested run is still needed before deciding whether the expert closes the 30-second Accuracy gap.

## Artifacts

- Script: `eval/experiment_death_aware_nested_transition_expert.py`
- Report: `out_m2/death_aware_global_mode_experiments/nested_transition_expert_30s_5outer_pilot/report.json`
- OOF predictions: `out_m2/death_aware_global_mode_experiments/nested_transition_expert_30s_5outer_pilot/nested_transition_oof.npz`
- Reused features: `out_m2/death_aware_global_mode_experiments/nested_transition_expert_30s_5outer_pilot/causal_train_features.npz`

The complete run finished in 249.506 seconds over 363 matches and 94,150 valid 30-second rows.
