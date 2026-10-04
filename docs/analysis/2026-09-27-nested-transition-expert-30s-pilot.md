# Nested transition expert: 30-second low-cost pilot

## Design and scope

This is a train-only method pilot using the cached issue-time causal feature matrix; it does not reparse match JSONL. It evaluates only 30 seconds with 2 outer and 2 inner `GroupKFold` splits by complete match, 20 LightGBM trees and one CPU thread. Red and blue views of a match remain together. In each outer fold, the state expert and nine-way current-mode × future-mode transition expert use the same splits. The transition mixture weight is selected only from inner OOF game-equal accuracy (Macro-F1 breaks ties), then scored on outer-held games.

The baseline is the **same low-cost state expert retrained on the same outer folds**, rather than the full 5-fold 260-tree candidate. This paired comparison is appropriate for this pilot but its absolute accuracy is not directly comparable to the previous 59.2% full-candidate OOF figure. Validation and test were not accessed.

## Results

All headline scores average games equally and average red/blue within each game. Delta CIs use 2,000 paired whole-game bootstrap replicates.

| Metric | Nested transition mixture | Same-fold state baseline | Delta, 95% CI |
|---|---:|---:|---:|
| Accuracy | 57.97% | 57.72% | +0.241 pp [+0.088, +0.397] |
| Macro-F1 | 38.02% | 37.72% | +0.298 pp [+0.159, +0.439] |
| Balanced accuracy | 39.04% | 38.77% | +0.271 pp [+0.139, +0.399] |
| Brier | 0.54125 | 0.54538 | −0.00414 [−0.00458, −0.00369] |
| LogLoss | 0.91354 | 0.92024 | −0.00670 [−0.00735, −0.00604] |

Both outer folds independently selected transition weight 0.25. By side, red Accuracy was 58.10% vs 57.87% and blue Accuracy was 57.83% vs 57.58% (mixture vs baseline). Both sides improved by roughly 0.23–0.25 percentage points.

## Interpretation

The 2×2 pilot shows a small positive transition-expert signal against its same-fold lightweight baseline. It does **not** show an improvement from the approximately 59.3% full-candidate result: both models are underpowered by the 20-tree setting and the two-fold design. Treat the paired +0.24pp as a reason to run a full nested evaluation, not as a stable production gain. Do not promote this pilot or use its coarse weights as final settings.

## Artifacts

- Script: `eval/experiment_death_aware_nested_transition_expert.py`
- Machine report: `out_m2/death_aware_global_mode_experiments/nested_transition_expert_30s_pilot/report.json`
- OOF predictions: `out_m2/death_aware_global_mode_experiments/nested_transition_expert_30s_pilot/nested_transition_oof.npz`
- Reused train feature cache: `out_m2/death_aware_global_mode_experiments/nested_transition_expert_30s_pilot/causal_train_features.npz`

Runtime was 69.379 seconds. The cached feature audit covers 101,410 train rows and 363 matches; 99,750 rows were valid and paired for the 30-second target.
