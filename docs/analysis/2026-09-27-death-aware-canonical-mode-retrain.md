# Death-aware canonical global-mode retraining (validation only)

Date: 2026-09-27

## Result

The production canonical-side checkpoint and archived compact labels were trained under a different global-mode contract. This experiment rebuilt the same train/validation rows under the currently explicit mobile-unit rule: a mobile unit is excluded from the global mode score when `hp <= 0` or `life_state == "destroyed"`. It did not read or rebuild test data and did not change the production checkpoint/default.

The candidate was trained with the frozen canonical configuration (GRU, 232 input features, hidden size 128, 2 layers, business-weighted loss, seed 0, 3 epochs, batch size 256, AdamW learning rate 0.002 and weight decay 0.0001). Training used CUDA on an NVIDIA GeForce RTX 5060 Laptop GPU. Best validation loss was 0.51839 at epoch 2 (epoch 3: 0.51968).

| Horizon | Candidate accuracy | Candidate macro-F1 | Death-aware persistence accuracy | Persistence macro-F1 | Paired game red/blue accuracy delta (95% CI) | Paired game red/blue macro-F1 delta (95% CI) |
|---:|---:|---:|---:|---:|---:|---:|
| 10s | 59.67% | 60.16% | 72.24% | 71.48% | −12.56 pp [−14.79, −10.38] | −13.09 pp [−15.17, −11.09] |
| 30s | 53.08% | 53.37% | 60.37% | 59.47% | −7.28 pp [−9.58, −4.87] | −6.92 pp [−8.78, −5.18] |
| 60s | 50.63% | 50.47% | 55.87% | 55.10% | −5.23 pp [−7.73, −2.80] | −4.52 pp [−6.12, −3.02] |
| 120s | 47.37% | 46.39% | 50.21% | 49.73% | −2.82 pp [−5.13, −0.55] | −2.35 pp [−3.57, −1.03] |

Each paired bootstrap replicate resamples whole games (2,000 replicates, seed 0); within each game, the red and blue observing-side deltas are averaged before game-level aggregation. The CIs exclude zero in the negative direction at every horizon. **This candidate is not suitable for promotion; the death-aware persistence baseline is better on both accuracy and macro-F1 at all four horizons.** The validation scores are development evidence, not a final blind-test estimate.

## Label/data contract

- Frozen split rows were taken from `out_m2/temporal_forecast_canonical_side/{train,validation}.npz` and matched, in order and exactly by `(game_id, side, issued_at)`, to `out_m2/forecast_dataset/{train,validation}.jsonl.gz`.
- Train: 101,410 rows. Validation: 34,114 rows / 122 games. Both red and blue observing sides are present in each validation game.
- Existing causal input tensors and every non-global target array were copied verbatim. Only `mode_y`, `base_mode`, and the global-mode contract version were regenerated. Hence this is deliberately a **global-mode-only label-contract experiment**, not a full relabel/retrain of entity, trend, or region heads.
- Validation target-label differences from the archived pre-death-aware labels: 3,164 / 33,378 at 10s; 3,156 / 31,674 at 30s; 3,078 / 29,234 at 60s; 2,854 / 24,354 at 120s. Current `base_mode` changed on 3,201 / 34,114 validation rows.
- Death-aware contract string stored in the NPZ is `relative_margin_v2_death_aware_hp_le_zero`. The annotated JSONL retains the observable/future truth and adds `death_aware_global_mode_labels` for auditability.
- Training and evaluation used only `train.npz` and `validation.npz`; `test_access=false` is recorded in the validation metric report. No test file was opened.

## Reproduction

Run from the repository root using the project CUDA environment:

```powershell
.venv_openjev_cuda\Scripts\python.exe eval\rebuild_death_aware_mode_splits.py
.venv_openjev_cuda\Scripts\python.exe -c "from forecast.train import train_model; print(train_model('out_m2/temporal_forecast_death_aware_candidate/train.npz','out_m2/temporal_forecast_death_aware_candidate/validation.npz','out_m2/temporal_forecast_death_aware_candidate/temporal_gru_v1.pt',epochs=3,batch_size=256,hidden=128,layers=2,lr=2e-3,weight_decay=1e-4,device='auto',seed=0,loss_profile='business_weighted',architecture='gru'))"
.venv_openjev_cuda\Scripts\python.exe eval\evaluate_death_aware_candidate_validation.py --checkpoint out_m2/temporal_forecast_death_aware_candidate/temporal_gru_v1.pt --validation out_m2/temporal_forecast_death_aware_candidate/validation.npz --output out_m2/temporal_forecast_death_aware_candidate/validation_metrics.json --device cuda --bootstrap 2000
```

The compact split builder `eval/run_temporal_forecast.py` was intentionally not used because it iterates over train, validation, and test. The specialized rebuild command accepts only the train/validation source directory and never names or opens a test path.

## SHA-256

| Artifact | SHA-256 |
|---|---|
| `out_m2/forecast_dataset/train.jsonl.gz` | `b14a18557252fb44515d926d0654f29428c18378878a8d591f4ff754b0b166a6` |
| `out_m2/forecast_dataset/validation.jsonl.gz` | `1b553ed2aba2d5933e9db1bfe3f8712bb8fc2c4322955a537c9a73c8d7d24a4f` |
| frozen canonical-side `train.npz` | `2107948791aebebeb50c314fa00a5249a855bac562ca4d1a9908cff1e4238d9e` |
| frozen canonical-side `validation.npz` | `677c074f5524fbeadd5f6b1158da4057fdadc75176c1beed9b96a523b41c934b` |
| candidate `train.npz` | `17a06608852b53f0ac72d2148d9583f1ca1c47a60c8abae0cc30482e7965a407` |
| candidate `validation.npz` | `f4ee2fa229f958ad5a1cc9c87b29aa4c52f4bc74f86c41c6dedcf50799c43f0e` |
| candidate checkpoint | `d9a75f669e2e60a7289050fd89a6beb5cfcf0355c0181fdd01d491d66618a8aa` |
| `validation_metrics.json` | `eaaf2cf069f25d1048434ca13402e192cc4fbb36735da1ac4a369c1e8aa91951` |

Machine-readable rebuild counts are in `out_m2/temporal_forecast_death_aware_candidate/data_rebuild_manifest.json`; model configuration/history and validation-only results are in the same directory's checkpoint and `validation_metrics.json`.
