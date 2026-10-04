# Train-OOF ID trajectory candidate

**Status: rejected for deployment; offline validation experiment only.** The independent evaluation used the existing train/validation split and did not open test data (`test_access=false`). It compares four forecasts for each alive, observable mobile robot with a stable `entity_id`: same-ID persistence, exact-3-second same-ID velocity extrapolation, the frozen canonical side+type model translated as a group offset to the unit, and a train-OOF-fitted blend of group offset and same-ID velocity.

## Protocol and limits

- 363 train games, 122 validation games. Train-only blend weights were fitted from five game-grouped OOF folds. Each OOF GRU was fixed at one epoch; validation was scored once and never used to select blend weights.
- Fold GRU configuration: GRU, hidden=128, layers=2, batch=256, lr=0.002, weight_decay=0.0001, business_weighted loss, seed=0. The frozen canonical checkpoint was `out_m2/temporal_forecast_canonical_side/temporal_gru_v1.pt`.
- Current units must be alive, observable, have a stable ID and valid XY. Velocity requires the same ID exactly 3 seconds earlier and is speed-capped by type; otherwise it falls back to persistence. Future truth is exact-horizon, exact-ID, and alive. Dead/destroyed rows are excluded; missing/dead observations break the velocity track; unmatched IDs are not imputed and lower coverage.
- The neural positional component is **not an individual-ID trajectory model**. It is a side+type slot/group displacement added to each unit's current position. The blend alpha is fit on train OOF by minimizing equal-game/equal-side coordinate MAE, with an alpha grid 0 to 1 by 0.05.
- The full canonical checkpoint has its historical validation-based checkpoint selection history. Thus the validation pass is untouched for this experiment's alpha choice, but not a pristine holdout for the backbone's model selection.

## Validation results

MAE and mean Euclidean error are metres; zone hit is exact functional-zone agreement. Coverage is exact future same-ID alive matches over eligible current units. `Δ MAE` is candidate minus persistence, with 95% game-cluster bootstrap CI over 122 games (red/blue paired within game, 2,000 replicates); positive is worse.

| Horizon | Train alpha (velocity share) | Coverage | Persistence MAE / zone hit | ID velocity MAE / zone hit | Group model MAE / zone hit | Blend MAE / zone hit | Blend Δ MAE vs persistence (95% CI) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 10s | 0.85 | 95.1% | 1.45 / 62.5% | 1.80 / 50.3% | 2.91 / 23.9% | 1.72 / 47.9% | +0.277 m [+0.267, +0.287] |
| 30s | 0.40 | 86.9% | 2.38 / 50.3% | 3.69 / 30.5% | 3.14 / 22.4% | 2.82 / 27.8% | +0.437 m [+0.397, +0.476] |
| 60s | 0.20 | 79.6% | 2.94 / 42.5% | 5.08 / 20.2% | 3.32 / 20.4% | 3.19 / 22.3% | +0.254 m [+0.196, +0.312] |
| 120s | 0.10 | 66.0% | 3.41 / 36.0% | 6.41 / 12.8% | 3.43 / 19.3% | 3.39 / 19.9% | -0.019 m [-0.086, +0.053] |

Velocity was available for 98.4–98.5% of matched validation samples, so its poor results are not mainly missing-history fallback. The 120s blend/persistence MAE difference is not statistically resolved; zone hit still drops by 16.1 percentage points for the blend. Persistence beats all candidates on zone hit at every horizon.

## Interpretation

The measured motion pattern is too discontinuous for linear extrapolation to be reliable: robots stop, turn, get blocked, switch objectives, die, or reappear. At 10s, staying in the current position is the strongest individual position baseline. At 30–60s, the group model reduces continuous error versus ID velocity but does not beat persistence. At 120s, all position forecasts approach persistence in metre error while zone classification remains weaker. Do not deploy either velocity or blend weights as defaults. The useful next experiment is a per-ID, short-horizon movement-state model with turning/stopping and legal-zone transition constraints, evaluated against persistence with the same exact-ID coverage policy.

## Reproduction

From the project root, reuse the already trained five fold checkpoints:

```powershell
.venv_openjev_cuda\Scripts\python.exe eval\trainfit_crossfit_identity_position.py --device cuda --folds 5 --batch-size 256 --bootstrap 2000 --reuse-fold-checkpoints
```

The script writes `out_m2/identity_trajectory_trainfit/report.json` and fold artifacts. SHA-256 hashes of the train/validation NPZ and JSONL files and the canonical checkpoint are recorded in that JSON; it also records `test_access=false`, the ID/death policy, exact alpha-fit metric, and the bootstrap protocol. No production defaults or unit-decision files were changed.
