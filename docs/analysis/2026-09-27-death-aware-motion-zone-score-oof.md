# Death-aware causal motion feature OOF experiment

## Scope and leakage audit

- Uses the death-aware **train** split only: 101,410 rows from 363 complete matches; 5-fold `GroupKFold` splits on full `game_id`, keeping both red and blue views of a match in one fold.
- Inputs are only each JSONL row’s issue-time `observable.self_entities` and `observable.opponent_entities`, plus direct-model historical inputs. Train NPZ/JSONL alignment was exact. There were zero `max_input_observed_at > issued_at` violations. No JSONL target keys were read by the feature extractor; NPZ labels are used only for training/scoring. Validation and test were not opened.
- Per-robot velocity uses the same entity ID’s immediately previous current-observable position. Actual elapsed time normalizes the displacement to a three-second interval (all matched intervals were exactly 3.0s; 1,171,274 usable deltas). Tracks do not bridge missing snapshots. Enemy positions come only from the radar-visible observable list.
- Current observable yaw, where present, is encoded as separate sine/cosine heading features and coverage. It is not treated as travel direction; missing enemy radar yaw has zero coverage. Current/previous canonical-zone transitions are encoded by destination-zone mass.
- Candidate adds these features to the frozen direct LightGBM temporal inputs and the existing zone/observable-score feature set. LightGBM parameters match that candidate; no tuning or search was done.

## Results

Accuracy and macro-F1 are game-equal with red/blue averaged within each match. “Existing” is the strongest prior train OOF candidate (zone mass + observable score history). Paired 95% CIs resample whole matches.

| Horizon | Motion accuracy | Existing accuracy | Δ accuracy (95% CI) | Motion macro-F1 | Existing macro-F1 | Δ macro-F1 (95% CI) |
|---|---:|---:|---:|---:|---:|---:|
| 10s | 69.56% | 69.33% | +0.23 pp (+0.06, +0.40) | 51.66% | 51.63% | +0.03 pp (-0.17, +0.24) |
| 30s | 59.32% | 59.19% | +0.12 pp (-0.12, +0.36) | 37.84% | 37.78% | +0.06 pp (-0.20, +0.32) |
| 60s | 55.37% | 55.48% | -0.11 pp (-0.45, +0.19) | 32.67% | 33.01% | -0.34 pp (-0.63, -0.06) |
| 120s | 52.29% | 52.26% | +0.03 pp (-0.43, +0.51) | 28.98% | 29.37% | -0.40 pp (-0.79, -0.00) |

Per-class recalls (attack / balanced / defense):

| Horizon | Motion candidate | Existing candidate |
|---|---|---|
| 10s | 50.40 / 55.08 / 51.04% | 50.59 / 54.71 / 51.23% |
| 30s | 42.52 / 32.73 / 43.48% | 42.42 / 32.87 / 43.05% |
| 60s | 39.28 / 26.37 / 39.91% | 39.49 / 26.69 / 40.04% |
| 120s | 38.74 / 22.32 / 37.96% | 38.64 / 22.55 / 38.44% |

## Assessment

Motion features produce a small but paired-CI-positive 10s accuracy gain (+0.23 pp); 30s and later accuracy CIs include zero. Macro-F1 is not improved reliably, and the 60s/120s macro-F1 differences are negative. Compared with direct LightGBM the accuracy rises, but that gain is largely already present in the existing zone/score candidate. This candidate therefore does **not** justify promotion; keep it as an experimental feature set, especially because the requested result is balanced across horizons and classes. The major remaining gap is 10s and 30s accuracy versus the desired target, while class-balanced performance remains weak.

These are train-only cross-fitted estimates, not validation or blind test results. Reusing prior OOF results to select this candidate creates post-hoc selection risk; independent future confirmation is needed before any deployment decision.

## Artifacts

- Script: `eval/experiment_death_aware_motion_oof.py`
- Machine-readable results/predictions: `out_m2/death_aware_global_mode_experiments/motion_zone_score_oof/report.json` and `motion_candidate_oof.npz`
- Feature-name static contract check passed: 240 columns; the reported names are in the same role/block order as the row builder.
