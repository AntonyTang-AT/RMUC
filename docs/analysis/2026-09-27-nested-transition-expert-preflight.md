# Nested transition expert preflight and compute limit

## Intended evaluation protocol

The candidate encodes each supervised row as one of nine `current_mode × future_mode` transition outcomes, trains a 9-class transition model, and marginalizes its joint probabilities into a future three-class distribution. A state expert predicts future mode directly from the same causal observable feature matrix. For every outer held-out set of complete matches, a coarse mixture weight is chosen only from 2-fold inner `GroupKFold` OOF predictions on the outer-training matches. The saved outer prediction is then evaluated once against a state-only expert trained and predicted on the identical outer split. Planned primary horizons are 10/30 seconds; longer horizons can be added after a low-cost signal is established.

The candidate script is `eval/experiment_death_aware_nested_transition_expert.py`. It defaults to 3 outer folds, 2 inner folds, 100 trees, 2 CPU threads, and lets the caller restrict horizons. The candidate and state expert use exactly the same folds. It does not use the previously rejected balanced binary transition-factor approach.

## Preflight run status

The first, higher-capacity version was stopped after its first outer fold completed because it consumed several minutes before useful results. The reduced 3-outer/2-inner/100-tree/2-thread version was then started and stopped at **4 minutes 21 seconds** under the requested time limit. It had parsed and saved the full train-only causal feature matrix, but had not finished its first outer fold. Therefore this run produced **no evaluable outer-held-out scores**, and it is not evidence for or against the transition expert.

The saved feature cache contains 101,410 aligned rows from 363 matches, generated from train NPZ plus current-observable train JSONL only. No validation or test data were opened. There was no completed outer score, and no outer OOF predictions were used to choose fusion weights.

The limiting cost is nested fitting, not data access: even for two horizons the protocol needs inner state and 9-class transition models plus outer state and transition refits per fold. The 100-tree CPU run still did not complete one of three outer folds within the 4:21 wall-time budget. Feature cache was retained at `out_m2/death_aware_global_mode_experiments/nested_transition_expert_light_10_30/causal_train_features.npz` so a future reduced run can skip JSONL parsing.

## Correct low-cost continuation

For a method-direction pilot that can finish quickly, use one horizon (10s), **2 outer × 2 inner whole-game folds**, 20–30 trees, and 1–2 threads. Keep both state and transition experts trained on identical features and splits; choose only from inner OOF, evaluate only outer-held games, and report that the small fold count/tree budget is a pilot rather than a stable estimate. If that pilot shows signal, schedule the full 5 outer × 3–4 inner fold run with the same predeclared recipe, then extend to 30/60/120s. Avoid dropping the inner loop or selecting any mixture weight from outer OOF, because that would reintroduce the leakage this experiment is designed to prevent.

This constrained run is not reported as a completed model comparison. The existing multiclass baseline remains the current candidate; the transition expert requires a faster inner model, less CPU compute, or a longer run window before its nested OOF benefit can be assessed.
