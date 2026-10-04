# Deployment Readiness Audit

Overall status: **not_ready**

## Forecast Heads

| Task | Horizon | Evidence | Status | Accuracy delta (pp) | Macro-F1 delta (pp) |
| --- | ---: | --- | --- | ---: | ---: |
| global_mode | 10 | unified_locked_test_report | pass | 0.240 | 0.205 |
| global_mode | 30 | unified_locked_test_report | pass | 0.160 | 0.014 |
| global_mode | 60 | unified_locked_test_report | pass | 1.827 | 0.496 |
| global_mode | 120 | unified_locked_test_report | pass | 2.315 | 0.378 |
| opponent_trend | 10 | unified_locked_test_report | pass | 8.072 | 26.504 |
| opponent_trend | 30 | unified_locked_test_report | pass | 15.764 | 33.199 |
| opponent_trend | 60 | unified_locked_test_report | pass | 25.160 | 34.669 |
| opponent_trend | 120 | unified_locked_test_report | pass | 30.398 | 33.460 |
| enemy_region | 10 | unified_locked_test_report | pass | 0.386 | 0.028 |
| enemy_region | 30 | unified_locked_test_report | pass | 1.376 | 0.120 |
| enemy_region | 60 | unified_locked_test_report | pass | 1.935 | 0.721 |
| enemy_region | 120 | unified_locked_test_report | pass | 1.093 | 0.052 |
| enemy_region | 10 | standalone_transition_promotion_artifact | superseded_by_unified_report | 0.386 | 0.028 |

## LLM Structural Evaluation

| Side | Forecast context | Parse rate | Conditional legal rate | Status |
| --- | --- | ---: | ---: | --- |
| 红 | True | 1.000 | 1.000 | not_ready |
| 蓝 | True | 0.700 | 1.000 | not_ready |
| 红 | False | 0.000 | 0.000 | not_ready |

Structural robustness gate: **fail**
Policy-value gate: **unverifiable_without_execution_or_counterfactual_labels**

Missing scenarios: 蓝/forecast=False
Structural failures: 蓝/forecast=True, 红/forecast=False

## Blocking Conditions

- LLM decision reports fail structural robustness or scenario coverage gates.
- No controller execution or causal policy-value labels support a deployment claim.
