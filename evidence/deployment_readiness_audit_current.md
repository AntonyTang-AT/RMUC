# Deployment Readiness Audit

Overall status: **not_ready**

## Forecast Heads

| Task | Horizon | Evidence | Directional status | Confidence status | Accuracy delta (pp) | Macro-F1 delta (pp) |
| --- | ---: | --- | --- | --- | ---: | ---: |
| global_mode | 10 | unified_locked_test_report | pass | directional_only | 0.240 | 0.205 |
| global_mode | 30 | unified_locked_test_report | pass | directional_only | 0.160 | 0.014 |
| global_mode | 60 | unified_locked_test_report | pass | directional_only | 1.827 | 0.496 |
| global_mode | 120 | unified_locked_test_report | pass | directional_only | 2.315 | 0.378 |
| opponent_trend | 10 | unified_locked_test_report | pass | confirmed | 8.072 | 26.504 |
| opponent_trend | 30 | unified_locked_test_report | pass | confirmed | 15.764 | 33.199 |
| opponent_trend | 60 | unified_locked_test_report | pass | confirmed | 25.160 | 34.669 |
| opponent_trend | 120 | unified_locked_test_report | pass | confirmed | 30.398 | 33.460 |
| enemy_region | 10 | unified_locked_test_report | pass | directional_only | 0.386 | 0.028 |
| enemy_region | 30 | unified_locked_test_report | pass | directional_only | 1.376 | 0.120 |
| enemy_region | 60 | unified_locked_test_report | pass | directional_only | 1.935 | 0.721 |
| enemy_region | 120 | unified_locked_test_report | pass | directional_only | 1.093 | 0.052 |
| enemy_region | 10 | standalone_transition_promotion_artifact | superseded_by_unified_report | not_applicable | 0.386 | 0.028 |

## LLM Structural Evaluation

| Side | Forecast context | Parse rate | Conditional legal rate | Status |
| --- | --- | ---: | ---: | --- |

Structural robustness gate: **fail**
LLM evidence source: **legacy_scenario_reports**
Policy-value gate: **unverifiable_without_execution_or_counterfactual_labels**

Missing scenarios: 红/forecast=False, 红/forecast=True, 蓝/forecast=False, 蓝/forecast=True

## Blocking Conditions

- No current LLM candidate acceptance artifact is available; legacy scenario reports are diagnostic only.
- No controller execution or causal policy-value labels support a deployment claim.
