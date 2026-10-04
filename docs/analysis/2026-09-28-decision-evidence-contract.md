# Decision Evidence Contract

## Purpose

This contract separates three claims that must not be conflated:

1. A model produced a parseable, schema-legal decision.
2. A downstream controller received and executed that decision.
3. Executing that decision improved a defined tactical or match-level outcome.

The current replay dataset supports the first claim, provides only observational
proxies for the second, and does not identify the third causally.

## Required Issue Record

Persist one immutable record at recommendation time:

```json
{
  "decision_id": "unique id",
  "match_id": 0,
  "issued_at_s": 0.0,
  "side": "red_or_blue",
  "decision_source": "tier0_rule|tier1_student|tier2_llm|fallback",
  "visible_input_sha256": "hash",
  "forecast_context_sha256": "hash_or_null",
  "decision": {"layered schema payload"},
  "guard": {"accepted": true, "reasons": []}
}
```

The record must contain only issue-time input hashes and the recommended
decision. It must not be retroactively overwritten with future outcome data.

## Required Controller Evidence

For each unit-level command derived from a recommendation, record:

```json
{
  "decision_id": "parent recommendation",
  "command_id": "controller command id",
  "unit_id": "actual unit identity",
  "issued_at_s": 0.0,
  "controller_received_at_s": 0.0,
  "controller_ack_at_s": 0.0,
  "ack_status": "accepted|rejected|timed_out",
  "executed_action": "actual command payload",
  "rejection_reason": "optional"
}
```

Without an acknowledgement and a unit identity, later movement or firing logs
are only correlated replay behavior, not proof that a recommendation executed.

## Outcome and Counterfactual Evidence

At fixed, predeclared horizons, link only telemetry that can be attributed to
the commanded unit and target. Examples include route progress, target-linked
damage, own survival, protected-building HP, and match score. The horizon,
missing-data policy, and metric denominators must be declared before scoring.

To claim policy value rather than response association, compare the candidate
against a predeclared baseline on matched initial states or randomized/shadow
assignment. Store assignment, baseline policy version, and safety aborts.

## Deployment Gate

Tier 2 can be considered for shadow deployment only when all of the following
are true:

1. Red and blue, forecast and no-forecast structural scenarios all parse and
   validate under the visibility contract.
2. The guard/fallback audit is green for invalid generation and missing-visible
   input cases.
3. A unified forecast report includes every runtime sidecar version.
4. Controller receipt and acknowledgement logs exist for the shadow run.

It can be considered for active policy comparison only after matched or
randomized counterfactual outcome evidence is available. Legacy-rule agreement
and replay-response proxies are diagnostics, not substitutes for this gate.
