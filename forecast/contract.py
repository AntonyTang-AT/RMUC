"""Frozen contract for the causal RM temporal forecast experiment.

This module is deliberately small and dependency free.  Training, evaluation,
and deployment can import it without accidentally importing a model or a
future label builder.
"""

from __future__ import annotations

FORECAST_CONTRACT_VERSION = "rm-temporal-contract-v1"
GLOBAL_MODE_LABEL_VERSION = "relative_margin_v2"
HORIZONS_SECONDS = (10, 30, 60, 120)
TIME_RESOLUTION_SECONDS = 1.0
ISSUE_STEP_SECONDS = 3.0

# Target conventions are shared by every entity head.
UNKNOWN_LABEL = -1
ABSENT_PRESENCE = 0.0
PRESENT_PRESENCE = 1.0
CENSORED_TARGET_POLICY = "mask_all_fields_with_unknown; do_not_impute_future"
DEATH_POLICY = "explicit_destroyed_or_hp_le_zero_only; missing_is_unknown"

# Only fields with a numeric or categorical truth target enter the composite
# score.  Rule-derived attack/mining/target text remains a separate product
# layer until a reliable event label is available.
SCORED_FIELDS = {
    "global_mode": {"kind": "categorical", "metric": "macro_f1"},
    "opponent_trend": {"kind": "categorical", "metric": "macro_f1"},
    "enemy_region": {"kind": "categorical", "metric": "macro_f1"},
    "entity_presence": {"kind": "categorical", "metric": "macro_f1"},
    "entity_zone": {"kind": "categorical", "metric": "macro_f1"},
    "entity_position": {"kind": "continuous", "metric": "mae"},
    "entity_hp": {"kind": "continuous", "metric": "mae"},
    "route_legality": {"kind": "constraint", "metric": "rate"},
}

# Initial business/evidence weights.  They are stored with every checkpoint
# and may only be changed after a fresh, same-contract evaluation.
FIELD_WEIGHTS = {
    "global_mode": 1.00,
    "opponent_trend": 1.00,
    "enemy_region": 0.45,
    "entity_presence": 1.25,
    "entity_zone": 0.85,
    "entity_position": 1.35,
    "entity_hp": 0.45,
    "route_legality": 0.55,
}
HORIZON_WEIGHTS = {10: 1.40, 30: 1.25, 60: 0.85, 120: 0.45}


def contract_metadata() -> dict:
    """Return JSON-serialisable metadata for reports/checkpoints."""
    return {
        "version": FORECAST_CONTRACT_VERSION,
        "global_mode_label_version": GLOBAL_MODE_LABEL_VERSION,
        "horizons_seconds": list(HORIZONS_SECONDS),
        "time_resolution_seconds": TIME_RESOLUTION_SECONDS,
        "issue_step_seconds": ISSUE_STEP_SECONDS,
        "unknown_label": UNKNOWN_LABEL,
        "censored_target_policy": CENSORED_TARGET_POLICY,
        "death_policy": DEATH_POLICY,
        "scored_fields": SCORED_FIELDS,
        "field_weights": FIELD_WEIGHTS,
        "horizon_weights": {str(k): v for k, v in HORIZON_WEIGHTS.items()},
    }
