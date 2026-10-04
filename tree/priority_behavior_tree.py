# -*- coding: utf-8 -*-
"""Four-level sentry behavior-tree priority gate.

T0 and T1 are allowed to override the radar/strategy intent written at T2.
T3 only refines the selected state into a local action; it cannot change that
state.  This mirrors the priority direction in the project requirement slide.
"""
from __future__ import annotations

from typing import Any, Mapping

from fastloop import behavior as fast_behavior


def _find_sentry(worldstate: Mapping[str, Any], team: str) -> Mapping[str, Any] | None:
    for unit in (worldstate.get("robots") or {}).values():
        if unit.get("team") == team and unit.get("type") == "哨兵":
            return unit
    return None


class PriorityBehaviorTree:
    """Evaluate T0 safety -> T1 task -> T2 strategy -> T3 combat."""

    def __init__(self, rules: fast_behavior.Rules | None = None):
        self.rules = rules or fast_behavior.Rules()

    def tick(self, worldstate: Mapping[str, Any], *, team: str = "红",
             semantic_situation: Mapping[str, Any] | None = None,
             strategic_state: str = "cruise") -> dict[str, Any]:
        sentry = dict(_find_sentry(worldstate or {}, team) or {})
        blackboard = {"t2.sentry_intent": strategic_state}
        hp_ratio = float(sentry.get("hp_ratio") or 0.0) if sentry else 1.0

        if sentry and (hp_ratio < self.rules.min_hp_to_attack or bool(sentry.get("vuln"))):
            selected_state = "def"
            priority_level = "T0_system_safety"
            reason = "sentry_low_hp_or_vulnerable"
        elif bool((semantic_situation or {}).get("base_threat")):
            selected_state = "def"
            priority_level = "T1_task_coordination"
            reason = "base_defense_task"
        else:
            selected_state = strategic_state if strategic_state in fast_behavior.SENTRY_STATES else "cruise"
            priority_level = "T2_strategic_goal"
            reason = "state_machine_strategy"

        local_snapshot = {"robots": {7: sentry}, "bot_state": selected_state}
        combat_action = fast_behavior.decide_fast(local_snapshot, self.rules)
        blackboard["t3.combat_action"] = dict(combat_action)
        return {
            "state": selected_state,
            "priority_level": priority_level,
            "reason": reason,
            "blackboard": blackboard,
            "combat_action": combat_action,
        }
