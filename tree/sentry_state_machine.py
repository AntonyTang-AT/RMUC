# -*- coding: utf-8 -*-
"""Explicit sentry finite-state machine for the three required macro states.

The state machine is deliberately small and deterministic.  It turns the
previously implicit rule ordering into named events and transitions so a
controller integration can inspect why a state was selected.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping

from fastloop import behavior as fast_behavior


class StateEnum(str, Enum):
    DEFEND = "def"
    HIT_BUILDING = "hitbuilding"
    CRUISE = "cruise"


class EventEnum(str, Enum):
    SAFETY_OVERRIDE = "safety_override"
    BASE_THREAT = "base_threat"
    ATTACK_OPPORTUNITY = "attack_opportunity"
    PATROL = "patrol"


# Every source state handles every event.  This is intentional: each decision
# tick represents the newest observable battlefield state, not a blind latch.
TRANSITIONS: dict[StateEnum, dict[EventEnum, StateEnum]] = {
    state: {
        EventEnum.SAFETY_OVERRIDE: StateEnum.DEFEND,
        EventEnum.BASE_THREAT: StateEnum.DEFEND,
        EventEnum.ATTACK_OPPORTUNITY: StateEnum.HIT_BUILDING,
        EventEnum.PATROL: StateEnum.CRUISE,
    }
    for state in StateEnum
}


_STRATEGIC_INTENTS = {
    StateEnum.DEFEND: "StrategicDefendBase",
    StateEnum.HIT_BUILDING: "StrategicAttackOutpost",
    StateEnum.CRUISE: "StrategicPatrol",
}


def _find_sentry(worldstate: Mapping[str, Any], team: str) -> Mapping[str, Any] | None:
    for unit in (worldstate.get("robots") or {}).values():
        if unit.get("team") == team and unit.get("type") == "哨兵":
            return unit
    return None


@dataclass(frozen=True)
class StateTransition:
    previous_state: str
    event: str
    state: str
    strategic_intent: str
    reason: str

    def as_dict(self) -> dict[str, str]:
        return {
            "previous_state": self.previous_state,
            "event": self.event,
            "state": self.state,
            "strategic_intent": self.strategic_intent,
            "reason": self.reason,
        }


class SentryStateMachine:
    """Finite-state owner for ``def`` / ``hitbuilding`` / ``cruise``."""

    def __init__(self, rules: fast_behavior.Rules | None = None):
        self.rules = rules or fast_behavior.Rules()
        self.current_state = StateEnum.CRUISE

    def reset(self) -> None:
        self.current_state = StateEnum.CRUISE

    def _event_for(self, worldstate: Mapping[str, Any], *, team: str,
                   semantic_situation: Mapping[str, Any] | None,
                   proposed_state: str | None) -> tuple[EventEnum, str]:
        sentry = _find_sentry(worldstate, team)
        if sentry is not None:
            hp_ratio = float(sentry.get("hp_ratio") or 0.0)
            if hp_ratio < self.rules.min_hp_to_attack or bool(sentry.get("vuln")):
                return EventEnum.SAFETY_OVERRIDE, "sentry_low_hp_or_vulnerable"

        if bool((semantic_situation or {}).get("base_threat")):
            return EventEnum.BASE_THREAT, "semantic_base_threat"

        if proposed_state == StateEnum.HIT_BUILDING.value:
            snapshot = {"robots": {7: dict(sentry or {})}, "bot_state": proposed_state}
            if fast_behavior.feasible(proposed_state, snapshot, self.rules):
                return EventEnum.ATTACK_OPPORTUNITY, "observable_attack_opportunity"
            return EventEnum.SAFETY_OVERRIDE, "attack_not_feasible"

        if proposed_state == StateEnum.DEFEND.value:
            return EventEnum.BASE_THREAT, "rule_defense_signal"
        return EventEnum.PATROL, "no_defense_or_attack_signal"

    def transition(self, worldstate: Mapping[str, Any], *, team: str = "红",
                   semantic_situation: Mapping[str, Any] | None = None,
                   proposed_state: str | None = None) -> StateTransition:
        event, reason = self._event_for(
            worldstate or {}, team=team, semantic_situation=semantic_situation,
            proposed_state=proposed_state,
        )
        previous = self.current_state
        next_state = TRANSITIONS[previous][event]
        self.current_state = next_state
        return StateTransition(
            previous_state=previous.value,
            event=event.value,
            state=next_state.value,
            strategic_intent=_STRATEGIC_INTENTS[next_state],
            reason=reason,
        )
