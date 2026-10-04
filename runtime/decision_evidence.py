"""Immutable records for decision, controller, and outcome evidence.

The runtime intentionally does not write files. Integrators can send these
plain dictionaries to a JSONL logger, database, or controller telemetry bus.
The schema preserves issue-time recommendations separately from later
execution and outcome observations so replay correlations cannot be mistaken
for causal value evidence.
"""
from __future__ import annotations

import hashlib
import json
import uuid
from typing import Any, Mapping

from labels import router as rt
from schema import layered as sl

DECISION_SOURCES = {'tier0_rule', 'tier1_student', 'tier2_llm', 'fallback'}
ACK_STATUSES = {'accepted', 'rejected', 'timed_out'}


def stable_sha256(value: Any) -> str:
    """Hash structured evidence independent of dictionary insertion order."""
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(',', ':')).encode('utf-8')).hexdigest()


def build_issue_record(*, match_id: int, side: str, decision_source: str,
                       visible_input: Mapping[str, Any], decision: Mapping[str, Any],
                       guard: Mapping[str, Any], forecast_context: Mapping[str, Any] | None = None,
                       decision_id: str | None = None) -> dict[str, Any]:
    """Build an immutable issue-time recommendation record."""
    if side not in ('红', '蓝'):
        raise ValueError('side must be 红 or 蓝')
    if decision_source not in DECISION_SOURCES:
        raise ValueError(f'unsupported decision_source: {decision_source}')
    clean = rt.strip_sidecar_for_validate(dict(decision))
    valid, errors = sl.validate(clean)
    if not valid:
        raise ValueError(f'cannot issue schema-invalid decision: {errors}')
    if not isinstance(visible_input, Mapping):
        raise ValueError('visible_input is required for a deployable issue record')
    return {
        'schema_version': 'decision-evidence-v1',
        'record_type': 'issue',
        'decision_id': decision_id or str(uuid.uuid4()),
        'match_id': int(match_id),
        'issued_at_s': float(clean.get('t', 0.0)),
        'side': side,
        'decision_source': decision_source,
        'visible_input_sha256': stable_sha256(visible_input),
        'forecast_context_sha256': stable_sha256(forecast_context)
        if forecast_context is not None else None,
        'decision': clean,
        'guard': {'accepted': bool(guard.get('accepted')),
                  'reasons': list(guard.get('reasons') or [])},
    }


def build_controller_record(*, decision_id: str, command_id: str, unit_id: str,
                            issued_at_s: float, controller_received_at_s: float | None,
                            controller_ack_at_s: float | None, ack_status: str,
                            executed_action: Mapping[str, Any] | None = None,
                            rejection_reason: str | None = None) -> dict[str, Any]:
    """Record whether an identified unit accepted a derived command."""
    if ack_status not in ACK_STATUSES:
        raise ValueError(f'unsupported ack_status: {ack_status}')
    if not decision_id or not command_id or not unit_id:
        raise ValueError('decision_id, command_id, and unit_id are required')
    return {
        'schema_version': 'decision-evidence-v1', 'record_type': 'controller',
        'decision_id': str(decision_id), 'command_id': str(command_id),
        'unit_id': str(unit_id), 'issued_at_s': float(issued_at_s),
        'controller_received_at_s': (float(controller_received_at_s)
                                     if controller_received_at_s is not None else None),
        'controller_ack_at_s': (float(controller_ack_at_s)
                                if controller_ack_at_s is not None else None),
        'ack_status': ack_status,
        'executed_action': dict(executed_action) if executed_action is not None else None,
        'rejection_reason': rejection_reason,
    }


def build_outcome_record(*, decision_id: str, unit_id: str, horizon_seconds: int,
                         telemetry: Mapping[str, Any], missing_fields=()) -> dict[str, Any]:
    """Attach fixed-horizon telemetry without modifying the issue record."""
    if not decision_id or not unit_id:
        raise ValueError('decision_id and unit_id are required')
    if int(horizon_seconds) <= 0:
        raise ValueError('horizon_seconds must be positive')
    return {
        'schema_version': 'decision-evidence-v1', 'record_type': 'outcome',
        'decision_id': str(decision_id), 'unit_id': str(unit_id),
        'horizon_seconds': int(horizon_seconds), 'telemetry': dict(telemetry),
        'missing_fields': sorted(str(value) for value in missing_fields),
    }


def validate_chain(issue: Mapping[str, Any], controller: Mapping[str, Any],
                   outcome: Mapping[str, Any]) -> tuple[bool, list[str]]:
    """Validate linkage only; it does not infer causal benefit."""
    reasons = []
    decision_id = issue.get('decision_id')
    if not decision_id or controller.get('decision_id') != decision_id:
        reasons.append('controller_decision_id_mismatch')
    if not decision_id or outcome.get('decision_id') != decision_id:
        reasons.append('outcome_decision_id_mismatch')
    if controller.get('ack_status') != 'accepted':
        reasons.append('command_not_accepted')
    if controller.get('unit_id') != outcome.get('unit_id'):
        reasons.append('unit_id_mismatch')
    if not isinstance(outcome.get('telemetry'), Mapping):
        reasons.append('outcome_telemetry_missing')
    return not reasons, reasons
