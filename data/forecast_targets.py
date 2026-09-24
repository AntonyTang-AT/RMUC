"""Exact-time forecast targets and side-specific, observable-only examples."""

from collections.abc import Mapping
from copy import deepcopy
from math import isfinite

from data.observable import MissingVisibilityProvider


_TEAMS = frozenset({'红', '蓝'})
_SENSOR_SOURCES = frozenset({'sensor', 'radar', 'camera', 'vision'})
_OBSERVATION_METADATA = frozenset({
    'entity_id', 'team', 'entity_type', 'source', 'observed_at', 'received_at',
    'confidence', 'presence_state', 'observability_state', 'age_seconds',
})


def _number(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value):
        raise ValueError(f'{label} must be a finite number')
    return value


def _horizons(horizons):
    result = tuple(horizons)
    if not result or len(result) != len(set(result)):
        raise ValueError('horizons must be a nonempty set of distinct positive seconds')
    for horizon in result:
        if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon <= 0:
            raise ValueError('horizons must be positive integer seconds')
    return result


def build_targets(global_truth_by_time, issued_at, horizons=(10, 30, 60, 120), side='红'):
    """Read only truth at each exact future timestamp; missing truth is censored.

    Truth stays in global team coordinates for audit and scoring. The caller's
    side changes feature roles, never the historical outcome itself.
    """
    if side not in _TEAMS:
        raise ValueError('side must be 红 or 蓝')
    _number(issued_at, 'issued_at')
    horizons = _horizons(horizons)
    if not isinstance(global_truth_by_time, Mapping):
        raise ValueError('global_truth_by_time must map timestamps to truth states')
    targets = {}
    for horizon in horizons:
        target_at = issued_at + horizon
        truth = global_truth_by_time.get(target_at)
        if truth is None:
            targets[horizon] = None
            continue
        if not isinstance(truth, Mapping):
            raise ValueError(f'truth at {target_at} must be a mapping')
        if 'target_at' in truth and truth['target_at'] != target_at:
            raise ValueError(f'truth timestamp disagrees with exact target time {target_at}')
        targets[horizon] = {'target_at': target_at, 'truth': deepcopy(dict(truth)), 'censored': False}
    return targets


def _validate_snapshot(snapshot, side):
    if not isinstance(snapshot, Mapping):
        raise ValueError('observable history must contain snapshots')
    if snapshot.get('self_team') != side:
        raise ValueError('observable snapshot self_team must equal side')
    issued_at = _number(snapshot.get('issued_at'), 'issued_at')
    entities = snapshot.get('entities')
    if not isinstance(entities, Mapping):
        raise ValueError('observable snapshot entities must be a mapping')
    grouped = {'self_entities': {}, 'opponent_entities': {}}
    max_observed_at = None
    for entity_id, entity in entities.items():
        if not isinstance(entity, Mapping):
            raise ValueError('observable entity must be a mapping')
        team = entity.get('team')
        if team not in _TEAMS:
            raise ValueError('observable entity must carry raw 红/蓝 team')
        source = entity.get('source')
        if source not in _SENSOR_SOURCES:
            raise MissingVisibilityProvider('observable entity lacks a sensor or radar source')
        observed_at = _number(entity.get('observed_at'), 'observed_at')
        received_at = _number(entity.get('received_at'), 'received_at')
        if observed_at > issued_at or received_at > issued_at:
            raise ValueError('future measurement cannot enter observable features')
        if received_at < observed_at:
            raise ValueError('received_at cannot precede observed_at')
        if entity.get('observability_state') not in ('current', 'stale', 'unknown', 'absent', 'destroyed'):
            raise ValueError('observable entity must have a valid observability_state')
        if entity.get('role', 'self' if team == side else 'opponent') != ('self' if team == side else 'opponent'):
            raise ValueError('observable entity role disagrees with side and raw team')
        group = 'self_entities' if team == side else 'opponent_entities'
        # The group already records role; stripping its duplicate field makes
        # red/blue swap symmetry directly comparable without changing team.
        allowed = (entity.items() if entity['observability_state'] == 'current'
                   else ((key, value) for key, value in entity.items()
                         if key in _OBSERVATION_METADATA))
        grouped[group][entity_id] = {key: deepcopy(value) for key, value in allowed
                                     if key != 'role'}
        max_observed_at = observed_at if max_observed_at is None else max(max_observed_at, observed_at)
    return issued_at, grouped, max_observed_at


def build_side_examples(game_id, observable_history, global_truth, side,
                        horizons=(10, 30, 60, 120)):
    """Join issued sensor snapshots with future truth without leaking truth into X.

    Call once per side using snapshots produced for that self_team. A missing
    history is a missing visibility provider, not an empty opponent roster.
    """
    if observable_history is None:
        raise MissingVisibilityProvider('observable history requires a visibility provider')
    if side not in _TEAMS:
        raise ValueError('side must be 红 or 蓝')
    horizons = _horizons(horizons)
    if not isinstance(global_truth, Mapping):
        raise ValueError('global_truth must map timestamps to truth states')
    history = (observable_history.values() if isinstance(observable_history, Mapping)
               else observable_history)
    examples = []
    seen_issued_at = set()
    for snapshot in history:
        issued_at, grouped, max_observed_at = _validate_snapshot(snapshot, side)
        if issued_at in seen_issued_at:
            raise ValueError(f'duplicate observable snapshot at {issued_at}')
        seen_issued_at.add(issued_at)
        examples.append({
            'game_id': game_id,
            'side': side,
            'issued_at': issued_at,
            'observable': grouped,
            'visibility_summary': deepcopy(snapshot.get('visibility_summary', {})),
            'max_input_observed_at': max_observed_at,
            'targets': build_targets(global_truth, issued_at, horizons, side),
        })
    return sorted(examples, key=lambda example: example['issued_at'])
