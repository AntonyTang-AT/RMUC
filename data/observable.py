"""Build deployment snapshots only from explicitly supplied sensor observations.

The global replay table is not a visibility source. Callers must supply a
sensor log or validated visibility provider before using this contract.
"""

from collections.abc import Mapping
from math import isfinite


class MissingVisibilityProvider(ValueError):
    """Deployment input was requested without observable measurements."""


_TEAMS = {'红': '蓝', '蓝': '红'}
_SENSOR_SOURCES = frozenset({'sensor', 'radar', 'camera', 'vision'})
_PRESENCE_STATES = frozenset({'present', 'unknown', 'absent', 'destroyed'})
_CONTRACT_FIELDS = frozenset({
    'entity_id', 'team', 'entity_type', 'observed_at', 'received_at',
    'source', 'confidence', 'presence_state', 'role', 'side',
    'observability_state', 'age_seconds',
})


def _finite_number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value):
        raise ValueError(f'{name} must be a finite number')
    return value


def build_observable_snapshot(observations, issued_at, self_team, max_age_by_source):
    """Return the latest causal, sensor-observable view for one decision time.

    ``observations`` may be empty when a provider reports no detections, but
    ``None`` means no provider was supplied and fails closed. Source age limits
    apply to observation time, not reception time. A stale track retains its
    metadata but exposes no current measured fields.
    """
    if observations is None:
        raise MissingVisibilityProvider('deployment mode requires a visibility provider')
    issued_at = _finite_number(issued_at, 'issued_at')
    if self_team not in _TEAMS:
        raise ValueError('self_team must be 红 or 蓝')
    if not isinstance(max_age_by_source, Mapping):
        raise ValueError('max_age_by_source must map sensor sources to maximum age')
    for source, age in max_age_by_source.items():
        if source not in _SENSOR_SOURCES:
            raise ValueError(f'unsupported observation source: {source!r}')
        if _finite_number(age, f'maximum age for {source}') < 0:
            raise ValueError('maximum age must be nonnegative')

    tracks = {}
    for row in observations:
        if not isinstance(row, Mapping):
            raise ValueError('each observation must be a mapping')
        for name in ('entity_id', 'team', 'entity_type', 'observed_at',
                     'received_at', 'source', 'confidence'):
            if name not in row:
                raise ValueError(f'observation missing {name}')
        entity_id = row['entity_id']
        if not isinstance(entity_id, (str, int)) or isinstance(entity_id, bool) or entity_id == '':
            raise ValueError('entity_id must be a nonempty string or integer')
        if not isinstance(row['team'], str) or row['team'] not in _TEAMS:
            raise ValueError('observation team must be 红 or 蓝')
        if not isinstance(row['entity_type'], str) or not row['entity_type']:
            raise ValueError('entity_type must be nonempty')
        observed_at = _finite_number(row['observed_at'], 'observed_at')
        received_at = _finite_number(row['received_at'], 'received_at')
        if observed_at > issued_at or received_at > issued_at:
            raise ValueError('future observation cannot enter a deployment snapshot')
        if received_at < observed_at:
            raise ValueError('received_at cannot precede observed_at')
        source = row['source']
        if not isinstance(source, str) or source not in _SENSOR_SOURCES or source not in max_age_by_source:
            raise ValueError(f'unconfigured sensor source: {source!r}')
        confidence = _finite_number(row['confidence'], 'confidence')
        if not 0 <= confidence <= 1:
            raise ValueError('confidence must be between 0 and 1')
        presence = row.get('presence_state', 'present')
        if not isinstance(presence, str) or presence not in _PRESENCE_STATES:
            raise ValueError(f'invalid presence_state: {presence!r}')
        previous_rows = tracks.get(entity_id, [])
        if previous_rows and previous_rows[0]['team'] != row['team']:
            raise ValueError(f'conflicting team for entity {entity_id!r}')
        tracks.setdefault(entity_id, []).append(dict(row))

    entities = {}
    state_counts = {'current': 0, 'stale': 0, 'unknown': 0, 'absent': 0, 'destroyed': 0}
    source_counts = {}
    for entity_id, entity_rows in tracks.items():
        # Filter by each measurement source's own lifetime before selecting
        # the newest usable estimate. An expired radar track must not hide a
        # still-current camera/sensor estimate for the same entity.
        newest_overall = max(entity_rows, key=lambda item: (item['observed_at'], item['received_at']))
        current_rows = [row for row in entity_rows
                        if issued_at - row['observed_at'] <= max_age_by_source[row['source']]]
        latest_current = max(current_rows, key=lambda item: (item['observed_at'], item['received_at'])) if current_rows else None
        latest_overall_state = newest_overall.get('presence_state', 'present')
        current_state = latest_current.get('presence_state', 'present') if latest_current else None
        latest_overall_stale = issued_at - newest_overall['observed_at'] > max_age_by_source[newest_overall['source']]

        # Keep a newer stale status transition from being overwritten by an
        # older fresh track. If the stale update agrees that the entity is
        # present, a still-valid coordinate from another source remains usable.
        if latest_current is None or (
            latest_overall_stale
            and (latest_overall_state in ('absent', 'destroyed')
                 or latest_overall_state != current_state)
        ):
            row = newest_overall
        else:
            row = latest_current
        source = row['source']
        age = issued_at - row['observed_at']
        stale = age > max_age_by_source[source]
        presence = row.get('presence_state', 'present')
        if stale:
            observability = 'stale'
            if presence == 'present':
                presence = 'unknown'
        elif presence == 'present':
            observability = 'current'
        else:
            observability = presence
        entity = {
            'entity_id': entity_id,
            'team': row['team'],
            'role': 'self' if row['team'] == self_team else 'opponent',
            'entity_type': row['entity_type'],
            'presence_state': presence,
            'observability_state': observability,
            'observed_at': row['observed_at'],
            'received_at': row['received_at'],
            'age_seconds': age,
            'source': source,
            'confidence': row['confidence'],
        }
        if observability == 'current':
            entity.update({key: value for key, value in row.items()
                           if key not in _CONTRACT_FIELDS})
        entities[entity_id] = entity
        state_counts[observability] += 1
        source_counts[source] = source_counts.get(source, 0) + 1

    return {
        'issued_at': issued_at,
        'self_team': self_team,
        'opponent_team': _TEAMS[self_team],
        'entities': entities,
        'visibility_summary': {
            'entity_count': len(entities),
            'by_observability_state': state_counts,
            'by_source': source_counts,
        },
    }
