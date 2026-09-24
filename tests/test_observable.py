import pytest

from data.observable import MissingVisibilityProvider, build_observable_snapshot


def observation(entity_id='enemy-1', team='蓝', observed_at=10, received_at=10,
                source='radar', confidence=.9, position=(2.0, 3.0), **extra):
    return {
        'entity_id': entity_id, 'team': team, 'entity_type': '英雄',
        'observed_at': observed_at, 'received_at': received_at,
        'source': source, 'confidence': confidence, 'position': position,
        **extra,
    }


def snapshot(observations, issued_at=12, self_team='红'):
    return build_observable_snapshot(
        observations, issued_at, self_team, {'radar': 2, 'sensor': 2},
    )


def test_snapshot_maps_self_opponent_without_changing_global_team():
    snap = snapshot([
        observation(entity_id='red-1', team='红'),
        observation(entity_id='blue-1', team='蓝'),
    ], self_team='蓝')
    assert snap['self_team'] == '蓝'
    assert snap['opponent_team'] == '红'
    assert snap['entities']['blue-1']['team'] == '蓝'
    assert snap['entities']['blue-1']['role'] == 'self'
    assert snap['entities']['red-1']['team'] == '红'
    assert snap['entities']['red-1']['role'] == 'opponent'


@pytest.mark.parametrize('field', ['observed_at', 'received_at'])
def test_future_measurement_is_rejected(field):
    row = observation()
    row[field] = 13
    with pytest.raises(ValueError):
        snapshot([row])


def test_stale_enemy_track_has_no_current_position():
    enemy = snapshot([observation(observed_at=2, received_at=2)])['entities']['enemy-1']
    assert enemy['presence_state'] == 'unknown'
    assert enemy['observability_state'] == 'stale'
    assert enemy['observed_at'] == 2
    assert enemy['age_seconds'] == 10
    assert 'position' not in enemy


def test_latest_received_track_is_used_and_unobserved_position_is_not_filled():
    rows = [
        observation(observed_at=11, received_at=12, position=(4, 5)),
        observation(observed_at=10, received_at=10, position=(1, 2)),
    ]
    enemy = snapshot(rows)['entities']['enemy-1']
    assert enemy['position'] == (4, 5)
    assert enemy['observed_at'] == 11
    assert enemy['age_seconds'] == 1
    no_position = observation(entity_id='enemy-2')
    del no_position['position']
    assert 'position' not in snapshot([no_position])['entities']['enemy-2']


def test_expired_source_does_not_hide_current_measurement_from_another_source():
    rows = [
        observation(observed_at=10, received_at=10, source='sensor', position=(4, 5)),
        observation(observed_at=11, received_at=11, source='radar', position=(90, 91)),
    ]
    snap = build_observable_snapshot(rows, 12, '红', {'sensor': 5, 'radar': 0})
    enemy = snap['entities']['enemy-1']
    assert enemy['source'] == 'sensor'
    assert enemy['observability_state'] == 'current'
    assert enemy['position'] == (4, 5)


def test_newer_destruction_confirmation_does_not_revive_older_live_track():
    rows = [
        observation(observed_at=10, received_at=10, source='sensor', position=(4, 5)),
        observation(observed_at=11, received_at=11, source='radar', position=(90, 91),
                    presence_state='destroyed'),
    ]
    snap = build_observable_snapshot(rows, 12, '红', {'sensor': 5, 'radar': 0})
    enemy = snap['entities']['enemy-1']
    assert enemy['presence_state'] == 'destroyed'
    assert 'position' not in enemy


def test_newer_confirmed_absence_does_not_revive_older_live_track():
    rows = [
        observation(observed_at=10, received_at=10, source='sensor', position=(4, 5)),
        observation(observed_at=11, received_at=11, source='radar', position=(90, 91),
                    presence_state='absent'),
    ]
    snap = build_observable_snapshot(rows, 12, '红', {'sensor': 5, 'radar': 0})
    enemy = snap['entities']['enemy-1']
    assert enemy['presence_state'] == 'absent'
    assert 'position' not in enemy


def test_newer_stale_present_report_does_not_preserve_older_absence():
    rows = [
        observation(observed_at=10, received_at=10, source='sensor', position=None,
                    presence_state='absent'),
        observation(observed_at=11, received_at=11, source='radar', position=(90, 91),
                    presence_state='present'),
    ]
    snap = build_observable_snapshot(rows, 12, '红', {'sensor': 5, 'radar': 0})
    enemy = snap['entities']['enemy-1']
    assert enemy['presence_state'] == 'unknown'
    assert enemy['observability_state'] == 'stale'
    assert 'position' not in enemy


def test_unknown_absent_and_destroyed_are_distinct_without_coordinates():
    rows = [
        observation(entity_id='unknown', presence_state='unknown'),
        observation(entity_id='absent', presence_state='absent'),
        observation(entity_id='destroyed', presence_state='destroyed'),
    ]
    snap = snapshot(rows)
    for state in ('unknown', 'absent', 'destroyed'):
        entity = snap['entities'][state]
        assert entity['presence_state'] == state
        assert entity['observability_state'] == state
        assert 'position' not in entity


def test_missing_provider_and_replay_truth_are_rejected():
    with pytest.raises(MissingVisibilityProvider):
        snapshot(None)
    with pytest.raises(ValueError):
        snapshot([observation(source='global_replay')])


@pytest.mark.parametrize('value', [-.1, 1.1, float('nan')])
def test_invalid_confidence_is_rejected(value):
    with pytest.raises(ValueError):
        snapshot([observation(confidence=value)])
