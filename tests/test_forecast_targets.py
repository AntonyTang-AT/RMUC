import pytest

from data.forecast_targets import build_side_examples, build_targets
from data.observable import MissingVisibilityProvider


def truth(t, red_x, blue_x):
    return {'target_at': t, 'entities': {
        'red': {'team': '红', 'x': red_x},
        'blue': {'team': '蓝', 'x': blue_x},
    }}


def observation(t=10, side='红'):
    return {'issued_at': t, 'self_team': side, 'entities': {
        'red': {'entity_id': 'red', 'team': '红', 'observed_at': t,
                'received_at': t, 'source': 'sensor',
                'observability_state': 'current', 'x': 1},
        'blue': {'entity_id': 'blue', 'team': '蓝', 'observed_at': t,
                 'received_at': t, 'source': 'radar',
                 'observability_state': 'current', 'x': 9},
    }}


def test_target_uses_exact_requested_future_time():
    truths = {10: truth(10, 1, 9), 19: truth(19, 5, 5), 20: truth(20, 2, 8)}
    result = build_targets(truths, 10, horizons=(10,), side='红')
    assert result[10] == {'target_at': 20, 'truth': truth(20, 2, 8), 'censored': False}


def test_missing_exact_time_is_censored_even_with_later_truth():
    result = build_targets({10: truth(10, 1, 9), 19: truth(19, 5, 5),
                            21: truth(21, 2, 8)}, 10, horizons=(10,), side='蓝')
    assert result[10] is None


def test_end_of_game_censors_each_horizon_independently():
    truths = {10: truth(10, 1, 9), 20: truth(20, 2, 8)}
    result = build_targets(truths, 10, horizons=(10, 30), side='红')
    assert result[10]['target_at'] == 20
    assert result[30] is None


def test_side_examples_swap_roles_but_keep_raw_teams_and_future_truth():
    truths = {10: truth(10, 1, 9), 20: truth(20, 2, 8)}
    red = build_side_examples(42, [observation()], truths, '红', horizons=(10,))
    blue = build_side_examples(42, [observation(side='蓝')], truths, '蓝', horizons=(10,))
    assert red[0]['observable']['self_entities'] == blue[0]['observable']['opponent_entities']
    assert red[0]['observable']['opponent_entities'] == blue[0]['observable']['self_entities']
    assert red[0]['observable']['self_entities']['red']['team'] == '红'
    assert blue[0]['observable']['self_entities']['blue']['team'] == '蓝'
    assert red[0]['targets'][10] == blue[0]['targets'][10]
    assert red[0]['issued_at'] == blue[0]['issued_at'] == 10
    assert red[0]['observable']['self_entities']['red']['x'] == 1
    assert red[0]['targets'][10]['truth']['entities']['red']['x'] == 2


def test_missing_observable_provider_fails_closed():
    with pytest.raises(MissingVisibilityProvider):
        build_side_examples(42, None, {20: truth(20, 2, 8)}, '红', horizons=(10,))


def test_future_measurement_cannot_enter_feature_snapshot():
    obs = observation()
    obs['entities']['blue']['observed_at'] = 11
    with pytest.raises(ValueError, match='future'):
        build_side_examples(42, [obs], {20: truth(20, 2, 8)}, '红', horizons=(10,))


def test_global_replay_source_cannot_masquerade_as_sensor_input():
    obs = observation()
    obs['entities']['blue']['source'] = 'global_truth'
    with pytest.raises(MissingVisibilityProvider):
        build_side_examples(42, [obs], {20: truth(20, 2, 8)}, '红', horizons=(10,))


def test_stale_or_unknown_measured_fields_are_removed_from_features():
    obs = observation()
    obs['entities']['blue'].update({
        'observability_state': 'stale', 'position': [999, 999], 'x': 999,
    })
    example = build_side_examples(42, [obs], {20: truth(20, 2, 8)}, '红', horizons=(10,))[0]
    enemy = example['observable']['opponent_entities']['blue']
    assert enemy['observability_state'] == 'stale'
    assert 'position' not in enemy and 'x' not in enemy
