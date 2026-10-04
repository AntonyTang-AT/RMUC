import json

import numpy as np

from forecast.entity import (ENTITY_SLOTS, ENTITY_TARGET_DIM, decode_entity_forecast,
                             ENTITY_ZONE_DIM, entity_target_vector,
                             entity_zone_target_vector)


def _example():
    return {
        "side": "红",
        "targets": {"10": {"truth": {"entities": {
            "r1": {"team": "红", "entity_type": "英雄", "position": [14.0, 7.5],
                   "hp": 100.0, "maxhp": 200.0},
            "b1": {"team": "蓝", "entity_type": "英雄", "position": [20.0, 7.5],
                   "hp": 200.0, "maxhp": 200.0},
        }}}},
    }


def test_entity_targets_are_fixed_slots_and_side_relative():
    value = entity_target_vector(_example(), 10)
    assert value.shape == (len(ENTITY_SLOTS), ENTITY_TARGET_DIM)
    self_hero = ENTITY_SLOTS.index(("self", "英雄"))
    opp_hero = ENTITY_SLOTS.index(("opponent", "英雄"))
    assert value[self_hero, 0] == 1.0
    assert value[opp_hero, 0] == 1.0
    assert value[self_hero, 3] == 0.5
    assert value[opp_hero, 1] > value[self_hero, 1]


def test_entity_target_excludes_explicitly_destroyed_lifecycle_but_keeps_unknown():
    example = {
        'side': '红',
        'targets': {'10': {'truth': {'entities': {
            'dead': {'team': '红', 'entity_type': '英雄', 'position': [4, 4],
                     'hp': None, 'life_state': 'destroyed'},
            'unknown': {'team': '红', 'entity_type': '英雄', 'position': [8, 4],
                        'hp': None, 'life_state': 'unknown'},
        }}}},
    }
    value = entity_target_vector(example, 10)
    self_hero = ENTITY_SLOTS.index(('self', '英雄'))
    assert value[self_hero, 0] == 1.0
    # The destroyed track is excluded; only the explicitly unknown track
    # contributes its position, so the slot remains forecastable.
    assert np.isclose(value[self_hero, 1] * 28.0, 8.0)


def test_entity_decoder_is_json_serializable_and_has_state():
    decoded = decode_entity_forecast(np.zeros((len(ENTITY_SLOTS), ENTITY_TARGET_DIM), dtype=np.float32))
    assert len(decoded) == len(ENTITY_SLOTS)
    assert all("position" in value and "hp_ratio" in value and "state" in value
               for value in decoded.values())
    json.dumps(decoded, ensure_ascii=False)

def test_entity_decoder_emits_zone_to_zone_transition():
    values = np.zeros((len(ENTITY_SLOTS), ENTITY_TARGET_DIM), dtype=np.float32)
    hero_idx = ENTITY_SLOTS.index(('self', '英雄'))
    # sigmoid(large) -> predicted position near (28,15), then projection must
    # still report a legal adjacent functional-zone move.
    values[hero_idx, 1] = 8.0
    values[hero_idx, 2] = 0.0
    decoded = decode_entity_forecast(
        values, observing_side='红',
        current_positions={'self:英雄': (10.0, 13.5)})
    item = decoded['self:英雄']
    assert item['from_zone'] == '红上公路'
    assert 'to_zone' in item and 'movement' in item
    assert ' → ' in item['movement']['description']
    assert item['movement']['legal_adjacent'] is True

def test_current_positions_decode_from_causal_feature_row():
    from forecast.entity import current_positions_from_feature_row
    row = np.zeros(len(ENTITY_SLOTS) * 10 + 7, dtype=np.float32)
    row[0] = 10.0 / 28.0
    row[1] = 13.5 / 15.0
    row[3] = 1.0
    row[9] = 1.0
    positions = current_positions_from_feature_row(row)
    assert positions['self:英雄'] == (10.0, 13.5)


def test_entity_zone_targets_and_zone_head_are_explicit():
    example = _example()
    target = entity_zone_target_vector(example, 10)
    self_hero = ENTITY_SLOTS.index(('self', '英雄'))
    assert target[self_hero] >= 0
    values = np.zeros((len(ENTITY_SLOTS), ENTITY_TARGET_DIM), dtype=np.float32)
    zone_logits = np.zeros((len(ENTITY_SLOTS), ENTITY_ZONE_DIM), dtype=np.float32)
    decoded = decode_entity_forecast(values, observing_side='红', zone_logits=zone_logits)
    assert decoded['self:英雄']['zone_head_predicted_zone'] is not None
    assert 1 <= len(decoded['self:英雄']['zone_head_probabilities']) <= 3


def test_entity_route_respects_speed_times_horizon():
    values = np.zeros((len(ENTITY_SLOTS), ENTITY_TARGET_DIM), dtype=np.float32)
    hero_idx = ENTITY_SLOTS.index(('self', '英雄'))
    values[hero_idx, 1] = 8.0
    values[hero_idx, 2] = 8.0
    decoded = decode_entity_forecast(
        values, observing_side='红', horizon_seconds=10,
        current_positions={'self:英雄': (10.0, 13.5)})
    item = decoded['self:英雄']
    movement = item['movement']
    assert movement['speed_mps'] == 0.4
    assert movement['max_travel_m'] == 4.0
    assert movement['distance_constraint'] == 'speed_x_horizon_disc'
    distance = ((item['position']['x'] - 10.0) ** 2 +
                (item['position']['y'] - 13.5) ** 2) ** 0.5
    assert distance <= movement['max_travel_m'] + 0.01


def test_long_horizon_can_use_multiple_legal_zone_edges():
    values = np.zeros((len(ENTITY_SLOTS), ENTITY_TARGET_DIM), dtype=np.float32)
    hero_idx = ENTITY_SLOTS.index(('self', '英雄'))
    values[hero_idx, 1] = 8.0
    values[hero_idx, 2] = 0.0
    current = {'self:英雄': (10.0, 13.5)}
    short = decode_entity_forecast(values, observing_side='红', horizon_seconds=10,
                                   current_positions=current)['self:英雄']['movement']
    long = decode_entity_forecast(values, observing_side='红', horizon_seconds=60,
                                  current_positions=current)['self:英雄']['movement']
    assert short['path_hops'] <= 1
    assert long['path_hops'] >= short['path_hops']
    assert long['legal_path'] is True


def test_reverse_position_projection_is_corrected_by_causal_velocity():
    values = np.zeros((len(ENTITY_SLOTS), ENTITY_TARGET_DIM), dtype=np.float32)
    hero_idx = ENTITY_SLOTS.index(('self', '英雄'))
    values[hero_idx, 1] = -8.0  # learned point lies behind current motion
    values[hero_idx, 2] = 0.0
    decoded = decode_entity_forecast(
        values, observing_side='红', horizon_seconds=10,
        current_positions={'self:英雄': (10.0, 13.5)},
        current_entity_context={'self:英雄': {
            'entity_type': '英雄', 'velocity_x': 0.6, 'velocity_y': 0.0,
            'speed_mps': 0.6, 'yaw': 180.0,
        }})
    item = decoded['self:英雄']
    assert item['direction_correction_applied'] is True
    assert item['position']['x'] >= 10.0
    assert item['movement']['motion_direction_deg'] == 0.0
    assert item['movement']['heading_deg'] == 180.0


def test_validation_position_fusion_uses_group_xy_velocity_and_fixed_horizon_weight():
    values = np.zeros((len(ENTITY_SLOTS), ENTITY_TARGET_DIM), dtype=np.float32)
    hero_idx = ENTITY_SLOTS.index(('self', '英雄'))
    # Neural x=10.8, 3-second group velocity extrapolates (10, 7.5) ->
    # (12, 7.5) at 10s; the .25 model weight gives x=11.7.
    hero_idx = ENTITY_SLOTS.index(('self', '英雄'))
    values[hero_idx, 1] = np.log((10.8 / 28.0) / (1.0 - 10.8 / 28.0))
    vectors = {'self:英雄': (0.2, 0.0)}
    decoded = decode_entity_forecast(
        values, observing_side='红', horizon_seconds=10,
        current_positions={'self:英雄': (10.0, 7.5)},
        current_entity_context={'self:英雄': {
            'entity_type': '英雄', 'velocity_x': -0.2, 'velocity_y': 0,
            'yaw': 90.0,
        }},
        position_fusion_weights={10: .25, 30: .75, 60: .75, 120: 1.0},
        position_motion_vectors=vectors)
    item = decoded['self:英雄']
    assert item['position']['x'] == 11.7
    assert item['position_fusion_applied'] is True
    assert item['position_fusion_model_weight'] == .25
    assert item['direction_correction_applied'] is False
    assert item['movement']['motion_direction_deg'] == 180.0  # causal velocity, not yaw
    assert item['movement']['heading_deg'] == 90.0


def test_validation_position_fusion_does_not_infer_motion_from_yaw():
    values = np.zeros((len(ENTITY_SLOTS), ENTITY_TARGET_DIM), dtype=np.float32)
    decoded = decode_entity_forecast(
        values, observing_side='红', horizon_seconds=10,
        current_positions={'self:英雄': (10.0, 7.5)},
        current_entity_context={'self:英雄': {'entity_type': '英雄', 'yaw': 90.0}},
        position_fusion_weights={10: .25}, position_motion_vectors={})
    item = decoded['self:英雄']
    assert item['position']['x'] == 11.0
    assert item['position']['y'] == 7.5
    assert item['position_fusion_applied'] is True
    assert item['position_fusion_velocity_available'] is False


def test_causal_group_centroid_velocity_is_exact_three_second_xy_difference():
    from forecast.entity import causal_group_centroid_motion

    def row(t, first, second, yaw):
        return {'issued_at': t, 'observable': {'self_entities': {
            'h1': {'entity_type': '英雄', 'position': [first, 4.0], 'yaw': yaw},
            'h2': {'entity_type': '英雄', 'position': [second, 4.0], 'yaw': -yaw},
            'dead': {'entity_type': '英雄', 'position': [100.0, 4.0], 'hp': 0},
        }, 'opponent_entities': {}}}

    previous = row(0.0, 2.0, 4.0, 0.0)
    current = row(3.0, 3.0, 5.0, 180.0)
    positions, vectors = causal_group_centroid_motion(current, previous)
    assert positions['self:英雄'] == (4.0, 4.0)
    assert np.allclose(vectors['self:英雄'], (1.0 / 3.0, 0.0))
    _, unavailable = causal_group_centroid_motion(row(6.1, 4.0, 6.0, 0.0), current)
    assert 'self:英雄' not in unavailable
