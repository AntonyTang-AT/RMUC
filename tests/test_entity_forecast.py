import json

import numpy as np

from forecast.entity import (ENTITY_SLOTS, ENTITY_TARGET_DIM, decode_entity_forecast,
                             entity_target_vector)


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


def test_entity_decoder_is_json_serializable_and_has_state():
    decoded = decode_entity_forecast(np.zeros((len(ENTITY_SLOTS), ENTITY_TARGET_DIM), dtype=np.float32))
    assert len(decoded) == len(ENTITY_SLOTS)
    assert all("position" in value and "hp_ratio" in value and "state" in value
               for value in decoded.values())
    json.dumps(decoded, ensure_ascii=False)
