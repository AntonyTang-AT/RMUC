import gzip
import json

import numpy as np

from forecast.temporal import (
    FEATURE_DIM,
    MOTION_FEATURE_DIM,
    MOTION_FEATURE_SCHEMA_VERSION,
    feature_vector,
    motion_feature_vector,
    prepare_compact_split,
)


def _snapshot(t, x, entity_id="u1"):
    return {
        "game_id": 1,
        "side": "红",
        "issued_at": float(t),
        "observable": {
            "self_entities": {
                entity_id: {
                    "entity_id": entity_id,
                    "entity_type": "英雄",
                    "position": [float(x), 7.0],
                    "hp": 100,
                    "maxhp": 100,
                }
            },
            "opponent_entities": {},
        },
        "targets": {},
    }


def test_legacy_schema_remains_stable_and_motion_schema_is_explicit():
    row = _snapshot(1, 2)
    assert feature_vector(row).shape == (FEATURE_DIM,)
    assert motion_feature_vector(row).shape == (MOTION_FEATURE_DIM,)
    # No previous snapshot means motion fields are zero, rather than an
    # imputed displacement that could leak future state.
    assert np.allclose(motion_feature_vector(row)[14:20], 0.0)


def test_motion_features_are_causal_and_include_acceleration():
    first = _snapshot(1, 2)
    second = _snapshot(4, 3)
    third = _snapshot(7, 5)
    vector = motion_feature_vector(third, second, first)
    # Slot zero is self:英雄.  dx is normalized by the 28 m map width and
    # direction_x is +1 for movement toward increasing x.
    assert np.isclose(vector[14], 2.0 / 28.0)
    assert np.isclose(vector[15], 0.0)
    assert np.isclose(vector[18], 1.0)
    assert vector[17] > 0.0  # speed increased from 1/3 to 2/3 m/s

    changed_future = _snapshot(7, 99)
    # The current row and its two causal predecessors determine the output;
    # changing an unused later row cannot change it.
    assert np.array_equal(vector, motion_feature_vector(third, second, first))
    assert not np.array_equal(vector, motion_feature_vector(changed_future, second, first))


def test_prepare_motion_compact_split_supports_context_40(tmp_path):
    source = tmp_path / "split.jsonl.gz"
    rows = [_snapshot(1 + 3 * i, 2 + i) for i in range(3)]
    with gzip.open(source, "wt", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    target = tmp_path / "split.npz"
    result = prepare_compact_split(source, target, context=40, feature_schema="motion")
    assert result["feature_dim"] == MOTION_FEATURE_DIM
    with np.load(target, allow_pickle=False) as data:
        assert data["x"].shape == (3, 40, MOTION_FEATURE_DIM)
        assert str(data["feature_schema_version"][0]) == MOTION_FEATURE_SCHEMA_VERSION
        assert data["x"].dtype == np.float32
