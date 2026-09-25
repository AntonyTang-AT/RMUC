"""Per-type entity trajectory targets and output helpers.

The global heads answer *what the match is doing*.  These slots answer *where
each robot class is and in what condition*.  Slots are relative to the
current observing side, so the same model works for red and blue views.
"""

from __future__ import annotations

import math

import numpy as np

from .temporal import ENTITY_TYPES, HORIZONS, MAP_X, MAP_Y


ENTITY_SIDES = ("self", "opponent")
ENTITY_SLOTS = tuple((side, entity_type) for side in ENTITY_SIDES
                     for entity_type in ENTITY_TYPES)
ENTITY_TARGET_FIELDS = ("presence", "x", "y", "hp_ratio", "count")
ENTITY_TARGET_DIM = len(ENTITY_TARGET_FIELDS)


def _number(value, default=0.0):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return float(default)
    return value if math.isfinite(value) else float(default)


def _entities(observable, group):
    values = (observable or {}).get(group) or {}
    return list(values.values()) if isinstance(values, dict) else list(values)


def _hp_ratio(entity):
    hp = _number(entity.get("hp"), 0.0)
    maxhp = _number(entity.get("maxhp"), 0.0)
    scale = maxhp if maxhp > 0 else 5000.0
    return max(0.0, min(1.0, hp / scale))


def _slot_value(entities, entity_type):
    matching = [entity for entity in entities
                if entity.get("entity_type") == entity_type]
    if not matching:
        return np.zeros(ENTITY_TARGET_DIM, dtype=np.float32)
    positions = [entity.get("position") for entity in matching
                 if isinstance(entity.get("position"), (list, tuple))
                 and len(entity.get("position")) >= 2]
    x = sum(_number(position[0]) for position in positions) / max(1, len(positions))
    y = sum(_number(position[1]) for position in positions) / max(1, len(positions))
    hp = sum(_hp_ratio(entity) for entity in matching) / len(matching)
    return np.asarray([
        1.0,
        max(0.0, min(1.0, x / MAP_X)),
        max(0.0, min(1.0, y / MAP_Y)),
        hp,
        max(0.0, min(1.0, len(matching) / 2.0)),
    ], dtype=np.float32)


def entity_target_vector(example, horizon):
    """Return ``[slots, fields]`` future targets for one causal example.

    A missing slot is an explicit absent prediction (presence=0); no future
    feature is copied into the input window.
    """
    targets = example.get("targets") or {}
    item = targets.get(str(horizon), targets.get(horizon))
    out = np.zeros((len(ENTITY_SLOTS), ENTITY_TARGET_DIM), dtype=np.float32)
    if not isinstance(item, dict) or item.get("censored"):
        out.fill(-1.0)
        return out
    truth = item.get("truth") or {}
    entities = list((truth.get("entities") or {}).values())
    side = example.get("side", "红")
    grouped = {"self": [], "opponent": []}
    for entity in entities:
        grouped["self" if entity.get("team") == side else "opponent"].append(entity)
    for index, (group, entity_type) in enumerate(ENTITY_SLOTS):
        out[index] = _slot_value(grouped[group], entity_type)
    return out


def slot_name(index):
    group, entity_type = ENTITY_SLOTS[int(index)]
    return f"{group}:{entity_type}"


def decode_entity_forecast(values, presence_temperature=1.0, observing_side=None):
    """Convert model output ``[slots, 5]`` to a JSON-serializable forecast."""
    values = np.asarray(values, dtype=np.float32)
    if values.shape != (len(ENTITY_SLOTS), ENTITY_TARGET_DIM):
        raise ValueError(f"expected {(len(ENTITY_SLOTS), ENTITY_TARGET_DIM)}, got {values.shape}")
    output = {}
    for index, (group, entity_type) in enumerate(ENTITY_SLOTS):
        raw = values[index]
        presence = 1.0 / (1.0 + math.exp(-float(raw[0]) / max(1e-6, presence_temperature)))
        continuous = 1.0 / (1.0 + math.exp(-float(raw[1])))
        continuous_y = 1.0 / (1.0 + math.exp(-float(raw[2])))
        continuous_hp = 1.0 / (1.0 + math.exp(-float(raw[3])))
        continuous_count = 1.0 / (1.0 + math.exp(-float(raw[4])))
        output[slot_name(index)] = {
            "side": group,
            "team": (observing_side if group == "self" else
                      ("蓝" if observing_side == "红" else "红"))
            if observing_side in ("红", "蓝") else None,
            "entity_type": entity_type,
            "presence_probability": round(presence, 4),
            "position": {
                "x": round(continuous * MAP_X, 3),
                "y": round(continuous_y * MAP_Y, 3),
            },
            "hp_ratio": round(continuous_hp, 4),
            "count": round(continuous_count * 2.0, 3),
            "state": "absent" if presence < 0.35 else (
                "critical" if continuous_hp < 0.3 else "active"),
        }
    return output
