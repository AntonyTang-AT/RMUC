"""Per-type entity trajectory targets and output helpers.

The global heads answer *what the match is doing*.  These slots answer *where
each robot class is and in what condition*.  Slots are relative to the
current observing side, so the same model works for red and blue views.
"""

from __future__ import annotations

import math

import numpy as np

from .temporal import (
    ENTITY_TYPES, HORIZONS, MAP_X, MAP_Y, FEATURES_PER_SLOT,
    MOTION_FEATURES_PER_SLOT, MOTION_FEATURE_DIM, FEATURE_DIM,
)
from worldstate import functional_zone as fz
from worldstate.canonical_zone import CANONICAL_ZONE_DIM, CANONICAL_ZONE_LABELS, locate_zone
from worldstate.unit_state import derive_unit_special_state, forecast_special_state


ENTITY_SIDES = ("self", "opponent")
ENTITY_SLOTS = tuple((side, entity_type) for side in ENTITY_SIDES
                     for entity_type in ENTITY_TYPES)
ENTITY_TARGET_FIELDS = ("presence", "x", "y", "hp_ratio", "count")
ENTITY_TARGET_DIM = len(ENTITY_TARGET_FIELDS)
ENTITY_ZONE_LABELS = tuple(fz.FUNCTIONAL_ZONE_LABELS)
ENTITY_ZONE_DIM = len(ENTITY_ZONE_LABELS)
ENTITY_CANONICAL_ZONE_LABELS = CANONICAL_ZONE_LABELS
ENTITY_CANONICAL_ZONE_DIM = CANONICAL_ZONE_DIM

# Effective movement priors from the project physics contract.  These are
# deliberately kept next to the forecast decoder so a trajectory prediction
# cannot jump across the map simply because the neural position head is noisy.
# They are priors, not hidden future observations; a sensor supplied
# ``speed_mps``/``velocity_mps`` on the current entity overrides the prior.
ROBOT_SPEED_MPS = {
    '哨兵': 0.10,
    '工程': 0.35,
    '英雄': 0.40,
    '步兵3': 0.70,
    '步兵4': 0.70,
    '空中': 2.50,
}


def _transit_origin_zone(entity_type, functional_zone):
    """Map a module footprint to its adjacent transit lane for route output.

    Interaction modules remain available as ``functional_zone``.  For a
    mobile unit that is merely passing through an energy-mechanism footprint,
    the route contract uses the connected road as its movement origin.  This
    keeps route labels stable across the legacy trajectory data while still
    preserving the exact module in the state sidecar.
    """
    if entity_type in ('英雄', '步兵3', '步兵4', '哨兵', '空中'):
        return {
            '红能量机关区': '红上公路',
            '蓝能量机关区': '蓝下公路',
        }.get(functional_zone, functional_zone)
    return functional_zone


def _number(value, default=0.0):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return float(default)
    return value if math.isfinite(value) else float(default)


def _entities(observable, group):
    values = (observable or {}).get(group) or {}
    return list(values.values()) if isinstance(values, dict) else list(values)


def causal_group_centroid_motion(current_example, previous_example=None,
                                 expected_delta_seconds=3.0):
    """Group mean XY and exact-cadence causal velocity from observable rows.

    This matches ``entity_target_vector``'s side/type group-mean position
    contract. It only inspects the two supplied observable snapshots. Yaw is
    deliberately ignored because the available field is muzzle/gimbal yaw,
    not chassis heading.
    """
    def points_by_slot(example):
        obs = (example or {}).get("observable") or {}
        result = {}
        for role, group_key in (("self", "self_entities"), ("opponent", "opponent_entities")):
            values = _entities(obs, group_key)
            for entity_type in ENTITY_TYPES:
                pts = []
                for entity in values:
                    if entity.get("entity_type") != entity_type or not _alive_truth(entity):
                        continue
                    pos = entity.get("position")
                    if not isinstance(pos, (list, tuple)) or len(pos) < 2:
                        continue
                    x, y = _number(pos[0], math.nan), _number(pos[1], math.nan)
                    if math.isfinite(x) and math.isfinite(y):
                        pts.append((x, y))
                if pts:
                    result[f"{role}:{entity_type}"] = (
                        sum(p[0] for p in pts) / len(pts),
                        sum(p[1] for p in pts) / len(pts))
        return result

    current_positions = points_by_slot(current_example)
    vectors = {}
    if previous_example is None:
        return current_positions, vectors
    dt = (_number((current_example or {}).get("issued_at"), 0.0) -
          _number((previous_example or {}).get("issued_at"), 0.0))
    if abs(dt - float(expected_delta_seconds)) > 1e-4:
        return current_positions, vectors
    previous_positions = points_by_slot(previous_example)
    for slot, current in current_positions.items():
        previous = previous_positions.get(slot)
        if previous is None:
            continue
        vx, vy = ((current[0] - previous[0]) / dt,
                  (current[1] - previous[1]) / dt)
        speed = math.hypot(vx, vy)
        entity_type = slot.split(":", 1)[1]
        cap = float(ROBOT_SPEED_MPS.get(entity_type, .35))
        if speed > cap and speed > 1e-8:
            scale = cap / speed
            vx, vy = vx * scale, vy * scale
        vectors[slot] = (vx, vy)
    return current_positions, vectors


def _hp_ratio(entity):
    hp = _number(entity.get("hp"), 0.0)
    maxhp = _number(entity.get("maxhp"), 0.0)
    scale = maxhp if maxhp > 0 else 5000.0
    return max(0.0, min(1.0, hp / scale))


def _alive_truth(entity):
    """Return false only for explicit death evidence in the supplied record.

    Life-state labels are produced by the visibility/lifecycle layer and are
    useful when a track has no numeric HP. Unknown, stale, and missing states
    remain eligible; they are not evidence of death.
    """
    hp = entity.get('hp')
    hp_dead = (isinstance(hp, (int, float)) and not isinstance(hp, bool)
               and math.isfinite(float(hp)) and hp <= 0)
    return not (hp_dead or entity.get('life_state') == 'destroyed')


def _slot_value(entities, entity_type):
    matching = [entity for entity in entities
                if entity.get("entity_type") == entity_type and _alive_truth(entity)]
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


def entity_zone_target_vector(example, horizon):
    """Return one causal functional-zone class per entity slot.

    Absent entities and positions censored by the visibility contract are
    masked with ``-1``.  The presence head remains responsible for existence;
    this head only teaches the model to choose a named functional zone.
    """
    out = np.full((len(ENTITY_SLOTS),), -1, dtype=np.int64)
    targets = example.get('targets') or {}
    item = targets.get(str(horizon), targets.get(horizon))
    if not isinstance(item, dict) or item.get('censored'):
        return out
    truth = item.get('truth') or {}
    entities = list((truth.get('entities') or {}).values())
    side = example.get('side', '红')
    grouped = {'self': [], 'opponent': []}
    for entity in entities:
        grouped['self' if entity.get('team') == side else 'opponent'].append(entity)
    for index, (group, entity_type) in enumerate(ENTITY_SLOTS):
        matching = [entity for entity in grouped[group]
                    if entity.get('entity_type') == entity_type and _alive_truth(entity)]
        if not matching:
            continue
        position = matching[0].get('position')
        if not isinstance(position, (list, tuple)) or len(position) < 2:
            continue
        entity = matching[0]
        entity_team = entity.get('team', side)
        if entity_type == '基地':
            zone = '红基地' if entity_team == '红' else '蓝基地'
        elif entity_type == '前哨站':
            zone = '红前哨区' if entity_team == '红' else '蓝前哨区'
        else:
            zone = fz.assign_functional_zone(float(position[0]), float(position[1]))
        if zone in ENTITY_ZONE_LABELS:
            out[index] = ENTITY_ZONE_LABELS.index(zone)
    return out


def canonical_zone_target_vector(example, horizon):
    """Return canonical JSON zone labels for each future entity slot."""
    out = np.full((len(ENTITY_SLOTS),), -1, dtype=np.int64)
    targets = example.get('targets') or {}
    item = targets.get(str(horizon), targets.get(horizon))
    if not isinstance(item, dict) or item.get('censored'):
        return out
    truth = item.get('truth') or {}
    entities = list((truth.get('entities') or {}).values())
    side = example.get('side', '红')
    grouped = {'self': [], 'opponent': []}
    for entity in entities:
        grouped['self' if entity.get('team') == side else 'opponent'].append(entity)
    for index, (group, entity_type) in enumerate(ENTITY_SLOTS):
        matching = [entity for entity in grouped[group]
                    if entity.get('entity_type') == entity_type and _alive_truth(entity)]
        if not matching:
            continue
        position = matching[0].get('position')
        if not isinstance(position, (list, tuple)) or len(position) < 2:
            continue
        zone = locate_zone(float(position[0]), float(position[1]))
        if zone in ENTITY_CANONICAL_ZONE_LABELS:
            out[index] = ENTITY_CANONICAL_ZONE_LABELS.index(zone)
    return out


def slot_name(index):
    group, entity_type = ENTITY_SLOTS[int(index)]
    return f"{group}:{entity_type}"


def _entity_speed_mps(entity_type, context_entity):
    """Resolve the causal speed available at issue time."""
    context_entity = context_entity or {}
    for key in ('speed_mps', 'speed', 'velocity_mps'):
        value = context_entity.get(key)
        if isinstance(value, (int, float)) and math.isfinite(float(value)) and float(value) >= 0:
            return float(value), 'sensor_or_state'
    velocity = context_entity.get('velocity')
    if isinstance(velocity, (list, tuple)) and len(velocity) >= 2:
        try:
            value = math.hypot(float(velocity[0]), float(velocity[1]))
            if math.isfinite(value):
                return max(0.0, value), 'sensor_or_state'
        except (TypeError, ValueError):
            pass
    vx, vy = context_entity.get('velocity_x'), context_entity.get('velocity_y')
    if isinstance(vx, (int, float)) and isinstance(vy, (int, float)):
        value = math.hypot(float(vx), float(vy))
        if math.isfinite(value):
            return max(0.0, value), 'causal_position_delta'
    return float(ROBOT_SPEED_MPS.get(entity_type, 0.35)), 'project_prior'


def _distance_xy(a, b):
    if a is None or b is None:
        return None
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


def _causal_motion_vector(context_entity):
    """Read a causal velocity vector and optional orientation metadata."""
    context_entity = context_entity or {}
    vx, vy = context_entity.get('velocity_x'), context_entity.get('velocity_y')
    if isinstance(vx, (int, float)) and isinstance(vy, (int, float)):
        vector = (float(vx), float(vy))
    else:
        velocity = context_entity.get('velocity')
        vector = (float(velocity[0]), float(velocity[1])) if (
            isinstance(velocity, (list, tuple)) and len(velocity) >= 2 and
            all(isinstance(value, (int, float)) for value in velocity[:2])) else None
    if vector is None:
        return None, None, context_entity.get('yaw')
    speed = math.hypot(*vector)
    if speed <= 1e-8:
        return None, 0.0, context_entity.get('yaw')
    return (vector[0] / speed, vector[1] / speed), speed, context_entity.get('yaw')


def _reachable_target_zone(unit, team, from_zone, raw_zone, raw_xy, from_xy,
                           horizon_seconds, speed_mps):
    """Choose a physically reachable route target for the requested horizon.

    A ten-second forecast is intentionally local (stay or one edge).  Longer
    forecasts may cross several functional-zone edges, but only when the
    accumulated centroid route length fits ``speed × horizon`` and every edge
    is legal for the unit/team.  This keeps the short forecast precise without
    making the 30/60/120 second forecasts artificially one-zone predictions.
    """
    if not from_zone:
        return from_zone, True, None, None, [from_zone] if from_zone else []
    horizon = max(0.0, float(horizon_seconds))
    max_distance = max(0.0, float(speed_mps) * horizon)
    # Decision semantics are every 3 s, but the useful spatial detail decays
    # at a ten-second scale: 10s=1 edge, 30s=3, 60s=6, 120s=12.
    max_hops = max(1, int(math.floor(horizon / 10.0 + 1e-9)))
    if horizon <= 10.0 + 1e-9:
        max_hops = 1
    reachable = {from_zone: (0.0, [from_zone])}
    frontier = [(from_zone, 0, 0.0, [from_zone])]
    while frontier:
        zone, hops, distance, path = frontier.pop(0)
        if hops >= max_hops:
            continue
        for nxt in fz.functional_zone_neighbors(zone):
            if not fz.can_move_between(unit, team, zone, nxt):
                continue
            a, b = fz.zone_centroid(zone), fz.zone_centroid(nxt)
            edge = _distance_xy(a, b) if a is not None and b is not None else 0.0
            next_distance = distance + max(0.0, float(edge or 0.0))
            if next_distance > max_distance + 1e-6:
                continue
            next_path = path + [nxt]
            old = reachable.get(nxt)
            if old is None or next_distance < old[0]:
                reachable[nxt] = (next_distance, next_path)
                frontier.append((nxt, hops + 1, next_distance, next_path))
    if raw_zone in reachable:
        chosen = raw_zone
    else:
        target = raw_xy or from_xy or (0.0, 0.0)
        def score(item):
            zone, (_, path) = item
            center = fz.zone_centroid(zone)
            # Prefer a target close to the model point, with route distance as
            # a tie-breaker so a long forecast does not jump across a wall.
            point_distance = (_distance_xy(center, target)
                              if center is not None else float('inf'))
            return point_distance, len(path), reachable[zone][0]
        chosen = min(reachable.items(), key=score)[0]
    route_distance, path = reachable[chosen]
    center = fz.zone_centroid(chosen)
    direct_distance = _distance_xy(from_xy, center)
    travel_time = (route_distance / speed_mps
                   if speed_mps > 1e-9 else (0.0 if route_distance == 0 else None))
    # Report route distance for reachability; retain the straight-line value
    # separately for diagnostics and UI context.
    return chosen, bool(chosen != raw_zone), route_distance, travel_time, path



def current_positions_from_feature_row(feature_row):
    """Decode current positions from the causal feature row for zone transitions."""
    values = np.asarray(feature_row, dtype=np.float32).reshape(-1)
    # Keep old 10-field compact rows readable for existing reports/checkpoints;
    # newly generated rows use the 14-field canonical-zone schema.
    if values.size >= MOTION_FEATURE_DIM:
        slot_width = MOTION_FEATURES_PER_SLOT
    elif values.size >= FEATURE_DIM:
        slot_width = FEATURES_PER_SLOT
    elif values.size >= len(ENTITY_SLOTS) * 10:
        slot_width = 10
    else:
        raise ValueError(f"feature row has {values.size} values; need at least {len(ENTITY_SLOTS) * 10}")
    out = {}
    for index, (group, entity_type) in enumerate(ENTITY_SLOTS):
        base = index * slot_width
        has_position = values[base + 3] > 0.5
        present = values[base + 9] > 0.5
        if not (has_position and present):
            continue
        out[f"{group}:{entity_type}"] = (
            float(values[base] * MAP_X), float(values[base + 1] * MAP_Y))
    return out

def decode_entity_forecast(values, presence_temperature=1.0, observing_side=None,
                           current_positions=None, current_entity_context=None,
                           current_entities=None, horizon_seconds=10,
                           zone_logits=None, canonical_zone_logits=None,
                           position_fusion_weights=None,
                           position_motion_vectors=None):
    """Convert model output to JSON with functional-zone movement contracts.

    ``current_positions`` is an optional mapping such as
    ``{"self:英雄": (x_m, y_m)}``. When supplied, a continuous model position
    is projected onto a speed- and horizon-reachable legal functional route
    and emitted as ``from_zone -> to_zone``. The raw model zone remains
    available for audit.
    """
    values = np.asarray(values, dtype=np.float32)
    if values.shape != (len(ENTITY_SLOTS), ENTITY_TARGET_DIM):
        raise ValueError(f"expected {(len(ENTITY_SLOTS), ENTITY_TARGET_DIM)}, got {values.shape}")
    current_positions = current_positions or {}
    current_entity_context = current_entity_context or {}
    current_entities = list(current_entities or current_entity_context.values())
    position_motion_vectors = position_motion_vectors or {}
    fusion_weight = None
    if isinstance(position_fusion_weights, dict):
        fusion_weight = position_fusion_weights.get(int(horizon_seconds))
        if fusion_weight is None:
            fusion_weight = position_fusion_weights.get(str(int(horizon_seconds)))
        if fusion_weight is not None:
            fusion_weight = max(0.0, min(1.0, float(fusion_weight)))
    if zone_logits is not None:
        zone_logits = np.asarray(zone_logits, dtype=np.float32)
        if zone_logits.shape != (len(ENTITY_SLOTS), ENTITY_ZONE_DIM):
            raise ValueError(f"expected zone logits {(len(ENTITY_SLOTS), ENTITY_ZONE_DIM)}, got {zone_logits.shape}")
    if canonical_zone_logits is not None:
        canonical_zone_logits = np.asarray(canonical_zone_logits, dtype=np.float32)
        if canonical_zone_logits.shape != (len(ENTITY_SLOTS), ENTITY_CANONICAL_ZONE_DIM):
            raise ValueError(
                f"expected canonical zone logits "
                f"{(len(ENTITY_SLOTS), ENTITY_CANONICAL_ZONE_DIM)}, "
                f"got {canonical_zone_logits.shape}")
    output = {}
    for index, (group, entity_type) in enumerate(ENTITY_SLOTS):
        raw = values[index]
        presence = 1.0 / (1.0 + math.exp(-float(raw[0]) / max(1e-6, presence_temperature)))
        continuous = 1.0 / (1.0 + math.exp(-float(raw[1])))
        continuous_y = 1.0 / (1.0 + math.exp(-float(raw[2])))
        continuous_hp = 1.0 / (1.0 + math.exp(-float(raw[3])))
        continuous_count = 1.0 / (1.0 + math.exp(-float(raw[4])))
        position = {"x": round(continuous * MAP_X, 3),
                    "y": round(continuous_y * MAP_Y, 3)}
        slot = f"{group}:{entity_type}"
        team = (observing_side if group == "self" else
                ("蓝" if observing_side == "红" else "红"))
        raw_zone = fz.assign_functional_zone(position["x"], position["y"])
        zone_head_predicted = None
        zone_head_probabilities = None
        zone_head_used = False
        if zone_logits is not None:
            logits = zone_logits[index] - np.max(zone_logits[index])
            probs = np.exp(logits)
            probs = probs / max(float(np.sum(probs)), 1e-12)
            order = np.argsort(probs)[::-1]
            zone_head_predicted = ENTITY_ZONE_LABELS[int(order[0])]
            zone_head_probabilities = {
                ENTITY_ZONE_LABELS[int(i)]: round(float(probs[int(i)]), 6)
                for i in order[:3]
            }
        canonical_zone_predicted = None
        canonical_zone_probabilities = None
        if canonical_zone_logits is not None:
            logits = canonical_zone_logits[index] - np.max(canonical_zone_logits[index])
            probs = np.exp(logits)
            probs = probs / max(float(np.sum(probs)), 1e-12)
            order = np.argsort(probs)[::-1]
            canonical_zone_predicted = ENTITY_CANONICAL_ZONE_LABELS[int(order[0])]
            canonical_zone_probabilities = {
                ENTITY_CANONICAL_ZONE_LABELS[int(i)]: round(float(probs[int(i)]), 6)
                for i in order[:3]
            }
        from_xy = current_positions.get(slot)
        current_functional_zone = (fz.assign_functional_zone(from_xy[0], from_xy[1])
                                   if from_xy is not None else None)
        current_canonical_zone = (locate_zone(from_xy[0], from_xy[1])
                                  if from_xy is not None else None)
        from_zone = (_transit_origin_zone(entity_type, current_functional_zone)
                     if current_functional_zone is not None else None)
        context_entity = current_entity_context.get(slot)
        if context_entity is None:
            context_entity = {
                "entity_type": entity_type,
                "team": team,
                "role": group,
            }
            if from_xy is not None:
                context_entity["position"] = list(from_xy)
        motion_direction, observed_speed, observed_yaw = _causal_motion_vector(context_entity)
        direction_correction = False
        position_fusion_applied = False
        position_fusion_velocity_available = False
        # Validation-derived candidate: blend the neural group-mean point with
        # a causal group-centroid extrapolation at a horizon-specific fixed
        # weight. It is opt-in and uses XY differences only; yaw is not used.
        if fusion_weight is not None and from_xy is not None:
            velocity = position_motion_vectors.get(slot)
            if (isinstance(velocity, (list, tuple)) and len(velocity) >= 2 and
                    all(isinstance(v, (int, float)) and math.isfinite(float(v))
                        for v in velocity[:2])):
                kinematic = (float(from_xy[0]) + float(velocity[0]) * max(0.0, float(horizon_seconds)),
                             float(from_xy[1]) + float(velocity[1]) * max(0.0, float(horizon_seconds)))
                position_fusion_velocity_available = True
            else:
                # Match offline validation scoring: absent causal velocity
                # falls back to current position for the motion component.
                kinematic = (float(from_xy[0]), float(from_xy[1]))
            position = {
                "x": round(fusion_weight * position["x"] + (1.0 - fusion_weight) * kinematic[0], 3),
                "y": round(fusion_weight * position["y"] + (1.0 - fusion_weight) * kinematic[1], 3),
            }
            position_fusion_applied = True
            raw_zone = fz.assign_functional_zone(position["x"], position["y"])
        # The learned position head can occasionally point behind a robot that
        # has an observed velocity.  Blend only those reverse projections with
        # a causal kinematic point; the blend decays with horizon because long
        # term steering is less certain.  No opponent yaw is fabricated.
        if fusion_weight is None and from_xy is not None and motion_direction is not None:
            raw_dx = float(position["x"]) - float(from_xy[0])
            raw_dy = float(position["y"]) - float(from_xy[1])
            forward_component = raw_dx * motion_direction[0] + raw_dy * motion_direction[1]
            if forward_component < -1e-4:
                raw_distance = math.hypot(raw_dx, raw_dy)
                travel = min(raw_distance,
                             max(0.0, float(observed_speed or 0.0)) *
                             max(0.0, float(horizon_seconds)))
                kinematic = (float(from_xy[0]) + motion_direction[0] * travel,
                             float(from_xy[1]) + motion_direction[1] * travel)
                blend = max(0.25, min(0.85, 1.0 - float(horizon_seconds) / 80.0))
                position = {
                    "x": round((1.0 - blend) * position["x"] + blend * kinematic[0], 3),
                    "y": round((1.0 - blend) * position["y"] + blend * kinematic[1], 3),
                }
                direction_correction = True
                raw_zone = fz.assign_functional_zone(position["x"], position["y"])
        # Preserve the continuous model point before the legal route/speed
        # projection below.  Evaluation and diagnostics must be able to
        # distinguish the learned forecast from the safety-feasible route
        # that is shown to a controller/UI.
        unprojected_position = dict(position)
        unprojected_canonical_zone = locate_zone(
            unprojected_position["x"], unprojected_position["y"])
        # Keep the continuous/adjacency projection as the production route.
        # The categorical head is retained as a candidate for audit and can
        # be enabled after it passes its acceptance gate; using it directly
        # here made the experimental V4 checkpoint less stable under class
        # imbalance.
        to_zone = raw_zone
        projected = False
        movement = None
        if from_zone is not None and team in ("红", "蓝"):
            speed_mps, speed_source = _entity_speed_mps(entity_type, context_entity)
            to_zone, projected, target_distance_m, travel_time_s, route_path = _reachable_target_zone(
                entity_type, team, from_zone, raw_zone,
                (position["x"], position["y"]), from_xy,
                horizon_seconds, speed_mps)
            max_travel_m = speed_mps * max(0.0, float(horizon_seconds))
            raw_distance_m = _distance_xy(from_xy, (position["x"], position["y"]))
            if from_xy is not None and raw_distance_m is not None and raw_distance_m > max_travel_m + 1e-6:
                # The continuous head is still retained as raw evidence, but
                # the actionable point is clipped to the physically reachable
                # disc around the observed position.
                if max_travel_m <= 1e-9:
                    position = {"x": round(float(from_xy[0]), 3),
                                "y": round(float(from_xy[1]), 3)}
                else:
                    ratio = max_travel_m / raw_distance_m
                    position = {
                        "x": round(float(from_xy[0]) + (position["x"] - float(from_xy[0])) * ratio, 3),
                        "y": round(float(from_xy[1]) + (position["y"] - float(from_xy[1])) * ratio, 3),
                    }
                projected = True
            movement = fz.transition_record(
                entity_type, team, from_zone, to_zone,
                probability=presence)
            movement["raw_to_zone"] = raw_zone
            movement["projection_applied"] = bool(projected)
            movement["speed_mps"] = round(float(speed_mps), 4)
            movement["speed_source"] = speed_source
            movement["horizon_seconds"] = float(horizon_seconds)
            movement["max_travel_m"] = round(float(max_travel_m), 4)
            movement["distance_to_target_m"] = (round(float(target_distance_m), 4)
                                                 if target_distance_m is not None else None)
            movement["estimated_travel_time_s"] = (round(float(travel_time_s), 4)
                                                    if travel_time_s is not None else None)
            movement["reachable_within_horizon"] = bool(
                target_distance_m is None or target_distance_m <= max_travel_m + 1e-6)
            movement["distance_constraint"] = 'speed_x_horizon_disc'
            movement["route_constraint"] = 'legal_functional_path_with_speed_x_horizon'
            movement["path_zones"] = list(route_path)
            movement["path_hops"] = max(0, len(route_path) - 1)
            movement["legal_path"] = bool(route_path)
            movement["motion_direction_deg"] = (round(
                math.degrees(math.atan2(motion_direction[1], motion_direction[0])), 3)
                if motion_direction is not None else None)
            movement["heading_deg"] = (round(float(observed_yaw), 3)
                                        if isinstance(observed_yaw, (int, float)) else None)
            movement["orientation_source"] = (
                'muzzle_yaw_only' if observed_yaw is not None else None)
            movement["direction_correction_applied"] = bool(direction_correction)
            movement["position_fusion_applied"] = bool(position_fusion_applied)
            movement["position_fusion_model_weight"] = fusion_weight
            movement["position_fusion_velocity_available"] = bool(position_fusion_velocity_available)
        elif zone_head_predicted is not None:
            # With no origin context there is no adjacency constraint to
            # apply, so a high level candidate is still useful to a UI.
            to_zone = zone_head_predicted
            zone_head_used = True
        # Unit-specific state is deliberately a sidecar.  The raw model head
        # remains compatible with old checkpoints while deployment can enrich
        # each slot with sensor/radar-derived attack, resource, defence, or
        # flight state.  A synthetic context is used only for the spatial
        # zone; missing protocol fields stay unknown in the sidecar.
        current_special = derive_unit_special_state(
            context_entity, current_entities, self_team=observing_side or "红")
        predicted_special = forecast_special_state(
            entity_type, current_special, horizon_seconds)
        predicted_canonical_zone = locate_zone(position["x"], position["y"])
        output[slot_name(index)] = {
            "side": group,
            "team": team if observing_side in ("红", "蓝") else None,
            "entity_type": entity_type,
            "presence_probability": round(presence, 4),
            # Explicitly expose lifecycle uncertainty to the decision layer;
            # absence/death is not inferred from a frozen position.
            "death_probability": round(1.0 - presence, 4),
            "position": position,
            "unprojected_position": unprojected_position,
            "unprojected_canonical_zone": unprojected_canonical_zone,
            "raw_predicted_zone": raw_zone,
            "zone_head_predicted_zone": zone_head_predicted,
            "zone_head_probabilities": zone_head_probabilities,
            "zone_head_used": zone_head_used,
            "canonical_zone_predicted_zone": canonical_zone_predicted,
            "canonical_zone_probabilities": canonical_zone_probabilities,
            "from_canonical_zone": current_canonical_zone,
            "to_canonical_zone": predicted_canonical_zone,
            "current_functional_zone": current_functional_zone,
            "from_zone": from_zone,
            "to_zone": to_zone,
            "movement": movement,
            "speed_mps": (round(float(_entity_speed_mps(entity_type, context_entity)[0]), 4)
                          if from_zone is not None and team in ("红", "蓝") else None),
            "motion_direction_deg": (round(
                math.degrees(math.atan2(motion_direction[1], motion_direction[0])), 3)
                if motion_direction is not None else None),
            "heading_deg": (round(float(observed_yaw), 3)
                            if isinstance(observed_yaw, (int, float)) else None),
            "orientation_source": ('muzzle_yaw_only' if observed_yaw is not None else None),
            "direction_correction_applied": bool(direction_correction),
            "position_fusion_applied": bool(position_fusion_applied),
            "position_fusion_model_weight": fusion_weight,
            "position_fusion_velocity_available": bool(position_fusion_velocity_available),
            "special_state": current_special,
            "special_state_forecast": predicted_special,
            "hp_ratio": round(continuous_hp, 4),
            "count": round(continuous_count * 2.0, 3),
            "state": "absent" if presence < 0.35 else (
                "critical" if continuous_hp < 0.3 else "active"),
        }
    return output
