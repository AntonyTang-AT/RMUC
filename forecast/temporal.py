"""Feature extraction, causal windows, and targets for local forecasting.

The module deliberately consumes only the ``observable`` part of a forecast
example for features.  ``targets`` are used only to construct supervised
labels and are never copied into the input window.
"""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from worldstate.canonical_zone import CANONICAL_ZONE_DIM, zone_features


HORIZONS = (10, 30, 60, 120)
MODES = ("进攻", "均衡", "防守")
TRENDS = ("推进", "稳定", "回撤")
ENTITY_TYPES = ("英雄", "工程", "步兵3", "步兵4", "空中", "哨兵", "前哨站", "基地")
MOBILE_TYPES = frozenset(ENTITY_TYPES[:6])
MAP_X = 28.0
MAP_Y = 15.0
MAX_TIME = 420.0
# ``FEATURES_PER_SLOT`` and ``FEATURE_DIM`` are the v2 compact schema.  Keep
# them stable so old checkpoints and reports can still be decoded.  The
# motion schema below is opt-in through ``schema="motion"`` and is used by
# the larger TCN/Transformer experiments.
FEATURES_PER_SLOT = 14
LEGACY_FEATURE_DIM = len(ENTITY_TYPES) * 2 * FEATURES_PER_SLOT + 8
N_GLOBAL = 8
FEATURE_DIM = len(ENTITY_TYPES) * 2 * FEATURES_PER_SLOT + N_GLOBAL
REGION_LABELS = tuple(f"x{x}y{y}" for x in range(4) for y in range(3))
GLOBAL_MODE_LABEL_VERSION = "relative_margin_v2"
FEATURE_SCHEMA_VERSION = "canonical-zone-v2-side-aware"

MOTION_FEATURE_NAMES = (
    "delta_x", "delta_y", "speed", "acceleration", "direction_x", "direction_y",
)
MOTION_FEATURES_PER_SLOT = FEATURES_PER_SLOT + len(MOTION_FEATURE_NAMES)
MOTION_N_GLOBAL = N_GLOBAL + 1  # causal time delta, in addition to the v2 globals
MOTION_FEATURE_DIM = (len(ENTITY_TYPES) * 2 * MOTION_FEATURES_PER_SLOT
                      + MOTION_N_GLOBAL)
MOTION_FEATURE_SCHEMA_VERSION = "canonical-zone-v3-motion"
MOTION_MAX_SPEED_MPS = 3.0
MOTION_MAX_ACCEL_MPS2 = 3.0
MOTION_DT_SCALE_SECONDS = 5.0

# The task set is intentionally stable across horizons.  Longer horizons
# become less specific through nested aggregation and a wider uncertainty
# budget; they do not silently drop classes or switch to a different label
# contract.
HORIZON_PROFILES = {
    10: {
        "detail_level": "high",
        "temporal_view": "12 raw snapshots",
        "spatial_view": "12-cell enemy centroid",
        "scope": "unit/building changes, enemy trend, short action candidates",
    },
    30: {
        "detail_level": "medium_high",
        "temporal_view": "6 two-snapshot bins",
        "spatial_view": "12-cell enemy centroid plus local region mass",
        "scope": "formation and local control changes",
    },
    60: {
        "detail_level": "medium",
        "temporal_view": "4 three-snapshot bins",
        "spatial_view": "12-cell distribution plus macro x-band",
        "scope": "area control, push depth, building exchange",
    },
    120: {
        "detail_level": "coarse",
        "temporal_view": "2 six-snapshot bins",
        "spatial_view": "12-cell distribution plus coarse side of map",
        "scope": "overall attack/defense direction and base risk",
    },
}


def _number(value, default=0.0):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return float(default)
    return float(value) if math.isfinite(float(value)) else float(default)


def _target(example, horizon):
    targets = example.get("targets") or {}
    value = targets.get(str(horizon), targets.get(horizon))
    if not isinstance(value, dict) or value.get("censored"):
        return None
    truth = value.get("truth")
    return truth if isinstance(truth, dict) else None


def _entities(observable, group):
    values = (observable or {}).get(group) or {}
    return list(values.values()) if isinstance(values, dict) else list(values)


def _centroid(entities):
    points = []
    for entity in entities:
        pos = entity.get("position")
        if isinstance(pos, (list, tuple)) and len(pos) >= 2:
            x, y = _number(pos[0], math.nan), _number(pos[1], math.nan)
            if math.isfinite(x) and math.isfinite(y):
                points.append((x, y))
    if not points:
        return None
    return (sum(x for x, _ in points) / len(points),
            sum(y for _, y in points) / len(points))


def _mobile(entities):
    return [e for e in entities if e.get("entity_type") in MOBILE_TYPES
            and e.get('life_state') != 'destroyed'
            and not (isinstance(e.get('hp'), (int, float))
                     and not isinstance(e.get('hp'), bool) and e.get('hp') <= 0)]


def _hp_ratio(entity):
    hp = _number(entity.get("hp"), 0.0)
    maxhp = _number(entity.get("maxhp"), 0.0)
    if maxhp > 0:
        return max(0.0, min(1.0, hp / maxhp))
    # Opponent max HP is intentionally hidden in the protocol replay.  This
    # fallback is only a scale for the observable feature, not a label.
    return max(0.0, min(1.0, hp / 5000.0))


def _selected_entity(entities, entity_type, entity_id=None):
    """Select the deterministic slot representative from one snapshot.

    The replay format may contain multiple units of one type.  The existing
    compact contract uses one stable representative per type; motion features
    must use the same representative and then match it by ``entity_id`` in
    earlier snapshots whenever that id is available.
    """
    matching = [e for e in entities if e.get("entity_type") == entity_type]
    if entity_id is not None:
        by_id = [e for e in matching if str(e.get("entity_id", "")) == str(entity_id)]
        if by_id:
            return sorted(by_id, key=lambda e: str(e.get("entity_id", "")))[0]
    return sorted(matching, key=lambda e: str(e.get("entity_id", "")))[0] if matching else None


def _same_slot_entity(entities, current_entity, entity_type):
    """Match a current slot to a previous snapshot without future access."""
    if current_entity is None:
        return None
    return _selected_entity(entities, entity_type, current_entity.get("entity_id"))


def _motion_features(current_entity, previous_entity, previous_previous_entity,
                     current_time, previous_time, previous_previous_time):
    """Return causal movement features for one entity slot.

    The six values are normalized signed displacement, speed, signed speed
    change, and the two direction components.  A missing prior observation is
    represented by zeros; no future or target fields are consulted.
    """
    if current_entity is None:
        return [0.0] * len(MOTION_FEATURE_NAMES)
    current_pos = current_entity.get("position")
    previous_pos = (previous_entity or {}).get("position")
    if (not isinstance(current_pos, (list, tuple)) or len(current_pos) < 2 or
            not isinstance(previous_pos, (list, tuple)) or len(previous_pos) < 2):
        return [0.0] * len(MOTION_FEATURE_NAMES)
    current_xy = (_number(current_pos[0], math.nan), _number(current_pos[1], math.nan))
    previous_xy = (_number(previous_pos[0], math.nan), _number(previous_pos[1], math.nan))
    if not all(math.isfinite(value) for value in (*current_xy, *previous_xy)):
        return [0.0] * len(MOTION_FEATURE_NAMES)
    dt = _number(current_time, 0.0) - _number(previous_time, 0.0)
    if dt <= 1e-6:
        return [0.0] * len(MOTION_FEATURE_NAMES)
    dx_m = current_xy[0] - previous_xy[0]
    dy_m = current_xy[1] - previous_xy[1]
    distance_m = math.hypot(dx_m, dy_m)
    speed_mps = distance_m / dt
    speed_norm = max(0.0, min(1.0, speed_mps / MOTION_MAX_SPEED_MPS))
    if distance_m > 1e-8:
        direction_x, direction_y = dx_m / distance_m, dy_m / distance_m
    else:
        direction_x = direction_y = 0.0

    acceleration_norm = 0.0
    previous_previous_pos = (previous_previous_entity or {}).get("position")
    dt_previous = _number(previous_time, 0.0) - _number(previous_previous_time, 0.0)
    if (isinstance(previous_previous_pos, (list, tuple)) and len(previous_previous_pos) >= 2
            and dt_previous > 1e-6):
        previous_previous_xy = (
            _number(previous_previous_pos[0], math.nan),
            _number(previous_previous_pos[1], math.nan),
        )
        if all(math.isfinite(value) for value in previous_previous_xy):
            previous_speed = math.hypot(
                previous_xy[0] - previous_previous_xy[0],
                previous_xy[1] - previous_previous_xy[1],
            ) / dt_previous
            acceleration_norm = max(
                -1.0,
                min(1.0, (speed_mps - previous_speed) / max(MOTION_MAX_ACCEL_MPS2, 1e-6)),
            )
    return [
        max(-1.0, min(1.0, dx_m / MAP_X)),
        max(-1.0, min(1.0, dy_m / MAP_Y)),
        speed_norm,
        acceleration_norm,
        direction_x,
        direction_y,
    ]


def _region(point):
    if point is None:
        return None
    x, y = point
    xi = min(3, max(0, int(x / (MAP_X / 4.0))))
    yi = min(2, max(0, int(y / (MAP_Y / 3.0))))
    return f"x{xi}y{yi}"


def _mode_from_groups(self_entities, opponent_entities, side="红"):
    """Derive the rebalanced, side-symmetric three-way forecast label.

    V1 counted absolute map halves and required an additional opponent-base
    condition for ``进攻``.  That produced almost no attack labels and made a
    persistence classifier look artificially strong.  V2 compares push margin
    in the acting team's coordinates and adds the base-health differential.
    Fixed thresholds keep the definition reproducible across splits.
    """
    score = _mode_score(self_entities, opponent_entities, side)
    if score <= -1.0:
        return "防守"
    if score >= 1.0:
        return "进攻"
    return "均衡"


def _mode_score(self_entities, opponent_entities, side="红"):
    """Continuous score behind the three-way mode label."""
    own_mobile = _mobile(self_entities)
    opp_mobile = _mobile(opponent_entities)
    if side == "蓝":
        own_push = sum(1 for e in own_mobile if (e.get("position") or [MAP_X / 2])[0] < MAP_X / 2)
        opp_push = sum(1 for e in opp_mobile if (e.get("position") or [MAP_X / 2])[0] >= MAP_X / 2)
    else:
        own_push = sum(1 for e in own_mobile if (e.get("position") or [MAP_X / 2])[0] > MAP_X / 2)
        opp_push = sum(1 for e in opp_mobile if (e.get("position") or [MAP_X / 2])[0] <= MAP_X / 2)
    own_base = next((e for e in self_entities if e.get("entity_type") == "基地"), None)
    opp_base = next((e for e in opponent_entities if e.get("entity_type") == "基地"), None)
    own_hp = _hp_ratio(own_base) if own_base else 1.0
    opp_hp = _hp_ratio(opp_base) if opp_base else 1.0
    return float(own_push - opp_push) + 2.0 * (opp_hp - own_hp)


def _boundary_quality(score, boundaries=(-1.0, 1.0), scale=1.25):
    """Downweight labels close to a hard rule boundary.

    A value near a threshold is a legitimate ambiguous state, not a bad
    sample.  It receives a soft quality weight instead of forcing the network
    to become overconfident about an arbitrary one-frame class flip.
    """
    distance = min(abs(float(score) - boundary) for boundary in boundaries)
    return float(np.clip(0.25 + distance / max(scale, 1e-6), 0.25, 1.0))


def _trend_quality(current_center, future_center, side):
    if current_center is None or future_center is None:
        return 0.55
    direction = 1.0 if side == "蓝" else -1.0
    delta_toward = (future_center[0] - current_center[0]) * direction
    # The trend thresholds are +/-0.8 m.  Stable samples near either cutoff
    # are inherently ambiguous and should not dominate the loss.
    return _boundary_quality(delta_toward, boundaries=(-0.8, 0.8), scale=1.0)


def feature_vector(example, previous_example=None, previous_previous_example=None,
                   schema="legacy"):
    """Encode one visible snapshot into a fixed float32 vector.

    ``schema="legacy"`` preserves the original 14 fields per entity and is
    the default for old checkpoints.  ``schema="motion"`` adds six causal
    movement fields per entity and one global inter-snapshot time delta.  The
    latter requires only earlier snapshots and therefore remains valid for
    deployment; omitted history is represented by zeros.
    """
    if schema not in ("legacy", "motion"):
        raise ValueError("schema must be 'legacy' or 'motion'")
    motion_schema = schema == "motion"
    observable = example.get("observable") or {}
    self_entities = _entities(observable, "self_entities")
    opponent_entities = _entities(observable, "opponent_entities")
    previous_observable = (previous_example or {}).get("observable") or {}
    previous_previous_observable = (previous_previous_example or {}).get("observable") or {}
    out = []
    for group_name, group in (("self_entities", self_entities),
                              ("opponent_entities", opponent_entities)):
        previous_group = _entities(previous_observable, group_name)
        previous_previous_group = _entities(previous_previous_observable, group_name)
        for entity_type in ENTITY_TYPES:
            entity = _selected_entity(group, entity_type)
            pos = entity.get("position") if entity else None
            has_pos = float(isinstance(pos, (list, tuple)) and len(pos) >= 2)
            x = _number(pos[0], 0.0) / MAP_X if has_pos else 0.0
            y = _number(pos[1], 0.0) / MAP_Y if has_pos else 0.0
            out.extend([
                max(0.0, min(1.0, x)), max(0.0, min(1.0, y)),
                _hp_ratio(entity or {}), has_pos,
                max(0.0, min(1.0, _number((entity or {}).get("age_seconds"), 0.0) / 120.0)),
                max(0.0, min(1.0, _number((entity or {}).get("confidence"), 0.0))),
                float(bool((entity or {}).get("radar_highlighted", False))),
                max(0.0, min(1.0, _number((entity or {}).get("team_coins_total"), 0.0) / 500.0)),
                max(0.0, min(1.0, _number((entity or {}).get("team_coins_left"), 0.0) / 500.0)),
                float(bool(entity)),
            ])
            # Four compact features bind the causal snapshot to the approved
            # 59-cell geometry without replacing the continuous position head.
            out.extend(zone_features(pos if has_pos else None))
            if motion_schema:
                previous_entity = _same_slot_entity(previous_group, entity, entity_type)
                previous_previous_entity = _same_slot_entity(
                    previous_previous_group, previous_entity, entity_type)
                out.extend(_motion_features(
                    entity,
                    previous_entity,
                    previous_previous_entity,
                    example.get("issued_at", 0.0),
                    (previous_example or {}).get("issued_at", 0.0),
                    (previous_previous_example or {}).get("issued_at", 0.0),
                ))
    t = _number(example.get("issued_at"), 0.0)
    all_entities = self_entities + opponent_entities
    visible_opp = sum(1 for e in opponent_entities if e.get("position") is not None)
    global_values = [
        max(0.0, min(1.0, t / MAX_TIME)),
        max(0.0, min(1.0, (MAX_TIME - t) / MAX_TIME)),
        max(0.0, min(1.0, _number((self_entities[0] if self_entities else {}).get("team_coins_total"), 0.0) / 500.0)),
        max(0.0, min(1.0, _number((self_entities[0] if self_entities else {}).get("team_coins_left"), 0.0) / 500.0)),
        min(1.0, len(all_entities) / 16.0),
        min(1.0, len(opponent_entities) / 6.0),
        visible_opp / max(1, len(opponent_entities)),
        # Global labels are side-relative (red pushes toward decreasing x,
        # blue toward increasing x).  Keep the observing side explicit so a
        # causal model does not have to infer orientation from symmetric
        # snapshots.
        1.0 if example.get("side", "红") == "蓝" else 0.0,
    ]
    if motion_schema:
        delta_t = max(0.0, t - _number((previous_example or {}).get("issued_at"), t))
        global_values.append(max(0.0, min(1.0, delta_t / MOTION_DT_SCALE_SECONDS)))
    out.extend(global_values)
    result = np.asarray(out, dtype=np.float32)
    expected_dim = MOTION_FEATURE_DIM if motion_schema else FEATURE_DIM
    if result.shape != (expected_dim,):
        raise ValueError(f"feature dimension {result.shape} != {(expected_dim,)}")
    return result


def motion_feature_vector(example, previous_example=None, previous_previous_example=None):
    """Encode a snapshot with causal per-entity motion features."""
    return feature_vector(example, previous_example, previous_previous_example,
                          schema="motion")


def baseline_labels(example):
    observable = example.get("observable") or {}
    self_entities = _mobile(_entities(observable, "self_entities")) + [
        e for e in _entities(observable, "self_entities") if e.get("entity_type") not in MOBILE_TYPES
        and e.get('life_state') != 'destroyed']
    opponent_entities = _mobile(_entities(observable, "opponent_entities")) + [
        e for e in _entities(observable, "opponent_entities") if e.get("entity_type") not in MOBILE_TYPES
        and e.get('life_state') != 'destroyed']
    return {
        "global_mode": _mode_from_groups(self_entities, opponent_entities, example.get("side", "红")),
        "opponent_trend": "稳定",
        "enemy_region": _region(_centroid(_mobile(opponent_entities))),
    }


def target_labels(example, horizon):
    """Return labels for one horizon, or None for a censored/unavailable target."""
    truth = _target(example, horizon)
    if truth is None:
        return None
    current = example.get("observable") or {}
    self_now = _entities(current, "self_entities")
    opp_now = _entities(current, "opponent_entities")
    entities = list((truth.get("entities") or {}).values())
    side = example.get("side")
    def future_entity_alive(entity):
        return (entity.get('entity_type') not in MOBILE_TYPES
                or not (isinstance(entity.get('hp'), (int, float))
                        and not isinstance(entity.get('hp'), bool)
                        and entity.get('hp') <= 0))
    self_future = [e for e in entities if e.get("team") == side and future_entity_alive(e)]
    opp_future = [e for e in entities if e.get("team") != side and future_entity_alive(e)]
    future_center = _centroid(_mobile(opp_future))
    current_center = _centroid(_mobile(opp_now))
    trend = "稳定"
    if current_center is not None and future_center is not None:
        direction = 1.0 if side == "蓝" else -1.0
        delta_toward = (future_center[0] - current_center[0]) * direction
        if delta_toward >= 0.8:
            trend = "推进"
        elif delta_toward <= -0.8:
            trend = "回撤"
    mode_score = _mode_score(self_future, opp_future, side)
    return {
        "global_mode": _mode_from_groups(self_future, opp_future, side),
        "opponent_trend": trend,
        "enemy_region": _region(future_center),
        "label_quality": {
            "global_mode": _boundary_quality(mode_score),
            "opponent_trend": _trend_quality(current_center, future_center, side),
            # Region is a coarse centroid bin; its uncertainty is handled by
            # class weighting and horizon smoothing rather than a fake margin.
            "enemy_region": 1.0,
        },
    }


def snapshot_hash(example):
    payload = json.dumps(example.get("observable", {}), ensure_ascii=False,
                         sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def prepare_compact_split(jsonl_gz, out_path, context=12, feature_schema="legacy"):
    """Convert verbose JSONL examples to compact causal windows for training.

    ``feature_schema="legacy"`` keeps the historical 14-field vectors.  Use
    ``feature_schema="motion"`` for the final multi-scale architecture; it
    builds each row from the current, previous, and previous-previous causal
    snapshots and supports long windows such as ``context=40``.
    """
    import gzip

    if context <= 0:
        raise ValueError("context must be positive")
    if feature_schema not in ("legacy", "motion"):
        raise ValueError("feature_schema must be 'legacy' or 'motion'")
    feature_dim = MOTION_FEATURE_DIM if feature_schema == "motion" else FEATURE_DIM
    schema_version = (MOTION_FEATURE_SCHEMA_VERSION if feature_schema == "motion"
                      else FEATURE_SCHEMA_VERSION)
    # The extractor writes complete game/side groups consecutively.  Stream one
    # group at a time so the verbose observable JSON is never retained for an
    # entire split (the train split is hundreds of MB uncompressed).
    parts = []
    group_count = 0
    current_key = None
    current_rows = []

    def flush_group(values):
        nonlocal group_count
        if not values:
            return
        values = sorted(values, key=lambda row: float(row["issued_at"]))
        local_n = len(values)
        local_x = np.zeros((local_n, context, feature_dim), dtype=np.float32)
        local_mode = np.full((local_n, len(HORIZONS)), -1, dtype=np.int64)
        local_trend = np.full((local_n, len(HORIZONS)), -1, dtype=np.int64)
        local_region = np.full((local_n, len(HORIZONS)), -1, dtype=np.int64)
        local_quality = np.full((local_n, len(HORIZONS), 3), -1.0,
                                dtype=np.float32)
        from .entity import (ENTITY_SLOTS, ENTITY_TARGET_DIM, entity_target_vector,
                             entity_zone_target_vector)
        local_entity = np.full(
            (local_n, len(HORIZONS), len(ENTITY_SLOTS), ENTITY_TARGET_DIM),
            -1.0, dtype=np.float32)
        local_entity_zone = np.full(
            (local_n, len(HORIZONS), len(ENTITY_SLOTS)),
            -1, dtype=np.int64)
        from .entity import canonical_zone_target_vector
        local_entity_canonical_zone = np.full(
            (local_n, len(HORIZONS), len(ENTITY_SLOTS)),
            -1, dtype=np.int64)
        local_base_mode = np.full(local_n, -1, dtype=np.int64)
        local_base_region = np.full(local_n, -1, dtype=np.int64)
        local_game_ids = np.full(local_n, int(values[0]["game_id"]), dtype=np.int64)
        local_sides = np.full(local_n, values[0]["side"], dtype="U1")
        local_issued = np.zeros(local_n, dtype=np.float32)
        feats = []
        for i, row in enumerate(values):
            previous = values[i - 1] if i >= 1 else None
            previous_previous = values[i - 2] if i >= 2 else None
            feats.append(feature_vector(
                row, previous, previous_previous, schema=feature_schema))
        for i, row in enumerate(values):
            start = max(0, i - context + 1)
            window = feats[start:i + 1]
            local_x[i, -len(window):] = np.asarray(window, dtype=np.float32)
            base = baseline_labels(row)
            local_base_mode[i] = MODES.index(base["global_mode"])
            if base["enemy_region"] in REGION_LABELS:
                local_base_region[i] = REGION_LABELS.index(base["enemy_region"])
            for h_idx, horizon in enumerate(HORIZONS):
                local_entity[i, h_idx] = entity_target_vector(row, horizon)
                local_entity_zone[i, h_idx] = entity_zone_target_vector(row, horizon)
                local_entity_canonical_zone[i, h_idx] = canonical_zone_target_vector(row, horizon)
                labels = target_labels(row, horizon)
                if labels is None:
                    continue
                local_mode[i, h_idx] = MODES.index(labels["global_mode"])
                local_trend[i, h_idx] = TRENDS.index(labels["opponent_trend"])
                if labels["enemy_region"] in REGION_LABELS:
                    local_region[i, h_idx] = REGION_LABELS.index(labels["enemy_region"])
                quality = labels.get("label_quality") or {}
                local_quality[i, h_idx] = [
                    float(quality.get("global_mode", 1.0)),
                    float(quality.get("opponent_trend", 1.0)),
                    float(quality.get("enemy_region", 1.0)),
                ]
            local_issued[i] = float(row["issued_at"])
        parts.append({"x": local_x, "mode_y": local_mode, "trend_y": local_trend,
                      "region_y": local_region, "entity_y": local_entity,
                      "entity_zone_y": local_entity_zone,
                      "entity_canonical_zone_y": local_entity_canonical_zone,
                      "label_quality": local_quality,
                      "base_mode": local_base_mode,
                      "base_region": local_base_region, "game_ids": local_game_ids,
                      "sides": local_sides, "issued_at": local_issued})
        group_count += 1

    with gzip.open(jsonl_gz, "rt", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                row = json.loads(line)
                key = (int(row["game_id"]), row["side"])
                if current_key is None:
                    current_key = key
                if key != current_key:
                    flush_group(current_rows)
                    current_rows = []
                    current_key = key
                current_rows.append(row)
    flush_group(current_rows)
    if not parts:
        raise ValueError(f"no examples found in {jsonl_gz}")
    arrays = {key: np.concatenate([part[key] for part in parts], axis=0)
              for key in parts[0]}
    n = len(arrays["x"])
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **arrays,
                        horizons=np.asarray(HORIZONS, dtype=np.int64),
                        feature_dim=np.asarray([feature_dim], dtype=np.int64),
                        global_mode_label_version=np.asarray([GLOBAL_MODE_LABEL_VERSION]),
                        feature_schema_version=np.asarray([schema_version]),
                        canonical_zone_dim=np.asarray([CANONICAL_ZONE_DIM], dtype=np.int64))
    return {"examples": n, "groups": group_count, "feature_dim": feature_dim,
            "feature_schema_version": schema_version,
            "context": int(context), "path": str(out_path)}


def load_compact(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def prepare_motion_compact_split(jsonl_gz, out_path, context=40):
    """Convenience wrapper for the final motion-aware training schema."""
    return prepare_compact_split(
        jsonl_gz, out_path, context=context, feature_schema="motion")
