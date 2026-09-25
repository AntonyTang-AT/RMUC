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


HORIZONS = (10, 30, 60, 120)
MODES = ("进攻", "均衡", "防守")
TRENDS = ("推进", "稳定", "回撤")
ENTITY_TYPES = ("英雄", "工程", "步兵3", "步兵4", "空中", "哨兵", "前哨站", "基地")
MOBILE_TYPES = frozenset(ENTITY_TYPES[:6])
MAP_X = 28.0
MAP_Y = 15.0
MAX_TIME = 420.0
FEATURES_PER_SLOT = 10
N_GLOBAL = 7
FEATURE_DIM = len(ENTITY_TYPES) * 2 * FEATURES_PER_SLOT + N_GLOBAL
REGION_LABELS = tuple(f"x{x}y{y}" for x in range(4) for y in range(3))


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
    return [e for e in entities if e.get("entity_type") in MOBILE_TYPES]


def _hp_ratio(entity):
    hp = _number(entity.get("hp"), 0.0)
    maxhp = _number(entity.get("maxhp"), 0.0)
    if maxhp > 0:
        return max(0.0, min(1.0, hp / maxhp))
    # Opponent max HP is intentionally hidden in the protocol replay.  This
    # fallback is only a scale for the observable feature, not a label.
    return max(0.0, min(1.0, hp / 5000.0))


def _region(point):
    if point is None:
        return None
    x, y = point
    xi = min(3, max(0, int(x / (MAP_X / 4.0))))
    yi = min(2, max(0, int(y / (MAP_Y / 3.0))))
    return f"x{xi}y{yi}"


def _mode_from_groups(self_entities, opponent_entities):
    """Derive the same coarse three-way state definition for labels/baseline."""
    own_mobile = _mobile(self_entities)
    opp_mobile = _mobile(opponent_entities)
    own_push = sum(1 for e in own_mobile if (e.get("position") or [MAP_X / 2])[0] > MAP_X / 2)
    opp_push = sum(1 for e in opp_mobile if (e.get("position") or [MAP_X / 2])[0] < MAP_X / 2)
    own_base = next((e for e in self_entities if e.get("entity_type") == "基地"), None)
    opp_base = next((e for e in opponent_entities if e.get("entity_type") == "基地"), None)
    own_hp = _hp_ratio(own_base) if own_base else 1.0
    opp_hp = _hp_ratio(opp_base) if opp_base else 1.0
    if own_hp < 0.5 or opp_push - own_push >= 2:
        return "防守"
    if own_push - opp_push >= 2 and opp_hp < 0.9:
        return "进攻"
    return "均衡"


def feature_vector(example):
    """Encode one visible snapshot into a fixed float32 vector."""
    observable = example.get("observable") or {}
    self_entities = _entities(observable, "self_entities")
    opponent_entities = _entities(observable, "opponent_entities")
    out = []
    for group in (self_entities, opponent_entities):
        for entity_type in ENTITY_TYPES:
            matching = [e for e in group if e.get("entity_type") == entity_type]
            entity = sorted(matching, key=lambda e: str(e.get("entity_id", "")))[0] if matching else None
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
    t = _number(example.get("issued_at"), 0.0)
    all_entities = self_entities + opponent_entities
    visible_opp = sum(1 for e in opponent_entities if e.get("position") is not None)
    out.extend([
        max(0.0, min(1.0, t / MAX_TIME)),
        max(0.0, min(1.0, (MAX_TIME - t) / MAX_TIME)),
        max(0.0, min(1.0, _number((self_entities[0] if self_entities else {}).get("team_coins_total"), 0.0) / 500.0)),
        max(0.0, min(1.0, _number((self_entities[0] if self_entities else {}).get("team_coins_left"), 0.0) / 500.0)),
        min(1.0, len(all_entities) / 16.0),
        min(1.0, len(opponent_entities) / 6.0),
        visible_opp / max(1, len(opponent_entities)),
    ])
    result = np.asarray(out, dtype=np.float32)
    if result.shape != (FEATURE_DIM,):
        raise ValueError(f"feature dimension {result.shape} != {(FEATURE_DIM,)}")
    return result


def baseline_labels(example):
    observable = example.get("observable") or {}
    self_entities = _entities(observable, "self_entities")
    opponent_entities = _entities(observable, "opponent_entities")
    return {
        "global_mode": _mode_from_groups(self_entities, opponent_entities),
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
    self_future = [e for e in entities if e.get("team") == side]
    opp_future = [e for e in entities if e.get("team") != side]
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
    return {
        "global_mode": _mode_from_groups(self_future, opp_future),
        "opponent_trend": trend,
        "enemy_region": _region(future_center),
    }


def snapshot_hash(example):
    payload = json.dumps(example.get("observable", {}), ensure_ascii=False,
                         sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def prepare_compact_split(jsonl_gz, out_path, context=12):
    """Convert verbose JSONL examples to compact causal windows for training."""
    import gzip

    if context <= 0:
        raise ValueError("context must be positive")
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
        local_x = np.zeros((local_n, context, FEATURE_DIM), dtype=np.float32)
        local_mode = np.full((local_n, len(HORIZONS)), -1, dtype=np.int64)
        local_trend = np.full((local_n, len(HORIZONS)), -1, dtype=np.int64)
        local_region = np.full((local_n, len(HORIZONS)), -1, dtype=np.int64)
        local_base_mode = np.full(local_n, -1, dtype=np.int64)
        local_base_region = np.full(local_n, -1, dtype=np.int64)
        local_game_ids = np.full(local_n, int(values[0]["game_id"]), dtype=np.int64)
        local_sides = np.full(local_n, values[0]["side"], dtype="U1")
        local_issued = np.zeros(local_n, dtype=np.float32)
        feats = [feature_vector(row) for row in values]
        for i, row in enumerate(values):
            start = max(0, i - context + 1)
            window = feats[start:i + 1]
            local_x[i, -len(window):] = np.asarray(window, dtype=np.float32)
            base = baseline_labels(row)
            local_base_mode[i] = MODES.index(base["global_mode"])
            if base["enemy_region"] in REGION_LABELS:
                local_base_region[i] = REGION_LABELS.index(base["enemy_region"])
            for h_idx, horizon in enumerate(HORIZONS):
                labels = target_labels(row, horizon)
                if labels is None:
                    continue
                local_mode[i, h_idx] = MODES.index(labels["global_mode"])
                local_trend[i, h_idx] = TRENDS.index(labels["opponent_trend"])
                if labels["enemy_region"] in REGION_LABELS:
                    local_region[i, h_idx] = REGION_LABELS.index(labels["enemy_region"])
            local_issued[i] = float(row["issued_at"])
        parts.append({"x": local_x, "mode_y": local_mode, "trend_y": local_trend,
                      "region_y": local_region, "base_mode": local_base_mode,
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
                        feature_dim=np.asarray([FEATURE_DIM], dtype=np.int64))
    return {"examples": n, "groups": group_count, "feature_dim": FEATURE_DIM,
            "path": str(out_path)}


def load_compact(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}
