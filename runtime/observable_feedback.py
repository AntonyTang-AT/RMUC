"""Causal maturity labels derived only from sensor/radar observations."""
from __future__ import annotations

import math

from forecast.temporal import _centroid, _mobile, baseline_labels


def _observable_example(snapshot, side):
    """Convert either dataset or deployment snapshot shape to a visible view."""
    if not isinstance(snapshot, dict):
        raise ValueError("snapshot must be a mapping")
    issue_time = snapshot.get("issued_at")
    observable = snapshot.get("observable")
    if isinstance(observable, dict):
        self_entities = observable.get("self_entities") or {}
        opponent_entities = observable.get("opponent_entities") or {}
        values = [*self_entities.values(), *opponent_entities.values()] if isinstance(
            self_entities, dict) and isinstance(opponent_entities, dict) else []
    else:
        values = list((snapshot.get("entities") or {}).values())

    own, enemy = [], []
    for entity in values:
        if not isinstance(entity, dict):
            continue
        if entity.get("observability_state", "current") != "current":
            continue
        if entity.get("source") not in ("sensor", "radar", "camera", "vision"):
            continue
        if entity.get("team") not in ("红", "蓝"):
            continue
        try:
            observed_at = float(entity.get("observed_at", issue_time or 0.0))
            received_at = float(entity.get("received_at", observed_at))
        except (TypeError, ValueError):
            continue
        if issue_time is not None and (
                observed_at > float(issue_time) + 1e-6 or
                received_at > float(issue_time) + 1e-6):
            continue
        position = entity.get("position")
        if not isinstance(position, (list, tuple)) or len(position) < 2:
            continue
        try:
            x, y = float(position[0]), float(position[1])
        except (TypeError, ValueError):
            continue
        if not (math.isfinite(x) and math.isfinite(y)):
            continue
        row = dict(entity)
        row["position"] = [x, y]
        team = row.get("team")
        role = row.get("role")
        is_own = (role == "self") if role in ("self", "opponent") else team == side
        (own if is_own else enemy).append(row)

    if issue_time is None:
        issue_time = max((float(row.get("observed_at", 0.0)) for row in values
                          if isinstance(row, dict)), default=0.0)
    example = {
        "side": side,
        "issued_at": float(issue_time),
        "observable": {"self_entities": own, "opponent_entities": enemy},
    }
    return example


def _visible_enemy_center(example):
    entities = example["observable"]["opponent_entities"]
    return _centroid(_mobile(entities))


def feedback_labels_from_observable(issue_snapshot, matured_snapshot, side, horizon):
    """Recompute matured labels without consulting any recorded future target.

    The caller supplies the current sensor/radar snapshot only after the exact
    horizon has elapsed. A later observed tick is acceptable, matching the
    online replay contract. Unobservable heads return ``None`` and are not
    scored or used for calibration.
    """
    if side not in ("红", "蓝"):
        raise ValueError("side must be 红 or 蓝")
    horizon = float(horizon)
    if not math.isfinite(horizon) or horizon <= 0:
        raise ValueError("horizon must be a positive finite number")
    issued = _observable_example(issue_snapshot, side)
    matured = _observable_example(matured_snapshot, side)
    if matured["issued_at"] + 1e-6 < issued["issued_at"] + horizon:
        raise ValueError("forecast horizon has not matured in observed time")

    own_now = issued["observable"]["self_entities"]
    enemy_now = issued["observable"]["opponent_entities"]
    own_later = matured["observable"]["self_entities"]
    enemy_later = matured["observable"]["opponent_entities"]
    labels = {"global_mode": None, "opponent_trend": None, "enemy_region": None}

    # Global mode is evaluable only when both sides contribute a currently
    # observed mobile unit; otherwise a missing radar return could look like
    # a real retreat or advantage.
    if _mobile(own_later) and _mobile(enemy_later):
        labels["global_mode"] = baseline_labels(matured)["global_mode"]
    if _mobile(enemy_later):
        labels["enemy_region"] = baseline_labels(matured)["enemy_region"]

    start_center = _visible_enemy_center(issued)
    end_center = _visible_enemy_center(matured)
    if start_center is not None and end_center is not None:
        toward = 1.0 if side == "蓝" else -1.0
        delta = (float(end_center[0]) - float(start_center[0])) * toward
        labels["opponent_trend"] = (
            "推进" if delta >= 0.8 else "回撤" if delta <= -0.8 else "稳定")
    return labels
