"""Causal feature builder and inference wrapper for the experimental mode XGB.

This module is deliberately separate from the production temporal forecaster.
At inference, callers must pass only observations available at the prediction
issue time, grouped by match and observing side.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from forecast.temporal import MODES

BASE_FEATURE_DIM = 20
WINDOWS = (6.0, 15.0, 30.0, 60.0)
EXTRA_FEATURE_NAMES = (
    "elapsed_match_fraction", "remaining_match_fraction",
    *(f"current_mode_{mode}" for mode in MODES),
    *(name for window in WINDOWS for name in (
        f"mode_score_delta_{int(window)}s_x8",
        f"push_margin_delta_{int(window)}s_x6",
        f"base_hp_margin_delta_{int(window)}s_x2",
        f"base_presence_margin_delta_{int(window)}s",
    )),
    "seconds_in_current_mode_fraction_120",
)
FEATURE_NAMES = tuple(f"observable_{i}" for i in range(BASE_FEATURE_DIM)) + EXTRA_FEATURE_NAMES
FEATURE_DIM = len(FEATURE_NAMES)


def observable_snapshot_base_features(example, previous_issued_at=None):
    """Reconstruct the candidate's 20 per-frame features from one visible snapshot.

    This intentionally mirrors ``experiment_global_mode_feature_ablation``'s
    observable summary contract. It consumes only the already visibility-masked
    ``observable`` part of a runtime example and one prior issue timestamp.
    Returns ``(features, current_mode_index)``; unavailable evidence follows
    the same zero/unknown conventions used by the training sidecar.
    """
    import math
    from forecast.temporal import (MAP_X, MOBILE_TYPES, _entities, _hp_ratio,
                                   _mode_from_groups, _mode_score, _mobile)

    observable = example.get("observable") or {}
    self_entities = _entities(observable, "self_entities")
    opp_entities = _entities(observable, "opponent_entities")
    side = example.get("side", "红")
    if side not in ("红", "蓝"):
        raise ValueError(f"unsupported observing side: {side!r}")
    self_mobile, opp_mobile = _mobile(self_entities), _mobile(opp_entities)
    self_base = next((e for e in self_entities if e.get("entity_type") == "基地"), None)
    opp_base = next((e for e in opp_entities if e.get("entity_type") == "基地"), None)
    self_base_ratio = _hp_ratio(self_base or {}) if self_base is not None else 1.0
    opp_base_ratio = _hp_ratio(opp_base or {}) if opp_base is not None else 1.0

    def valid_xy(pos):
        if not isinstance(pos, (list, tuple)) or len(pos) < 2:
            return False
        try:
            return math.isfinite(float(pos[0])) and math.isfinite(float(pos[1]))
        except (TypeError, ValueError):
            return False

    def known_overline(mobiles, own):
        count = 0
        for entity in mobiles:
            pos = entity.get("position")
            if not valid_xy(pos):
                continue
            x = float(pos[0])
            across = (x < MAP_X / 2.0 if own else x > MAP_X / 2.0) if side == "蓝" else (
                x > MAP_X / 2.0 if own else x < MAP_X / 2.0)
            count += int(across)
        return count

    def rule_push(mobiles, own):
        count = 0
        for entity in mobiles:
            pos = entity.get("position") or [MAP_X / 2.0]
            try:
                x = float(pos[0]) if pos else MAP_X / 2.0
            except (TypeError, ValueError):
                x = MAP_X / 2.0
            condition = (x < MAP_X / 2.0 if own else x >= MAP_X / 2.0) if side == "蓝" else (
                x > MAP_X / 2.0 if own else x <= MAP_X / 2.0)
            count += int(condition)
        return count

    def coverage(entities, mobiles):
        n = len(mobiles)
        if not n:
            return 0.0, 0.0, 0.0
        positioned = sum(valid_xy(e.get("position")) for e in mobiles)
        hp_known = sum(isinstance(e.get("hp"), (int, float)) and
                       not isinstance(e.get("hp"), bool) and
                       math.isfinite(float(e["hp"])) for e in mobiles)
        types = len({e.get("entity_type") for e in mobiles
                     if e.get("entity_type") in MOBILE_TYPES})
        return positioned / n, hp_known / n, types / max(1, len(MOBILE_TYPES))

    self_push, opp_push = rule_push(self_mobile, True), rule_push(opp_mobile, False)
    score = _mode_score(self_entities, opp_entities, side)
    mode = _mode_from_groups(self_entities, opp_entities, side)
    now = float(example.get("issued_at", 0.0))
    dt = max(0.0, now - (now if previous_issued_at is None else float(previous_issued_at)))
    self_pos, self_hp_cov, self_type = coverage(self_entities, self_mobile)
    opp_pos, opp_hp_cov, opp_type = coverage(opp_entities, opp_mobile)
    vec = np.asarray([
        known_overline(self_mobile, True) / 6.0,
        known_overline(opp_mobile, False) / 6.0,
        self_push / 6.0, opp_push / 6.0,
        self_base_ratio, opp_base_ratio, score / 8.0,
        len(self_mobile) / 6.0, len(opp_mobile) / 6.0,
        self_pos, opp_pos, self_hp_cov, opp_hp_cov, self_type, opp_type,
        int(self_base is not None), int(opp_base is not None),
        int(self_base is not None and isinstance(self_base.get("hp"), (int, float)) and
            not isinstance(self_base.get("hp"), bool)),
        int(opp_base is not None and isinstance(opp_base.get("hp"), (int, float)) and
            not isinstance(opp_base.get("hp"), bool)),
        dt / 5.0,
    ], dtype=np.float32)
    if vec.shape != (BASE_FEATURE_DIM,) or not np.isfinite(vec).all():
        raise ValueError("failed to construct finite 20D observable summary")
    return vec, MODES.index(mode)


def build_mode_change_features(base_features, issued_at, current_modes, group_ids=None):
    """Build a causal feature matrix from current observable vectors.

    Inputs are one record per issue time. History is isolated by ``group_ids``
    (normally ``(game_id, side)``); if omitted, all rows are one stream. A row
    can see itself and earlier rows only. Times within a group must be unique.
    """
    base = np.asarray(base_features, dtype=np.float32)
    times = np.asarray(issued_at, dtype=np.float64)
    modes = np.asarray(current_modes, dtype=np.int64)
    if base.ndim != 2 or base.shape[1] != BASE_FEATURE_DIM:
        raise ValueError(f"base_features must have shape (N, {BASE_FEATURE_DIM})")
    n = base.shape[0]
    if times.shape != (n,) or modes.shape != (n,):
        raise ValueError("issued_at and current_modes must have shape (N,)")
    if not np.isfinite(base).all() or not np.isfinite(times).all():
        raise ValueError("features and issue times must be finite")
    if np.any((modes < 0) | (modes >= len(MODES))):
        raise ValueError("current_modes must be valid MODES indexes")
    if group_ids is None:
        group_keys = [0] * n
    else:
        raw_groups = list(group_ids)
        if len(raw_groups) != n:
            raise ValueError("group_ids must contain one key per row")
        group_keys = [tuple(g.tolist()) if isinstance(g, np.ndarray) else g
                      for g in raw_groups]
    if len(group_keys) != n:
        raise ValueError("group_ids must have shape (N,)")

    extra = np.zeros((n, len(EXTRA_FEATURE_NAMES)), dtype=np.float32)
    # Convert arbitrary hashable group IDs to row-index lists without sorting
    # the IDs themselves (which may be tuples or mixed types).
    grouped: dict[object, list[int]] = {}
    for row, key in enumerate(group_keys):
        grouped.setdefault(key, []).append(row)
    mode_offset = 2
    delta_offset = mode_offset + len(MODES)
    duration_col = delta_offset + len(WINDOWS) * 4
    for rows in grouped.values():
        ix = np.asarray(sorted(rows, key=lambda i: times[i]), dtype=np.int64)
        ts, values, seq_modes = times[ix], base[ix], modes[ix]
        if len(ts) > 1 and np.any(np.diff(ts) <= 0):
            raise ValueError("issue times must be unique within each group")
        same_start = 0
        for j, row in enumerate(ix):
            now = ts[j]
            mode = int(seq_modes[j])
            extra[row, 0] = np.clip(now / 420.0, 0.0, 1.0)
            extra[row, 1] = np.clip((420.0 - now) / 420.0, 0.0, 1.0)
            extra[row, mode_offset + mode] = 1.0
            if j == 0 or seq_modes[j] != seq_modes[j - 1]:
                same_start = j
            extra[row, duration_col] = min(1.0, (now - ts[same_start]) / 120.0)
            for wi, window in enumerate(WINDOWS):
                # First observed sample on/after the trailing-window boundary,
                # matching the training candidate's established definition.
                k = int(np.searchsorted(ts[:j + 1], now - window, side="left"))
                if k >= j:
                    continue
                past = values[k]
                off = delta_offset + wi * 4
                extra[row, off] = np.clip((values[j, 6] - past[6]) * 8.0, -8.0, 8.0)
                extra[row, off + 1] = np.clip(((values[j, 2] - values[j, 3]) -
                                                (past[2] - past[3])) * 6.0, -6.0, 6.0)
                extra[row, off + 2] = np.clip(((values[j, 5] - values[j, 4]) -
                                                (past[5] - past[4])) * 2.0, -2.0, 2.0)
                extra[row, off + 3] = np.clip((values[j, 7] - values[j, 8]) -
                                               (past[7] - past[8]), -1.0, 1.0)
    return np.concatenate((base, extra), axis=1)


def build_latest_mode_change_features(base_features, issued_at, current_modes):
    """Efficiently build the final row of one causal match/side stream."""
    base = np.asarray(base_features, dtype=np.float32)
    times = np.asarray(issued_at, dtype=np.float64)
    modes = np.asarray(current_modes, dtype=np.int64)
    if base.ndim != 2 or base.shape[1] != BASE_FEATURE_DIM or len(base) == 0:
        raise ValueError(f"base_features must have shape (N, {BASE_FEATURE_DIM}) and N > 0")
    if times.shape != (len(base),) or modes.shape != (len(base),):
        raise ValueError("issued_at and current_modes must have shape (N,)")
    if not np.isfinite(base).all() or not np.isfinite(times).all():
        raise ValueError("features and issue times must be finite")
    if np.any((modes < 0) | (modes >= len(MODES))):
        raise ValueError("current_modes must be valid MODES indexes")
    if len(times) > 1 and np.any(np.diff(times) <= 0):
        raise ValueError("issue times must be unique and increasing within the stream")
    now, mode = times[-1], int(modes[-1])
    extra = np.zeros(len(EXTRA_FEATURE_NAMES), dtype=np.float32)
    mode_offset = 2
    delta_offset = mode_offset + len(MODES)
    duration_col = delta_offset + len(WINDOWS) * 4
    extra[0] = np.clip(now / 420.0, 0.0, 1.0)
    extra[1] = np.clip((420.0 - now) / 420.0, 0.0, 1.0)
    extra[mode_offset + mode] = 1.0
    same_start = len(modes) - 1
    while same_start > 0 and modes[same_start - 1] == mode:
        same_start -= 1
    extra[duration_col] = min(1.0, (now - times[same_start]) / 120.0)
    for wi, window in enumerate(WINDOWS):
        k = int(np.searchsorted(times, now - window, side="left"))
        if k >= len(times) - 1:
            continue
        past, current = base[k], base[-1]
        off = delta_offset + wi * 4
        extra[off] = np.clip((current[6] - past[6]) * 8.0, -8.0, 8.0)
        extra[off + 1] = np.clip(((current[2] - current[3]) -
                                   (past[2] - past[3])) * 6.0, -6.0, 6.0)
        extra[off + 2] = np.clip(((current[5] - current[4]) -
                                   (past[5] - past[4])) * 2.0, -2.0, 2.0)
        extra[off + 3] = np.clip((current[7] - current[8]) -
                                  (past[7] - past[8]), -1.0, 1.0)
    return np.concatenate((base[-1], extra))


class ModeChangeXGBPredictor:
    """CPU-safe loader for the experimental XGBoost mode candidate."""

    def __init__(self, model_path: str | Path, persistence_alpha: float = 0.95):
        if not 0.0 <= persistence_alpha <= 1.0:
            raise ValueError("persistence_alpha must be in [0, 1]")
        from xgboost import XGBClassifier
        self.model = XGBClassifier()
        self.model.load_model(str(model_path))
        booster = self.model.get_booster()
        if booster.num_features() != FEATURE_DIM:
            raise ValueError(f"candidate expects {booster.num_features()} features; expected {FEATURE_DIM}")
        classes = np.asarray(self.model.classes_)
        if not np.array_equal(classes, np.arange(len(MODES))):
            raise ValueError(f"unexpected XGBoost class order: {classes.tolist()}")
        self.persistence_alpha = float(persistence_alpha)

    def predict_proba(self, features, current_modes=None):
        """Return class probabilities; optionally blend the causal prior."""
        x = np.asarray(features, dtype=np.float32)
        if x.ndim == 1:
            x = x[None, :]
        if x.ndim != 2 or x.shape[1] != FEATURE_DIM or not np.isfinite(x).all():
            raise ValueError(f"features must be finite with shape (N, {FEATURE_DIM})")
        probs = np.asarray(self.model.predict_proba(x), dtype=np.float32)
        if current_modes is None:
            return probs
        modes = np.asarray(current_modes, dtype=np.int64)
        if modes.shape != (len(x),) or np.any((modes < 0) | (modes >= len(MODES))):
            raise ValueError("current_modes must contain one valid mode index per row")
        prior = np.full_like(probs, 0.02 / (len(MODES) - 1))
        prior[np.arange(len(x)), modes] = 0.98
        return self.persistence_alpha * probs + (1.0 - self.persistence_alpha) * prior
