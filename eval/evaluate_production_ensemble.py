"""Evaluate the exact causal production ensemble on the locked canonical split.

The component reports in ``out_m2`` are useful for model selection, but they
do not prove that the runtime route is better after the GRU, persistence
fusion, and LightGBM sidecars are combined.  This evaluator reproduces that
route on one fixed test split and reports global and per-robot metrics in one
file.  It never reads target fields when building features.

The report intentionally keeps two entity views:

* ``production`` follows the position policy in the runtime manifest, uses
  sidecar zone heads, and keeps GRU presence/HP heads;
* ``persistence`` uses the current visible slot as the baseline.

HP persistence is included as an explicit baseline so the HP MAE is no longer
reported without context.  The primary output is a diagnostic artifact; it
does not change runtime promotion gates.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from lightgbm import Booster

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from forecast.calibration import calibrate_horizon_probabilities
from forecast.entity import ENTITY_SLOTS
from forecast.temporal import HORIZONS, MODES, REGION_LABELS, TRENDS
from forecast.train import load_model, load_compact
from worldstate.canonical_zone import CANONICAL_ZONE_LABELS, locate_zone
from worldstate.functional_zone import (
    FUNCTIONAL_ZONE_LABELS, assign_functional_zone, functional_zone_neighbors,
)


DEFAULT_ROOT = ROOT / "out_m2" / "temporal_forecast_canonical_side"
DEFAULT_OUT = ROOT / "out_m2" / "production_ensemble_eval" / "report.json"
HORIZON_INDEX = {h: i for i, h in enumerate(HORIZONS)}
SLOT_COUNT = len(ENTITY_SLOTS)
MAP_X, MAP_Y = 28.0, 15.0


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _booster(path: Path) -> Booster:
    # LightGBM's Windows loader is more reliable with a cwd-relative path
    # when the path contains non-ASCII directory names.
    try:
        relative = os.path.relpath(str(path), os.getcwd())
        return Booster(model_file=relative)
    except Exception:
        return Booster(model_file=str(path))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50.0, 50.0)))


def _softmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    return e / np.maximum(e.sum(axis=-1, keepdims=True), 1e-12)


def _persistence_probs(base: np.ndarray, classes: int) -> np.ndarray:
    base = np.asarray(base, dtype=np.int64)
    out = np.full((len(base), classes), 0.02 / max(1, classes - 1), dtype=np.float64)
    valid = (base >= 0) & (base < classes)
    out[np.arange(len(base))[valid], base[valid]] = 0.98
    return out


def _region_marginal_probs(region_probs: np.ndarray) -> dict[str, np.ndarray]:
    """Collapse a 4x3 region distribution into horizontal and vertical bands."""
    values = np.asarray(region_probs, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(REGION_LABELS):
        raise ValueError(f"expected region probabilities [N, 12], got {values.shape}")
    grid = values.reshape(len(values), 4, 3)
    return {"x_band": grid.sum(axis=2), "y_band": grid.sum(axis=1)}


def _enemy_region_transition_probs(data: dict[str, np.ndarray], manifest: dict[str, Any]) -> np.ndarray:
    """Reproduce a frozen horizon-specific transition sidecar from its manifest."""
    classes = len(REGION_LABELS)
    alpha = float(manifest["alpha_transition"])
    bucket_seconds = int(manifest["time_bucket_seconds"])
    table = manifest.get("table") or {}
    base = np.asarray(data["base_region"], dtype=np.int64)
    sides = np.asarray(data["sides"]).astype(str)
    modes = np.asarray(data["base_mode"], dtype=np.int64)
    issued = np.asarray(data["issued_at"], dtype=np.float64)
    persistence = _persistence_probs(base, classes)
    out = np.empty((len(base), classes), dtype=np.float64)
    uniform = np.full(classes, 1.0 / classes, dtype=np.float64)
    for index in range(len(base)):
        key = json.dumps([
            int(base[index]), str(sides[index]),
            int(max(0.0, float(issued[index])) // bucket_seconds), int(modes[index]),
        ], ensure_ascii=False, separators=(",", ":"))
        transition = np.asarray(table.get(key, uniform), dtype=np.float64)
        transition /= max(float(transition.sum()), 1e-12)
        out[index] = alpha * transition + (1.0 - alpha) * persistence[index]
    return out / np.maximum(out.sum(axis=1, keepdims=True), 1e-12)


def _bias_prob(prob: np.ndarray, bias: list[float] | tuple[float, ...]) -> np.ndarray:
    logp = np.log(np.clip(np.asarray(prob, dtype=np.float64), 1e-8, 1.0))
    logp = logp + np.asarray(bias, dtype=np.float64).reshape(1, -1)
    logp = logp - logp.max(axis=1, keepdims=True)
    out = np.exp(logp)
    return out / np.maximum(out.sum(axis=1, keepdims=True), 1e-12)


def _canonical_features(data: dict[str, np.ndarray]) -> np.ndarray:
    x = np.asarray(data["x"], dtype=np.float32)
    cur, prev = x[:, -1], x[:, -2]
    mean3, mean6 = x[:, -3:].mean(axis=1), x[:, -6:].mean(axis=1)
    d1, d3 = np.clip(cur - prev, -2.0, 2.0), np.clip(cur - x[:, -4], -2.0, 2.0)
    base = np.zeros((len(x), 3), dtype=np.float32)
    mode = np.asarray(data["base_mode"], dtype=np.int64)
    valid = (mode >= 0) & (mode < 3)
    base[np.arange(len(x))[valid], mode[valid]] = 1.0
    time = np.asarray(data["issued_at"], dtype=np.float32).reshape(-1, 1) / 420.0
    return np.concatenate([cur, mean3, mean6, d1, d3, base, time], axis=1).astype(np.float32)


def _global_features(data: dict[str, np.ndarray]) -> np.ndarray:
    x = np.asarray(data["x"], dtype=np.float32)
    cur, prev = x[:, -1], x[:, -2]
    mean3, mean6 = x[:, -3:].mean(axis=1), x[:, -6:].mean(axis=1)
    d1, d3 = np.clip(cur - prev, -2.0, 2.0), np.clip(cur - x[:, -4], -2.0, 2.0)
    slot_delta = d1[:, :-8].reshape(len(x), -1, 14)
    motion = np.concatenate([
        slot_delta[..., :2].mean(axis=1),
        np.abs(slot_delta[..., :2]).mean(axis=1),
        (slot_delta[..., 9] > 0.05).mean(axis=1, keepdims=True),
    ], axis=1).astype(np.float32)
    base = np.zeros((len(x), 3), dtype=np.float32)
    mode = np.asarray(data["base_mode"], dtype=np.int64)
    valid = (mode >= 0) & (mode < 3)
    base[np.arange(len(x))[valid], mode[valid]] = 1.0
    time = np.asarray(data["issued_at"], dtype=np.float32).reshape(-1, 1) / 420.0
    return np.concatenate([cur, mean3, mean6, d1, d3, motion, base, time], axis=1).astype(np.float32)


def _metrics(y: np.ndarray, probs: np.ndarray, sides: np.ndarray,
             labels: tuple[str, ...]) -> dict[str, Any]:
    y = np.asarray(y, dtype=np.int64)
    valid = (y >= 0) & (y < len(labels))
    pred = np.argmax(probs, axis=1)
    rows = {}
    for side in ("红", "蓝"):
        mask = valid & (np.asarray(sides).astype(str) == side)
        rows[side] = {
            "accuracy": float(accuracy_score(y[mask], pred[mask])) if mask.any() else None,
            "macro_f1": float(f1_score(y[mask], pred[mask], labels=np.arange(len(labels)),
                                       average="macro", zero_division=0)) if mask.any() else None,
            "n": int(mask.sum()),
        }
    valid_rows = [rows[s] for s in ("红", "蓝") if rows[s]["accuracy"] is not None]
    acc = float(np.mean([r["accuracy"] for r in valid_rows])) if valid_rows else None
    f1 = float(np.mean([r["macro_f1"] for r in valid_rows])) if valid_rows else None
    truth = np.eye(len(labels), dtype=np.float64)[np.clip(y, 0, len(labels) - 1)]
    brier = float(np.mean(np.sum((np.asarray(probs) - truth) ** 2, axis=1)[valid])) if valid.any() else None
    return {"accuracy": acc, "macro_f1": f1, "brier": brier, "by_side": rows,
            "n": int(valid.sum())}


def _macro_f1_from_confusions(confusions: np.ndarray) -> np.ndarray:
    """Return fixed-label macro-F1 for ``[..., truth, prediction]`` matrices."""
    matrix = np.asarray(confusions, dtype=np.float64)
    true_positive = np.diagonal(matrix, axis1=-2, axis2=-1)
    false_positive = matrix.sum(axis=-2) - true_positive
    false_negative = matrix.sum(axis=-1) - true_positive
    denominator = 2.0 * true_positive + false_positive + false_negative
    scores = np.divide(2.0 * true_positive, denominator,
                       out=np.zeros_like(true_positive), where=denominator > 0.0)
    return scores.mean(axis=-1)


def _paired_game_bootstrap(y: np.ndarray, production_probs: np.ndarray,
                           persistence_probs: np.ndarray, sides: np.ndarray,
                           game_ids: np.ndarray, labels: tuple[str, ...], *,
                           replicates: int = 2000, seed: int = 20260929) -> dict[str, Any]:
    """Compare production with persistence while preserving whole red/blue games.

    Adjacent windows from one game are correlated.  Each bootstrap draw
    therefore samples complete games and keeps their red/blue views paired.
    """
    y = np.asarray(y, dtype=np.int64)
    production_probs = np.asarray(production_probs, dtype=np.float64)
    persistence_probs = np.asarray(persistence_probs, dtype=np.float64)
    side_values = np.asarray(sides).astype(str)
    game_values = np.asarray(game_ids, dtype=np.int64)
    n_classes = len(labels)
    valid = (y >= 0) & (y < n_classes)
    games = np.unique(game_values)
    confusions = np.zeros((len(games), 2, 2, n_classes, n_classes), dtype=np.int32)
    brier_sums = np.zeros((len(games), 2, 2), dtype=np.float64)
    counts = np.zeros((len(games), 2), dtype=np.int32)
    game_index = {int(game): index for index, game in enumerate(games)}
    truth = np.eye(n_classes, dtype=np.float64)[np.clip(y, 0, n_classes - 1)]

    for side_index, side in enumerate(("红", "蓝")):
        side_mask = valid & (side_values == side)
        for game in games:
            game_mask = side_mask & (game_values == game)
            if not game_mask.any():
                continue
            index = game_index[int(game)]
            target = y[game_mask]
            counts[index, side_index] = int(len(target))
            for model_index, probs in enumerate((production_probs, persistence_probs)):
                predicted = np.argmax(probs[game_mask], axis=1)
                np.add.at(confusions[index, side_index, model_index], (target, predicted), 1)
                brier_sums[index, side_index, model_index] = float(np.sum(
                    (probs[game_mask] - truth[game_mask]) ** 2))

    complete = np.flatnonzero((counts > 0).all(axis=1))
    if not len(complete):
        return {
            "unit_of_resampling": "complete game_id with paired red and blue views",
            "n_complete_games": 0,
            "replicates": int(replicates),
            "seed": int(seed),
            "confidence_level": 0.95,
            "interval_method": "paired_game_percentile_bootstrap",
            "multiplicity_adjustment": "none_pointwise_intervals",
            "available": False,
            "reason": "no game has scorable rows for both red and blue views",
        }

    confusions = confusions[complete]
    brier_sums = brier_sums[complete]
    counts = counts[complete]

    def delta_from_aggregate(confusion: np.ndarray, sums: np.ndarray,
                             denominators: np.ndarray) -> dict[str, float]:
        side_accuracy, side_f1, side_brier = [], [], []
        for side_index in range(2):
            matrices = confusion[side_index]
            accuracy = np.trace(matrices, axis1=-2, axis2=-1) / np.maximum(
                1, matrices.sum(axis=(-2, -1)))
            f1 = _macro_f1_from_confusions(matrices)
            brier = sums[side_index] / np.maximum(1, denominators[side_index])
            side_accuracy.append(float(accuracy[0] - accuracy[1]))
            side_f1.append(float(f1[0] - f1[1]))
            side_brier.append(float(brier[0] - brier[1]))
        return {
            "accuracy": float(np.mean(side_accuracy)),
            "macro_f1": float(np.mean(side_f1)),
            "brier": float(np.mean(side_brier)),
        }

    point = delta_from_aggregate(confusions.sum(axis=0), brier_sums.sum(axis=0),
                                 counts.sum(axis=0))
    samples = {"accuracy": [], "macro_f1": [], "brier": []}
    rng = np.random.default_rng(int(seed))
    batch_size = 64
    for start in range(0, int(replicates), batch_size):
        size = min(batch_size, int(replicates) - start)
        draws = rng.integers(0, len(complete), size=(size, len(complete)))
        selected_confusions = confusions[draws].sum(axis=1)
        selected_brier_sums = brier_sums[draws].sum(axis=1)
        selected_counts = counts[draws].sum(axis=1)
        side_deltas = {"accuracy": [], "macro_f1": [], "brier": []}
        for side_index in range(2):
            matrices = selected_confusions[:, side_index]
            accuracy = np.trace(matrices, axis1=-2, axis2=-1) / np.maximum(
                1, matrices.sum(axis=(-2, -1)))
            f1 = _macro_f1_from_confusions(matrices)
            brier = selected_brier_sums[:, side_index] / np.maximum(
                1, selected_counts[:, side_index])[:, None]
            side_deltas["accuracy"].append(accuracy[:, 0] - accuracy[:, 1])
            side_deltas["macro_f1"].append(f1[:, 0] - f1[:, 1])
            side_deltas["brier"].append(brier[:, 0] - brier[:, 1])
        for metric, values in side_deltas.items():
            samples[metric].extend(np.mean(np.stack(values, axis=1), axis=1).tolist())

    metrics = {}
    for metric, values in samples.items():
        interval = [float(value) for value in np.percentile(values, [2.5, 97.5])]
        lower_is_better = metric == "brier"
        metrics[metric] = {
            "point_delta_production_minus_persistence": point[metric],
            "ci95": interval,
            "lower_is_better": lower_is_better,
            "ci_excludes_zero_in_favor_of_production": bool(
                interval[1] < 0.0 if lower_is_better else interval[0] > 0.0),
            "bootstrap_probability_favors_production": float(np.mean(
                np.asarray(values) < 0.0 if lower_is_better else np.asarray(values) > 0.0)),
        }
    return {
        "unit_of_resampling": "complete game_id with paired red and blue views",
        "n_complete_games": int(len(complete)),
        "replicates": int(replicates),
        "seed": int(seed),
        "confidence_level": 0.95,
        "interval_method": "paired_game_percentile_bootstrap",
        "multiplicity_adjustment": "none_pointwise_intervals",
        "available": True,
        "metrics": metrics,
    }


def _zone_id(name: str | None, labels: tuple[str, ...]) -> int:
    try:
        return labels.index(name) if name is not None else -1
    except ValueError:
        return -1


def _zone_metrics(pred: np.ndarray, truth: np.ndarray, current: np.ndarray,
                  present_mask: np.ndarray, labels: tuple[str, ...]) -> dict[str, Any]:
    valid = present_mask & (truth >= 0) & (truth < len(labels))
    exact = valid & (pred >= 0)
    adjacent = exact & ((pred == truth) | np.asarray([
        labels[int(t)] in functional_zone_neighbors(labels[int(p)])
        if 0 <= int(p) < len(labels) and 0 <= int(t) < len(labels) else False
        for p, t in zip(pred, truth)
    ], dtype=bool)) if len(labels) == len(FUNCTIONAL_ZONE_LABELS) else exact
    return {
        "exact_accuracy": float(np.mean(pred[exact] == truth[exact])) if exact.any() else None,
        "adjacent_or_exact_accuracy": float(np.mean(adjacent[valid])) if valid.any() else None,
        "scored": int(exact.sum()),
        "target_valid": int(valid.sum()),
    }


def _binary_quality(pred_dead: np.ndarray, truth_dead: np.ndarray,
                    valid: np.ndarray) -> dict[str, Any]:
    """Precision/recall/F1 for explicit future absence (death/destruction)."""
    pred_dead = np.asarray(pred_dead, dtype=bool)[valid]
    truth_dead = np.asarray(truth_dead, dtype=bool)[valid]
    tp = int(np.sum(pred_dead & truth_dead))
    fp = int(np.sum(pred_dead & ~truth_dead))
    fn = int(np.sum(~pred_dead & truth_dead))
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2.0 * precision * recall / max(1e-12, precision + recall)
    return {"precision": float(precision), "recall": float(recall),
            "f1": float(f1), "support": int(truth_dead.sum()),
            "false_positive": fp, "false_negative": fn}


def _current_zone_arrays(data: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    latest = np.asarray(data["x"][:, -1, :SLOT_COUNT * 14], dtype=np.float32).reshape(-1, SLOT_COUNT, 14)
    current_present = (latest[..., 3] > 0.5) & (latest[..., 9] > 0.5)
    functional = np.full((len(latest), SLOT_COUNT), -1, dtype=np.int64)
    canonical = np.full((len(latest), SLOT_COUNT), -1, dtype=np.int64)
    for i in range(len(latest)):
        for slot in range(SLOT_COUNT):
            if not current_present[i, slot]:
                continue
            x, y = float(latest[i, slot, 0] * MAP_X), float(latest[i, slot, 1] * MAP_Y)
            functional[i, slot] = _zone_id(assign_functional_zone(x, y), FUNCTIONAL_ZONE_LABELS)
            canonical[i, slot] = _zone_id(locate_zone(x, y), CANONICAL_ZONE_LABELS)
    return latest, current_present, functional, canonical


def evaluate(*, root: Path = DEFAULT_ROOT, checkpoint: Path | None = None,
             output: Path = DEFAULT_OUT, batch_size: int = 512,
             device: str = "cpu", bootstrap_replicates: int = 2000) -> dict[str, Any]:
    root = Path(root)
    checkpoint = Path(checkpoint or root / "temporal_gru_v1.pt")
    data = load_compact(root / "test.npz")
    n = len(data["x"])
    dev = torch.device(device)
    model, checkpoint_data = load_model(checkpoint, device=dev)
    model.eval()

    # Keep neural outputs in compact arrays; the batch size prevents the 7GB
    # workstation limit from being exceeded.
    neural_logits: dict[str, dict[int, np.ndarray]] = {
        "global_mode": {}, "opponent_trend": {}, "enemy_region": {}}
    entity_raw: dict[int, np.ndarray] = {}
    for h in HORIZONS:
        entity_raw[h] = np.empty((n, SLOT_COUNT, 5), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, n, int(batch_size)):
            end = min(n, start + int(batch_size))
            out = model(torch.from_numpy(np.asarray(data["x"][start:end])).to(dev))
            for task in neural_logits:
                for h in HORIZONS:
                    values = out[h][task].detach().cpu().numpy()
                    if h not in neural_logits[task]:
                        neural_logits[task][h] = np.empty((n, values.shape[-1]), dtype=np.float32)
                    neural_logits[task][h][start:end] = values
                    if task == "global_mode":
                        entity_raw[h][start:end] = out[h]["entity_state"].detach().cpu().numpy()

    # Load the frozen production manifests.  These are the same paths used by
    # resolve_temporal_forecaster("auto").
    mode_manifest_path = ROOT / "out_m2" / "canonical_mode_lightgbm" / "production_manifest.json"
    mode_manifest = _load_json(mode_manifest_path)
    mode_x = _canonical_features(data)
    mode_probs: dict[int, np.ndarray] = {}
    mode_dir = mode_manifest_path.parent / "models"
    for h in HORIZONS:
        cfg = mode_manifest["horizons"][str(h)]
        raw = _booster(mode_dir / Path(cfg["model_path"]).name).predict(mode_x)
        alpha = float(cfg.get("alpha", cfg.get("validation", {}).get("chosen", {}).get("alpha", 1.0)))
        bias = cfg.get("class_logit_bias", [0.0, 0.0, 0.0])
        mode_probs[h] = _bias_prob(alpha * raw + (1.0 - alpha) *
                                    _persistence_probs(data["base_mode"], 3), bias)

    # Trend/region sidecar; disabled region horizons intentionally fall back
    # to the exact neural + persistence route from the checkpoint.
    task_manifest_path = ROOT / "out_m2" / "global_task_lightgbm" / "production_manifest.json"
    task_manifest = _load_json(task_manifest_path)
    task_x = _global_features(data)
    persistence_blend = checkpoint_data.get("persistence_blend") or {}
    alphas = persistence_blend.get("alphas") or {}
    global_probs: dict[str, dict[int, np.ndarray]] = {"opponent_trend": {}, "enemy_region": {}}
    for task, labels in (("opponent_trend", TRENDS), ("enemy_region", REGION_LABELS)):
        for h in HORIZONS:
            raw = _softmax(neural_logits[task][h])
            # Runtime calibration is per sample across horizons.
            calibrated = np.empty_like(raw, dtype=np.float64)
            for i in range(n):
                one = {hh: _softmax(neural_logits[task][hh][i:i + 1])[0]
                       for hh in HORIZONS}
                calibrated[i] = np.asarray(calibrate_horizon_probabilities(one)[h], dtype=np.float64)
            base = (np.full(n, TRENDS.index("稳定"), dtype=np.int64)
                    if task == "opponent_trend" else np.asarray(data["base_region"], dtype=np.int64))
            alpha = float(alphas.get(f"{task}@{h}s", 1.0))
            prior = _persistence_probs(base, len(labels))
            global_probs[task][h] = alpha * calibrated + (1.0 - alpha) * prior
            cfg = (task_manifest.get("tasks", {}).get(task, {}).get(str(h)) or {})
            if cfg.get("enabled"):
                booster = _booster(task_manifest_path.parent / "models" / Path(cfg["model_path"]).name)
                raw_side = booster.predict(task_x)
                side_alpha = float(cfg.get("alpha", 1.0))
                global_probs[task][h] = side_alpha * raw_side + (1.0 - side_alpha) * prior

    transition_manifests = {
        10: ROOT / "out_m2" / "enemy_region_transition" / "production_manifest_10.json",
        30: ROOT / "out_m2" / "enemy_region_transition" / "production_manifest.json",
    }
    for horizon, transition_manifest_path in transition_manifests.items():
        if not transition_manifest_path.exists():
            continue
        transition_manifest = _load_json(transition_manifest_path)
        if (transition_manifest.get("enabled")
                and int(transition_manifest.get("horizon_seconds", horizon)) == horizon):
            global_probs["enemy_region"][horizon] = _enemy_region_transition_probs(
                data, transition_manifest)

    # Entity sidecar test arrays are generated from the same locked test NPZ
    # and contain the validation-selected candidate (including its blend gate).
    sidecar_path = ROOT / "out_m2" / "entity_motion_lightgbm" / "report.npz"
    entity_manifest_path = ROOT / "out_m2" / "entity_motion_lightgbm" / "production_manifest.json"
    entity_manifest = _load_json(entity_manifest_path)
    sidecar = np.load(sidecar_path, allow_pickle=False)
    latest, current_present, current_functional, current_canonical = _current_zone_arrays(data)
    entity_report: dict[str, Any] = {}
    for h in HORIZONS:
        hi = HORIZON_INDEX[h]
        target = np.asarray(data["entity_y"][:, hi], dtype=np.float32)
        valid = target[..., 0] >= 0
        truth_present = target[..., 0] > 0.5
        # Neural presence, HP and raw position are kept authoritative.
        raw = entity_raw[h]
        pred_presence = _sigmoid(raw[..., 0])
        neural_pos = _sigmoid(raw[..., 1:3])
        pred_hp = _sigmoid(raw[..., 3])
        use_side_position = bool(
            entity_manifest.get("horizons", {}).get(str(h), {}).get(
                "position_enabled", True))
        positions = (np.asarray(sidecar[f"h{h}_test_position"], dtype=np.float32)
                     if use_side_position else neural_pos)
        side_zone = np.asarray(sidecar[f"h{h}_test_functional_zone"], dtype=np.int64)
        side_canon = np.asarray(sidecar[f"h{h}_test_canonical_zone"], dtype=np.int64)
        # Position and HP are scored only for future-present units. Presence
        # is scored over all non-censored slots.
        presence_acc = float(np.mean((pred_presence[valid] >= 0.5) == truth_present[valid])) if valid.any() else None
        # Position, HP and zone labels are only actionable for a currently
        # tracked unit.  A future spawn/respawn is scored by the lifecycle
        # head, not by a fabricated current coordinate.
        present = valid & truth_present & current_present
        pos_abs = np.abs(positions - target[..., 1:3])[present]
        hp_abs = np.abs(pred_hp - target[..., 3])[present]
        pos_mae = float(pos_abs.mean()) if pos_abs.size else None
        hp_mae = float(hp_abs.mean()) if hp_abs.size else None
        cur_pos = latest[..., :2]
        base_pos_abs = np.abs(cur_pos - target[..., 1:3])[present]
        base_hp = latest[..., 2]
        base_hp_abs = np.abs(base_hp - target[..., 3])[present]
        ftruth = np.asarray(data["entity_zone_y"][:, hi], dtype=np.int64)
        ctruth = np.asarray(data["entity_canonical_zone_y"][:, hi], dtype=np.int64)
        fmask = present & (ftruth >= 0) & (side_zone >= 0)
        cmask = present & (ctruth >= 0) & (side_canon >= 0)
        fexact = float(np.mean(side_zone[fmask] == ftruth[fmask])) if fmask.any() else None
        cexact = float(np.mean(side_canon[cmask] == ctruth[cmask])) if cmask.any() else None
        # A graph-aware score for functional zones: exact or one legal edge.
        adjacent_hits = 0
        for p, t in zip(side_zone[fmask].tolist(), ftruth[fmask].tolist()):
            if int(p) == int(t):
                adjacent_hits += 1
                continue
            p_name, t_name = FUNCTIONAL_ZONE_LABELS[int(p)], FUNCTIONAL_ZONE_LABELS[int(t)]
            adjacent_hits += int(t_name in functional_zone_neighbors(p_name))
        adjacent = float(adjacent_hits / int(fmask.sum())) if fmask.any() else None
        death_valid = valid
        death = _binary_quality(pred_presence < 0.5, ~truth_present, death_valid)
        death_baseline = _binary_quality(~current_present, ~truth_present, death_valid)
        entity_report[str(h)] = {
            "position": {"mae_normalized": pos_mae, "baseline_mae_normalized": float(base_pos_abs.mean()) if base_pos_abs.size else None,
                         "delta_vs_persistence": (pos_mae - float(base_pos_abs.mean())) if base_pos_abs.size else None,
                         "position_sidecar_enabled": bool(use_side_position)},
            "presence": {"accuracy": presence_acc, "baseline_accuracy": float(np.mean(current_present[valid] == truth_present[valid])) if valid.any() else None},
            "death": {"production": death, "persistence": death_baseline},
            "hp": {"mae": hp_mae, "persistence_mae": float(base_hp_abs.mean()) if base_hp_abs.size else None,
                   "delta_vs_persistence": (hp_mae - float(base_hp_abs.mean())) if base_hp_abs.size else None},
            "functional_zone": {"exact_accuracy": fexact, "adjacent_or_exact_accuracy": adjacent,
                                "baseline_exact_accuracy": float(np.mean(current_functional[fmask] == ftruth[fmask])) if fmask.any() else None},
            "canonical_zone": {"exact_accuracy": cexact,
                               "baseline_exact_accuracy": float(np.mean(current_canonical[cmask] == ctruth[cmask])) if cmask.any() else None},
            "scored": {"presence": int(valid.sum()), "position": int(present.sum()),
                       "functional_zone": int(fmask.sum()), "canonical_zone": int(cmask.sum())},
        }

    global_report = {"global_mode": {}, "opponent_trend": {}, "enemy_region": {}}
    for task_index, (task, labels) in enumerate(
            (("global_mode", MODES), ("opponent_trend", TRENDS), ("enemy_region", REGION_LABELS))):
        for h in HORIZONS:
            y = np.asarray(data[{"global_mode": "mode_y", "opponent_trend": "trend_y", "enemy_region": "region_y"}[task]][:, HORIZON_INDEX[h]])
            probs = mode_probs[h] if task == "global_mode" else global_probs[task][h]
            base = (np.asarray(data["base_mode"]) if task == "global_mode" else
                    (np.full(n, TRENDS.index("稳定"), dtype=np.int64) if task == "opponent_trend" else np.asarray(data["base_region"])))
            base_probs = _persistence_probs(base, len(labels))
            global_report[task][str(h)] = {
                "production": _metrics(y, probs, data["sides"], labels),
                "persistence": _metrics(y, base_probs, data["sides"], labels),
                "paired_game_bootstrap": _paired_game_bootstrap(
                    y, probs, base_probs, data["sides"], data["game_ids"], labels,
                    replicates=bootstrap_replicates,
                    seed=20260929 + 100 * task_index + int(h)),
            }

    # Exact 12-cell Macro-F1 is weak because several edge cells are rare.
    # Score its spatial marginals for broad-pressure forecasts without changing
    # the production distribution or its runtime route.
    region_marginals: dict[str, dict[str, Any]] = {"x_band": {}, "y_band": {}}
    x_labels = tuple(f"x{i}" for i in range(4))
    y_labels = tuple(f"y{i}" for i in range(3))
    for horizon_index, horizon in enumerate(HORIZONS):
        region_y = np.asarray(data["region_y"][:, horizon_index], dtype=np.int64)
        base_region = np.asarray(data["base_region"], dtype=np.int64)
        production_probabilities = _region_marginal_probs(
            global_probs["enemy_region"][horizon])
        persistence_probabilities = _region_marginal_probs(
            _persistence_probs(base_region, len(REGION_LABELS)))
        for axis, labels, target in (
                ("x_band", x_labels, np.where(region_y >= 0, region_y // 3, -1)),
                ("y_band", y_labels, np.where(region_y >= 0, region_y % 3, -1))):
            production_probs = production_probabilities[axis]
            persistence_probs = persistence_probabilities[axis]
            region_marginals[axis][str(horizon)] = {
                "production": _metrics(target, production_probs, data["sides"], labels),
                "persistence": _metrics(target, persistence_probs, data["sides"], labels),
                "paired_game_bootstrap": _paired_game_bootstrap(
                    target, production_probs, persistence_probs, data["sides"],
                    data["game_ids"], labels, replicates=bootstrap_replicates,
                    seed=20261029 + 100 * horizon_index + int(axis == "y_band")),
                "class_labels": list(labels),
            }

    report = {
        "schema": "production-ensemble-evaluation-v1",
        "split": str(root / "test.npz"),
        "locked_test_games": int(len(np.unique(data["game_ids"]))),
        "rows": int(n),
        "checkpoint": str(checkpoint),
        "global": global_report,
        "enemy_region_marginals": region_marginals,
        "entity": entity_report,
        "runtime_policy": {
            "global_mode": "canonical_mode_lightgbm",
            "opponent_trend": "global_task_lightgbm",
            "enemy_region": ("enemy_region_transition_table at enabled horizons; global_task_lightgbm only when enabled; otherwise neural+persistence"),
            "entity_position": {
                str(h): ("entity_motion_lightgbm" if bool(
                    entity_manifest.get("horizons", {}).get(str(h), {}).get(
                        "position_enabled", True)) else "neural")
                for h in HORIZONS
            },
            "entity_zone": "entity_motion_lightgbm all horizons",
            "entity_presence_hp": "neural lifecycle head",
        },
        "limitations": [
            "This locked-test report is canonical-side and must be rerun on the mature replay ledger before claiming final deployment performance.",
            "Zone exact accuracy and graph-distance/adjacent accuracy are both reported; legal projection is not counted as correctness.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output", default=str(DEFAULT_OUT))
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    args = parser.parse_args()
    report = evaluate(root=Path(args.root), checkpoint=Path(args.checkpoint) if args.checkpoint else None,
                      output=Path(args.output), batch_size=args.batch_size, device=args.device,
                      bootstrap_replicates=args.bootstrap_replicates)
    print(json.dumps({"output": args.output, "rows": report["rows"],
                      "locked_test_games": report["locked_test_games"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
