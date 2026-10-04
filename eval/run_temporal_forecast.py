"""Train/evaluate the local multi-horizon forecaster against persistence baselines."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from forecast.temporal import (HORIZONS, MODES, REGION_LABELS, TRENDS,
                               HORIZON_PROFILES, FEATURE_DIM, FEATURE_SCHEMA_VERSION,
                               MOTION_FEATURE_DIM, MOTION_FEATURES_PER_SLOT,
                               MOTION_DT_SCALE_SECONDS, MAP_X, MAP_Y,
                               load_compact, prepare_compact_split)
from forecast.train import (TASKS, LOSS_PROFILES, load_model, train_model)
from forecast.model import HORIZON_POOL
from forecast.calibration import CALIBRATION_VERSION, calibrate_horizon_probabilities
from forecast.entity import ENTITY_SLOTS, decode_entity_forecast, current_positions_from_feature_row
from worldstate.canonical_zone import CANONICAL_ZONE_LABELS, locate_zone
from worldstate.functional_zone import assign_functional_zone_normalized
from eval.benchmark import evaluate_forecast_ledger, forecast_key


LABELS = {"global_mode": MODES, "opponent_trend": TRENDS, "enemy_region": REGION_LABELS}


VALIDATION_POSITION_FUSION_WEIGHTS = {10: 0.25, 30: 0.75, 60: 0.75, 120: 1.0}


def _motion_vectors_from_feature_row(feature_row):
    """Decode causal per-slot velocity from the motion compact schema.

    The motion schema stores displacement normalized by map width/height and
    the issue-to-previous snapshot interval as a normalized global feature.
    This helper reconstructs m/s without using any future target fields.  It
    returns an empty mapping for legacy compact rows or missing intervals.
    """
    values = np.asarray(feature_row, dtype=np.float32).reshape(-1)
    if values.size < MOTION_FEATURE_DIM:
        return {}
    global_start = len(ENTITY_SLOTS) * MOTION_FEATURES_PER_SLOT
    delta_t = float(values[global_start + 8]) * float(MOTION_DT_SCALE_SECONDS)
    if not np.isfinite(delta_t) or delta_t <= 1e-6:
        return {}
    vectors = {}
    for index, (group, entity_type) in enumerate(ENTITY_SLOTS):
        base = index * MOTION_FEATURES_PER_SLOT
        has_position = values[base + 3] > 0.5
        present = values[base + 9] > 0.5
        if not (has_position and present):
            continue
        dx = float(values[base + 14]) * float(MAP_X)
        dy = float(values[base + 15]) * float(MAP_Y)
        if np.isfinite(dx) and np.isfinite(dy):
            vectors[f"{group}:{entity_type}"] = (dx / delta_t, dy / delta_t)
    return vectors


def _weighted_acceptance(report, model_name='temporal_gru_v1', profile='business_weighted'):
    """Return a business-weighted score without hiding raw metrics.

    Accuracy/F1 remain separately reported.  This summary answers the product
    question: did the model improve the fields that matter most for an action?
    """
    weights = LOSS_PROFILES.get(profile, LOSS_PROFILES['business_weighted'])
    task_w, horizon_w = weights['task'], weights['horizon']
    accs = report.get('accuracy_by_task_horizon', {}).get(model_name, {})
    f1s = report.get('macro_f1_by_task_horizon', {}).get(model_name, {})
    terms = []
    for task in LABELS:
        for horizon in HORIZONS:
            bucket = f'{task}@{horizon}s'
            if bucket not in accs or bucket not in f1s:
                continue
            value = 0.6 * float(accs[bucket]['accuracy']) + 0.4 * float(f1s[bucket]['macro_f1'])
            weight = float(task_w.get(task, 1.0)) * float(horizon_w.get(horizon, 1.0))
            terms.append((weight, value))
    global_score = (sum(weight * value for weight, value in terms) /
                    max(1e-9, sum(weight for weight, _ in terms)))
    entity_terms = []
    # Entity metrics are stored per model.  Older reports only had the
    # production model's map; retain that fallback for compatibility, but do
    # not silently reuse it for a baseline model.
    entity_report = report.get('entity_forecast_metrics_by_model', {}).get(model_name)
    if entity_report is None and model_name == report.get('entity_metrics_model', model_name):
        entity_report = report.get('entity_forecast_metrics', {})
    entity_report = entity_report or {}
    for horizon in HORIZONS:
        value = entity_report.get(f'{horizon}s')
        if not value:
            continue
        # Accuracy uses the unprojected neural point/zone.  Legal movement is
        # reported as a separate feasibility signal and is not allowed to
        # turn a route projection into apparent predictive accuracy.
        position_mae = value.get('raw_position_mae_normalized',
                                 value.get('position_mae_normalized', 1.0))
        zone_accuracy = value.get('raw_functional_zone_accuracy',
                                  value.get('functional_zone_accuracy', 0.0))
        entity_value = (0.40 * value['presence_accuracy'] +
                        0.45 * max(0.0, 1.0 - position_mae) +
                        0.15 * zone_accuracy)
        weight = float(horizon_w.get(horizon, 1.0))
        entity_terms.append((weight, entity_value))
    entity_score = (sum(weight * value for weight, value in entity_terms) /
                    max(1e-9, sum(weight for weight, _ in entity_terms))) if entity_terms else None
    return {
        'profile': profile,
        'global_score': round(float(global_score), 6),
        'entity_score': None if entity_score is None else round(float(entity_score), 6),
        'formula': 'global=.6*accuracy+.4*macro_f1; entity=.4*presence+.45*(1-raw_position_mae)+.15*raw_zone; legal_route_reported_separately',
        'weights': {'task': dict(task_w), 'horizon': {str(k): v for k, v in horizon_w.items()}},
    }


def _probs(index, n, eps=0.02):
    if index is None or index < 0:
        return None
    base = eps / max(1, n - 1)
    values = [base] * n
    values[int(index)] = 1.0 - eps
    return values


def _row(game_id, side, issued, horizon, task, probs, model, snapshot_hash,
         entity_forecasts=None):
    labels = list(LABELS[task])
    profile = HORIZON_PROFILES[int(horizon)]
    row = {
        "prediction_id": f"{model}:{game_id}:{side}:{issued}:{horizon}:{task}",
        "game_id": int(game_id), "side": str(side), "issued_at": float(issued),
        "valid_at": float(issued) + int(horizon), "horizon": int(horizon),
        "task": task, "entity_or_region": "global" if task != "enemy_region" else "opponent_centroid",
        "target_schema_version": "temporal-forecast-v1",
        "input_snapshot_hash": snapshot_hash,
        "position_contract": {
            "coordinate_frame": "rmuc2026_28x15m",
            "zone_schema": "functional-zone-rmuc2026-v1",
            "movement_field": "from_zone -> to_zone",
            "illegal_raw_transition": "project_to_speed_reachable_legal_route_and_keep_raw_to_zone",
        },
        "special_state_contract": {
            "schema": "specialized-unit-state-v1",
            "unknown_policy": "missing radar/sensor fields remain unknown",
            "inference_policy": "possible targets may use visible geometry and are marked as inferred",
        },
        "visibility_summary": {
            "provider": "rmuc_2026_referee_radar_protocol_v1",
            "detail_level": profile["detail_level"],
            "temporal_view": profile["temporal_view"],
            "spatial_view": profile["spatial_view"],
            "scope": profile["scope"],
            "base_tasks_stable": True,
        },
        "model_version": model,
        "calibration_version": (CALIBRATION_VERSION if model in ("temporal_gru_v1", "temporal_hybrid_v1")
                                 else "baseline-v1"),
        "labels": labels,
        "probabilities": {label: float(value) for label, value in zip(labels, probs)},
    }
    if entity_forecasts is not None:
        row["entity_forecasts"] = entity_forecasts
    return row


def _evaluate(model_path, test_npz, out_path, device="auto", batch_size=512,
              model_name="temporal_gru_v1", include_entity_forecasts=False,
              position_fusion=False):
    dev = torch.device("cuda" if device == "auto" and torch.cuda.is_available()
                       else ("cpu" if device == "auto" else device))
    model, checkpoint = load_model(model_path, device=dev)
    data = load_compact(test_npz)
    loader = torch.utils.data.DataLoader(torch.from_numpy(data["x"]), batch_size=batch_size,
                                         shuffle=False, pin_memory=(dev.type == "cuda"))
    predictions = []
    expected = []
    truth = {}
    base_rows = []
    baseline_unavailable = set()
    accuracy_counts = {model_name: {}, "persistence_v1": {}}
    confusion = {model_name: {}, "persistence_v1": {}}
    target_counts = {}
    quality_accuracy = {model_name: {}, "persistence_v1": {}}
    # Keep the learned and persistence entity evaluations separate.  The
    # previous report exposed one map and then reused it for both models,
    # making their weighted entity scores identical even when predictions
    # differed.
    entity_metrics_by_model = {model_name: {}, "persistence_v1": {}}
    entity_metrics = entity_metrics_by_model[model_name]
    cursor = 0
    with torch.no_grad():
        for xb in loader:
            outputs = model(xb.to(dev, non_blocking=True))
            for row_idx in range(len(xb)):
                i = cursor + row_idx
                gid, side, issued = int(data["game_ids"][i]), str(data["sides"][i]), float(data["issued_at"][i])
                snap_hash = hashlib.sha256(f"{gid}:{side}:{issued}".encode()).hexdigest()
                # Keep the task/class contract identical across horizons.  If
                # a farther horizon is more confident than a nearer one, only
                # widen its probability vector toward uniform.
                calibrated = {}
                for task, (_key, _n_classes) in TASKS.items():
                    raw = {
                        horizon: torch.softmax(outputs[horizon][task][row_idx], dim=-1)
                        .detach().cpu().numpy()
                        for horizon in HORIZONS
                    }
                    calibrated[task] = calibrate_horizon_probabilities(raw)
                entity_by_horizon = {}
                current_positions = current_positions_from_feature_row(data["x"][i, -1])
                position_motion_vectors = (_motion_vectors_from_feature_row(data["x"][i, -1])
                                           if position_fusion else None)
                if getattr(model, "entity_slots", 0):
                    for horizon in HORIZONS:
                        raw_entity = outputs[horizon]["entity_state"][row_idx].detach().cpu().numpy()
                        zone_logits = None
                        canonical_zone_logits = None
                        if "entity_zone_state" in outputs[horizon]:
                            zone_logits = outputs[horizon]["entity_zone_state"][row_idx].detach().cpu().numpy()
                        if "entity_canonical_zone_state" in outputs[horizon]:
                            canonical_zone_logits = outputs[horizon]["entity_canonical_zone_state"][row_idx].detach().cpu().numpy()
                        entity_by_horizon[horizon] = decode_entity_forecast(
                            raw_entity, observing_side=side,
                            current_positions=current_positions,
                            horizon_seconds=horizon,
                            zone_logits=zone_logits,
                            canonical_zone_logits=canonical_zone_logits,
                            position_fusion_weights=(
                                VALIDATION_POSITION_FUSION_WEIGHTS
                                if position_fusion else None),
                            position_motion_vectors=position_motion_vectors)
                        target_entity = data.get("entity_y")
                        if target_entity is not None:
                            target = target_entity[i, HORIZONS.index(horizon)]
                            pred_presence = np.asarray([
                                value["presence_probability"]
                                for value in entity_by_horizon[horizon].values()
                            ])
                            valid = target[:, 0] >= 0
                            metric = entity_metrics.setdefault(str(horizon), {
                                "presence_correct": 0, "presence_n": 0,
                                "position_abs_sum": 0.0, "position_n": 0,
                                "raw_position_abs_sum": 0.0, "raw_position_n": 0,
                                "hp_abs_sum": 0.0, "hp_n": 0,
                                "by_side": {},
                                "by_slot": {},
                                "zone_correct": 0, "zone_n": 0,
                                "raw_zone_correct": 0, "raw_zone_n": 0,
                                "canonical_zone_correct": 0, "canonical_zone_n": 0,
                                "raw_canonical_zone_correct": 0, "raw_canonical_zone_n": 0,
                                "transition_legal": 0, "transition_n": 0,
                            })
                            truth_presence = target[:, 0] > 0.5
                            metric["presence_correct"] += int(
                                np.sum((pred_presence[valid] >= 0.5) == truth_presence[valid]))
                            metric["presence_n"] += int(np.sum(valid))
                            for slot_index, (slot_side, _entity_type) in enumerate(ENTITY_SLOTS):
                                if not valid[slot_index] or not truth_presence[slot_index]:
                                    continue
                                pred = entity_by_horizon[horizon][f"{slot_side}:{_entity_type}"]
                                px = pred["position"]["x"] / 28.0
                                py = pred["position"]["y"] / 15.0
                                tx, ty = target[slot_index, 1], target[slot_index, 2]
                                metric["position_abs_sum"] += abs(px - tx) + abs(py - ty)
                                metric["position_n"] += 2
                                raw_position = pred.get("unprojected_position", pred["position"])
                                raw_px = float(raw_position["x"]) / 28.0
                                raw_py = float(raw_position["y"]) / 15.0
                                metric["raw_position_abs_sum"] += abs(raw_px - tx) + abs(raw_py - ty)
                                metric["raw_position_n"] += 2
                                metric["hp_abs_sum"] += abs(pred["hp_ratio"] - target[slot_index, 3])
                                metric["hp_n"] += 1
                                side_metric = metric["by_side"].setdefault(slot_side, {
                                    "presence_correct": 0, "presence_n": 0,
                                    "position_abs_sum": 0.0, "position_n": 0,
                                    "hp_abs_sum": 0.0, "hp_n": 0,
                                    "zone_correct": 0, "zone_n": 0,
                                    "canonical_zone_correct": 0, "canonical_zone_n": 0,
                                    "transition_legal": 0, "transition_n": 0,
                                })
                                side_metric["presence_correct"] += int(
                                    (pred_presence[slot_index] >= 0.5) == truth_presence[slot_index])
                                side_metric["presence_n"] += 1
                                side_metric["position_abs_sum"] += abs(px - tx) + abs(py - ty)
                                side_metric["position_n"] += 2
                                side_metric["hp_abs_sum"] += abs(pred["hp_ratio"] - target[slot_index, 3])
                                side_metric["hp_n"] += 1
                                slot_key = f"{slot_side}:{_entity_type}"
                                slot_metric = metric["by_slot"].setdefault(slot_key, {
                                    "presence_correct": 0, "presence_n": 0,
                                    "position_abs_sum": 0.0, "position_n": 0,
                                    "hp_abs_sum": 0.0, "hp_n": 0,
                                    "zone_correct": 0, "zone_n": 0,
                                    "canonical_zone_correct": 0, "canonical_zone_n": 0,
                                    "transition_legal": 0, "transition_n": 0,
                                })
                                slot_metric["presence_correct"] += int(
                                    (pred_presence[slot_index] >= 0.5) == truth_presence[slot_index])
                                slot_metric["presence_n"] += 1
                                slot_metric["position_abs_sum"] += abs(px - tx) + abs(py - ty)
                                slot_metric["position_n"] += 2
                                slot_metric["hp_abs_sum"] += abs(pred["hp_ratio"] - target[slot_index, 3])
                                slot_metric["hp_n"] += 1
                                target_zone = assign_functional_zone_normalized(
                                    float(target[slot_index, 1]), float(target[slot_index, 2]))
                                side_metric["zone_correct"] += int(pred.get("to_zone") == target_zone)
                                side_metric["zone_n"] += 1
                                canonical_target = data.get("entity_canonical_zone_y")
                                if canonical_target is not None:
                                    canonical_label = int(canonical_target[i, HORIZONS.index(horizon), slot_index])
                                    if canonical_label >= 0 and pred.get("canonical_zone_predicted_zone") is not None:
                                        side_metric["canonical_zone_correct"] += int(
                                            pred["canonical_zone_predicted_zone"] ==
                                            CANONICAL_ZONE_LABELS[canonical_label])
                                        side_metric["canonical_zone_n"] += 1
                                if pred.get("movement") is not None:
                                    side_metric["transition_legal"] += int(
                                        pred["movement"].get(
                                            "legal_path",
                                            pred["movement"].get("legal_adjacent", False)))
                                    side_metric["transition_n"] += 1
                                slot_metric["zone_correct"] += int(pred.get("to_zone") == target_zone)
                                slot_metric["zone_n"] += 1
                                if canonical_target is not None:
                                    canonical_label = int(canonical_target[i, HORIZONS.index(horizon), slot_index])
                                    if canonical_label >= 0 and pred.get("canonical_zone_predicted_zone") is not None:
                                        slot_metric["canonical_zone_correct"] += int(
                                            pred["canonical_zone_predicted_zone"] ==
                                            CANONICAL_ZONE_LABELS[canonical_label])
                                        slot_metric["canonical_zone_n"] += 1
                                if pred.get("movement") is not None:
                                    slot_metric["transition_legal"] += int(
                                        pred["movement"].get(
                                            "legal_path",
                                            pred["movement"].get("legal_adjacent", False)))
                                    slot_metric["transition_n"] += 1
                                metric["zone_correct"] += int(pred.get("to_zone") == target_zone)
                                metric["zone_n"] += 1
                                metric["raw_zone_correct"] += int(pred.get("raw_predicted_zone") == target_zone)
                                metric["raw_zone_n"] += 1
                                if canonical_target is not None:
                                    canonical_label = int(canonical_target[i, HORIZONS.index(horizon), slot_index])
                                    if canonical_label >= 0 and pred.get("canonical_zone_predicted_zone") is not None:
                                        metric["canonical_zone_correct"] += int(
                                            pred["canonical_zone_predicted_zone"] ==
                                            CANONICAL_ZONE_LABELS[canonical_label])
                                        metric["canonical_zone_n"] += 1
                                raw_canonical_target = pred.get("unprojected_canonical_zone")
                                if canonical_target is not None:
                                    canonical_label = int(canonical_target[i, HORIZONS.index(horizon), slot_index])
                                    if canonical_label >= 0 and raw_canonical_target is not None:
                                        metric["raw_canonical_zone_correct"] += int(
                                            raw_canonical_target == CANONICAL_ZONE_LABELS[canonical_label])
                                        metric["raw_canonical_zone_n"] += 1
                                if pred.get("movement") is not None:
                                    metric["transition_legal"] += int(
                                        pred["movement"].get(
                                            "legal_path",
                                            pred["movement"].get("legal_adjacent", False)))
                                    metric["transition_n"] += 1
                # Persistence entity baseline: keep the current observed
                # slot position/zone and mark no movement.  This is computed
                # from the issue-time feature row only and is kept separate
                # from the learned entity metrics.
                target_entity = data.get("entity_y")
                if target_entity is not None:
                    for horizon in HORIZONS:
                        target = target_entity[i, HORIZONS.index(horizon)]
                        valid = target[:, 0] >= 0
                        metric = entity_metrics_by_model["persistence_v1"].setdefault(str(horizon), {
                            "presence_correct": 0, "presence_n": 0,
                            "position_abs_sum": 0.0, "position_n": 0,
                            "raw_position_abs_sum": 0.0, "raw_position_n": 0,
                            "hp_abs_sum": 0.0, "hp_n": 0,
                            "zone_correct": 0, "zone_n": 0,
                            "raw_zone_correct": 0, "raw_zone_n": 0,
                            "canonical_zone_correct": 0, "canonical_zone_n": 0,
                            "raw_canonical_zone_correct": 0, "raw_canonical_zone_n": 0,
                            "transition_legal": 0, "transition_n": 0,
                            "by_side": {}, "by_slot": {},
                        })
                        truth_presence = target[:, 0] > 0.5
                        for slot_index, (slot_side, entity_type) in enumerate(ENTITY_SLOTS):
                            if not valid[slot_index]:
                                continue
                            slot = f"{slot_side}:{entity_type}"
                            present = slot in current_positions
                            metric["presence_correct"] += int(present == truth_presence[slot_index])
                            metric["presence_n"] += 1
                            if not truth_presence[slot_index] or not present:
                                continue
                            px, py = current_positions[slot]
                            tx, ty = target[slot_index, 1], target[slot_index, 2]
                            error = abs(px / 28.0 - tx) + abs(py / 15.0 - ty)
                            metric["position_abs_sum"] += error
                            metric["position_n"] += 2
                            metric["raw_position_abs_sum"] += error
                            metric["raw_position_n"] += 2
                            target_zone = assign_functional_zone_normalized(float(tx), float(ty))
                            current_zone = assign_functional_zone_normalized(float(px / 28.0), float(py / 15.0))
                            metric["zone_correct"] += int(current_zone == target_zone)
                            metric["zone_n"] += 1
                            metric["raw_zone_correct"] += int(current_zone == target_zone)
                            metric["raw_zone_n"] += 1
                            canonical_target = data.get("entity_canonical_zone_y")
                            if canonical_target is not None:
                                canonical_label = int(canonical_target[i, HORIZONS.index(horizon), slot_index])
                                if canonical_label >= 0:
                                    current_canonical = locate_zone(px, py)
                                    metric["canonical_zone_correct"] += int(
                                        current_canonical == CANONICAL_ZONE_LABELS[canonical_label])
                                    metric["canonical_zone_n"] += 1
                                    metric["raw_canonical_zone_correct"] += int(
                                        current_canonical == CANONICAL_ZONE_LABELS[canonical_label])
                                    metric["raw_canonical_zone_n"] += 1
                            metric["transition_legal"] += 1
                            metric["transition_n"] += 1

                for h_idx, horizon in enumerate(HORIZONS):
                    for task, (key, n_classes) in TASKS.items():
                        y = int(data[key][i, h_idx])
                        if y < 0:
                            continue
                        key_tuple = (gid, side, issued, horizon, task,
                                     "global" if task != "enemy_region" else "opponent_centroid")
                        expected.append(key_tuple)
                        truth[key_tuple] = LABELS[task][y]
                        probs = calibrated[task][horizon]
                        bucket = f"{task}@{horizon}s"
                        predicted_label = int(np.argmax(probs))
                        item = accuracy_counts[model_name].setdefault(
                            bucket, {"correct": 0, "n": 0, "by_side": {}})
                        side_item = item["by_side"].setdefault(str(side), {"correct": 0, "n": 0})
                        item["correct"] += int(predicted_label == y)
                        item["n"] += 1
                        side_item["correct"] += int(predicted_label == y)
                        side_item["n"] += 1
                        if "label_quality" in data:
                            quality_value = float(data["label_quality"][i, h_idx, {
                                "global_mode": 0, "opponent_trend": 1,
                                "enemy_region": 2}[task]])
                            qitem = quality_accuracy[model_name].setdefault(
                                bucket, {"correct_weight": 0.0, "weight": 0.0})
                            qitem["correct_weight"] += quality_value * int(predicted_label == y)
                            qitem["weight"] += quality_value
                        cm = confusion[model_name].setdefault(
                            bucket, np.zeros((n_classes, n_classes), dtype=np.int64))
                        cm[y, predicted_label] += 1
                        target_counts.setdefault(bucket, np.zeros(n_classes, dtype=np.int64))[y] += 1
                        predictions.append(_row(
                            gid, side, issued, horizon, task, probs,
                            model_name, snap_hash,
                            # Entity metrics are accumulated independently;
                            # embedding 16 decoded entities in every global
                            # ledger row multiplies memory for large tests.
                            entity_forecasts=(entity_by_horizon.get(horizon)
                                              if include_entity_forecasts and task == "global_mode"
                                              else None)))
                        if task == "global_mode":
                            bi = int(data["base_mode"][i])
                        elif task == "enemy_region":
                            bi = int(data["base_region"][i])
                        else:
                            bi = 1  # persistence baseline: no evidence of movement
                        if bi >= 0:
                            item = accuracy_counts["persistence_v1"].setdefault(
                                bucket, {"correct": 0, "n": 0, "by_side": {}})
                            side_item = item["by_side"].setdefault(str(side), {"correct": 0, "n": 0})
                            item["correct"] += int(bi == y)
                            item["n"] += 1
                            side_item["correct"] += int(bi == y)
                            side_item["n"] += 1
                            if "label_quality" in data:
                                quality_value = float(data["label_quality"][i, h_idx, {
                                    "global_mode": 0, "opponent_trend": 1,
                                    "enemy_region": 2}[task]])
                                qitem = quality_accuracy["persistence_v1"].setdefault(
                                    bucket, {"correct_weight": 0.0, "weight": 0.0})
                                qitem["correct_weight"] += quality_value * int(bi == y)
                                qitem["weight"] += quality_value
                            cm = confusion["persistence_v1"].setdefault(
                                bucket, np.zeros((n_classes, n_classes), dtype=np.int64))
                            cm[y, bi] += 1
                        base_probs = _probs(bi if bi >= 0 else None, n_classes)
                        if base_probs is not None:
                            base_rows.append(_row(gid, side, issued, horizon, task, base_probs,
                                                  "persistence_v1", snap_hash))
                        else:
                            # A persistence baseline is undefined when the
                            # issue-time base label is censored/missing.  Do
                            # not turn that into a fake wrong prediction or
                            # an invalid model record.
                            baseline_unavailable.add(key_tuple)
            cursor += len(xb)
    now = max(float(row["valid_at"]) for row in predictions) + 1.0
    report = evaluate_forecast_ledger(
        predictions + base_rows, truth, expected, now=now, n_bootstrap=300, seed=0,
        model_versions=(model_name, "persistence_v1"),
        unavailable_keys_by_model={"persistence_v1": baseline_unavailable},
    )
    report["model_config"] = {
        "input_dim": checkpoint["input_dim"], "hidden": checkpoint["hidden"],
        "layers": checkpoint["layers"], "history": checkpoint.get("history", []),
        "entity_slots": int(checkpoint.get("entity_slots", 0)),
        "zone_classes": int(checkpoint.get("zone_classes", 0)),
        "canonical_zone_classes": int(checkpoint.get("canonical_zone_classes", 0)),
        "feature_schema_version": str(data.get("feature_schema_version", np.asarray(["unknown"]))[0]),
        "loss_profile": checkpoint.get("loss_profile", "legacy_unknown"),
        "loss_config": checkpoint.get("loss_config"),
        "position_fusion": {
            "enabled": bool(position_fusion),
            "weights": (dict(VALIDATION_POSITION_FUSION_WEIGHTS)
                        if position_fusion else None),
            "motion_source": "causal_compact_motion_schema"
            if position_fusion else None,
        },
    }
    versions = data.get("global_mode_label_version")
    report["labeling"] = {
        "global_mode_label_version": str(versions[0]) if versions is not None else "unknown",
        "horizon_pool": {str(key): int(value) for key, value in HORIZON_POOL.items()},
        "horizon_profiles": HORIZON_PROFILES,
        "stable_tasks": list(LABELS),
        "calibration_version": CALIBRATION_VERSION,
        "entity_slots": len(ENTITY_SLOTS),
    }
    report["accuracy_by_task_horizon"] = {}
    for model_name, buckets in accuracy_counts.items():
        report["accuracy_by_task_horizon"][model_name] = {}
        for bucket, values in buckets.items():
            report["accuracy_by_task_horizon"][model_name][bucket] = {
                "accuracy": values["correct"] / max(1, values["n"]),
                "correct": values["correct"],
                "n": values["n"],
                "by_side": {
                    side: {
                        "accuracy": side_values["correct"] / max(1, side_values["n"]),
                        "correct": side_values["correct"], "n": side_values["n"],
                    }
                    for side, side_values in values["by_side"].items()
                },
            }
    if any(quality_accuracy.values()):
        report["quality_weighted_accuracy_by_task_horizon"] = {
            model_name: {
                bucket: {
                    "accuracy": float(values["correct_weight"] /
                                      max(values["weight"], 1e-9)),
                    "weight": float(values["weight"]),
                }
                for bucket, values in buckets.items()
            }
            for model_name, buckets in quality_accuracy.items()
        }
    report["macro_f1_by_task_horizon"] = {}
    for model_name, buckets in confusion.items():
        report["macro_f1_by_task_horizon"][model_name] = {}
        for bucket, matrix in buckets.items():
            scores = []
            for class_idx in range(matrix.shape[0]):
                tp = float(matrix[class_idx, class_idx])
                fp = float(matrix[:, class_idx].sum() - tp)
                fn = float(matrix[class_idx, :].sum() - tp)
                denom = 2.0 * tp + fp + fn
                scores.append((2.0 * tp / denom) if denom else 0.0)
            report["macro_f1_by_task_horizon"][model_name][bucket] = {
                "macro_f1": float(np.mean(scores)),
                "per_class": {LABELS[bucket.split('@', 1)[0]][i]: float(v)
                              for i, v in enumerate(scores)},
            }
    report["target_distribution_by_task_horizon"] = {
        bucket: {label: int(count) for label, count in zip(
            LABELS[bucket.split('@', 1)[0]], counts)}
        for bucket, counts in target_counts.items()
    }
    if entity_metrics:
        report["entity_forecast_metrics"] = {}
        for horizon, values in entity_metrics.items():
            report["entity_forecast_metrics"][f"{horizon}s"] = {
                "slot_count": len(ENTITY_SLOTS),
                "presence_accuracy": float(values["presence_correct"] / max(1, values["presence_n"])),
                "position_mae_normalized": float(values["position_abs_sum"] / max(1, values["position_n"])),
                "raw_position_mae_normalized": float(values["raw_position_abs_sum"] / max(1, values["raw_position_n"])),
                "hp_mae": float(values["hp_abs_sum"] / max(1, values["hp_n"])),
                "functional_zone_accuracy": float(values["zone_correct"] / max(1, values["zone_n"])),
                "raw_functional_zone_accuracy": float(values["raw_zone_correct"] / max(1, values["raw_zone_n"])),
                "canonical_zone_accuracy": float(values["canonical_zone_correct"] / max(1, values["canonical_zone_n"])),
                "raw_canonical_zone_accuracy": float(values["raw_canonical_zone_correct"] / max(1, values["raw_canonical_zone_n"])),
                "transition_legal_rate": float(values["transition_legal"] / max(1, values["transition_n"])),
                "by_side": {
                    side: {
                        "presence_accuracy": float(side_values["presence_correct"] / max(1, side_values["presence_n"])),
                        "position_mae_normalized": float(side_values["position_abs_sum"] / max(1, side_values["position_n"])),
                        "hp_mae": float(side_values["hp_abs_sum"] / max(1, side_values["hp_n"])),
                        "functional_zone_accuracy": float(side_values["zone_correct"] / max(1, side_values["zone_n"])),
                        "canonical_zone_accuracy": float(side_values["canonical_zone_correct"] / max(1, side_values["canonical_zone_n"])),
                        "transition_legal_rate": float(side_values["transition_legal"] / max(1, side_values["transition_n"])),
                    }
                    for side, side_values in values["by_side"].items()
                },
                "by_slot": {
                    slot: {
                        "presence_accuracy": float(slot_values["presence_correct"] / max(1, slot_values["presence_n"])),
                        "position_mae_normalized": float(slot_values["position_abs_sum"] / max(1, slot_values["position_n"])),
                        "hp_mae": float(slot_values["hp_abs_sum"] / max(1, slot_values["hp_n"])),
                        "functional_zone_accuracy": float(slot_values["zone_correct"] / max(1, slot_values["zone_n"])),
                        "canonical_zone_accuracy": float(slot_values["canonical_zone_correct"] / max(1, slot_values["canonical_zone_n"])),
                        "transition_legal_rate": float(slot_values["transition_legal"] / max(1, slot_values["transition_n"])),
                    }
                    for slot, slot_values in values["by_slot"].items()
                },
            }
        # Baseline metrics are intentionally emitted separately.  Its HP and
        # per-slot breakdown are unavailable from the compact persistence
        # contract, so they remain null/empty rather than borrowing model
        # values.
        baseline_report = {}
        for horizon, values in entity_metrics_by_model["persistence_v1"].items():
            baseline_report[f"{horizon}s"] = {
                "slot_count": len(ENTITY_SLOTS),
                "presence_accuracy": float(values["presence_correct"] / max(1, values["presence_n"])),
                "position_mae_normalized": float(values["position_abs_sum"] / max(1, values["position_n"])),
                "raw_position_mae_normalized": float(values["raw_position_abs_sum"] / max(1, values["raw_position_n"])),
                "hp_mae": None,
                "functional_zone_accuracy": float(values["zone_correct"] / max(1, values["zone_n"])),
                "raw_functional_zone_accuracy": float(values["raw_zone_correct"] / max(1, values["raw_zone_n"])),
                "canonical_zone_accuracy": float(values["canonical_zone_correct"] / max(1, values["canonical_zone_n"])),
                "raw_canonical_zone_accuracy": float(values["raw_canonical_zone_correct"] / max(1, values["raw_canonical_zone_n"])),
                "transition_legal_rate": float(values["transition_legal"] / max(1, values["transition_n"])),
                "by_side": {}, "by_slot": {},
            }
        learned_entity_model = next(
            (name for name in accuracy_counts if name != "persistence_v1"), model_name)
        report["entity_forecast_metrics_by_model"] = {
            learned_entity_model: report["entity_forecast_metrics"],
            "persistence_v1": baseline_report,
        }
        report["entity_metrics_model"] = learned_entity_model
    report["weighted_acceptance"] = {
        model_name: _weighted_acceptance(report, model_name=model_name)
        for model_name in report.get("models", {})
        if model_name in report.get("accuracy_by_task_horizon", {})
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default="out_m2/forecast_dataset")
    parser.add_argument("--work-dir", default="out_m2/temporal_forecast")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--context", type=int, default=12)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--loss-profile", default="business_weighted",
                        choices=("baseline", "business_weighted"),
                        help="importance/class-imbalance weighting used when training")
    parser.add_argument("--resume-from", default=None,
                        help="resume model weights while keeping the same feature/loss contract")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--include-entity-forecasts", action="store_true",
                        help="embed decoded entity objects in every global ledger row; high memory")
    parser.add_argument("--position-fusion-candidate", action="store_true",
                        help="evaluate the validation-derived causal velocity/model position fusion")
    args = parser.parse_args()
    work = Path(args.work_dir)
    compact = {}
    for split in ("train", "validation", "test"):
        target = work / f"{split}.npz"
        compatible = False
        if target.exists():
            try:
                with np.load(target, allow_pickle=False) as cached:
                    compatible = (
                        "entity_canonical_zone_y" in cached.files and
                        int(cached["feature_dim"][0]) == int(FEATURE_DIM) and
                        "feature_schema_version" in cached.files and
                        str(cached["feature_schema_version"][0]) == FEATURE_SCHEMA_VERSION)
            except (OSError, ValueError, KeyError, IndexError):
                compatible = False
        if not compatible:
            print(prepare_compact_split(Path(args.dataset_dir) / f"{split}.jsonl.gz", target, args.context), flush=True)
        compact[split] = target
    model_path = work / "temporal_gru_v1.pt"
    if not args.skip_train:
        print(train_model(compact["train"], compact["validation"], model_path,
                           epochs=args.epochs, batch_size=args.batch_size,
                           device=args.device, loss_profile=args.loss_profile,
                           resume_from=args.resume_from), flush=True)
    report = _evaluate(model_path, compact["test"], work / "temporal_forecast_report.json",
                       device=args.device, batch_size=args.batch_size,
                       include_entity_forecasts=args.include_entity_forecasts,
                       position_fusion=args.position_fusion_candidate)
    print(json.dumps({"report": str(work / "temporal_forecast_report.json"),
                      "scored": report.get("scored_count"),
                      "models": list(report.get("models", {}))}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
