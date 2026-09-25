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
                               HORIZON_PROFILES, load_compact, prepare_compact_split)
from forecast.train import TASKS, load_model, train_model
from forecast.model import HORIZON_POOL
from forecast.calibration import CALIBRATION_VERSION, enforce_monotonic_horizon_confidence
from forecast.entity import ENTITY_SLOTS, decode_entity_forecast
from eval.benchmark import evaluate_forecast_ledger, forecast_key


LABELS = {"global_mode": MODES, "opponent_trend": TRENDS, "enemy_region": REGION_LABELS}


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
        "visibility_summary": {
            "provider": "rmuc_2026_referee_radar_protocol_v1",
            "detail_level": profile["detail_level"],
            "temporal_view": profile["temporal_view"],
            "spatial_view": profile["spatial_view"],
            "scope": profile["scope"],
            "base_tasks_stable": True,
        },
        "model_version": model,
        "calibration_version": (CALIBRATION_VERSION if model == "temporal_gru_v1"
                                 else "baseline-v1"),
        "labels": labels,
        "probabilities": {label: float(value) for label, value in zip(labels, probs)},
    }
    if entity_forecasts is not None:
        row["entity_forecasts"] = entity_forecasts
    return row


def _evaluate(model_path, test_npz, out_path, device="auto", batch_size=512):
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
    accuracy_counts = {"temporal_gru_v1": {}, "persistence_v1": {}}
    confusion = {"temporal_gru_v1": {}, "persistence_v1": {}}
    target_counts = {}
    entity_metrics = {}
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
                    calibrated[task] = enforce_monotonic_horizon_confidence(raw)
                entity_by_horizon = {}
                if getattr(model, "entity_slots", 0):
                    for horizon in HORIZONS:
                        raw_entity = outputs[horizon]["entity_state"][row_idx].detach().cpu().numpy()
                        entity_by_horizon[horizon] = decode_entity_forecast(
                            raw_entity, observing_side=side)
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
                                "hp_abs_sum": 0.0, "hp_n": 0,
                                "by_side": {},
                                "by_slot": {},
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
                                metric["hp_abs_sum"] += abs(pred["hp_ratio"] - target[slot_index, 3])
                                metric["hp_n"] += 1
                                side_metric = metric["by_side"].setdefault(slot_side, {
                                    "presence_correct": 0, "presence_n": 0,
                                    "position_abs_sum": 0.0, "position_n": 0,
                                    "hp_abs_sum": 0.0, "hp_n": 0,
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
                                })
                                slot_metric["presence_correct"] += int(
                                    (pred_presence[slot_index] >= 0.5) == truth_presence[slot_index])
                                slot_metric["presence_n"] += 1
                                slot_metric["position_abs_sum"] += abs(px - tx) + abs(py - ty)
                                slot_metric["position_n"] += 2
                                slot_metric["hp_abs_sum"] += abs(pred["hp_ratio"] - target[slot_index, 3])
                                slot_metric["hp_n"] += 1
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
                        item = accuracy_counts["temporal_gru_v1"].setdefault(
                            bucket, {"correct": 0, "n": 0, "by_side": {}})
                        side_item = item["by_side"].setdefault(str(side), {"correct": 0, "n": 0})
                        item["correct"] += int(predicted_label == y)
                        item["n"] += 1
                        side_item["correct"] += int(predicted_label == y)
                        side_item["n"] += 1
                        cm = confusion["temporal_gru_v1"].setdefault(
                            bucket, np.zeros((n_classes, n_classes), dtype=np.int64))
                        cm[y, predicted_label] += 1
                        target_counts.setdefault(bucket, np.zeros(n_classes, dtype=np.int64))[y] += 1
                        predictions.append(_row(gid, side, issued, horizon, task, probs,
                                                "temporal_gru_v1", snap_hash,
                                                entity_forecasts=(entity_by_horizon.get(horizon)
                                                                  if task == "global_mode" else None)))
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
                            cm = confusion["persistence_v1"].setdefault(
                                bucket, np.zeros((n_classes, n_classes), dtype=np.int64))
                            cm[y, bi] += 1
                        base_probs = _probs(bi if bi >= 0 else None, n_classes)
                        if base_probs is not None:
                            base_rows.append(_row(gid, side, issued, horizon, task, base_probs,
                                                  "persistence_v1", snap_hash))
            cursor += len(xb)
    now = max(float(row["valid_at"]) for row in predictions) + 1.0
    report = evaluate_forecast_ledger(
        predictions + base_rows, truth, expected, now=now, n_bootstrap=300, seed=0,
        model_versions=("temporal_gru_v1", "persistence_v1"),
    )
    report["model_config"] = {
        "input_dim": checkpoint["input_dim"], "hidden": checkpoint["hidden"],
        "layers": checkpoint["layers"], "history": checkpoint.get("history", []),
        "entity_slots": int(checkpoint.get("entity_slots", 0)),
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
                "hp_mae": float(values["hp_abs_sum"] / max(1, values["hp_n"])),
                "by_side": {
                    side: {
                        "presence_accuracy": float(side_values["presence_correct"] / max(1, side_values["presence_n"])),
                        "position_mae_normalized": float(side_values["position_abs_sum"] / max(1, side_values["position_n"])),
                        "hp_mae": float(side_values["hp_abs_sum"] / max(1, side_values["hp_n"])),
                    }
                    for side, side_values in values["by_side"].items()
                },
                "by_slot": {
                    slot: {
                        "presence_accuracy": float(slot_values["presence_correct"] / max(1, slot_values["presence_n"])),
                        "position_mae_normalized": float(slot_values["position_abs_sum"] / max(1, slot_values["position_n"])),
                        "hp_mae": float(slot_values["hp_abs_sum"] / max(1, slot_values["hp_n"])),
                    }
                    for slot, slot_values in values["by_slot"].items()
                },
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
    parser.add_argument("--skip-train", action="store_true")
    args = parser.parse_args()
    work = Path(args.work_dir)
    compact = {}
    for split in ("train", "validation", "test"):
        target = work / f"{split}.npz"
        if not target.exists():
            print(prepare_compact_split(Path(args.dataset_dir) / f"{split}.jsonl.gz", target, args.context), flush=True)
        compact[split] = target
    model_path = work / "temporal_gru_v1.pt"
    if not args.skip_train:
        print(train_model(compact["train"], compact["validation"], model_path,
                           epochs=args.epochs, batch_size=args.batch_size,
                           device=args.device), flush=True)
    report = _evaluate(model_path, compact["test"], work / "temporal_forecast_report.json",
                       device=args.device, batch_size=args.batch_size)
    print(json.dumps({"report": str(work / "temporal_forecast_report.json"),
                      "scored": report.get("scored_count"),
                      "models": list(report.get("models", {}))}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
