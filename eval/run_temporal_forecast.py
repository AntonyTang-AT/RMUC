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
                               load_compact, prepare_compact_split)
from forecast.train import TASKS, load_model, train_model
from forecast.model import HORIZON_POOL
from eval.benchmark import evaluate_forecast_ledger, forecast_key


LABELS = {"global_mode": MODES, "opponent_trend": TRENDS, "enemy_region": REGION_LABELS}


def _probs(index, n, eps=0.02):
    if index is None or index < 0:
        return None
    base = eps / max(1, n - 1)
    values = [base] * n
    values[int(index)] = 1.0 - eps
    return values


def _row(game_id, side, issued, horizon, task, probs, model, snapshot_hash):
    labels = list(LABELS[task])
    return {
        "prediction_id": f"{model}:{game_id}:{side}:{issued}:{horizon}:{task}",
        "game_id": int(game_id), "side": str(side), "issued_at": float(issued),
        "valid_at": float(issued) + int(horizon), "horizon": int(horizon),
        "task": task, "entity_or_region": "global" if task != "enemy_region" else "opponent_centroid",
        "target_schema_version": "temporal-forecast-v1",
        "input_snapshot_hash": snapshot_hash,
        "visibility_summary": {"provider": "rmuc_2026_referee_radar_protocol_v1"},
        "model_version": model, "calibration_version": "uncalibrated-v1",
        "labels": labels,
        "probabilities": {label: float(value) for label, value in zip(labels, probs)},
    }


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
    cursor = 0
    with torch.no_grad():
        for xb in loader:
            outputs = model(xb.to(dev, non_blocking=True))
            for row_idx in range(len(xb)):
                i = cursor + row_idx
                gid, side, issued = int(data["game_ids"][i]), str(data["sides"][i]), float(data["issued_at"][i])
                snap_hash = hashlib.sha256(f"{gid}:{side}:{issued}".encode()).hexdigest()
                for h_idx, horizon in enumerate(HORIZONS):
                    for task, (key, n_classes) in TASKS.items():
                        y = int(data[key][i, h_idx])
                        if y < 0:
                            continue
                        key_tuple = (gid, side, issued, horizon, task,
                                     "global" if task != "enemy_region" else "opponent_centroid")
                        expected.append(key_tuple)
                        truth[key_tuple] = LABELS[task][y]
                        logits = outputs[horizon][task][row_idx]
                        probs = torch.softmax(logits, dim=-1).cpu().numpy().tolist()
                        bucket = f"{task}@{horizon}s"
                        predicted_label = int(torch.argmax(logits).item())
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
                                                "temporal_gru_v1", snap_hash))
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
    }
    versions = data.get("global_mode_label_version")
    report["labeling"] = {
        "global_mode_label_version": str(versions[0]) if versions is not None else "unknown",
        "horizon_pool": {str(key): int(value) for key, value in HORIZON_POOL.items()},
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
