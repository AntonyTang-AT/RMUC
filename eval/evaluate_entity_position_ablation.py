"""Evaluate causal group-centroid position decoding on a held-out split.

This is deliberately separate from the large forecast ledger evaluator.  It
scores only entity positions, so a candidate decoder can be checked without
allocating hundreds of thousands of global ledger rows.  The velocity
definition is identical to the validation ablation: visible same-type group
centroid, exact previous 3-second snapshot, speed capped by the type prior.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from forecast.temporal import HORIZONS, load_compact
from forecast.entity import ENTITY_SLOTS
from worldstate.functional_zone import assign_functional_zone_normalized
from eval.crossfit_validation_velocity_position import load_velocity, predict_frozen


DEFAULT_WEIGHTS = {10: 0.25, 30: 0.75, 60: 0.75, 120: 1.0}
MAP = np.asarray([28.0, 15.0], dtype=np.float32)


def _zone_accuracy(points, truth, mask):
    correct = 0
    total = 0
    for row, col in np.argwhere(mask):
        predicted = assign_functional_zone_normalized(
            float(points[row, col, 0]), float(points[row, col, 1]))
        expected = assign_functional_zone_normalized(
            float(truth[row, col, 0]), float(truth[row, col, 1]))
        correct += int(predicted == expected)
        total += 1
    return correct / max(1, total), total


def evaluate(checkpoint, test_npz, test_jsonl, output, device="cpu", batch_size=1024):
    data = load_compact(test_npz)
    current_m, velocity_mps, has_velocity = load_velocity(test_jsonl, data)
    pred = predict_frozen(checkpoint, data["x"], device=device, batch_size=batch_size)
    target = np.asarray(data["entity_y"], dtype=np.float32)
    current_norm = current_m / MAP[None, None, :]
    results = {}
    for h_idx, horizon in enumerate(HORIZONS):
        truth = target[:, h_idx, :, 1:3]
        mask = ((target[:, h_idx, :, 0] > 0.5) &
                np.isfinite(current_norm).all(axis=-1))
        raw = pred[:, h_idx]
        kinematic_m = current_m + velocity_mps * float(horizon)
        kinematic_m = np.where(has_velocity[..., None], kinematic_m, current_m)
        kinematic = kinematic_m / MAP[None, None, :]
        weight = float(DEFAULT_WEIGHTS[int(horizon)])
        fused = weight * raw + (1.0 - weight) * kinematic
        rows = {}
        for name, points in (("persistence", current_norm),
                             ("neural", raw), ("causal_fusion", fused)):
            selected_error = np.abs(points - truth)[mask]
            zone_acc, zone_n = _zone_accuracy(points, truth, mask)
            rows[name] = {
                "position_mae_normalized": float(np.mean(selected_error))
                if selected_error.size else None,
                "functional_zone_accuracy": float(zone_acc),
                "scored_slots": int(mask.sum()),
                "zone_scored_slots": int(zone_n),
            }
        results[f"{horizon}s"] = {
            "weights_model": weight,
            "velocity_available_rate": float(np.mean(has_velocity[mask]))
            if np.any(mask) else 0.0,
            "methods": rows,
        }
    report = {
        "schema_version": "entity-position-ablation-test-v1",
        "checkpoint": str(checkpoint),
        "test_npz": str(test_npz),
        "test_jsonl": str(test_jsonl),
        "velocity_contract": "visible same-type group centroid and exact previous 3s snapshot; no future/yaw/target",
        "weights": dict(DEFAULT_WEIGHTS),
        "results": results,
    }
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--test-npz", required=True)
    parser.add_argument("--test-jsonl", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=1024)
    args = parser.parse_args()
    report = evaluate(args.checkpoint, args.test_npz, args.test_jsonl,
                      args.output, device=args.device, batch_size=args.batch_size)
    print(json.dumps(report["results"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
