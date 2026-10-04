"""Tune causal persistence fusion for a trained forecast checkpoint.

The model is evaluated on a validation split first.  For every task/horizon,
only the mixture coefficient is selected; the test split is touched once
afterward.  The persistence prior is computed from the current compact row,
so this script never uses a matured target while choosing a coefficient.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from forecast.calibration import calibrate_horizon_probabilities
from forecast.entity import current_positions_from_feature_row
from forecast.temporal import HORIZONS, MODES, TRENDS, REGION_LABELS, load_compact
from forecast.train import TASKS, load_model, LOSS_PROFILES


LABELS = {
    "global_mode": ("mode_y", len(MODES)),
    "opponent_trend": ("trend_y", len(TRENDS)),
    "enemy_region": ("region_y", len(REGION_LABELS)),
}


def _macro_f1(y_true, y_pred, n_classes):
    scores = []
    for cls in range(n_classes):
        tp = np.sum((y_true == cls) & (y_pred == cls))
        fp = np.sum((y_true != cls) & (y_pred == cls))
        fn = np.sum((y_true == cls) & (y_pred != cls))
        den = 2 * tp + fp + fn
        scores.append(float(2 * tp / den) if den else 0.0)
    return float(np.mean(scores))


def _predict(model_path, compact_path, device="auto", batch_size=1024):
    dev = torch.device("cuda" if device == "auto" and torch.cuda.is_available()
                       else ("cpu" if device == "auto" else device))
    model, _ = load_model(model_path, device=dev)
    data = load_compact(compact_path)
    outputs = {task: {h: [] for h in HORIZONS} for task in LABELS}
    loader = torch.utils.data.DataLoader(torch.from_numpy(data["x"]),
                                         batch_size=batch_size, shuffle=False,
                                         pin_memory=(dev.type == "cuda"))
    with torch.no_grad():
        for xb in loader:
            pred = model(xb.to(dev, non_blocking=True))
            for task in LABELS:
                for h in HORIZONS:
                    values = torch.softmax(pred[h][task], dim=-1).cpu().numpy()
                    outputs[task][h].append(values)
    outputs = {task: {h: np.concatenate(values, axis=0)
                      for h, values in buckets.items()}
               for task, buckets in outputs.items()}
    # Apply the same horizon calibration used by runtime/evaluation.
    calibrated = {task: {h: np.zeros_like(values)
                         for h, values in buckets.items()}
                 for task, buckets in outputs.items()}
    n = len(data["x"])
    for i in range(n):
        for task in LABELS:
            sample = {h: outputs[task][h][i] for h in HORIZONS}
            fixed = calibrate_horizon_probabilities(sample)
            for h in HORIZONS:
                calibrated[task][h][i] = fixed[h]
    return data, calibrated, str(dev)


def _prior(data, task, i, n_classes):
    if task == "global_mode":
        index = int(data["base_mode"][i])
    elif task == "enemy_region":
        index = int(data["base_region"][i])
    else:
        index = 1  # 稳定
    prior = np.full(n_classes, 0.02 / max(1, n_classes - 1), dtype=np.float32)
    if 0 <= index < n_classes:
        prior[index] = 0.98
        return prior
    return None


def _score(data, probs, alphas=None):
    result = {}
    terms = []
    weights = LOSS_PROFILES["business_weighted"]
    for task, (key, n_classes) in LABELS.items():
        for h_idx, h in enumerate(HORIZONS):
            y = data[key][:, h_idx].astype(np.int64)
            valid = y >= 0
            p = probs[task][h].copy()
            alpha = 1.0 if alphas is None else float(alphas.get(f"{task}@{h}s", 1.0))
            if alpha < 1.0:
                for i in np.flatnonzero(valid):
                    prior = _prior(data, task, int(i), n_classes)
                    if prior is not None:
                        p[i] = alpha * p[i] + (1.0 - alpha) * prior
            pred = np.argmax(p, axis=1)
            acc = float(np.mean(pred[valid] == y[valid])) if np.any(valid) else 0.0
            f1 = _macro_f1(y[valid], pred[valid], n_classes) if np.any(valid) else 0.0
            pooled_score = 0.6 * acc + 0.4 * f1
            by_side = {}
            sides = np.asarray(data.get("sides", np.full(len(y), "unknown")))
            for side in ("红", "蓝"):
                side_valid = valid & (sides == side)
                if not np.any(side_valid):
                    continue
                side_acc = float(np.mean(pred[side_valid] == y[side_valid]))
                side_f1 = _macro_f1(y[side_valid], pred[side_valid], n_classes)
                by_side[side] = {
                    "accuracy": side_acc, "macro_f1": side_f1,
                    "score": 0.6 * side_acc + 0.4 * side_f1,
                    "n": int(np.sum(side_valid)),
                }
            # Select one alpha per task/horizon after giving each observing
            # side equal influence, regardless of how many rows it contributes.
            balanced_side_score = (float(np.mean([x["score"] for x in by_side.values()]))
                                   if by_side else 0.0)
            bucket = f"{task}@{h}s"
            result[bucket] = {"accuracy": acc, "macro_f1": f1,
                              "score": pooled_score,
                              "balanced_side_score": balanced_side_score,
                              "by_side": by_side,
                              "alpha": alpha, "n": int(np.sum(valid))}
            terms.append((float(weights["task"].get(task, 1.0)) *
                          float(weights["horizon"].get(h, 1.0)),
                          pooled_score))
    balanced_terms = [
        (float(weights["task"].get(task, 1.0)) * float(weights["horizon"].get(h, 1.0)),
         result[f"{task}@{h}s"]["balanced_side_score"])
        for task in LABELS for h in HORIZONS
    ]
    result["global_score"] = sum(w * v for w, v in terms) / max(1e-9, sum(w for w, _ in terms))
    result["balanced_side_global_score"] = (
        sum(w * v for w, v in balanced_terms) / max(1e-9, sum(w for w, _ in balanced_terms)))
    result["by_side_global_score"] = {}
    for side in ("红", "蓝"):
        side_terms = []
        for task in LABELS:
            for h in HORIZONS:
                bucket = result[f"{task}@{h}s"]["by_side"].get(side)
                if bucket:
                    weight = float(weights["task"].get(task, 1.0)) * float(weights["horizon"].get(h, 1.0))
                    side_terms.append((weight, bucket["score"]))
        if side_terms:
            result["by_side_global_score"][side] = (
                sum(w * v for w, v in side_terms) / sum(w for w, _ in side_terms))
    return result


def tune(model_path, validation_path, test_path, output_path, device="auto", batch_size=1024):
    val_data, val_probs, used_device = _predict(model_path, validation_path, device, batch_size)
    available_sides = set(np.asarray(val_data.get("sides", [])).tolist())
    if not {"红", "蓝"}.issubset(available_sides):
        raise ValueError("validation split must contain both red and blue observing sides")
    candidates = np.linspace(0.0, 1.0, 11)
    alphas = {}
    val_buckets = {}
    for task, (key, n_classes) in LABELS.items():
        for h in HORIZONS:
            best = None
            for alpha in candidates:
                score = _score(val_data, val_probs, {f"{task}@{h}s": float(alpha)})[f"{task}@{h}s"]["balanced_side_score"]
                if best is None or score > best[0] + 1e-12:
                    best = (score, float(alpha))
            alphas[f"{task}@{h}s"] = best[1]
            val_buckets[f"{task}@{h}s"] = best[0]
    tuned_val = _score(val_data, val_probs, alphas)
    test_data, test_probs, _ = _predict(model_path, test_path, device, batch_size)
    tuned_test = _score(test_data, test_probs, alphas)
    raw_test = _score(test_data, test_probs)
    report = {
        "schema_version": "causal-persistence-fusion-v2",
        "model": str(model_path), "device": used_device,
        "grid": [float(x) for x in candidates],
        "alphas": alphas,
        "validation": {"global_score": tuned_val["global_score"],
                        "balanced_side_global_score": tuned_val["balanced_side_global_score"],
                        "by_side_global_score": tuned_val["by_side_global_score"],
                        "per_bucket": {k: tuned_val[k] for k in val_buckets},
                        "selected_balanced_side_bucket_scores": val_buckets},
        "test": {"raw": raw_test, "tuned": tuned_test},
        "causal_contract": {
            "uses_current_snapshot_baseline": True,
            "uses_future_rows_at_issue_time": False,
            "method": "alpha*model_probability+(1-alpha)*smoothed_current_label_prior",
        },
    }
    report["test"]["raw_by_side_global_score"] = raw_test["by_side_global_score"]
    report["test"]["tuned_by_side_global_score"] = tuned_test["by_side_global_score"]
    report["test"]["raw_pooled_global_score"] = raw_test["global_score"]
    report["test"]["tuned_pooled_global_score"] = tuned_test["global_score"]
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def _write_deployment_checkpoint(source_path, output_path, alphas, *, tuning_summary=None):
    """Save a separate checkpoint carrying validation-selected causal alphas."""
    source_path, output_path = Path(source_path), Path(output_path)
    if source_path.resolve() == output_path.resolve():
        raise ValueError("checkpoint-output must differ from the training checkpoint")
    if set(alphas) != {f"{task}@{h}s" for task in LABELS for h in HORIZONS}:
        raise ValueError("fusion alphas must contain every task/horizon bucket")
    clean_alphas = {}
    for key, value in alphas.items():
        value = float(value)
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"invalid fusion alpha for {key}: {value}")
        clean_alphas[key] = value
    checkpoint = torch.load(source_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise ValueError("source checkpoint must be a model checkpoint mapping")
    deployed = copy.deepcopy(checkpoint)
    deployed["persistence_blend"] = {
        "schema": "causal-persistence-fusion-v2",
        "enabled": True,
        "method": "alpha*model_probability+(1-alpha)*smoothed_current_label_prior",
        "selection_objective": "equal_red_blue_mean_of_0.6_accuracy_plus_0.4_macro_f1",
        "tuning_split": "validation",
        "source_model_checkpoint": str(source_path),
        "alphas": clean_alphas,
        "tuning_summary": dict(tuning_summary or {}),
        "causal_contract": {
            "uses_current_snapshot_baseline": True,
            "uses_future_rows_at_issue_time": False,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(deployed, output_path)
    return deployed["persistence_blend"]


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--validation", required=True)
    p.add_argument("--test", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--checkpoint-output", default=None,
                   help="optional separate deployment checkpoint with validation-tuned fusion alphas")
    p.add_argument("--device", default="auto")
    p.add_argument("--batch-size", type=int, default=1024)
    args = p.parse_args(argv)
    report = tune(model_path=args.model, validation_path=args.validation,
                  test_path=args.test, output_path=args.output,
                  device=args.device, batch_size=args.batch_size)
    if args.checkpoint_output:
        fusion = _write_deployment_checkpoint(
            args.model, args.checkpoint_output, report["alphas"],
            tuning_summary={
                "validation_pooled_global_score": report["validation"]["global_score"],
                "validation_balanced_side_global_score": report["validation"]["balanced_side_global_score"],
                "validation_by_side_global_score": report["validation"]["by_side_global_score"],
                "test_pooled_global_score": report["test"]["tuned"]["global_score"],
                "test_by_side_global_score": report["test"]["tuned"]["by_side_global_score"],
            })
    print(json.dumps({"output": args.output,
                      "checkpoint_output": args.checkpoint_output,
                      "validation_global_score": report["validation"]["global_score"],
                      "validation_balanced_side_global_score": report["validation"]["balanced_side_global_score"],
                      "validation_by_side_global_score": report["validation"]["by_side_global_score"],
                      "test_raw_global_score": report["test"]["raw"]["global_score"],
                      "test_tuned_global_score": report["test"]["tuned"]["global_score"]},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
