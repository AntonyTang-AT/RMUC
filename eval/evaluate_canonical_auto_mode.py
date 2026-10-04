"""Evaluate frozen canonical-side auto global-mode fusion, without tuning.

This inference path applies softmax to head logits, then the checkpoint's
embedded validation-selected alpha. It reports pooled and red/blue metrics.
It has no training or alpha-selection code.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from eval.train_canonical_mode_candidate import _probabilities
from forecast.temporal import MODES
from forecast.train import load_model


def _metrics(y: np.ndarray, p: np.ndarray) -> dict:
    pred = p.argmax(1)
    cm = np.zeros((len(MODES), len(MODES)), dtype=np.int64)
    for truth, guess in zip(y, pred):
        cm[int(truth), int(guess)] += 1
    per_class = {}
    f1 = []
    for i, label in enumerate(MODES):
        tp = int(cm[i, i])
        fp = int(cm[:, i].sum() - tp)
        fn = int(cm[i, :].sum() - tp)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        score = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1.append(score)
        per_class[label] = {
            "precision": precision, "recall": recall, "f1": score,
            "support": int(cm[i, :].sum()),
        }
    clipped = np.clip(p, 1e-12, 1.0)
    return {
        "n": int(len(y)), "accuracy": float(np.mean(pred == y)),
        "macro_f1": float(np.mean(f1)),
        "brier": float(np.mean(np.sum((p - np.eye(len(MODES))[y]) ** 2, axis=1))),
        "log_loss": float(-np.mean(np.log(clipped[np.arange(len(y)), y]))),
        "per_class": per_class, "confusion_matrix": cm.tolist(),
    }


def run(args):
    model, checkpoint = load_model(args.checkpoint, device="cpu")
    blend = checkpoint.get("persistence_blend") or {}
    alphas = {h: float(blend.get("alphas", {}).get(f"global_mode@{h}s", 1.0))
              for h in (10, 30, 60, 120)}
    data, probs, device = _probabilities(args.checkpoint, args.test, args.device,
                                         args.batch_size)
    base = np.asarray(data["base_mode"]).astype(int)
    sides = np.asarray(data["sides"])
    report = {
        "schema": "canonical-auto-global-mode-eval-v1",
        "checkpoint": str(args.checkpoint), "test_split": str(args.test),
        "test_access": "frozen checkpoint; no parameter or threshold selection",
        "inference": "softmax(logits) -> horizon calibration -> embedded persistence blend",
        "device": device, "n_games": int(len(np.unique(data["game_ids"]))),
        "alpha_by_horizon": {f"{h}s": alphas[h] for h in alphas},
        "horizons": {},
    }
    for j, h in enumerate((10, 30, 60, 120)):
        y = np.asarray(data["mode_y"][:, j]).astype(int)
        valid = y >= 0
        prior = np.full((len(y), len(MODES)), 0.01, dtype=np.float64)
        good = (base >= 0) & (base < len(MODES))
        prior[np.arange(len(base))[good], base[good]] = 0.98
        fused = alphas[h] * probs[h] + (1 - alphas[h]) * prior
        fused /= np.maximum(fused.sum(1, keepdims=True), 1e-12)
        horizon = {"pooled": _metrics(y[valid], fused[valid])}
        for side, name in (("红", "red"), ("蓝", "blue")):
            mask = valid & (sides == side)
            horizon[name] = _metrics(y[mask], fused[mask])
        persistence = prior[valid]
        horizon["persistence_accuracy"] = float(
            np.mean(persistence.argmax(1) == y[valid]))
        report["horizons"][f"{h}s"] = horizon
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": args.output,
                      "metrics": {h: {k: v["pooled"][k] for k in
                                       ("n", "accuracy", "macro_f1", "brier", "log_loss")}
                                  for h, v in report["horizons"].items()}}, ensure_ascii=False))
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="out_m2/temporal_forecast_canonical_side/temporal_gru_v1.pt")
    parser.add_argument("--test", default="out_m2/temporal_forecast_canonical_side/test.npz")
    parser.add_argument("--output", default="out_m2/temporal_forecast_canonical_side/runtime_default_mode_test_metrics.json")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=512)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
