"""Validation-only hyperparameter sweep for the 30 s mode XGB.

The test split is deliberately never opened by this script.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

try:
    from eval.train_mode_change_xgb import _side_metrics, load_split
except ModuleNotFoundError:  # direct `python eval/tune_mode_change_xgb.py`
    from train_mode_change_xgb import _side_metrics, load_split
from forecast.temporal import MODES


CONFIGS = (
    {"name": "candidate_baseline", "max_depth": 4, "min_child_weight": 10,
     "learning_rate": .03, "reg_lambda": 5., "colsample_bytree": .9},
    {"name": "shallower_regularized", "max_depth": 3, "min_child_weight": 15,
     "learning_rate": .03, "reg_lambda": 10., "colsample_bytree": .8},
    {"name": "deeper_regularized", "max_depth": 5, "min_child_weight": 15,
     "learning_rate": .02, "reg_lambda": 8., "colsample_bytree": .85},
)


def run(root: Path, device: str, out_path: Path):
    x_train, y_train, *_ = load_split(root, "train")
    x_val, y_val, current, sides, games = load_split(root, "validation")
    train_mask, val_mask = y_train >= 0, y_val >= 0
    from xgboost import XGBClassifier

    results = []
    for config in CONFIGS:
        params = {k: v for k, v in config.items() if k != "name"}
        model = XGBClassifier(
            objective="multi:softprob", num_class=len(MODES), eval_metric="mlogloss",
            n_estimators=700, **params, subsample=.8, random_state=17, n_jobs=8,
            tree_method="hist", device=device, early_stopping_rounds=50,
        )
        model.fit(x_train[train_mask], y_train[train_mask],
                  eval_set=[(x_val[val_mask], y_val[val_mask])], verbose=False)
        probabilities = model.predict_proba(x_val)
        for alpha in (.90, .95, 1.0):
            blended = probabilities.copy()
            for row in np.flatnonzero(val_mask):
                label = int(current[row])
                prior = np.full(len(MODES), .02 / (len(MODES) - 1), dtype=np.float32)
                prior[label] = .98
                blended[row] = alpha * probabilities[row] + (1 - alpha) * prior
            metrics = _side_metrics(y_val, blended.argmax(axis=1), sides)
            metrics.update({"config": config["name"], "alpha": alpha,
                            "best_iteration": int(model.best_iteration),
                            "best_validation_mlogloss": float(model.best_score),
                            "n_validation_games": int(len(np.unique(games[val_mask])))})
            results.append(metrics)
    results.sort(key=lambda r: (r["balanced_accuracy"], r["balanced_macro_f1"]), reverse=True)
    report = {"test_opened": False, "device": device, "results": results}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"test_opened": False, "top_results": results[:5]},
                     ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="out_m2/temporal_hybrid_mode_stats_ablation")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", default="out_m2/mode_change_xgb_validation_sweep/report.json")
    args = parser.parse_args()
    run(Path(args.root), args.device, Path(args.out))


if __name__ == "__main__":
    main()
