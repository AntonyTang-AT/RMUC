"""Evaluate a per-current-mode transition gate on the canonical contract.

The gate starts from the XGBoost probability distribution and only accepts a
change away from the current mode when the winning margin is large enough for
that current mode. Thresholds are selected on validation and frozen before the
test split is opened.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from forecast.temporal import MODES
from eval.train_canonical_mode_xgb import _features, _load, _metrics


def _gate(prob: np.ndarray, current: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    raw = np.argmax(prob, axis=1).astype(np.int64)
    out = current.astype(np.int64, copy=True)
    for c in range(len(MODES)):
        rows = np.flatnonzero(current == c)
        if rows.size == 0:
            continue
        candidates = raw[rows]
        changed = candidates != c
        margins = prob[rows, candidates] - prob[rows, c]
        accept = changed & (margins >= float(thresholds[c]))
        out[rows[accept]] = candidates[accept]
    return out


def run(args):
    root = Path(args.root)
    _, _, _, _, _, _ = _load(root, "train")
    vx, vy, vbase, vsides, vgames, vt = _load(root, "validation")
    tx, ty, tbase, tsides, tgames, tt = _load(root, "test")
    from xgboost import XGBClassifier
    model = XGBClassifier()
    model.load_model(str(args.model))
    vp = model.predict_proba(_features(vx, vbase, vt))
    tp = model.predict_proba(_features(tx, tbase, tt))
    vy1, ty1 = vy[:, args.horizon_index], ty[:, args.horizon_index]
    valid_v, valid_t = vy1 >= 0, ty1 >= 0
    baseline_v = _metrics(vy1, vbase, vsides)
    candidates = []
    grid = np.linspace(0.0, 0.60, 13)
    for a in grid:
        for b in grid:
            for c in grid:
                thresholds = np.asarray([a, b, c], dtype=np.float32)
                pred = _gate(vp, vbase, thresholds)
                m = _metrics(vy1, pred, vsides)
                m["thresholds"] = thresholds.tolist()
                m["override_rate"] = float(np.mean(pred[valid_v] != vbase[valid_v]))
                candidates.append(m)
    # Freeze the best Pareto point, prioritising both metrics over a single
    # accuracy gain from sacrificing the balanced class.
    pareto = [m for m in candidates
              if m["accuracy"] >= baseline_v["accuracy"]
              and m["macro_f1"] >= baseline_v["macro_f1"]]
    chosen = max(pareto or candidates,
                 key=lambda m: (m["score"], m["accuracy"], m["macro_f1"]))
    report = {
        "schema": "canonical-side-mode-transition-gate-v1",
        "root": str(root),
        "model": str(args.model),
        "horizon_seconds": [10, 30, 60, 120][args.horizon_index],
        "validation": {"persistence": baseline_v, "chosen": chosen,
                       "candidate_count": len(candidates)},
        "test_access": True,
    }
    pred_t = _gate(tp, tbase, np.asarray(chosen["thresholds"], dtype=np.float32))
    candidate_t = _metrics(ty1, pred_t, tsides)
    baseline_t = _metrics(ty1, tbase, tsides)
    report["test"] = {"candidate": candidate_t, "persistence": baseline_t,
                       "n_games": int(len(np.unique(tgames[valid_t]))),
                       "thresholds_frozen_on_validation": chosen["thresholds"]}
    report["test_deployment_gate"] = {
        "passed": bool(candidate_t["accuracy"] > baseline_t["accuracy"]
                        and candidate_t["macro_f1"] >= baseline_t["macro_f1"]),
        "requires_accuracy_and_macro_f1": True,
        "runtime_promoted": False,
    }
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("horizon_seconds", "validation", "test", "test_deployment_gate")}, ensure_ascii=False, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="out_m2/temporal_forecast_canonical_side")
    ap.add_argument("--model", required=True)
    ap.add_argument("--horizon-index", type=int, default=1, choices=range(4))
    ap.add_argument("--out", required=True)
    run(ap.parse_args())


if __name__ == "__main__":
    main()
