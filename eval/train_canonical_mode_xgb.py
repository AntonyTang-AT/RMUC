"""Train a causal XGBoost expert on the canonical-side forecast contract.

The previous mode-change XGB experiment used a different label archive.  This
script intentionally reads the canonical-side train/validation/test NPZ files
directly, so the candidate and the persistence baseline share exactly the same
rows, labels, sides and game splits.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from forecast.temporal import MODES


def _load(root: Path, split: str):
    with np.load(root / f"{split}.npz", allow_pickle=False) as a:
        x = np.asarray(a["x"], dtype=np.float32)
        y = np.asarray(a["mode_y"], dtype=np.int64)
        base = np.asarray(a["base_mode"], dtype=np.int64)
        sides = np.asarray(a["sides"])
        games = np.asarray(a["game_ids"], dtype=np.int64)
        issued = np.asarray(a["issued_at"], dtype=np.float32)
    return x, y, base, sides, games, issued


def _features(x: np.ndarray, base: np.ndarray, issued: np.ndarray) -> np.ndarray:
    """Use only the latest visible frame and causal finite differences.

    The last frame is the issue-time state.  The preceding frame is available
    before the prediction is made; the difference therefore contains no future
    information.  Including the one-hot current mode makes the expert a
    transition model while retaining the continuous state features.
    """
    if x.ndim != 3 or x.shape[1] < 2:
        raise ValueError(f"expected [N,T,F] history, got {x.shape}")
    last = x[:, -1, :]
    previous = x[:, -2, :]
    delta = np.clip(last - previous, -2.0, 2.0)
    cur = np.zeros((len(base), len(MODES)), dtype=np.float32)
    valid = (base >= 0) & (base < len(MODES))
    cur[np.arange(len(base))[valid], base[valid]] = 1.0
    t = np.asarray(issued, dtype=np.float32).reshape(-1, 1) / 420.0
    return np.concatenate([last, delta, cur, t], axis=1).astype(np.float32, copy=False)


def _metrics(y, pred, sides):
    valid = y >= 0
    rows = {}
    for side in ("红", "蓝"):
        m = valid & (sides == side)
        rows[side] = {
            "accuracy": float(accuracy_score(y[m], pred[m])),
            "macro_f1": float(f1_score(y[m], pred[m], labels=np.arange(len(MODES)), average="macro", zero_division=0)),
            "n": int(m.sum()),
        }
    rows["accuracy"] = float(np.mean([rows[s]["accuracy"] for s in ("红", "蓝")]))
    rows["macro_f1"] = float(np.mean([rows[s]["macro_f1"] for s in ("红", "蓝")]))
    rows["score"] = 0.6 * rows["accuracy"] + 0.4 * rows["macro_f1"]
    return rows


def _fused(prob, base, alpha):
    prior = np.full_like(prob, 0.01, dtype=np.float32)
    valid = (base >= 0) & (base < len(MODES))
    prior[np.arange(len(base))[valid], base[valid]] = 0.98
    prior /= prior.sum(axis=1, keepdims=True)
    return alpha * prob + (1.0 - alpha) * prior


def run(args):
    root = Path(args.root)
    train_x, train_y, train_base, train_side, train_games, train_t = _load(root, "train")
    val_x, val_y, val_base, val_side, val_games, val_t = _load(root, "validation")
    tr_x = _features(train_x, train_base, train_t)
    va_x = _features(val_x, val_base, val_t)
    tr_valid = train_y[:, args.horizon_index] >= 0
    va_valid = val_y[:, args.horizon_index] >= 0
    ytr = train_y[:, args.horizon_index]
    yva = val_y[:, args.horizon_index]
    from xgboost import XGBClassifier
    model = XGBClassifier(
        objective="multi:softprob", num_class=len(MODES), eval_metric="mlogloss",
        n_estimators=args.n_estimators, max_depth=4, learning_rate=0.04,
        min_child_weight=8, subsample=0.85, colsample_bytree=0.75,
        reg_lambda=8.0, random_state=20260928, n_jobs=8,
        tree_method="hist", device=args.device, early_stopping_rounds=50,
    )
    sample_weight = np.ones(int(tr_valid.sum()), dtype=np.float32)
    if args.balanced_weight != 1.0:
        sample_weight[ytr[tr_valid] == 1] = float(args.balanced_weight)
    model.fit(tr_x[tr_valid], ytr[tr_valid], sample_weight=sample_weight,
              eval_set=[(va_x[va_valid], yva[va_valid])], verbose=False)
    p = model.predict_proba(va_x)
    candidates = []
    for alpha in np.linspace(0.0, 1.0, 21):
        pred = _fused(p, val_base, float(alpha)).argmax(axis=1)
        m = _metrics(yva, pred, val_side)
        m["alpha"] = float(alpha)
        candidates.append(m)
    persistence = _metrics(yva, val_base, val_side)
    # Use a Pareto gate before ranking by the business score.  A candidate
    # which gains accuracy by sacrificing Macro-F1 is not an acceptable
    # replacement for the baseline, especially for the minority mode.
    pareto = [m for m in candidates
              if m["accuracy"] >= persistence["accuracy"]
              and m["macro_f1"] >= persistence["macro_f1"]]
    chosen = max(pareto or candidates,
                 key=lambda m: (m["score"], m["accuracy"], m["macro_f1"]))
    report = {
        "schema": "canonical-side-causal-xgb-v1",
        "root": str(root),
        "horizon_index": int(args.horizon_index),
        "horizon_seconds": [10, 30, 60, 120][args.horizon_index],
        "balanced_class_weight": float(args.balanced_weight),
        "feature_contract": "latest canonical frame + causal one-step difference + current mode one-hot + normalized issue time",
        "device": args.device,
        "features": int(tr_x.shape[1]),
        "best_iteration": int(model.best_iteration),
        "validation": {"persistence": persistence, "chosen": chosen, "sweep": candidates},
        "test_access": False,
    }
    # Require both accuracy and macro-F1 to improve before opening the lock.
    gate = (chosen["accuracy"] > persistence["accuracy"] and chosen["macro_f1"] >= persistence["macro_f1"])
    report["promotion_gate"] = {"passed": bool(gate), "requires_accuracy_and_macro_f1": True}
    if gate and args.test:
        test_x, test_y, test_base, test_side, test_games, test_t = _load(root, "test")
        te_x = _features(test_x, test_base, test_t)
        yt = test_y[:, args.horizon_index]
        valid = yt >= 0
        pt = model.predict_proba(te_x)
        fused = _fused(pt, test_base, chosen["alpha"])
        pred = fused.argmax(axis=1)
        report["test"] = {
            "candidate": _metrics(yt, pred, test_side),
            "persistence": _metrics(yt, test_base, test_side),
            "n_games": int(len(np.unique(test_games[valid]))),
            "alpha_frozen_on_validation": chosen["alpha"],
        }
        candidate_test = report["test"]["candidate"]
        baseline_test = report["test"]["persistence"]
        report["test_deployment_gate"] = {
            "passed": bool(candidate_test["accuracy"] > baseline_test["accuracy"]
                            and candidate_test["macro_f1"] >= baseline_test["macro_f1"]),
            "requires_accuracy_and_macro_f1": True,
            "runtime_promoted": False,
        }
        report["test_access"] = True
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    model.save_model(str(out.with_suffix(".model.ubj")))
    print(json.dumps({k: report[k] for k in ("horizon_seconds", "validation", "promotion_gate", "test_access", "test") if k in report}, ensure_ascii=False, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="out_m2/temporal_forecast_canonical_side")
    ap.add_argument("--horizon-index", type=int, required=True, choices=range(4))
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n-estimators", type=int, default=600)
    ap.add_argument("--balanced-weight", type=float, default=1.0)
    ap.add_argument("--out", required=True)
    run(ap.parse_args())


if __name__ == "__main__":
    main()
