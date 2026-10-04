"""Canonical-side LightGBM expert with a locked-test Pareto gate.

This is deliberately separate from the runtime GRU.  It uses only the causal
issue-time history in the canonical-side NPZ and never tunes on the locked
test split.  A candidate is eligible only when accuracy and Macro-F1 both beat
the persistence baseline on the frozen test split.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupKFold


MODES = ("进攻", "均衡", "防守")


def load_split(root: Path, split: str):
    with np.load(root / f"{split}.npz", allow_pickle=False) as d:
        return {k: np.asarray(d[k]) for k in d.files}


def features(d: dict[str, np.ndarray]) -> np.ndarray:
    x = np.asarray(d["x"], dtype=np.float32)
    if x.ndim != 3 or x.shape[1] < 4:
        raise ValueError(f"expected [N,T,F] history, got {x.shape}")
    cur = x[:, -1]
    prev = x[:, -2]
    mean3 = x[:, -3:].mean(axis=1)
    mean6 = x[:, -6:].mean(axis=1)
    d1 = np.clip(cur - prev, -2.0, 2.0)
    d3 = np.clip(cur - x[:, -4], -2.0, 2.0)
    base = np.zeros((len(x), len(MODES)), dtype=np.float32)
    current = np.asarray(d["base_mode"], dtype=np.int64)
    valid = (current >= 0) & (current < len(MODES))
    base[np.arange(len(x))[valid], current[valid]] = 1.0
    t = np.asarray(d["issued_at"], dtype=np.float32).reshape(-1, 1) / 420.0
    return np.concatenate([cur, mean3, mean6, d1, d3, base, t], axis=1).astype(np.float32)


def persistence_probs(base: np.ndarray) -> np.ndarray:
    base = np.asarray(base, dtype=np.int64)
    p = np.full((len(base), 3), 0.01, dtype=np.float64)
    ok = (base >= 0) & (base < 3)
    p[np.arange(len(base))[ok], base[ok]] = 0.98
    return p / p.sum(axis=1, keepdims=True)


def metrics(y: np.ndarray, pred: np.ndarray, sides: np.ndarray) -> dict:
    y = np.asarray(y, dtype=np.int64)
    pred = np.asarray(pred, dtype=np.int64)
    valid = y >= 0
    per_side = {}
    for side in ("红", "蓝"):
        m = valid & (np.asarray(sides).astype(str) == side)
        per_side[side] = {
            "accuracy": float(accuracy_score(y[m], pred[m])),
            "macro_f1": float(f1_score(y[m], pred[m], labels=np.arange(3),
                                        average="macro", zero_division=0)),
            "n": int(m.sum()),
        }
    acc = float(np.mean([per_side[s]["accuracy"] for s in ("红", "蓝")]))
    f1 = float(np.mean([per_side[s]["macro_f1"] for s in ("红", "蓝")]))
    return {**per_side, "accuracy": acc, "macro_f1": f1,
            "score": 0.6 * acc + 0.4 * f1, "n": int(valid.sum())}


def fused(prob: np.ndarray, base: np.ndarray, alpha: float) -> np.ndarray:
    prior = persistence_probs(base)
    return float(alpha) * prob + (1.0 - float(alpha)) * prior


def bias_prob(prob: np.ndarray, bias: tuple[float, float, float]) -> np.ndarray:
    """Apply a frozen validation-selected class logit bias."""
    logp = np.log(np.clip(np.asarray(prob, dtype=np.float64), 1e-8, 1.0))
    logp = logp + np.asarray(bias, dtype=np.float64).reshape(1, 3)
    logp = logp - logp.max(axis=1, keepdims=True)
    out = np.exp(logp)
    return out / out.sum(axis=1, keepdims=True)


def make_model(horizon_index: int, threads: int) -> LGBMClassifier:
    return LGBMClassifier(
        objective="multiclass", num_class=3, n_estimators=320,
        learning_rate=0.035, num_leaves=15, min_child_samples=90,
        reg_lambda=5.0, colsample_bytree=0.85, subsample=0.9,
        subsample_freq=1, verbosity=-1, n_jobs=int(threads),
        random_state=20260928 + horizon_index,
    )


def train_one(tr, va, te, horizon_index: int, threads: int,
              selection: str = "validation",
              objective: str = "score",
              model_dir: Path | None = None) -> dict:
    X, Xv, Xt = features(tr), features(va), features(te)
    y, yv, yt = (tr["mode_y"][:, horizon_index].astype(np.int64),
                 va["mode_y"][:, horizon_index].astype(np.int64),
                 te["mode_y"][:, horizon_index].astype(np.int64))
    mt, mv, mm = y >= 0, yv >= 0, yt >= 0
    if selection == "oof":
        # Tune only on train-group OOF; both sides of a game remain in one fold.
        oof = np.full((len(y), 3), np.nan, dtype=np.float64)
        rows = np.flatnonzero(mt)
        groups = np.asarray(tr["game_ids"], dtype=np.int64)[mt]
        for fit_idx, held_idx in GroupKFold(n_splits=5).split(X[mt], y[mt], groups=groups):
            fold_model = make_model(horizon_index, threads)
            fold_model.fit(X[rows[fit_idx]], y[rows[fit_idx]])
            oof[rows[held_idx]] = fold_model.predict_proba(X[rows[held_idx]])
        select_prob = oof[mt]
        select_y, select_base = y[mt], tr["base_mode"][mt]
        select_sides = tr["sides"][mt]
    else:
        select_prob = None
        select_y, select_base = yv[mv], va["base_mode"][mv]
        select_sides = va["sides"][mv]
    model = make_model(horizon_index, threads)
    model.fit(X[mt], y[mt])
    model_path = None
    if model_dir is not None:
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / f"global_mode_{[10, 30, 60, 120][horizon_index]}s.txt"
        model.booster_.save_model(str(model_path))
    pv = model.predict_proba(Xv)
    sweep_prob = select_prob if selection == "oof" else pv[mv]
    persistence_v = metrics(select_y, select_base, select_sides)
    candidates = []
    # Small, pre-registered class-logit sweep.  Biases are selected on the
    # validation split only and frozen before the locked test is opened.
    bias_values = (-0.08, -0.04, 0.0, 0.04, 0.08)
    for alpha in np.linspace(0.0, 1.0, 21):
        for b1 in bias_values:
            for b2 in bias_values:
                bias = (0.0, float(b1), float(b2))
                adjusted = bias_prob(fused(sweep_prob, select_base, alpha), bias)
                pred = adjusted.argmax(axis=1)
                row = metrics(select_y, pred, select_sides)
                row["alpha"] = float(alpha)
                row["class_logit_bias"] = list(bias)
                candidates.append(row)
    pareto = [r for r in candidates
              if r["accuracy"] >= persistence_v["accuracy"]
              and r["macro_f1"] >= persistence_v["macro_f1"]]
    if objective == "macro":
        chosen = max(pareto or candidates,
                     key=lambda r: (r["macro_f1"], r["accuracy"], -r["alpha"]))
    else:
        chosen = max(pareto or candidates,
                     key=lambda r: (r["score"], r["accuracy"], r["macro_f1"], -r["alpha"]))

    pt = model.predict_proba(Xt)
    pred_raw = pt.argmax(axis=1)
    pred_fused = bias_prob(
        fused(pt, te["base_mode"], chosen["alpha"]),
        tuple(chosen.get("class_logit_bias") or (0.0, 0.0, 0.0)),
    ).argmax(axis=1)
    test = {
        "raw": metrics(yt, pred_raw, te["sides"]),
        "candidate": metrics(yt, pred_fused, te["sides"]),
        "persistence": metrics(yt, te["base_mode"], te["sides"]),
        "alpha_frozen_on_validation": float(chosen["alpha"]),
        "n_games": int(len(np.unique(te["game_ids"][mm]))),
    }
    c, b = test["candidate"], test["persistence"]
    gate = bool(c["accuracy"] > b["accuracy"] and c["macro_f1"] > b["macro_f1"])
    return {
        "horizon_seconds": [10, 30, 60, 120][horizon_index],
        "features": int(X.shape[1]),
        "model_path": str(model_path) if model_path is not None else None,
        "validation": {"persistence": persistence_v, "chosen": chosen,
                        "pareto_count": int(len(pareto)),
                        "selection_source": selection,
                        "selection_objective": objective},
        "test": test,
        "test_deployment_gate": {
            "passed": gate,
            "requires_accuracy_and_macro_f1": True,
            "runtime_promoted": False,
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="out_m2/temporal_forecast_canonical_side")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--selection", choices=("validation", "oof"), default="validation")
    ap.add_argument("--objective", choices=("score", "macro"), default="score")
    ap.add_argument("--horizon-index", type=int, choices=range(4), default=None)
    ap.add_argument("--out", default="out_m2/canonical_mode_lightgbm/report.json")
    ap.add_argument("--model-dir", default=None)
    args = ap.parse_args()
    root = Path(args.root)
    tr, va, te = load_split(root, "train"), load_split(root, "validation"), load_split(root, "test")
    reports = {}
    indices = [args.horizon_index] if args.horizon_index is not None else range(4)
    for i in indices:
        reports[str([10, 30, 60, 120][i])] = train_one(
            tr, va, te, i, args.threads, args.selection, args.objective,
            Path(args.model_dir) if args.model_dir else None)
        print(json.dumps(reports[str([10, 30, 60, 120][i])], ensure_ascii=False), flush=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"schema": "canonical-side-causal-lightgbm-v1",
                               "root": str(root), "test_access": True,
                               "reports": reports}, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
