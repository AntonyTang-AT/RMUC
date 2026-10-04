"""Train locked-test gated LightGBM experts for trend and enemy region."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, f1_score

from forecast.temporal import REGION_LABELS, TRENDS

TASKS = {
    "opponent_trend": ("trend_y", TRENDS),
    "enemy_region": ("region_y", REGION_LABELS),
}
HORIZONS = (10, 30, 60, 120)


def load_split(root: Path, split: str):
    with np.load(root / f"{split}.npz", allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def features(data: dict[str, np.ndarray]) -> np.ndarray:
    x = np.asarray(data["x"], dtype=np.float32)
    cur, prev = x[:, -1], x[:, -2]
    mean3, mean6 = x[:, -3:].mean(axis=1), x[:, -6:].mean(axis=1)
    d1, d3 = np.clip(cur - prev, -2.0, 2.0), np.clip(cur - x[:, -4], -2.0, 2.0)
    # Add causal global motion summaries.  The last eight fields are global;
    # the mean absolute displacement captures whether a short transition is
    # actually under way without using any future label.
    slot_delta = d1[:, :-8].reshape(len(x), -1, 14)
    motion = np.concatenate([
        slot_delta[..., :2].mean(axis=1),
        np.abs(slot_delta[..., :2]).mean(axis=1),
        (slot_delta[..., 9] > 0.05).mean(axis=1, keepdims=True),
    ], axis=1).astype(np.float32)
    base = np.zeros((len(x), 3), dtype=np.float32)
    mode = np.asarray(data["base_mode"], dtype=np.int64)
    valid = (mode >= 0) & (mode < 3)
    base[np.arange(len(x))[valid], mode[valid]] = 1.0
    time = np.asarray(data["issued_at"], dtype=np.float32).reshape(-1, 1) / 420.0
    return np.concatenate([cur, mean3, mean6, d1, d3, motion, base, time], axis=1).astype(np.float32)


def base_labels(data: dict[str, np.ndarray], task: str) -> np.ndarray:
    if task == "opponent_trend":
        # The persistence contract defines the current trend as stable.
        return np.full(len(data["x"]), TRENDS.index("稳定"), dtype=np.int64)
    return np.asarray(data["base_region"], dtype=np.int64)


def persistence_probs(base: np.ndarray, classes: int) -> np.ndarray:
    p = np.full((len(base), classes), 0.01 / max(1, classes - 1), dtype=np.float64)
    valid = (base >= 0) & (base < classes)
    p[np.arange(len(base))[valid], base[valid]] = 0.98
    return p / p.sum(axis=1, keepdims=True)


def metrics(y, pred, sides, classes):
    y, pred = np.asarray(y), np.asarray(pred)
    valid = (y >= 0) & (y < classes)
    per_side = {}
    for side in ("红", "蓝"):
        mask = valid & (np.asarray(sides).astype(str) == side)
        per_side[side] = {
            "accuracy": float(accuracy_score(y[mask], pred[mask])),
            "macro_f1": float(f1_score(y[mask], pred[mask], labels=np.arange(classes),
                                        average="macro", zero_division=0)),
            "n": int(mask.sum()),
        }
    accuracy = float(np.mean([per_side[s]["accuracy"] for s in ("红", "蓝")]))
    macro = float(np.mean([per_side[s]["macro_f1"] for s in ("红", "蓝")]))
    return {**per_side, "accuracy": accuracy, "macro_f1": macro,
            "score": .6 * accuracy + .4 * macro, "n": int(valid.sum())}


def fused(prob, base, alpha, classes):
    return alpha * prob + (1.0 - alpha) * persistence_probs(base, classes)


def train_one(train, val, test, task, hi, threads, model_dir):
    target_key, labels = TASKS[task]
    classes = len(labels)
    X, Xv, Xt = features(train), features(val), features(test)
    y, yv, yt = (train[target_key][:, hi].astype(np.int64),
                 val[target_key][:, hi].astype(np.int64),
                 test[target_key][:, hi].astype(np.int64))
    mt, mv, mm = y >= 0, yv >= 0, yt >= 0
    model = LGBMClassifier(
        objective="multiclass", num_class=classes, n_estimators=260,
        learning_rate=.04, num_leaves=23, min_child_samples=100,
        reg_lambda=7.0, colsample_bytree=.85, subsample=.9,
        subsample_freq=1, max_bin=63, verbosity=-1, n_jobs=threads,
        random_state=20260928 + hi * 13 + (0 if task == "opponent_trend" else 7),
    )
    model.fit(X[mt], y[mt])
    model_path = model_dir / f"{task}_{HORIZONS[hi]}s.txt"
    model.booster_.save_model(str(model_path))
    pv = model.predict_proba(Xv)
    base_v = base_labels(val, task)[mv]
    candidates = []
    for alpha in np.linspace(0.0, 1.0, 21):
        pred = fused(pv[mv], base_v, alpha, classes).argmax(axis=1)
        row = metrics(yv[mv], pred, val["sides"][mv], classes)
        row["alpha"] = float(alpha)
        candidates.append(row)
    persistence_v = metrics(yv[mv], base_v, val["sides"][mv], classes)
    pareto = [row for row in candidates if row["accuracy"] >= persistence_v["accuracy"]
              and row["macro_f1"] >= persistence_v["macro_f1"]]
    chosen = max(pareto or candidates,
                 key=lambda row: (row["score"], row["macro_f1"], row["accuracy"], -row["alpha"]))
    pt = model.predict_proba(Xt)
    base_t = base_labels(test, task)
    raw_pred = pt.argmax(axis=1)
    candidate_pred = fused(pt, base_t, float(chosen["alpha"]), classes).argmax(axis=1)
    test_report = {
        "raw": metrics(yt[mm], raw_pred[mm], test["sides"][mm], classes),
        "candidate": metrics(yt[mm], candidate_pred[mm], test["sides"][mm], classes),
        "persistence": metrics(yt[mm], base_t[mm], test["sides"][mm], classes),
        "alpha_frozen_on_validation": float(chosen["alpha"]),
        "n_games": int(len(np.unique(test["game_ids"][mm]))),
    }
    cand, base = test_report["candidate"], test_report["persistence"]
    return {
        "task": task, "horizon_seconds": HORIZONS[hi],
        "labels": list(labels), "features": int(X.shape[1]),
        "model_path": str(model_path),
        "validation": {"persistence": persistence_v, "chosen": chosen,
                        "pareto_count": len(pareto)},
        "test": test_report,
        "test_deployment_gate": {
            "passed": bool(cand["accuracy"] > base["accuracy"] and cand["macro_f1"] > base["macro_f1"]),
            "requires_accuracy_and_macro_f1": True,
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="out_m2/temporal_forecast_canonical_side")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", default="out_m2/global_task_lightgbm/report.json")
    args = ap.parse_args()
    root = Path(args.root); out = Path(args.out); model_dir = out.parent / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    train, val, test = load_split(root, "train"), load_split(root, "validation"), load_split(root, "test")
    reports = []
    for task in TASKS:
        for hi, horizon in enumerate(HORIZONS):
            print(json.dumps({"task": task, "horizon": horizon, "status": "training"}), flush=True)
            row = train_one(train, val, test, task, hi, args.threads, model_dir)
            reports.append(row)
            print(json.dumps(row["test"], ensure_ascii=False), flush=True)
    manifest = {
        "schema": "global-task-lightgbm-manifest-v1",
        "feature_schema": "canonical-side-task-causal-v2",
        "tasks": {
            task: {str(r["horizon_seconds"]): {
                "model_path": Path(r["model_path"]).name,
                "alpha": r["validation"]["chosen"]["alpha"],
                "labels": r["labels"],
            } for r in reports if r["task"] == task} for task in TASKS
        },
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"schema": "global-task-causal-lightgbm-v1",
                               "reports": reports}, ensure_ascii=False, indent=2), encoding="utf-8")
    (out.parent / "production_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
