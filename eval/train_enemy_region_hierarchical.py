"""Train a causal hierarchical enemy-region expert.

The flat 12-cell classifier is dominated by the two central cells.  This
candidate predicts the coarse x-band and y-band separately, then combines
their calibrated probabilities into the 12-cell distribution.  Promotion is
strict: a horizon is enabled only when both accuracy and Macro-F1 beat the
current-region persistence baseline on the locked test after validation-only
alpha selection.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, f1_score

from forecast.temporal import REGION_LABELS, HORIZONS
from eval.train_global_task_lightgbm import features as global_features


def load(root: Path, split: str):
    with np.load(root / f"{split}.npz", allow_pickle=False) as z:
        return {k: np.asarray(z[k]) for k in z.files}


def metrics(y, pred, sides):
    y, pred = np.asarray(y), np.asarray(pred)
    valid = (y >= 0) & (y < len(REGION_LABELS))
    by_side = {}
    for side in ("红", "蓝"):
        mask = valid & (np.asarray(sides).astype(str) == side)
        by_side[side] = {
            "accuracy": float(accuracy_score(y[mask], pred[mask])),
            "macro_f1": float(f1_score(y[mask], pred[mask], labels=np.arange(len(REGION_LABELS)),
                                       average="macro", zero_division=0)),
            "n": int(mask.sum()),
        }
    acc = float(np.mean([by_side[s]["accuracy"] for s in ("红", "蓝")]))
    f1 = float(np.mean([by_side[s]["macro_f1"] for s in ("红", "蓝")]))
    return {"accuracy": acc, "macro_f1": f1, "score": .6 * acc + .4 * f1,
            "by_side": by_side, "n": int(valid.sum())}


def _fit(X, y, seed, leaves=31, class_weight=None):
    model = LGBMClassifier(
        objective="multiclass", n_estimators=180, learning_rate=.045,
        num_leaves=leaves, min_child_samples=90, reg_lambda=10.0,
        colsample_bytree=.85, subsample=.9, subsample_freq=1,
        max_bin=63, verbosity=-1, n_jobs=4, random_state=seed,
        class_weight=class_weight,
    )
    model.fit(X, y)
    return model


def _combined(px, py):
    # x-band is the first digit in x0y0...x3y2; y-band is the second.
    return (px[:, :, None] * py[:, None, :]).reshape(len(px), 12)


def train_one(train, val, test, hi, out_dir, seed):
    X, Xv, Xt = global_features(train), global_features(val), global_features(test)
    y = train["region_y"][:, hi].astype(np.int64)
    yv = val["region_y"][:, hi].astype(np.int64)
    yt = test["region_y"][:, hi].astype(np.int64)
    mt, mv, mm = y >= 0, yv >= 0, yt >= 0
    # Inverse-square-root weights preserve central-cell precision while giving
    # rare cells a learnable gradient instead of letting them disappear.
    counts_x = np.bincount((y[mt] // 3), minlength=4).astype(np.float64)
    counts_y = np.bincount((y[mt] % 3), minlength=3).astype(np.float64)
    wx = {i: float(np.sqrt(counts_x.sum() / max(1.0, counts_x[i]))) for i in range(4)}
    wy = {i: float(np.sqrt(counts_y.sum() / max(1.0, counts_y[i]))) for i in range(3)}
    mx = _fit(X[mt], y[mt] // 3, seed, class_weight=wx)
    my = _fit(X[mt], y[mt] % 3, seed + 1, class_weight=wy)
    pval = _combined(mx.predict_proba(Xv), my.predict_proba(Xv))
    ptest = _combined(mx.predict_proba(Xt), my.predict_proba(Xt))
    base_v, base_t = val["base_region"].astype(np.int64), test["base_region"].astype(np.int64)
    persistence_v = metrics(yv[mv], base_v[mv], val["sides"][mv])
    candidates = []
    for alpha in np.linspace(0.0, 1.0, 21):
        prior_v = np.zeros_like(pval)
        valid_base = (base_v >= 0) & (base_v < 12)
        prior_v[valid_base] = .02 / 11.0
        prior_v[np.arange(len(base_v))[valid_base], base_v[valid_base]] = .98
        probs = alpha * pval + (1.0 - alpha) * prior_v
        row = metrics(yv[mv], probs[mv].argmax(axis=1), val["sides"][mv])
        row["alpha"] = float(alpha)
        candidates.append(row)
    pareto = [r for r in candidates if r["accuracy"] >= persistence_v["accuracy"]
              and r["macro_f1"] >= persistence_v["macro_f1"]]
    chosen = max(pareto or candidates,
                 key=lambda r: (r["score"], r["macro_f1"], r["accuracy"], -r["alpha"]))
    prior_t = np.zeros_like(ptest)
    valid_base_t = (base_t >= 0) & (base_t < 12)
    prior_t[valid_base_t] = .02 / 11.0
    prior_t[np.arange(len(base_t))[valid_base_t], base_t[valid_base_t]] = .98
    candidate_t = chosen["alpha"] * ptest + (1.0 - chosen["alpha"]) * prior_t
    test_report = {
        "candidate": metrics(yt[mm], candidate_t[mm].argmax(axis=1), test["sides"][mm]),
        "persistence": metrics(yt[mm], base_t[mm], test["sides"][mm]),
        "raw": metrics(yt[mm], ptest[mm].argmax(axis=1), test["sides"][mm]),
    }
    gate = (test_report["candidate"]["accuracy"] > test_report["persistence"]["accuracy"]
            and test_report["candidate"]["macro_f1"] > test_report["persistence"]["macro_f1"])
    out_dir.mkdir(parents=True, exist_ok=True)
    mx.booster_.save_model(str(out_dir / f"enemy_region_x_{HORIZONS[hi]}s.txt"))
    my.booster_.save_model(str(out_dir / f"enemy_region_y_{HORIZONS[hi]}s.txt"))
    return {
        "horizon_seconds": HORIZONS[hi], "validation": {"persistence": persistence_v,
            "chosen": chosen, "pareto_count": len(pareto)},
        "test": test_report, "gate": bool(gate),
        "models": {"x": f"enemy_region_x_{HORIZONS[hi]}s.txt",
                    "y": f"enemy_region_y_{HORIZONS[hi]}s.txt"},
        "class_weight": {"x": wx, "y": wy},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="out_m2/temporal_forecast_canonical_side")
    ap.add_argument("--output", default="out_m2/enemy_region_hierarchical/report.json")
    args = ap.parse_args()
    root, out = Path(args.root), Path(args.output)
    tr, va, te = load(root, "train"), load(root, "validation"), load(root, "test")
    rows = [train_one(tr, va, te, hi, out.parent / "models", 20260928 + hi * 101)
            for hi in range(4)]
    manifest = {
        "schema": "enemy-region-hierarchical-manifest-v1",
        "feature_schema": "canonical-side-task-causal-v2",
        "tasks": {str(row["horizon_seconds"]): {
            "x_model": row["models"]["x"], "y_model": row["models"]["y"],
            "alpha": row["validation"]["chosen"]["alpha"],
            "enabled": bool(row["gate"]),
        } for row in rows},
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"schema": "enemy-region-hierarchical-v1", "rows": rows},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    (out.parent / "production_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
