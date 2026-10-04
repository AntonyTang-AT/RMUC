"""Game-split XGBoost probe for 30-second global-mode forecasting.

Uses only causal per-frame observable statistics and their trailing changes.
It is an isolated candidate: production forecasts are not modified. Hyper-
parameters and the persistence blend are selected on validation; the locked
test split is evaluated only if the validation gate is passed.
"""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from forecast.temporal import MODES
from forecast.mode_change_xgb import build_mode_change_features


def load_split(root: Path, split: str):
    npz_path = root / f"{split}.npz"
    with np.load(npz_path, allow_pickle=False) as a:
        y = np.array(a["mode_y"][:, 1], dtype=np.int64)
        current = np.array(a["base_mode"], dtype=np.int64)
        sides = np.array(a["sides"])
        games = np.array(a["game_ids"], dtype=np.int64)
        issued = np.array(a["issued_at"], dtype=np.float64)
    n = len(y)
    base = np.zeros((n, 20), dtype=np.float32)
    groups: dict[tuple[int, str], list[tuple[int, float]]] = {}
    sidecar = root / f"{split}_observable_summary.jsonl.gz"
    with gzip.open(sidecar, "rt", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            i = int(row["row_index"])
            if not 0 <= i < n or int(row["game_id"]) != int(games[i]):
                raise ValueError(f"row alignment mismatch at {split}:{i}")
            if str(row["side"]) != str(sides[i]) or abs(float(row["issued_at"]) - issued[i]) > 1e-4:
                raise ValueError(f"side/time alignment mismatch at {split}:{i}")
            label = row.get("current_global_mode")
            if label not in MODES or int(current[i]) != MODES.index(label):
                raise ValueError(f"current-mode parity mismatch at {split}:{i}")
            base[i] = np.asarray(row["feature_vector"], dtype=np.float32)
            groups.setdefault((int(games[i]), str(sides[i])), []).append((i, float(issued[i])))

    # Shared causal transformation is also available to a future online
    # caller. Targets are never passed into the feature builder.
    group_ids = np.empty(n, dtype=np.int64)
    for group_id, entries in enumerate(groups.values()):
        for i, _ in entries:
            group_ids[i] = group_id
    x = build_mode_change_features(base, issued, current, group_ids)
    return x, y, current, sides, games


def _macro_f1(y, pred):
    return float(f1_score(y, pred, labels=np.arange(len(MODES)), average="macro", zero_division=0))


def _side_metrics(y, pred, sides):
    valid = y >= 0
    output = {}
    for side in ("红", "蓝"):
        mask = valid & (sides == side)
        output[side] = {
            "accuracy": float(accuracy_score(y[mask], pred[mask])),
            "macro_f1": _macro_f1(y[mask], pred[mask]),
            "n": int(mask.sum()),
        }
    output["balanced_accuracy"] = float(np.mean([output[s]["accuracy"] for s in ("红", "蓝")]))
    output["balanced_macro_f1"] = float(np.mean([output[s]["macro_f1"] for s in ("红", "蓝")]))
    return output


def run(args):
    train_root, val_root = Path(args.train_root), Path(args.validation_root)
    x_train, y_train, *_ = load_split(train_root, "train")
    x_val, y_val, base_val, side_val, game_val = load_split(val_root, "validation")
    train_valid, val_valid = y_train >= 0, y_val >= 0
    if not {"红", "蓝"}.issubset(set(map(str, side_val))):
        raise ValueError("validation must contain red and blue observing sides")
    from xgboost import XGBClassifier
    model = XGBClassifier(
        objective="multi:softprob", num_class=len(MODES), eval_metric="mlogloss",
        n_estimators=500, max_depth=4, learning_rate=0.03,
        min_child_weight=10, subsample=0.8, colsample_bytree=0.9,
        reg_lambda=5.0, random_state=17, n_jobs=8,
        tree_method="hist", device=args.device,
        early_stopping_rounds=40,
    )
    model.fit(x_train[train_valid], y_train[train_valid],
              eval_set=[(x_val[val_valid], y_val[val_valid])], verbose=False)
    p = model.predict_proba(x_val)
    candidates = []
    for alpha in np.linspace(0.0, 1.0, 21):
        mixed = p.copy()
        for i in np.flatnonzero(val_valid):
            cur = int(base_val[i])
            if 0 <= cur < len(MODES):
                prior = np.full(len(MODES), 0.02 / (len(MODES) - 1), dtype=np.float32)
                prior[cur] = 0.98
                mixed[i] = alpha * p[i] + (1.0 - alpha) * prior
        pred = mixed.argmax(axis=1)
        metrics = _side_metrics(y_val, pred, side_val)
        metrics["alpha"] = float(alpha)
        metrics["selection_score"] = 0.6 * metrics["balanced_accuracy"] + 0.4 * metrics["balanced_macro_f1"]
        candidates.append(metrics)
    chosen = max(candidates, key=lambda row: (row["selection_score"], row["balanced_accuracy"]))
    persistence = _side_metrics(y_val, base_val, side_val)
    raw = max(candidates, key=lambda row: row["alpha"])
    current_fusion_validation_accuracy = 0.6059  # game-group OOF, locked alpha=.6
    gate_pass = (chosen["balanced_accuracy"] >= 0.615 and
                 chosen["balanced_accuracy"] - current_fusion_validation_accuracy >= 0.005)
    report = {
        "model": "xgboost_multiclass_mode_30s",
        "device": args.device,
        "features": int(x_train.shape[1]),
        "n_train": int(train_valid.sum()),
        "n_validation": int(val_valid.sum()),
        "n_validation_games": int(len(np.unique(game_val))),
        "best_iteration": int(model.best_iteration),
        "best_validation_mlogloss": float(model.best_score),
        "validation_persistence": persistence,
        "validation_raw_alpha_1": raw,
        "validation_selected_fusion": chosen,
        "alpha_sweep": candidates,
        "promotion_gate": {
            "required_validation_balanced_accuracy": 0.615,
            "current_fusion_validation_balanced_accuracy": current_fusion_validation_accuracy,
            "required_improvement_over_current_fusion": 0.005,
            "test_accessed": False,
            "decision": "eligible_for_locked_test_once" if gate_pass else "reject_before_test",
        },
    }
    if gate_pass and args.test_root:
        # The test split is opened only after all model/alpha choices are
        # frozen on validation. Save the holdout result and its paired-game
        # uncertainty; do not use it to change the model or alpha.
        x_test, y_test, base_test, side_test, game_test = load_split(
            Path(args.test_root), "test")
        p_test = model.predict_proba(x_test)
        alpha = float(chosen["alpha"])
        fused = p_test.copy()
        valid_test = y_test >= 0
        for i in np.flatnonzero(valid_test):
            cur = int(base_test[i])
            if 0 <= cur < len(MODES):
                prior = np.full(len(MODES), 0.02 / (len(MODES) - 1), dtype=np.float32)
                prior[cur] = 0.98
                fused[i] = alpha * p_test[i] + (1.0 - alpha) * prior
        test_metrics = _side_metrics(y_test, fused.argmax(axis=1), side_test)
        test_persistence = _side_metrics(y_test, base_test, side_test)
        game_ids = np.unique(game_test[valid_test])
        correct = {}
        for game in game_ids:
            correct[int(game)] = {}
            gm = valid_test & (game_test == game)
            for side in ("红", "蓝"):
                sm = gm & (side_test == side)
                if sm.any():
                    correct[int(game)][side] = (
                        int(np.sum(fused.argmax(axis=1)[sm] == y_test[sm])), int(sm.sum()),
                        int(np.sum(base_test[sm] == y_test[sm])),
                    )
        rng = np.random.default_rng(20260927)
        ids = np.asarray(list(correct), dtype=np.int64)
        bootstrap = []
        for _ in range(2000):
            draw = rng.choice(ids, size=len(ids), replace=True)
            candidate_sides, baseline_sides = [], []
            for side in ("红", "蓝"):
                numer = sum(correct[int(game)].get(side, (0, 0, 0))[0] for game in draw)
                denom = sum(correct[int(game)].get(side, (0, 0, 0))[1] for game in draw)
                base_numer = sum(correct[int(game)].get(side, (0, 0, 0))[2] for game in draw)
                if denom:
                    candidate_sides.append(numer / denom)
                    baseline_sides.append(base_numer / denom)
            if len(candidate_sides) == 2:
                bootstrap.append((float(np.mean(candidate_sides)),
                                  float(np.mean(candidate_sides) - np.mean(baseline_sides))))
        bs = np.asarray(bootstrap, dtype=np.float64)
        report["locked_test_once"] = {
            "alpha_frozen_from_validation": alpha,
            "n_games": int(len(ids)),
            "metrics": test_metrics,
            "persistence": test_persistence,
            "balanced_accuracy_95ci": np.quantile(bs[:, 0], [0.025, 0.975]).tolist(),
            "paired_delta_vs_persistence_95ci": np.quantile(bs[:, 1], [0.025, 0.975]).tolist(),
            "bootstrap_replicates": int(len(bs)),
            "test_used_for_tuning": False,
        }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    model.save_model(str(out.with_name(out.stem + ".model.ubj")))
    print(json.dumps({k: report[k] for k in ("device", "features", "best_iteration",
                                           "best_validation_mlogloss", "validation_persistence",
                                           "validation_selected_fusion", "promotion_gate")},
                     ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-root", default="out_m2/temporal_hybrid_mode_stats_ablation")
    parser.add_argument("--validation-root", default="out_m2/temporal_hybrid_mode_stats_ablation")
    parser.add_argument("--test-root", default="out_m2/temporal_hybrid_mode_stats_ablation")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", default="out_m2/mode_change_xgb_probe/report.json")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
