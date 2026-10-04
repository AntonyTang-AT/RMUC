"""Train-only five-fold GroupKFold cumulative-ordinal LightGBM experiment.

Ordered labels are defense < balanced < attack. Two binary threshold heads
estimate P(Y >= balanced) and P(Y >= attack); probabilities are projected to
the monotone constraint before recovering the three class probabilities.
Only death-aware train NPZ/JSONL and saved train OOF baseline predictions are
read. No validation/test data are accepted by this CLI.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import log_loss

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from eval.experiment_death_aware_zone_mass_oof import extract_train_aggregates
from eval.experiment_death_aware_mode_lightgbm import features as direct_features
from forecast.temporal import HORIZONS, load_compact


def classifier(seed: int, threads: int) -> LGBMClassifier:
    # Fixed compact configuration; no parameter search or validation selection.
    return LGBMClassifier(
        objective="binary", n_estimators=140, learning_rate=.045,
        num_leaves=15, max_depth=-1, min_child_samples=90, reg_lambda=4.,
        colsample_bytree=.8, subsample=.85, subsample_freq=1,
        verbosity=-1, n_jobs=threads, random_state=seed,
    )


def project_cumulative(p_ge_balanced: np.ndarray, p_ge_attack: np.ndarray):
    """Euclidean projection of two cumulative probabilities onto p2 <= p1."""
    p1 = np.asarray(p_ge_balanced, dtype=np.float64).copy()
    p2 = np.asarray(p_ge_attack, dtype=np.float64).copy()
    violate = p2 > p1
    midpoint = .5 * (p1[violate] + p2[violate])
    p1[violate] = midpoint
    p2[violate] = midpoint
    probs = np.column_stack((1.0 - p1, p1 - p2, p2))
    probs = np.clip(probs, 0.0, 1.0)
    probs /= np.maximum(probs.sum(axis=1, keepdims=True), 1e-12)
    return probs, int(violate.sum())


def row_metrics(y, p):
    y = np.asarray(y, dtype=np.int64)
    p = np.asarray(p, dtype=np.float64).copy()
    p /= np.maximum(p.sum(axis=1, keepdims=True), 1e-12)
    pred = p.argmax(axis=1)
    recalls, f1s = [], []
    cm = np.zeros((3, 3), dtype=np.int64)
    np.add.at(cm, (y, pred), 1)
    for c in range(3):
        tp = int(cm[c, c]); fp = int(cm[:, c].sum() - tp); fn = int(cm[c, :].sum() - tp)
        recalls.append(tp / max(1, int(cm[c, :].sum())))
        f1s.append(2 * tp / max(1, 2 * tp + fp + fn))
    onehot = np.eye(3, dtype=np.float64)[y]
    return {
        "n": int(len(y)), "accuracy": float(np.mean(pred == y)),
        "macro_f1": float(np.mean(f1s)), "balanced_accuracy": float(np.mean(recalls)),
        "brier": float(np.mean(np.sum((p - onehot) ** 2, axis=1))),
        "log_loss": float(log_loss(y, p, labels=[0, 1, 2])),
        "class_recall_defense_balanced_attack": [float(x) for x in recalls],
        "confusion_true_rows_pred_cols": cm.tolist(),
    }


def by_game(y, p, gids, sides, valid):
    out = {}
    for gid in np.unique(gids[valid]):
        side_values = []
        for side in ("红", "蓝"):
            mask = valid & (gids == gid) & (sides == side)
            if mask.any():
                side_values.append(row_metrics(y[mask], p[mask]))
        if len(side_values) == 2:
            out[int(gid)] = {k: float(np.mean([x[k] for x in side_values])) for k in
                             ("accuracy", "macro_f1", "balanced_accuracy", "brier", "log_loss")}
    return out


def paired_game_bootstrap(candidate, baseline, seed=0, reps=2000):
    ids = sorted(set(candidate) & set(baseline))
    rng = np.random.default_rng(seed)
    ix = rng.integers(0, len(ids), size=(reps, len(ids)))
    result = {"paired_games": len(ids), "replicates": reps, "delta_candidate_minus_multiclass": {}}
    for metric in ("accuracy", "macro_f1", "balanced_accuracy", "brier", "log_loss"):
        delta = np.asarray([candidate[g][metric] - baseline[g][metric] for g in ids])
        boot = delta[ix].mean(axis=1)
        result["delta_candidate_minus_multiclass"][metric] = {
            "mean": float(delta.mean()),
            "ci95": [float(np.quantile(boot, .025)), float(np.quantile(boot, .975))],
        }
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-npz", required=True)
    ap.add_argument("--train-jsonl", required=True)
    ap.add_argument("--baseline-oof", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--seed", type=int, default=270927)
    ap.add_argument("--bootstrap-reps", type=int, default=2000)
    a = ap.parse_args()
    started = time.perf_counter()

    train, extra, feature_audit = extract_train_aggregates(a.train_npz, a.train_jsonl)
    baseline = np.load(a.baseline_oof, allow_pickle=False)
    for key in ("game_ids", "sides", "issued_at", "mode_y", "base_mode"):
        if not np.array_equal(np.asarray(train[key]), np.asarray(baseline[key])):
            raise ValueError(f"baseline OOF/train alignment failure for {key}")
    X = np.concatenate((direct_features(train), extra), axis=1).astype(np.float32, copy=False)
    yall = np.asarray(train["mode_y"], dtype=np.int64)
    gids = np.asarray(train["game_ids"], dtype=np.int64)
    sides = np.asarray(train["sides"]).astype(str)
    issued = np.asarray(train["issued_at"], dtype=np.float64)
    base_probs = np.asarray(baseline["aggregate_candidate"], dtype=np.float64)
    baseline_folds = np.asarray(baseline["fold_ids"], dtype=np.int64)
    if not np.array_equal(np.asarray(train["mode_y"]), np.asarray(baseline["mode_y"])):
        raise ValueError("OOF baseline labels differ")
    for gid in np.unique(gids):
        if len(set(baseline_folds[gids == gid].tolist())) != 1:
            raise ValueError("red and blue baseline OOF rows are not in the same whole-game fold")
    if feature_audit["rows_aligned"] != len(gids) or feature_audit["current_mode_parity"] != "exact":
        raise ValueError("causal feature extraction audit failed")

    cv = GroupKFold(n_splits=a.folds)
    folds = np.full(len(gids), -1, dtype=np.int8)
    candidate = np.full((len(gids), len(HORIZONS), 3), np.nan, dtype=np.float32)
    threshold_violation_count = np.zeros(len(HORIZONS), dtype=np.int64)
    fold_rows = []
    all_rows = np.arange(len(gids))
    for fold, (trloc, teloc) in enumerate(cv.split(X, groups=gids)):
        tr, held = all_rows[trloc], all_rows[teloc]
        folds[held] = fold
        horinfo = {}
        for j, h in enumerate(HORIZONS):
            tr_ok = yall[tr, j] >= 0
            te_ok = yall[held, j] >= 0
            trr, herr = tr[tr_ok], held[te_ok]
            yy = yall[trr, j]
            if set(np.unique(yy).tolist()) != {0, 1, 2}:
                raise ValueError(f"training fold {fold} lacks an ordinal class at {h}s")
            threshold_prob = []
            for k, threshold in enumerate((1, 2)):
                target = (yy >= threshold).astype(np.int8)
                model = classifier(a.seed + fold * 17 + j * 3 + k, a.threads)
                model.fit(X[trr], target)
                classes = list(model.classes_)
                if classes != [0, 1]:
                    raise ValueError(f"threshold model missing a binary class: {classes}")
                threshold_prob.append(model.predict_proba(X[herr])[:, 1])
            pp, nviol = project_cumulative(threshold_prob[0], threshold_prob[1])
            candidate[herr, j] = pp.astype(np.float32)
            threshold_violation_count[j] += nviol
            horinfo[f"{h}s"] = {"train_rows": int(len(trr)), "heldout_rows": int(len(herr)), "raw_order_violations": int(nviol)}
        fold_rows.append({"fold": fold, "train_games": int(len(np.unique(gids[tr]))),
                          "heldout_games": int(len(np.unique(gids[held]))), "horizons": horinfo})
        print(json.dumps({"fold": fold, "heldout_games": fold_rows[-1]["heldout_games"], "horizons": horinfo}, ensure_ascii=False), flush=True)

    if not np.array_equal(folds.astype(np.int64), baseline_folds):
        raise ValueError("candidate GroupKFold assignments differ from multiclass OOF baseline")

    results = {}
    for j, h in enumerate(HORIZONS):
        y = yall[:, j]
        valid = (y >= 0) & np.isfinite(candidate[:, j]).all(axis=1) & np.isfinite(base_probs[:, j]).all(axis=1)
        pred = candidate[:, j]
        bp = base_probs[:, j]
        cand_game = by_game(y, pred, gids, sides, valid)
        base_game = by_game(y, bp, gids, sides, valid)
        pooled_candidate = row_metrics(y[valid], pred[valid])
        pooled_baseline = row_metrics(y[valid], bp[valid])
        game_metrics = ("accuracy", "macro_f1", "balanced_accuracy", "brier", "log_loss")
        results[f"{h}s"] = {
            "valid_rows": int(valid.sum()),
            "valid_games_paired": len(set(cand_game) & set(base_game)),
            "candidate_pooled": pooled_candidate,
            "baseline_pooled": pooled_baseline,
            "candidate_game_equal_red_blue_equal": {m: float(np.mean([v[m] for v in cand_game.values()])) for m in game_metrics},
            "baseline_game_equal_red_blue_equal": {m: float(np.mean([v[m] for v in base_game.values()])) for m in game_metrics},
            "paired_game_bootstrap_delta": paired_game_bootstrap(cand_game, base_game, a.seed + j * 101, a.bootstrap_reps),
            "monotonicity_projected_rows": int(threshold_violation_count[j]),
        }
    report = {
        "schema": "death-aware-cumulative-ordinal-fivefold-group-oof-v1",
        "access": {"opened_splits": ["train"], "validation_access": False, "test_access": False},
        "train_npz": str(Path(a.train_npz).resolve()), "train_jsonl": str(Path(a.train_jsonl).resolve()),
        "baseline_oof": str(Path(a.baseline_oof).resolve()),
        "n_rows": int(len(gids)), "n_games": int(len(np.unique(gids))), "fold_count": a.folds,
        "label_order": ["防守", "均衡", "进攻"],
        "candidate": "two cumulative binary LightGBM heads: P(Y>=均衡), P(Y>=进攻); Euclidean projection to P(Y>=进攻)<=P(Y>=均衡); recover (1-p1,p1-p2,p2)",
        "candidate_features": "same causal direct temporal + current observable zone/score aggregate features as the multiclass baseline",
        "candidate_model_fixed_parameters": {"n_estimators": 140, "learning_rate": .045, "num_leaves": 15, "min_child_samples": 90, "reg_lambda": 4., "colsample_bytree": .8, "subsample": .85},
        "fold_contract": "GroupKFold by full game_id, red and blue in same outer fold; each binary head trained only on outer-train fold labels; same row/fold eligibility as saved zone+observable-score multiclass baseline",
        "feature_audit": feature_audit,
        "folds": fold_rows, "horizons": results,
        "runtime_seconds": round(time.perf_counter() - started, 3),
    }
    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    np.savez_compressed(out / "ordinal_oof_probabilities.npz", game_ids=gids, sides=sides, issued_at=issued,
                        mode_y=yall, fold_ids=folds, ordinal_candidate=candidate.astype(np.float32),
                        multiclass_baseline=base_probs.astype(np.float32), monotonicity_projected_rows=threshold_violation_count)
    print(json.dumps({h: {"candidate": v["candidate_game_equal_red_blue_equal"],
                          "baseline": v["baseline_game_equal_red_blue_equal"],
                          "delta_ci": v["paired_game_bootstrap_delta"]}
                      for h, v in results.items()}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
