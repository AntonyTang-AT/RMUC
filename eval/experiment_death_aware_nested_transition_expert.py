"""Nested train-only OOF experiment for multiclass mode-transition experts.

The transition expert predicts one of the nine current_mode x future_mode
classes and marginalizes its joint probabilities into a future-mode vector.
Its mixture weight with a persistence-aware multiclass state expert is selected
only from inner whole-game GroupKFold predictions inside each outer training
fold. Outer predictions are evaluated once. The saved zone/observable-score
multiclass OOF prediction is the outer comparator.
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
from forecast.temporal import HORIZONS


WEIGHT_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)


def make_model(classes: int, seed: int, threads: int, n_estimators: int):
    return LGBMClassifier(
        objective="multiclass", num_class=classes, n_estimators=n_estimators,
        learning_rate=.05, num_leaves=11, max_depth=-1, min_child_samples=100,
        reg_lambda=5., colsample_bytree=.85, subsample=.85, subsample_freq=1,
        verbosity=-1, n_jobs=threads, random_state=seed,
    )


def class_probs(model, X, nclasses):
    raw = model.predict_proba(X)
    out = np.zeros((len(X), nclasses), dtype=np.float64)
    for j, cls in enumerate(model.classes_):
        out[:, int(cls)] = raw[:, j]
    return out


def transition_to_future(joint):
    """Map joint class id current*3+future into a 3-way future distribution."""
    out = np.zeros((len(joint), 3), dtype=np.float64)
    for current in range(3):
        for future in range(3):
            out[:, future] += joint[:, current * 3 + future]
    out /= np.maximum(out.sum(axis=1, keepdims=True), 1e-12)
    return out


def row_metrics(y, p):
    y = np.asarray(y, dtype=np.int64)
    p = np.asarray(p, dtype=np.float64).copy()
    p /= np.maximum(p.sum(axis=1, keepdims=True), 1e-12)
    pred = p.argmax(axis=1)
    cm = np.zeros((3, 3), dtype=np.int64)
    np.add.at(cm, (y, pred), 1)
    recalls, f1s = [], []
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
        "class_recall_defense_balanced_attack": [float(r) for r in recalls],
    }


def game_scores(y, p, gids, sides, valid):
    out = {}
    for gid in np.unique(gids[valid]):
        side_rows = []
        for side in ("红", "蓝"):
            m = valid & (gids == gid) & (sides == side)
            if m.any():
                side_rows.append(row_metrics(y[m], p[m]))
        if len(side_rows) == 2:
            out[int(gid)] = {k: float(np.mean([x[k] for x in side_rows])) for k in
                             ("accuracy", "macro_f1", "balanced_accuracy", "brier", "log_loss")}
    return out


def game_side_scores(y, p, gids, sides, valid):
    out = {"红": {}, "蓝": {}}
    for side in ("红", "蓝"):
        for gid in np.unique(gids[valid & (sides == side)]):
            m = valid & (gids == gid) & (sides == side)
            if m.any():
                out[side][int(gid)] = row_metrics(y[m], p[m])
    metrics = ("accuracy", "macro_f1", "balanced_accuracy", "brier", "log_loss")
    return {side: {"games": len(rows), "metrics_game_equal": {
        metric: float(np.mean([score[metric] for score in rows.values()])) for metric in metrics
    }} for side, rows in out.items()}


def paired_bootstrap(candidate, baseline, seed=0, reps=2000):
    ids = sorted(set(candidate) & set(baseline))
    rng = np.random.default_rng(seed)
    ix = rng.integers(0, len(ids), size=(reps, len(ids)))
    out = {"paired_games": len(ids), "replicates": reps, "delta_transition_mixture_minus_baseline": {}}
    for metric in ("accuracy", "macro_f1", "balanced_accuracy", "brier", "log_loss"):
        diff = np.asarray([candidate[g][metric] - baseline[g][metric] for g in ids])
        boot = diff[ix].mean(axis=1)
        out["delta_transition_mixture_minus_baseline"][metric] = {
            "mean": float(diff.mean()), "ci95": [float(np.quantile(boot, .025)), float(np.quantile(boot, .975))]}
    return out


def select_weight(y, state_p, transition_p, gids, sides, valid):
    best = None
    diagnostics = {}
    for weight in WEIGHT_GRID:
        mix = (1.0 - weight) * state_p + weight * transition_p
        gs = game_scores(y, mix, gids, sides, valid)
        acc = float(np.mean([v["accuracy"] for v in gs.values()]))
        f1 = float(np.mean([v["macro_f1"] for v in gs.values()]))
        diagnostics[str(weight)] = {"game_equal_accuracy": acc, "game_equal_macro_f1": f1}
        key = (acc, f1, -weight)
        if best is None or key > best[0]:
            best = (key, float(weight))
    return best[1], diagnostics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-npz", required=True)
    ap.add_argument("--train-jsonl", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--inner-folds", type=int, default=2)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--n-estimators", type=int, default=100)
    ap.add_argument("--seed", type=int, default=270928)
    ap.add_argument("--bootstrap-reps", type=int, default=2000)
    ap.add_argument("--horizons", default="10,30,60,120", help="comma separated horizons in seconds")
    a = ap.parse_args()
    start = time.perf_counter()
    requested = [int(x) for x in a.horizons.split(",")]
    horizon_indices = [list(HORIZONS).index(h) for h in requested]

    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    cache_path = out / "causal_train_features.npz"
    audit_path = out / "feature_audit.json"
    if cache_path.exists() and audit_path.exists():
        cache = np.load(cache_path, allow_pickle=False)
        train = {k: cache[k] for k in ("x", "mode_y", "base_mode", "game_ids", "sides", "issued_at")}
        extra = cache["extra_features"]
        feature_audit = json.loads(audit_path.read_text(encoding="utf-8"))
    else:
        train, extra, feature_audit = extract_train_aggregates(a.train_npz, a.train_jsonl)
        np.savez_compressed(cache_path, x=train["x"], mode_y=train["mode_y"], base_mode=train["base_mode"],
                            game_ids=train["game_ids"], sides=train["sides"], issued_at=train["issued_at"],
                            extra_features=extra)
        audit_path.write_text(json.dumps(feature_audit, ensure_ascii=False, indent=2), encoding="utf-8")
    X = np.concatenate((direct_features(train), extra), axis=1).astype(np.float32, copy=False)
    yall = np.asarray(train["mode_y"], dtype=np.int64)
    base_mode = np.asarray(train["base_mode"], dtype=np.int64)
    gids = np.asarray(train["game_ids"], dtype=np.int64)
    sides = np.asarray(train["sides"]).astype(str)
    base_oof = np.full((len(gids), len(HORIZONS), 3), np.nan, dtype=np.float32)
    if feature_audit["rows_aligned"] != len(gids) or feature_audit["current_mode_parity"] != "exact":
        raise ValueError("feature extraction parity audit failed")

    n = len(gids)
    candidate_oof = np.full((n, len(HORIZONS), 3), np.nan, dtype=np.float32)
    outer_fold_ids = np.full(n, -1, dtype=np.int8)
    horizon_reports = {f"{h}s": {"folds": []} for h in requested}
    outer_cv = GroupKFold(n_splits=a.folds)
    all_idx = np.arange(n)
    outer_num = 0
    for outer_fold, (otr_local, ote_local) in enumerate(outer_cv.split(X, groups=gids)):
        otr, ote = all_idx[otr_local], all_idx[ote_local]
        outer_fold_ids[ote] = outer_fold
        inner_cv = GroupKFold(n_splits=a.inner_folds)
        fold_report = {"outer_fold": outer_fold, "outer_train_games": int(np.unique(gids[otr]).size),
                       "outer_heldout_games": int(np.unique(gids[ote]).size), "horizons": {}}
        for j in horizon_indices:
            h = list(HORIZONS)[j]
            outer_train_rows = otr[yall[otr, j] >= 0]
            outer_test_rows = ote[yall[ote, j] >= 0]
            inner_state = np.full((len(outer_train_rows), 3), np.nan, dtype=np.float64)
            inner_transition = np.full_like(inner_state, np.nan)
            inner_fold_ids = np.full(len(outer_train_rows), -1, dtype=np.int8)
            local_groups = gids[outer_train_rows]
            for inner_fold, (itr_local, ite_local) in enumerate(inner_cv.split(outer_train_rows, groups=local_groups)):
                itr = outer_train_rows[itr_local]
                ite = outer_train_rows[ite_local]
                inner_fold_ids[ite_local] = inner_fold
                ytr = yall[itr, j]
                state = make_model(3, a.seed + outer_fold * 10000 + j * 100 + inner_fold * 2, a.threads, a.n_estimators)
                state.fit(X[itr], ytr)
                inner_state[ite_local] = class_probs(state, X[ite], 3)

                transition_target = base_mode[itr] * 3 + ytr
                transition = make_model(9, a.seed + outer_fold * 10000 + j * 100 + inner_fold * 2 + 1, a.threads, a.n_estimators)
                transition.fit(X[itr], transition_target)
                joint = class_probs(transition, X[ite], 9)
                inner_transition[ite_local] = transition_to_future(joint)
            inner_valid = np.isfinite(inner_state).all(axis=1) & np.isfinite(inner_transition).all(axis=1)
            if not np.all(inner_valid) or np.any(inner_fold_ids < 0):
                raise ValueError("inner OOF predictions did not cover every outer-train eligible row")
            weight, weight_scores = select_weight(
                yall[outer_train_rows, j], inner_state, inner_transition,
                gids[outer_train_rows], sides[outer_train_rows], inner_valid)

            ytr = yall[outer_train_rows, j]
            transition_target = base_mode[outer_train_rows] * 3 + ytr
            outer_state = make_model(3, a.seed + outer_fold * 10000 + j * 100 + 998, a.threads, a.n_estimators)
            outer_state.fit(X[outer_train_rows], ytr)
            state_test = class_probs(outer_state, X[outer_test_rows], 3)
            base_oof[outer_test_rows, j] = state_test.astype(np.float32)

            outer_transition = make_model(9, a.seed + outer_fold * 10000 + j * 100 + 999, a.threads, a.n_estimators)
            outer_transition.fit(X[outer_train_rows], transition_target)
            transition_test = transition_to_future(class_probs(outer_transition, X[outer_test_rows], 9))
            candidate_oof[outer_test_rows, j] = ((1.0 - weight) * state_test + weight * transition_test).astype(np.float32)
            fold_report["horizons"][f"{h}s"] = {
                "inner_selected_transition_weight": weight,
                "inner_weight_scores": weight_scores,
                "inner_train_games": int(np.unique(gids[outer_train_rows]).size),
                "outer_heldout_rows": int(len(outer_test_rows)),
            }
        for h in requested:
            horizon_reports[f"{h}s"]["folds"].append(fold_report["horizons"][f"{h}s"] | {"outer_fold": outer_fold})
        outer_num += 1
        print(json.dumps(fold_report, ensure_ascii=False), flush=True)

    for j in horizon_indices:
        h = list(HORIZONS)[j]
        y = yall[:, j]
        valid = (y >= 0) & np.isfinite(candidate_oof[:, j]).all(axis=1) & np.isfinite(base_oof[:, j]).all(axis=1)
        cand = candidate_oof[:, j]
        base = base_oof[:, j]
        cg = game_scores(y, cand, gids, sides, valid)
        bg = game_scores(y, base, gids, sides, valid)
        keys = ("accuracy", "macro_f1", "balanced_accuracy", "brier", "log_loss")
        horizon_reports[f"{h}s"].update({
            "valid_rows": int(valid.sum()), "paired_games": len(set(cg) & set(bg)),
            "outer_candidate_pooled": row_metrics(y[valid], cand[valid]),
            "outer_baseline_pooled": row_metrics(y[valid], base[valid]),
            "outer_candidate_game_equal": {k: float(np.mean([v[k] for v in cg.values()])) for k in keys},
            "outer_baseline_game_equal": {k: float(np.mean([v[k] for v in bg.values()])) for k in keys},
            "outer_candidate_by_side": game_side_scores(y, cand, gids, sides, valid),
            "outer_baseline_by_side": game_side_scores(y, base, gids, sides, valid),
            "outer_paired_game_bootstrap": paired_bootstrap(cg, bg, a.seed + j * 701, a.bootstrap_reps),
        })

    report = {
        "schema": "death-aware-nested-multiclass-transition-expert-v1",
        "access": {"opened_splits": ["train"], "validation_access": False, "test_access": False},
        "n_rows": int(n), "n_games": int(np.unique(gids).size), "outer_folds": a.folds,
        "inner_folds": a.inner_folds, "requested_horizons_seconds": requested,
        "candidate_definition": "9-way current_mode*3+future_mode transition classification; marginalize joint probabilities over current mode to get future-mode distribution",
        "state_expert": "current-observable zone+score feature multiclass LightGBM trained in the same outer/inner folds as transition expert",
        "fusion": {"form": "(1-w)*state_expert + w*transition_expert", "weights": list(WEIGHT_GRID),
                   "selection": "maximize game-equal inner-OOF accuracy, then Macro-F1, then prefer lower transition weight; selected independently per outer fold and horizon"},
        "model_parameters": {"n_estimators": a.n_estimators, "learning_rate": .05, "num_leaves": 11,
                             "min_child_samples": 100, "reg_lambda": 5., "colsample_bytree": .85,
                             "subsample": .85},
        "feature_audit": feature_audit,
        "outer_fold_contract": "GroupKFold by full game_id; red and blue stay together; state and transition experts use identical splits; fusion weights only use inner OOF from outer-train games",
        "bootstrap": {"unit": "whole game; red/blue scores averaged within game", "replicates": a.bootstrap_reps},
        "horizons": horizon_reports,
        "runtime_seconds": round(time.perf_counter() - start, 3),
    }
    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    np.savez_compressed(out / "nested_transition_oof.npz", game_ids=gids, sides=sides,
                        issued_at=np.asarray(train["issued_at"]), mode_y=yall, fold_ids=outer_fold_ids,
                        transition_mixture=candidate_oof, multiclass_baseline=base_oof)
    print(json.dumps({f"{h}s": {"candidate": report["horizons"][f"{h}s"].get("outer_candidate_game_equal"),
                          "baseline": report["horizons"][f"{h}s"].get("outer_baseline_game_equal"),
                          "delta_ci": report["horizons"][f"{h}s"].get("outer_paired_game_bootstrap"),
                          "inner_weights": [x["inner_selected_transition_weight"] for x in report["horizons"][f"{h}s"]["folds"]]}
                      for h in requested}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
