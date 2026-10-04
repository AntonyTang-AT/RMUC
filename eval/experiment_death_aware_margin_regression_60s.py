"""Compare continuous 60s relative-margin regression to direct classification.

Strict scope: death-aware train/validation artifacts only. No test path or test
dataset is referenced. OOF predictions use GroupKFold by complete game_id.
"""
from __future__ import annotations

import argparse
import gzip
import json
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import (accuracy_score, f1_score, mean_absolute_error,
                             mean_squared_error, r2_score)
from sklearn.model_selection import GroupKFold
from lightgbm import LGBMClassifier, LGBMRegressor

ROOT = Path(__file__).resolve().parents[1]
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from eval.experiment_death_aware_mode_lightgbm import features, persistence_probs, scores, temp_scale, fit_temperature
from forecast.temporal import HORIZONS, MODES, MOBILE_TYPES, _entities, _mode_score, baseline_labels


CONTRACT = "relative_margin_v2_death_aware_hp_le_zero"
HORIZON = 60
HORIZON_INDEX = HORIZONS.index(HORIZON)


def _continuous_scores(npz_path: Path, jsonl_path: Path):
    """Read only one named train/validation JSONL and assert exact label parity."""
    with np.load(npz_path, allow_pickle=False) as archive:
        game_ids = archive["game_ids"].astype(np.int64)
        sides = archive["sides"].astype(str)
        issued = archive["issued_at"].astype(np.float64)
        base_mode = archive["base_mode"].astype(np.int64)
        labels = archive["mode_y"][:, HORIZON_INDEX].astype(np.int64)
        version = str(archive["global_mode_label_version"][0])
    if version != CONTRACT:
        raise ValueError(f"unexpected label contract {version!r} in {npz_path}")

    current_score = np.full(len(game_ids), np.nan, dtype=np.float32)
    future_score = np.full(len(game_ids), np.nan, dtype=np.float32)
    cursor = 0
    with gzip.open(jsonl_path, "rt", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            key = (int(row["game_id"]), str(row["side"]), round(float(row["issued_at"]), 5))
            expected = (int(game_ids[cursor]), str(sides[cursor]), round(float(issued[cursor]), 5))
            if key != expected:
                raise ValueError(f"row key mismatch at {cursor}: {key!r} != {expected!r}")
            current_groups = row.get("observable") or {}
            own_now = _entities(current_groups, "self_entities")
            opp_now = _entities(current_groups, "opponent_entities")
            current_score[cursor] = _mode_score(own_now, opp_now, sides[cursor])
            current_label = baseline_labels(row)["global_mode"]
            if MODES.index(current_label) != int(base_mode[cursor]):
                raise ValueError(f"death-aware current label mismatch at row {cursor}")

            target = (row.get("targets") or {}).get(str(HORIZON))
            if labels[cursor] >= 0:
                if not isinstance(target, dict) or target.get("censored"):
                    raise ValueError(f"NPZ has scoreable 60s label but JSONL is censored at row {cursor}")
                truth = target.get("truth") or {}
                entities = list((truth.get("entities") or {}).values())
                side = sides[cursor]
                def alive_for_score(e):
                    return (e.get("entity_type") not in MOBILE_TYPES
                            or not (isinstance(e.get("hp"), (int, float))
                                    and not isinstance(e.get("hp"), bool) and e.get("hp") <= 0))
                own_future = [e for e in entities if e.get("team") == side and alive_for_score(e)]
                opp_future = [e for e in entities if e.get("team") != side and alive_for_score(e)]
                future_score[cursor] = _mode_score(own_future, opp_future, side)
                predicted_label = 0 if future_score[cursor] >= 1 else (2 if future_score[cursor] <= -1 else 1)
                if predicted_label != int(labels[cursor]):
                    raise ValueError(f"death-aware 60s target label mismatch at row {cursor}")
            cursor += 1
    if cursor != len(game_ids):
        raise ValueError(f"JSONL rows {cursor} do not match NPZ rows {len(game_ids)}")
    return {
        "game_ids": game_ids, "sides": sides, "base_mode": base_mode,
        "labels": labels, "current_score": current_score,
        "future_score": future_score,
    }


def _classifier(seed, threads):
    return LGBMClassifier(objective="multiclass", num_class=3, n_estimators=260,
        learning_rate=.045, num_leaves=15, min_child_samples=90, reg_lambda=4.,
        colsample_bytree=.8, subsample=.85, subsample_freq=1, class_weight=None,
        verbosity=-1, n_jobs=threads, random_state=seed)


def _regressor(seed, threads):
    return LGBMRegressor(objective="regression_l2", n_estimators=260,
        learning_rate=.045, num_leaves=15, min_child_samples=90, reg_lambda=4.,
        colsample_bytree=.8, subsample=.85, subsample_freq=1,
        verbosity=-1, n_jobs=threads, random_state=seed)


def _score_to_class(score):
    return np.where(score >= 1.0, 0, np.where(score <= -1.0, 2, 1)).astype(np.int64)


def _hard_metrics(y, pred):
    per_class = {}
    f1s = []
    for i, label in enumerate(MODES):
        tp = int(np.sum((y == i) & (pred == i)))
        fp = int(np.sum((y != i) & (pred == i)))
        fn = int(np.sum((y == i) & (pred != i)))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {"precision": precision, "recall": recall, "f1": f1,
                            "support": int(np.sum(y == i))}
        f1s.append(f1)
    return {"n": int(len(y)), "accuracy": float(accuracy_score(y, pred)),
            "macro_f1": float(np.mean(f1s)), "per_class": per_class}


def _game_values(y, predictions, groups, sides, valid, probability_predictions=None,
                 current_score=None, true_delta=None, pred_delta=None):
    out = {}
    for gid in np.unique(groups[valid]):
        side_rows = []
        for side in ("红", "蓝"):
            m = valid & (groups == gid) & (sides == side)
            if not m.any():
                continue
            item = _hard_metrics(y[m], predictions[m])
            if probability_predictions is not None:
                item.update({k: scores(y[m], probability_predictions[m])[k]
                             for k in ("brier", "log_loss", "ece")})
            if true_delta is not None and pred_delta is not None:
                item["delta_mae"] = float(mean_absolute_error(true_delta[m], pred_delta[m]))
            side_rows.append(item)
        if len(side_rows) == 2:
            keys = ("accuracy", "macro_f1", "brier", "log_loss", "ece", "delta_mae")
            out[int(gid)] = {k: float(np.mean([v[k] for v in side_rows]))
                             for k in keys if k in side_rows[0] and k in side_rows[1]}
    return out


def _paired_bootstrap(a, b, seed=60, reps=2000):
    ids = sorted(set(a) & set(b))
    rng = np.random.default_rng(seed)
    result = {"n_games_red_blue_paired": len(ids), "replicates": reps, "seed": seed}
    if not ids:
        return result
    for metric in ("accuracy", "macro_f1", "brier", "log_loss", "ece", "delta_mae"):
        if not all(metric in a[g] and metric in b[g] for g in ids):
            continue
        delta = np.asarray([a[g][metric] - b[g][metric] for g in ids])
        draws = rng.integers(0, len(ids), size=(reps, len(ids)))
        boot = delta[draws].mean(axis=1)
        result[metric] = {"mean_delta_a_minus_b": float(delta.mean()),
                          "game_cluster_bootstrap_95_ci": [float(np.quantile(boot, .025)),
                                                           float(np.quantile(boot, .975))]}
    return result


def run(train_npz: Path, train_jsonl: Path, val_npz: Path, val_jsonl: Path,
        output: Path, folds=5, seed=60, threads=8, bootstrap=2000):
    started = time.perf_counter()
    with np.load(train_npz, allow_pickle=False) as a:
        train = {k: a[k].copy() for k in a.files}
    with np.load(val_npz, allow_pickle=False) as a:
        val = {k: a[k].copy() for k in a.files}
    if str(train["global_mode_label_version"][0]) != CONTRACT or str(val["global_mode_label_version"][0]) != CONTRACT:
        raise ValueError("train/validation artifacts do not share the death-aware label contract")

    train_targets = _continuous_scores(train_npz, train_jsonl)
    val_targets = _continuous_scores(val_npz, val_jsonl)
    X = features(train); Xv = features(val)
    train_y = train_targets["labels"]
    train_valid = (train_y >= 0) & np.isfinite(train_targets["current_score"]) & np.isfinite(train_targets["future_score"])
    train_delta = train_targets["future_score"] - train_targets["current_score"]
    val_y = val_targets["labels"]
    val_valid = (val_y >= 0) & np.isfinite(val_targets["current_score"]) & np.isfinite(val_targets["future_score"])
    val_delta = val_targets["future_score"] - val_targets["current_score"]
    groups = train_targets["game_ids"]; sides = train_targets["sides"]
    vgroups = val_targets["game_ids"]; vsides = val_targets["sides"]
    if train_valid.sum() < 100 or val_valid.sum() < 100:
        raise ValueError("insufficient scoreable 60s rows")

    oof_class_prob = np.full((len(train_y), 3), np.nan, dtype=np.float64)
    oof_reg_delta = np.full(len(train_y), np.nan, dtype=np.float64)
    cv = GroupKFold(n_splits=folds)
    valid_rows = np.flatnonzero(train_valid)
    for fold, (tr_idx, ho_idx) in enumerate(cv.split(X[train_valid], train_y[train_valid], groups[train_valid])):
        tr = valid_rows[tr_idx]; ho = valid_rows[ho_idx]
        cls = _classifier(seed + fold, threads)
        reg = _regressor(seed + fold, threads)
        cls.fit(X[tr], train_y[tr])
        reg.fit(X[tr], train_delta[tr])
        oof_class_prob[ho] = cls.predict_proba(X[ho])
        oof_reg_delta[ho] = reg.predict(X[ho])

    oof_valid = train_valid & np.isfinite(oof_class_prob[:, 0]) & np.isfinite(oof_reg_delta)
    oof_y = train_y[oof_valid]
    oof_base = persistence_probs(train_targets["base_mode"])[oof_valid]
    oof_class_p = oof_class_prob[oof_valid]
    oof_class_pred = oof_class_p.argmax(1)
    oof_reg_delta_v = oof_reg_delta[oof_valid]
    oof_reg_class = _score_to_class(train_targets["current_score"][oof_valid] + oof_reg_delta_v)
    oof_persistence = oof_base.argmax(1)
    oof_summary = {
        "rows": int(oof_valid.sum()), "games": int(len(np.unique(groups[oof_valid]))),
        "persistence": _hard_metrics(oof_y, oof_persistence),
        "direct_lightgbm_classifier_raw": _hard_metrics(oof_y, oof_class_pred),
        "direct_lightgbm_classifier_probability_quality": scores(oof_y, oof_class_p),
        "delta_regression_thresholded": _hard_metrics(oof_y, oof_reg_class),
        "delta_continuous": {
            "mae": float(mean_absolute_error(train_delta[oof_valid], oof_reg_delta_v)),
            "rmse": float(mean_squared_error(train_delta[oof_valid], oof_reg_delta_v) ** .5),
            "r2": float(r2_score(train_delta[oof_valid], oof_reg_delta_v)),
        },
    }
    game_oof_persist = _game_values(oof_y, oof_persistence, groups[oof_valid], sides[oof_valid],
                                    np.ones(oof_valid.sum(), bool), oof_base)
    game_oof_cls = _game_values(oof_y, oof_class_pred, groups[oof_valid], sides[oof_valid],
                                np.ones(oof_valid.sum(), bool), oof_class_p)
    game_oof_reg = _game_values(oof_y, oof_reg_class, groups[oof_valid], sides[oof_valid],
                                np.ones(oof_valid.sum(), bool), true_delta=train_delta[oof_valid],
                                pred_delta=oof_reg_delta_v)
    oof_ci = {
        "classifier_minus_persistence": _paired_bootstrap(game_oof_cls, game_oof_persist, seed, bootstrap),
        "regression_minus_persistence": _paired_bootstrap(game_oof_reg, game_oof_persist, seed + 1, bootstrap),
        "regression_minus_classifier": _paired_bootstrap(game_oof_reg, game_oof_cls, seed + 2, bootstrap),
    }

    # Freeze all settings using train OOF only; then fit each model on all train rows.
    temperature = fit_temperature(oof_y, oof_class_p)
    cls_final = _classifier(seed, threads).fit(X[train_valid], train_y[train_valid])
    reg_final = _regressor(seed, threads).fit(X[train_valid], train_delta[train_valid])
    raw_cls_val = cls_final.predict_proba(Xv[val_valid])
    calibrated_cls_val = temp_scale(raw_cls_val, temperature)
    reg_delta_val = reg_final.predict(Xv[val_valid])
    reg_class_val = _score_to_class(val_targets["current_score"][val_valid] + reg_delta_val)
    val_prior = persistence_probs(val_targets["base_mode"])[val_valid]
    val_persist_class = val_prior.argmax(1)
    val_class_pred = calibrated_cls_val.argmax(1)
    val_ci_y = val_y[val_valid]
    val_summary = {
        "rows": int(val_valid.sum()), "games": int(len(np.unique(vgroups[val_valid]))),
        "persistence": {**_hard_metrics(val_ci_y, val_persist_class),
                        **{k: scores(val_ci_y, val_prior)[k] for k in ("brier", "log_loss", "ece")}},
        "direct_lightgbm_classifier_temp_from_train_oof": {
            **_hard_metrics(val_ci_y, val_class_pred),
            **{k: scores(val_ci_y, calibrated_cls_val)[k] for k in ("brier", "log_loss", "ece")},
        },
        "delta_regression_thresholded": _hard_metrics(val_ci_y, reg_class_val),
        "delta_regression_continuous": {
            "mae": float(mean_absolute_error(val_delta[val_valid], reg_delta_val)),
            "rmse": float(mean_squared_error(val_delta[val_valid], reg_delta_val) ** .5),
            "r2": float(r2_score(val_delta[val_valid], reg_delta_val)),
        },
        "classifier_temperature_from_train_oof": temperature,
        "regression_probability_calibration": "not_reported; delta regressor is scored as continuous value and thresholded class, with no probability mapping fitted on validation",
    }
    game_val_persist = _game_values(val_ci_y, val_persist_class, vgroups[val_valid], vsides[val_valid],
                                    np.ones(val_valid.sum(), bool), val_prior)
    game_val_cls = _game_values(val_ci_y, val_class_pred, vgroups[val_valid], vsides[val_valid],
                                np.ones(val_valid.sum(), bool), calibrated_cls_val)
    game_val_reg = _game_values(val_ci_y, reg_class_val, vgroups[val_valid], vsides[val_valid],
                                np.ones(val_valid.sum(), bool), true_delta=val_delta[val_valid],
                                pred_delta=reg_delta_val)
    val_ci = {
        "classifier_minus_persistence": _paired_bootstrap(game_val_cls, game_val_persist, seed + 10, bootstrap),
        "regression_minus_persistence": _paired_bootstrap(game_val_reg, game_val_persist, seed + 11, bootstrap),
        "regression_minus_classifier": _paired_bootstrap(game_val_reg, game_val_cls, seed + 12, bootstrap),
    }

    report = {
        "schema": "death-aware-60s-relative-margin-regression-v1",
        "scope": "60s only; train and validation only; no production/default changes",
        "test_access": False,
        "train_npz": str(train_npz), "train_jsonl": str(train_jsonl),
        "validation_npz": str(val_npz), "validation_jsonl": str(val_jsonl),
        "label_contract": CONTRACT,
        "target": "future 60s _mode_score - current _mode_score; classify future score at [-1,+1] thresholds",
        "label_parity": "current and 60s target scores recomputed from JSONL; categorical death-aware labels verified row-by-row against NPZ",
        "features": "same compact causal feature recipe and LGBM params as death-aware direct LightGBM report",
        "grouping": f"{folds}-fold GroupKFold by complete game_id; red/blue views of a game stay together",
        "bootstrap": f"{bootstrap} paired game-cluster resamples; average red and blue per game before aggregation",
        "oof_train": oof_summary, "oof_paired_ci": oof_ci,
        "validation": val_summary, "validation_paired_ci": val_ci,
        "runtime_seconds": float(time.perf_counter() - started),
        "limitations": [
            "The test split was not read.",
            "The direct classifier uses the same death-aware LightGBM family and features as the existing validation experiment; hyperparameters were not tuned here.",
            "Regression thresholding yields hard classes; no regression class probabilities or calibration claim is made.",
            "Validation is an existing reused development split, not an untouched final blind test.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-npz", default="out_m2/temporal_forecast_death_aware_candidate/train.npz")
    parser.add_argument("--train-jsonl", default="out_m2/temporal_forecast_death_aware_candidate/train.death_aware.jsonl.gz")
    parser.add_argument("--validation-npz", default="out_m2/temporal_forecast_death_aware_candidate/validation.npz")
    parser.add_argument("--validation-jsonl", default="out_m2/temporal_forecast_death_aware_candidate/validation.death_aware.jsonl.gz")
    parser.add_argument("--output", default="out_m2/death_aware_global_mode_experiments/margin_regression_60s/report.json")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()
    report = run(Path(args.train_npz), Path(args.train_jsonl),
                 Path(args.validation_npz), Path(args.validation_jsonl),
                 Path(args.output), args.folds, args.seed, args.threads, args.bootstrap)
    print(json.dumps({"output": args.output, "runtime_seconds": report["runtime_seconds"],
                      "oof_train": report["oof_train"], "oof_paired_ci": report["oof_paired_ci"],
                      "validation": report["validation"], "validation_paired_ci": report["validation_paired_ci"]},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
