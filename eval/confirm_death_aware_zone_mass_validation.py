"""One-time development validation confirmation of a frozen visible-state model.

The feature set was frozen from train-only experiments before this script is
run. This script accepts train and validation files only, fits the direct and
aggregate LightGBM candidates on the full train split, scores validation once,
and never opens a test file. The validation split has been viewed in prior
project work, so this is development confirmation, not a blind final result.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
import sys

import numpy as np
from lightgbm import LGBMClassifier

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from forecast.temporal import HORIZONS, MODES, load_compact
from eval.experiment_death_aware_mode_lightgbm import features as direct_features
from eval.experiment_death_aware_zone_mass_oof import (extract_train_aggregates,
    metrics, game_scores, bootstrap_delta)


def make_model(seed, threads):
    return LGBMClassifier(objective="multiclass", num_class=3, n_estimators=260,
        learning_rate=.045, num_leaves=15, max_depth=-1, min_child_samples=90,
        reg_lambda=4., colsample_bytree=.8, subsample=.85, subsample_freq=1,
        class_weight=None, verbosity=-1, n_jobs=threads, random_state=seed)


def prediction(model, x):
    raw = model.predict_proba(x)
    out = np.zeros((len(x), 3), dtype=np.float64)
    for col, cls in enumerate(model.classes_):
        out[:, int(cls)] = raw[:, col]
    return out


def run(train_npz, train_jsonl, validation_npz, validation_jsonl, output,
        *, threads=8, seed=1927, reps=2000):
    started = time.perf_counter()
    train, train_extra, train_audit = extract_train_aggregates(
        train_npz, train_jsonl, expected_split="train")
    val, val_extra, val_audit = extract_train_aggregates(
        validation_npz, validation_jsonl, expected_split="validation")
    train_contract = str(train["global_mode_label_version"][0])
    val_contract = str(val["global_mode_label_version"][0])
    if train_contract != val_contract or "death_aware" not in train_contract:
        raise ValueError(f"train/validation contract mismatch: {train_contract} vs {val_contract}")
    train_gids = np.asarray(train["game_ids"], dtype=np.int64)
    val_gids = np.asarray(val["game_ids"], dtype=np.int64)
    if set(train_gids.tolist()) & set(val_gids.tolist()):
        raise ValueError("train and development validation contain overlapping game IDs")
    train_sides = np.asarray(train["sides"]).astype(str)
    val_sides = np.asarray(val["sides"]).astype(str)
    y_train = np.asarray(train["mode_y"], dtype=np.int64)
    y_val = np.asarray(val["mode_y"], dtype=np.int64)
    base_train = np.asarray(train["base_mode"], dtype=np.int64)
    base_val = np.asarray(val["base_mode"], dtype=np.int64)
    x0_train, x0_val = direct_features(train), direct_features(val)
    x1_train = np.concatenate([x0_train, train_extra], axis=1)
    x1_val = np.concatenate([x0_val, val_extra], axis=1)
    reports = {}
    for j, h in enumerate(HORIZONS):
        tr_ok = (y_train[:, j] >= 0) & (base_train >= 0) & (base_train < 3)
        va_ok = (y_val[:, j] >= 0) & (base_val >= 0) & (base_val < 3)
        if len(np.unique(y_train[tr_ok, j])) != 3:
            raise ValueError(f"train horizon {h} lacks a class")
        direct = make_model(seed + j, threads)
        aggregate = make_model(seed + j, threads)
        direct.fit(x0_train[tr_ok], y_train[tr_ok, j])
        aggregate.fit(x1_train[tr_ok], y_train[tr_ok, j])
        pred_direct = prediction(direct, x0_val[va_ok])
        pred_aggregate = prediction(aggregate, x1_val[va_ok])
        yv = y_val[va_ok, j]
        persist = np.full((int(va_ok.sum()), 3), .01, dtype=np.float64)
        bv = base_val[va_ok]
        persist[np.arange(len(bv)), bv] = .98
        direct_full = np.full((len(y_val), 3), np.nan, dtype=np.float64); direct_full[va_ok] = pred_direct
        aggregate_full = np.full((len(y_val), 3), np.nan, dtype=np.float64); aggregate_full[va_ok] = pred_aggregate
        persist_full = np.full((len(y_val), 3), np.nan, dtype=np.float64); persist_full[va_ok] = persist
        direct_games = game_scores(y_val[:, j], direct_full, val_gids, val_sides, va_ok)
        aggregate_games = game_scores(y_val[:, j], aggregate_full, val_gids, val_sides, va_ok)
        persist_games = game_scores(y_val[:, j], persist_full, val_gids, val_sides, va_ok)
        game_metrics = {}
        for name, gs in (("aggregate_candidate", aggregate_games),
                         ("direct_lightgbm", direct_games), ("persistence", persist_games)):
            keys = next(iter(gs.values())).keys()
            game_metrics[name] = {k: float(np.mean([v[k] for v in gs.values()])) for k in keys}
        reports[f"{h}s"] = {
            "train_rows": int(tr_ok.sum()), "validation_rows": int(va_ok.sum()),
            "pooled": {"aggregate_candidate": metrics(yv, pred_aggregate),
                       "direct_lightgbm": metrics(yv, pred_direct),
                       "persistence": metrics(yv, persist)},
            "game_equal_red_blue_equal": game_metrics,
            "paired_game_bootstrap_aggregate_minus_direct": bootstrap_delta(aggregate_games, direct_games, seed + 100 + j, reps),
            "paired_game_bootstrap_aggregate_minus_persistence": bootstrap_delta(aggregate_games, persist_games, seed + 200 + j, reps),
            "models": {"direct": {"n_estimators": 260, "learning_rate": .045, "num_leaves": 15,
                                    "min_child_samples": 90, "reg_lambda": 4.},
                       "aggregate": {"n_estimators": 260, "learning_rate": .045, "num_leaves": 15,
                                       "min_child_samples": 90, "reg_lambda": 4.}}}
        print(json.dumps({h: game_metrics}, ensure_ascii=False), flush=True)
    out = Path(output); out.parent.mkdir(parents=True, exist_ok=True)
    result = {"schema": "death-aware-zone-mass-one-time-development-validation-v1",
        "validation_access": True, "validation_scored_once": True, "test_access": False,
        "test_files_opened": False, "confirmation_status": "development validation only; validation has been inspected in prior project work; not blind final evaluation",
        "label_contract": train_contract,
        "train": {"npz": str(Path(train_npz).resolve()), "jsonl": str(Path(train_jsonl).resolve()),
                  "rows": int(len(train_gids)), "games": int(len(np.unique(train_gids))), "feature_audit": train_audit},
        "validation": {"npz": str(Path(validation_npz).resolve()), "jsonl": str(Path(validation_jsonl).resolve()),
                       "rows": int(len(val_gids)), "games": int(len(np.unique(val_gids))), "feature_audit": val_audit},
        "overlap_train_validation_game_ids": 0,
        "features_frozen_from": "train-only visible aggregate + observable score trend OOF candidate; no validation-based changes",
        "feature_boundary": "only current/causal observable; enemy source follows radar visibility; no opponent resources or future targets; opponent unobserved units omitted",
        "aggregation_dim": int(train_extra.shape[1]),
        "metrics": "pooled rows and game-equal, within-game red/blue-equal; 2,000 paired game bootstrap replicates",
        "horizons": reports, "runtime_seconds": time.perf_counter() - started}
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train-npz", required=True); p.add_argument("--train-jsonl", required=True)
    p.add_argument("--validation-npz", required=True); p.add_argument("--validation-jsonl", required=True)
    p.add_argument("--output", required=True); p.add_argument("--threads", type=int, default=8)
    p.add_argument("--seed", type=int, default=1927); p.add_argument("--bootstrap-reps", type=int, default=2000)
    a = p.parse_args()
    r = run(a.train_npz, a.train_jsonl, a.validation_npz, a.validation_jsonl,
            a.output, threads=a.threads, seed=a.seed, reps=a.bootstrap_reps)
    print(json.dumps({h: v["game_equal_red_blue_equal"] for h, v in r["horizons"].items()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
