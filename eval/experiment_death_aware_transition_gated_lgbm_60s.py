"""One fixed transition-gated LightGBM experiment for death-aware 60s mode.

Uses only the canonical death-aware train and validation compact NPZs. OOF
splits keep complete games together. Validation is evaluated once after all
design choices are fixed. This script never opens test artifacts.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from forecast.temporal import HORIZONS, MODES, load_compact

CLASSES = np.arange(3)


def features(data):
    x = data["x"].astype(np.float32)
    cur = x[:, -1]
    base = np.eye(3, dtype=np.float32)[np.clip(data["base_mode"], 0, 2)]
    return np.concatenate([cur, x[:, -3:].mean(1), x[:, -6:].mean(1),
                           cur - x[:, -2], cur - x[:, -4], base], axis=1)


def aligned_proba(model, x):
    raw = model.predict_proba(x)
    out = np.zeros((len(x), 3), dtype=np.float64)
    for j, c in enumerate(model.classes_):
        if int(c) in (0, 1, 2):
            out[:, int(c)] = raw[:, j]
    return out


def persistence_probs(base):
    p = np.full((len(base), 3), .01, dtype=np.float64)
    p[np.arange(len(base)), base.astype(int)] = .98
    return p


def metrics(y, p):
    y = np.asarray(y, dtype=np.int64)
    p = np.asarray(p, dtype=np.float64)
    pred = p.argmax(1)
    onehot = np.eye(3)[y]
    class_metrics = {}
    recalls = []
    f1s = []
    for c in CLASSES:
        tp = int(np.sum((pred == c) & (y == c)))
        fp = int(np.sum((pred == c) & (y != c)))
        fn = int(np.sum((pred != c) & (y == c)))
        support = int(np.sum(y == c))
        predicted = int(np.sum(pred == c))
        precision = tp / predicted if predicted else 0.
        recall = tp / support if support else 0.
        f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.
        if support:
            recalls.append(recall)
        f1s.append(f1)
        class_metrics[MODES[c]] = {"support": support, "predicted": predicted,
                                  "precision": float(precision), "recall": float(recall),
                                  "f1": float(f1)}
    return {
        "n": int(len(y)), "accuracy": float(np.mean(pred == y)),
        "balanced_accuracy": float(np.mean(recalls)) if recalls else 0.,
        "macro_f1": float(np.mean(f1s)),
        "brier": float(np.mean(np.sum((p - onehot) ** 2, axis=1))),
        "log_loss": float(log_loss(y, p, labels=CLASSES)),
        "classes": class_metrics,
    }


def paired_game_bootstrap(y, probs_a, probs_b, gids, sides, seed=607, reps=2000):
    """Paired bootstrap games, averaging red/blue metric before resampling."""
    ids = np.unique(gids)
    per = {"a": {}, "b": {}}
    for name, p in (("a", probs_a), ("b", probs_b)):
        for gid in ids:
            side_values = []
            for side in ("红", "蓝"):
                m = (gids == gid) & (sides == side)
                if m.any():
                    side_values.append(metrics(y[m], p[m]))
            if len(side_values) == 2:
                per[name][gid] = {k: float(np.mean([z[k] for z in side_values])
                                         if k not in ("classes", "n") else 0.)
                                  for k in ("accuracy", "balanced_accuracy", "macro_f1", "brier")}
    common = np.array(sorted(set(per["a"]) & set(per["b"])))
    if not len(common):
        raise ValueError("No games with both red and blue sides for paired bootstrap")
    rng = np.random.default_rng(seed)
    take = rng.integers(0, len(common), size=(reps, len(common)))
    out = {}
    for metric in ("accuracy", "balanced_accuracy", "macro_f1", "brier"):
        d = np.array([per["a"][g][metric] - per["b"][g][metric] for g in common])
        boot = d[take].mean(1)
        out[metric] = {"delta_candidate_minus_comparator": float(d.mean()),
                       "paired_game_bootstrap_95_ci": [float(np.quantile(boot, .025)),
                                                       float(np.quantile(boot, .975))]}
    out["paired_games_red_blue"] = int(len(common))
    return out


def make_model(objective, seed, threads, *, binary=False):
    params = dict(n_estimators=260, learning_rate=.045, num_leaves=15,
                  max_depth=-1, min_child_samples=90, reg_lambda=4.,
                  colsample_bytree=.8, subsample=.85, subsample_freq=1,
                  verbosity=-1, n_jobs=threads, random_state=seed)
    if binary:
        return LGBMClassifier(objective="binary", **params)
    return LGBMClassifier(objective="multiclass", num_class=3, **params)


def transition_probs(Xtr, ytr, basetr, Xte, basete, seed, threads):
    switched = ytr != basetr
    n_changed = int(switched.sum())
    if n_changed == 0:
        raise ValueError("Training fold has no mode switches; cannot fit destination stage")
    change_model = make_model("binary", seed, threads, binary=True)
    change_model.fit(Xtr, switched.astype(np.int8))
    classes = list(change_model.classes_)
    p_switch = (change_model.predict_proba(Xte)[:, classes.index(1)]
                if 1 in classes else np.zeros(len(Xte)))

    destination_model = make_model("multiclass", seed + 100, threads)
    destination_model.fit(Xtr[switched], ytr[switched])
    p_dest = aligned_proba(destination_model, Xte)
    # Conditional destination must exclude current mode, then renormalize.
    row = np.arange(len(Xte))
    basete = np.asarray(basete, dtype=np.int64)
    p_dest[row, basete] = 0.
    den = p_dest.sum(1, keepdims=True)
    empty = den[:, 0] <= 1e-12
    p_dest = np.divide(p_dest, np.maximum(den, 1e-12), out=np.zeros_like(p_dest), where=den > 1e-12)
    if empty.any():
        for i in np.flatnonzero(empty):
            allowed = [c for c in CLASSES if c != int(basete[i])]
            p_dest[i, allowed] = .5
    p = p_switch[:, None] * p_dest
    p[row, basete] += 1. - p_switch
    # Guard against numerical drift and record exact normalization in tests.
    p = np.clip(p, 0., 1.)
    p /= np.maximum(p.sum(1, keepdims=True), 1e-12)
    return p, {"train_switch_rows": n_changed,
               "train_switch_fraction": float(switched.mean()),
               "destination_model_classes": [int(c) for c in destination_model.classes_],
               "destination_empty_rows_uniform_fallback": int(empty.sum())}


def run(train_path, val_path, output, folds=5, threads=8):
    started = time.perf_counter()
    train = load_compact(train_path)
    val = load_compact(val_path)
    contract = str(train["global_mode_label_version"][0])
    if contract != str(val["global_mode_label_version"][0]):
        raise ValueError("train/validation label contracts differ")
    j = list(HORIZONS).index(60)
    y = np.asarray(train["mode_y"][:, j], dtype=np.int64)
    base = np.asarray(train["base_mode"], dtype=np.int64)
    valid = (y >= 0) & (base >= 0) & (base < 3)
    X = features(train)
    gids, sides = np.asarray(train["game_ids"])[valid], np.asarray(train["sides"])[valid]
    yy, bb, xx = y[valid], base[valid], X[valid]
    unique_games = np.unique(gids)
    if len(unique_games) < folds:
        raise ValueError(f"Only {len(unique_games)} games for {folds}-fold GroupKFold")

    direct_oof = np.full((len(yy), 3), np.nan)
    transition_oof = np.full((len(yy), 3), np.nan)
    cv = GroupKFold(n_splits=folds)
    fold_details = []
    for fold, (tr, te) in enumerate(cv.split(xx, yy, groups=gids)):
        direct = make_model("multiclass", 23 + fold, threads)
        direct.fit(xx[tr], yy[tr])
        direct_oof[te] = aligned_proba(direct, xx[te])
        transition_oof[te], info = transition_probs(xx[tr], yy[tr], bb[tr], xx[te], bb[te],
                                                    23 + fold, threads)
        fold_details.append({"fold": fold, "train_games": int(len(np.unique(gids[tr]))),
                             "heldout_games": int(len(np.unique(gids[te]))), **info})
    if not np.isfinite(direct_oof).all() or not np.isfinite(transition_oof).all():
        raise ValueError("OOF predictions contain missing values")
    if not np.allclose(transition_oof.sum(1), 1., atol=1e-8):
        raise ValueError("Transition probability composition failed normalization")
    persist_oof = persistence_probs(bb)
    oof = {"transition_gated": transition_oof, "direct_lightgbm": direct_oof,
           "smoothed_persistence": persist_oof}
    oof_metrics = {name: metrics(yy, p) for name, p in oof.items()}
    oof_cis = {
        name: paired_game_bootstrap(yy, transition_oof, p, gids, sides, seed=707 + i)
        for i, (name, p) in enumerate((("direct_lightgbm", direct_oof),
                                        ("smoothed_persistence", persist_oof)))
    }

    # One frozen validation comparison: fit fixed candidates on all train rows.
    direct_all = make_model("multiclass", 23, threads)
    direct_all.fit(xx, yy)
    p_direct_v = aligned_proba(direct_all, features(val))
    vy0 = np.asarray(val["mode_y"][:, j], dtype=np.int64)
    vb0 = np.asarray(val["base_mode"], dtype=np.int64)
    vv = (vy0 >= 0) & (vb0 >= 0) & (vb0 < 3)
    vx, vy, vb = features(val)[vv], vy0[vv], vb0[vv]
    p_direct_v = p_direct_v[vv]
    p_transition_v, val_fit_info = transition_probs(xx, yy, bb, vx, vb, 10023, threads)
    p_persist_v = persistence_probs(vb)
    vgids, vsides = np.asarray(val["game_ids"])[vv], np.asarray(val["sides"])[vv]
    validation = {"transition_gated": metrics(vy, p_transition_v),
                  "direct_lightgbm": metrics(vy, p_direct_v),
                  "smoothed_persistence": metrics(vy, p_persist_v)}
    validation_cis = {
        name: paired_game_bootstrap(vy, p_transition_v, p, vgids, vsides, seed=1707 + i)
        for i, (name, p) in enumerate((("direct_lightgbm", p_direct_v),
                                        ("smoothed_persistence", p_persist_v)))
    }

    report = {
        "schema": "death-aware-transition-gated-lightgbm-60s-v1",
        "test_access": False,
        "train_npz": str(train_path), "validation_npz": str(val_path),
        "label_contract": contract, "horizon_seconds": 60,
        "train_rows": int(len(yy)), "train_games": int(len(unique_games)),
        "validation_rows": int(len(vy)),
        "validation_games": int(len(np.unique(vgids))),
        "features": "current 232, trailing 3/6 means, causal 1/3 deltas, current-mode one-hot",
        "design": "fixed 5-fold GroupKFold by complete game_id; direct 3-class LGBM comparator; binary switch LGBM then destination 3-class LGBM fit only on switch rows; force destination != current and renormalize; compose (1-p_switch)*onehot(current)+p_switch*conditional_destination",
        "model_parameters": {"n_estimators": 260, "learning_rate": .045,
                             "num_leaves": 15, "min_child_samples": 90,
                             "reg_lambda": 4., "colsample_bytree": .8,
                             "subsample": .85, "subsample_freq": 1,
                             "class_weight": None},
        "folds": fold_details,
        "oof_train_groupcv": oof_metrics,
        "oof_paired_game_bootstrap_delta_vs_transition_candidate": oof_cis,
        "frozen_validation_once": validation,
        "validation_paired_game_bootstrap_delta_vs_transition_candidate": validation_cis,
        "validation_transition_fit": val_fit_info,
        "limitations": ["validation is a reused development split and cannot establish a final unbiased promotion claim",
                        "only one fixed 60s candidate tested; no hyperparameter/threshold search",
                        "test split was not opened", "class supports and per-class recall/F1 are reported to expose class sacrifice"],
        "runtime_seconds": None,
    }
    report["runtime_seconds"] = time.perf_counter() - started
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train", required=True)
    p.add_argument("--validation", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--threads", type=int, default=8)
    a = p.parse_args()
    r = run(a.train, a.validation, a.output, a.folds, a.threads)
    print(json.dumps({"oof_train_groupcv": r["oof_train_groupcv"],
                      "oof_delta_ci": r["oof_paired_game_bootstrap_delta_vs_transition_candidate"],
                      "validation": r["frozen_validation_once"],
                      "validation_delta_ci": r["validation_paired_game_bootstrap_delta_vs_transition_candidate"],
                      "runtime_seconds": r["runtime_seconds"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
