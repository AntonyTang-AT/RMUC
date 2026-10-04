"""Train-only GroupKFold OOF test of an explicit mode-transition factor.

The candidate estimates P(future mode != current mode) with a fold-local
binary LightGBM, then allocates that probability across alternative classes
using the already-held-out direct LightGBM probabilities. It reads only the
death-aware training NPZ and the prior train-only OOF probability artifact.
No validation/test split is loaded, scored, or tuned on.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
import sys

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from forecast.temporal import HORIZONS, MODES, load_compact
from eval.experiment_death_aware_mode_lightgbm import features


def metric_rows(y: np.ndarray, p: np.ndarray) -> dict:
    pred = p.argmax(1)
    conf = np.zeros((3, 3), dtype=np.int64)
    np.add.at(conf, (y, pred), 1)
    cls = {}
    f1s = []
    for c, name in enumerate(MODES):
        tp = int(conf[c, c]); fp = int(conf[:, c].sum() - tp); fn = int(conf[c].sum() - tp)
        recall = tp / int(conf[c].sum()) if conf[c].sum() else 0.
        precision = tp / int(conf[:, c].sum()) if conf[:, c].sum() else 0.
        f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.
        f1s.append(f1)
        cls[name] = {"support": int(conf[c].sum()), "precision": precision,
                     "recall": recall, "f1": f1}
    return {"n": int(len(y)), "accuracy": float(np.mean(pred == y)),
            "macro_f1": float(np.mean(f1s)), "per_class": cls,
            "confusion_true_rows_pred_cols": conf.tolist()}


def game_side_scores(y, p, gids, sides, valid):
    out = {}
    for gid in np.unique(gids[valid]):
        ss = []
        for side in ("红", "蓝"):
            m = valid & (gids == gid) & (sides == side)
            if m.any():
                s = metric_rows(y[m], p[m])
                ss.append(s)
        if len(ss) == 2:
            result = {k: float(np.mean([s[k] for s in ss])) for k in ("accuracy", "macro_f1")}
            for mode in MODES:
                result[f"recall_{mode}"] = float(np.mean([s["per_class"][mode]["recall"] for s in ss]))
            out[int(gid)] = result
    return out


def paired_bootstrap(candidate, comparator, seed, reps=2000):
    ids = sorted(set(candidate) & set(comparator))
    rng = np.random.default_rng(seed)
    ix = rng.integers(0, len(ids), size=(reps, len(ids)))
    out = {"paired_games": len(ids), "replicates": reps, "delta_candidate_minus_comparator": {}}
    metrics = ["accuracy", "macro_f1"] + [f"recall_{m}" for m in MODES]
    for key in metrics:
        d = np.asarray([candidate[g][key] - comparator[g][key] for g in ids])
        b = d[ix].mean(1)
        out["delta_candidate_minus_comparator"][key] = {
            "mean": float(d.mean()), "ci95": [float(np.quantile(b, .025)), float(np.quantile(b, .975))]}
    return out


def allocate_switch_probability(direct_p, base, p_switch):
    direct_p = np.asarray(direct_p, dtype=np.float64)
    base = np.asarray(base, dtype=np.int64)
    ps = np.clip(np.asarray(p_switch, dtype=np.float64), 1e-5, 1 - 1e-5)
    out = np.zeros_like(direct_p)
    for i in range(len(base)):
        b = int(base[i])
        alt = [c for c in range(3) if c != b]
        mass = direct_p[i, alt].sum()
        share = direct_p[i, alt] / mass if mass > 1e-12 else np.full(2, .5)
        out[i, b] = 1 - ps[i]
        out[i, alt] = ps[i] * share
    return out


def run(train_path, prior_oof_path, outdir, *, folds=5, threads=8, seed=1927, reps=2000):
    started = time.perf_counter()
    data = load_compact(train_path)
    versions = np.asarray(data["global_mode_label_version"]).astype(str)
    if len(np.unique(versions)) != 1 or "death_aware" not in versions[0]:
        raise ValueError(f"expected a single death-aware label contract, found {versions}")
    prior = np.load(prior_oof_path, allow_pickle=False)
    gids = np.asarray(data["game_ids"], dtype=np.int64)
    sides = np.asarray(data["sides"]).astype(str)
    yall = np.asarray(data["mode_y"], dtype=np.int64)
    baseall = np.asarray(data["base_mode"], dtype=np.int64)
    if not np.array_equal(gids, prior["game_ids"]) or not np.array_equal(yall, prior["mode_y"]):
        raise ValueError("prior OOF rows do not align with the death-aware training NPZ")
    direct_all = np.asarray(prior["lightgbm"], dtype=np.float64)
    x = features(data)
    cand_all = np.full_like(direct_all, np.nan)
    switch_all = np.full((len(gids), 4), np.nan, dtype=np.float32)
    fold_ids = np.full(len(gids), -1, dtype=np.int8)
    fold_reports = []
    cv = GroupKFold(n_splits=folds)
    for fold, (tr, held) in enumerate(cv.split(x, groups=gids)):
        fold_ids[held] = fold
        hor = {}
        for j, h in enumerate(HORIZONS):
            train_ok = (yall[tr, j] >= 0) & (baseall[tr] >= 0) & (baseall[tr] < 3)
            held_ok = (yall[held, j] >= 0) & (baseall[held] >= 0) & (baseall[held] < 3)
            trr, hrr = tr[train_ok], held[held_ok]
            yswitch = (yall[trr, j] != baseall[trr]).astype(np.int8)
            if len(np.unique(yswitch)) != 2:
                raise ValueError(f"fold {fold} horizon {h}: transition target lacks both classes")
            model = LGBMClassifier(objective="binary", n_estimators=220,
                learning_rate=.045, num_leaves=15, max_depth=-1, min_child_samples=90,
                reg_lambda=4., colsample_bytree=.8, subsample=.85, subsample_freq=1,
                class_weight="balanced", verbosity=-1, n_jobs=threads, random_state=seed + fold)
            model.fit(x[trr], yswitch)
            ps = model.predict_proba(x[hrr])[:, 1]
            dp = direct_all[hrr, j]
            candidate = allocate_switch_probability(dp, baseall[hrr], ps)
            cand_all[hrr, j] = candidate
            switch_all[hrr, j] = ps.astype(np.float32)
            hor[f"{h}s"] = {"train_rows": int(len(trr)), "heldout_rows": int(len(hrr)),
                             "train_switch_rate": float(yswitch.mean()),
                             "heldout_switch_rate": float(np.mean(yall[hrr, j] != baseall[hrr])),
                             "predicted_switch_probability_mean": float(ps.mean())}
        fold_reports.append({"fold": fold, "heldout_games": int(len(np.unique(gids[held]))),
                             "horizons": hor})
        print(json.dumps({"fold": fold, "horizons": hor}, ensure_ascii=False), flush=True)

    reports = {}
    for j, h in enumerate(HORIZONS):
        valid = (yall[:, j] >= 0) & np.isfinite(cand_all[:, j, 0])
        yy, cp, dp = yall[valid, j], cand_all[valid, j], direct_all[valid, j]
        basep = np.full((len(yy), 3), .01, dtype=np.float64)
        bb = baseall[valid]
        basep[np.arange(len(bb)), bb] = .98
        cs = game_side_scores(yall[:, j], cand_all[:, j], gids, sides, valid)
        ds = game_side_scores(yall[:, j], direct_all[:, j], gids, sides, valid)
        # Build full aligned persistence probabilities for game-level bootstrap.
        persist_full = np.full((len(gids), 3), np.nan, dtype=np.float64)
        legal = (baseall >= 0) & (baseall < 3)
        persist_full[legal] = .01
        persist_full[np.flatnonzero(legal), baseall[legal]] = .98
        persist_valid = valid & legal
        bs = game_side_scores(yall[:, j], persist_full, gids, sides, persist_valid)
        reports[f"{h}s"] = {
            "rows": int(valid.sum()), "folds": fold_reports,
            "pooled": {"transition_factor": metric_rows(yy, cp),
                       "direct_lightgbm_oof": metric_rows(yy, dp),
                       "persistence": metric_rows(yy, basep)},
            "game_equal_red_blue_equal": {
                "transition_factor": {k: float(np.mean([v[k] for v in cs.values()])) for k in next(iter(cs.values()))},
                "direct_lightgbm_oof": {k: float(np.mean([v[k] for v in ds.values()])) for k in next(iter(ds.values()))},
                "persistence": {k: float(np.mean([v[k] for v in bs.values()])) for k in next(iter(bs.values()))},
            },
            "paired_game_bootstrap_vs_direct_lightgbm": paired_bootstrap(cs, ds, seed + 100 + j, reps),
            "paired_game_bootstrap_vs_persistence": paired_bootstrap(cs, bs, seed + 200 + j, reps),
        }
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    report = {"schema": "death-aware-transition-factor-train-group-oof-v1",
        "test_access": False, "validation_access": False, "validation_or_test_files_opened": False,
        "train_npz": str(Path(train_path).resolve()), "prior_oof_probability_file": str(Path(prior_oof_path).resolve()),
        "label_contract": versions[0], "train_rows": int(len(gids)), "train_games": int(len(np.unique(gids)),),
        "fold_count": folds, "grouping": "GroupKFold by whole game_id; red/blue views stay in same fold",
        "candidate": "fold-local balanced binary LightGBM estimates future mode != current mode; total transition mass is allocated using normalized held-out direct LGBM probabilities among the two non-current classes; no calibration, threshold tuning, or validation selection",
        "binary_lightgbm": {"n_estimators": 220, "learning_rate": .045, "num_leaves": 15,
            "min_child_samples": 90, "reg_lambda": 4., "class_weight": "balanced"},
        "folds": fold_reports, "horizons": reports, "runtime_seconds": time.perf_counter()-started}
    (outdir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    np.savez_compressed(outdir / "transition_factor_oof.npz", game_ids=gids, sides=sides,
        issued_at=np.asarray(data["issued_at"]), mode_y=yall, base_mode=baseall,
        fold_ids=fold_ids, transition_factor=cand_all.astype(np.float32),
        direct_lightgbm=direct_all.astype(np.float32), persistence=np.asarray(prior["persistence"]))
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train", required=True); p.add_argument("--prior-oof", required=True)
    p.add_argument("--output", required=True); p.add_argument("--folds", type=int, default=5)
    p.add_argument("--threads", type=int, default=8); p.add_argument("--seed", type=int, default=1927)
    p.add_argument("--bootstrap-reps", type=int, default=2000)
    a = p.parse_args()
    r = run(a.train, a.prior_oof, a.output, folds=a.folds, threads=a.threads,
            seed=a.seed, reps=a.bootstrap_reps)
    print(json.dumps({h: v["game_equal_red_blue_equal"] for h, v in r["horizons"].items()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
