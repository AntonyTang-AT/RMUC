"""Nested, train-only evaluation of a tiny logit-intercept bias grid.

Outer folds hold out complete games. Inside each outer-training split, a fixed
game-level 80/20 split is used: fit games train the frozen zone+score LightGBM,
calibration games select one of seven predeclared three-class logit offsets.
Only outer-held games estimate generalization. Validation and test files are
not accepted or opened by this script.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from forecast.temporal import HORIZONS, MODES, load_compact
from eval.experiment_death_aware_mode_lightgbm import features as direct_features
from eval.experiment_death_aware_zone_mass_oof import (
    extract_train_aggregates, game_scores, metrics, bootstrap_delta,
)

# Constrained, symmetric three-class offsets. Each nonzero candidate promotes
# one class by 0.30 logit and reduces the other two by 0.15; the inverse
# candidates demote one class. This is intentionally a seven-point grid.
BIAS_GRID = [
    (0.0, 0.0, 0.0),
    (0.30, -0.15, -0.15), (-0.15, 0.30, -0.15), (-0.15, -0.15, 0.30),
    (-0.30, 0.15, 0.15), (0.15, -0.30, 0.15), (0.15, 0.15, -0.30),
]


def estimator(seed: int, threads: int) -> LGBMClassifier:
    return LGBMClassifier(
        objective="multiclass", num_class=3, n_estimators=260,
        learning_rate=.045, num_leaves=15, max_depth=-1,
        min_child_samples=90, reg_lambda=4., colsample_bytree=.8,
        subsample=.85, subsample_freq=1, class_weight=None,
        verbosity=-1, n_jobs=threads, random_state=seed,
    )


def predict_proba3(model, X):
    raw = model.predict_proba(X)
    p = np.zeros((len(X), 3), dtype=np.float64)
    for col, cls in enumerate(model.classes_):
        p[:, int(cls)] = raw[:, col]
    return p


def apply_bias(prob, bias):
    logits = np.log(np.clip(np.asarray(prob, dtype=np.float64), 1e-9, 1.0))
    logits += np.asarray(bias, dtype=np.float64)[None, :]
    logits -= logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True)


def balanced_game_accuracy(y, p, gids, sides, valid):
    gs = game_scores(y, p, gids, sides, valid)
    return float(np.mean([v["accuracy"] for v in gs.values()])) if gs else float("nan")


def choose_bias(y, prob, gids, sides, valid):
    # Select solely on the inner calibration games. Ties favor higher Macro-F1,
    # then the smallest offset norm, then the declared grid order.
    candidates = []
    for ix, bias in enumerate(BIAS_GRID):
        adjusted = apply_bias(prob, bias)
        gs = game_scores(y, adjusted, gids, sides, valid)
        acc = float(np.mean([v["accuracy"] for v in gs.values()])) if gs else -1.
        macro = float(np.mean([v["macro_f1"] for v in gs.values()])) if gs else -1.
        candidates.append({"index": ix, "bias": list(bias), "accuracy": acc,
                           "macro_f1": macro, "offset_l1": float(np.abs(bias).sum())})
    best = max(candidates, key=lambda r: (r["accuracy"], r["macro_f1"], -r["offset_l1"], -r["index"]))
    return tuple(best["bias"]), candidates


def per_game_detailed(y, p, gids, sides, valid):
    """Game-equal, red/blue-equal summaries including class recall."""
    return game_scores(y, p, gids, sides, valid)


def summarize_game_scores(gs):
    if not gs:
        return {}
    keys = next(iter(gs.values()))
    return {k: float(np.mean([v[k] for v in gs.values()])) for k in keys}


def run(train_npz, train_jsonl, direct_oof, zone_score_oof, output,
        *, folds=5, inner_fit_fraction=.8, threads=8, seed=260927, reps=4000):
    started = time.perf_counter()
    d, zone_score, feature_audit = extract_train_aggregates(train_npz, train_jsonl)
    baseline = np.load(direct_oof, allow_pickle=False)
    previous = np.load(zone_score_oof, allow_pickle=False)
    for ref in (baseline, previous):
        for key in ("game_ids", "sides", "issued_at", "mode_y", "base_mode"):
            if not np.array_equal(np.asarray(d[key]), np.asarray(ref[key])):
                raise ValueError(f"alignment mismatch in {key}")
    X = np.concatenate([direct_features(d), zone_score], axis=1)
    game_ids = np.asarray(d["game_ids"], dtype=np.int64)
    sides = np.asarray(d["sides"]).astype(str)
    y_all = np.asarray(d["mode_y"], dtype=np.int64)
    base_mode = np.asarray(d["base_mode"], dtype=np.int64)

    # Also ensure supplied direct OOF agrees with its canonical raw predictions
    # at the aggregate level only; candidate comparisons below use outer refits.
    direct_ref_key = "direct_lightgbm" if "direct_lightgbm" in baseline.files else "lightgbm"
    direct_ref = np.asarray(baseline[direct_ref_key], dtype=np.float64)
    persistence_ref = np.asarray(baseline["persistence"], dtype=np.float64)
    prior_zone = np.asarray(previous["aggregate_candidate"], dtype=np.float64)

    cv = GroupKFold(n_splits=folds)
    pred = {f"{h}s": {name: np.full((len(game_ids), 3), np.nan, dtype=np.float64)
                for name in ("outer_raw", "nested_bias", "persistence")}
            for h in HORIZONS}
    outer_fold = np.full(len(game_ids), -1, dtype=np.int8)
    folds_report = []
    grid_records = {f"{h}s": [] for h in HORIZONS}

    for fold, (outer_train, held) in enumerate(cv.split(X, groups=game_ids)):
        outer_fold[held] = fold
        fit_games, cal_games = next(GroupShuffleSplit(
            n_splits=1, train_size=inner_fit_fraction, random_state=seed + fold
        ).split(np.unique(game_ids[outer_train]),
                groups=np.unique(game_ids[outer_train])))
        # GroupShuffleSplit index is over unique IDs; map whole games back to rows.
        train_game_list = np.unique(game_ids[outer_train])
        fit_ids, cal_ids = train_game_list[fit_games], train_game_list[cal_games]
        fit_rows = outer_train[np.isin(game_ids[outer_train], fit_ids)]
        cal_rows = outer_train[np.isin(game_ids[outer_train], cal_ids)]
        held_report = {"fold": fold, "outer_fit_games": int(len(fit_ids)),
                       "inner_calibration_games": int(len(cal_ids)),
                       "outer_held_games": int(len(np.unique(game_ids[held]))),
                       "horizons": {}}

        for j, h in enumerate(HORIZONS):
            train_mask = (y_all[:, j] >= 0) & (base_mode >= 0) & (base_mode < 3)
            fit = fit_rows[train_mask[fit_rows]]
            cal = cal_rows[train_mask[cal_rows]]
            test = held[train_mask[held]]
            if len(np.unique(y_all[fit, j])) != 3 or len(np.unique(y_all[cal, j])) != 3:
                raise ValueError(f"fold {fold} horizon {h}: missing class in fit/calibration split")
            model = estimator(seed + fold, threads)
            model.fit(X[fit], y_all[fit, j])
            cal_p = predict_proba3(model, X[cal])
            cal_valid = np.ones(len(cal), dtype=bool)
            chosen, all_grid = choose_bias(y_all[cal, j], cal_p, game_ids[cal], sides[cal], cal_valid)
            test_p = predict_proba3(model, X[test])
            hkey = f"{h}s"
            pred[hkey]["outer_raw"][test] = test_p
            pred[hkey]["nested_bias"][test] = apply_bias(test_p, chosen)
            # Persistence is scored on exactly the outer-held issue rows.
            prior = np.full((len(test), 3), .01, dtype=np.float64)
            prior[np.arange(len(test)), base_mode[test]] = .98
            pred[hkey]["persistence"][test] = prior
            cal_raw = metrics(y_all[cal, j], cal_p)
            cal_best = apply_bias(cal_p, chosen)
            cal_best_m = metrics(y_all[cal, j], cal_best)
            grid_records[hkey].append({"fold": fold, "selected_bias": list(chosen),
                "calibration_raw": cal_raw,
                "calibration_selected": cal_best_m,
                "all_predeclared_offsets": all_grid})
            held_report["horizons"][hkey] = {"fit_rows": int(len(fit)),
                "calibration_rows": int(len(cal)), "held_rows": int(len(test)),
                "selected_bias": list(chosen), "calibration_games": int(len(np.unique(game_ids[cal])))}
        folds_report.append(held_report)
        print(json.dumps({"outer_fold": fold, "fit_games": len(fit_ids),
                          "calibration_games": len(cal_ids), "held_games": held_report["outer_held_games"]}), flush=True)

    results = {}
    for j, h in enumerate(HORIZONS):
        hkey = f"{h}s"
        valid = (y_all[:, j] >= 0) & np.isfinite(pred[hkey]["outer_raw"][:, 0])
        result_by_model = {}
        game_results = {}
        for name in ("outer_raw", "nested_bias", "persistence"):
            game_results[name] = per_game_detailed(y_all[:, j], pred[hkey][name],
                                                   game_ids, sides, valid)
            result_by_model[name] = {"pooled": metrics(y_all[valid, j], pred[hkey][name][valid]),
                                     "game_equal_red_blue_equal": summarize_game_scores(game_results[name])}
        # Reference estimates are informational only (not part of nested scores).
        ref_results = {}
        for name, p in (("prior_direct_lightgbm_oof", direct_ref[:, j]),
                        ("prior_zone_score_oof", prior_zone[:, j])):
            mask = (y_all[:, j] >= 0) & np.isfinite(p[:, 0])
            gs = per_game_detailed(y_all[:, j], p, game_ids, sides, mask)
            ref_results[name] = {"pooled": metrics(y_all[mask, j], p[mask]),
                                 "game_equal_red_blue_equal": summarize_game_scores(gs)}
        # paired cluster bootstrap against the same outer-fit raw model and persistence
        paired = {}
        for candidate, reference in (("nested_bias", "outer_raw"), ("nested_bias", "persistence")):
            boot = bootstrap_delta(
                game_results[candidate], game_results[reference], seed=seed + 100 + j + (0 if reference == "outer_raw" else 10), reps=reps)
            delta_key = next(k for k in boot if k.startswith("delta_"))
            boot[f"delta_{candidate}_minus_{reference}"] = boot.pop(delta_key)
            paired[f"{candidate}_minus_{reference}"] = boot
        results[hkey] = {"valid_rows": int(valid.sum()), "outer_models": result_by_model,
                         "paired_game_bootstrap": paired, "historical_oof_references_not_nested": ref_results}

    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    report = {
        "schema": "death-aware-nested-game-logit-bias-train-only-v1",
        "data_access": {"opened_split": ["train"], "validation_opened": False,
                        "test_opened": False, "targets_from_train_npz_only": True},
        "train_npz": str(Path(train_npz).resolve()), "train_jsonl": str(Path(train_jsonl).resolve()),
        "label_contract": str(d["global_mode_label_version"][0]),
        "train_games": int(len(np.unique(game_ids))), "train_rows": int(len(game_ids)),
        "features": "direct temporal features plus current-observable zone mass/alive/HP/formation/self-resource and causal observable-score trend features",
        "estimator": {"name": "LightGBM multiclass", "parameters": {
            "n_estimators": 260, "learning_rate": .045, "num_leaves": 15,
            "min_child_samples": 90, "reg_lambda": 4.0, "colsample_bytree": .8,
            "subsample": .85, "subsample_freq": 1, "class_weight": None}},
        "nested_design": "Outer 5-fold GroupKFold by full game. Within each outer-training games, deterministic 80/20 GroupShuffleSplit by full game. Fit the model only on inner-fit games; select one offset on calibration games; evaluate only on untouched outer-held games.",
        "inner_selection_metric": "game-equal, red/blue-equal accuracy; tie-break Macro-F1, then lower L1 offset, then declared grid order",
        "predeclared_offset_grid": [list(x) for x in BIAS_GRID],
        "grid_search_scope": "7 fixed triplets, independently selected per horizon and outer fold; no model/features/threshold tuning",
        "comparison_policy": "Outer raw argmax and persistence use exactly the same held games; primary paired CI resamples whole games.",
        "historical_oof_caution": "Direct and zone-score OOF references are included only as context. The nested candidate trains on 80% of each outer-training fold, so its point estimates are not directly comparable to full-training-fold OOF values.",
        "feature_audit": feature_audit, "folds": folds_report,
        "calibration_grid_results_all_folds": grid_records,
        "horizons": results, "runtime_seconds": time.perf_counter() - started,
    }
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    np.savez_compressed(out / "nested_bias_oof.npz", game_ids=game_ids, sides=sides,
        issued_at=np.asarray(d["issued_at"]), mode_y=y_all, base_mode=base_mode,
        outer_fold_ids=outer_fold,
        **{f"{h}s_{name}": arr.astype(np.float32) for hkey, models in pred.items()
           for name, arr in models.items() for h in [hkey[:-1]]})
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train-npz", required=True)
    p.add_argument("--train-jsonl", required=True)
    p.add_argument("--direct-oof", required=True)
    p.add_argument("--zone-score-oof", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--inner-fit-fraction", type=float, default=.8)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--seed", type=int, default=260927)
    p.add_argument("--bootstrap-reps", type=int, default=4000)
    a = p.parse_args()
    report = run(a.train_npz, a.train_jsonl, a.direct_oof, a.zone_score_oof,
                 a.output, folds=a.folds, inner_fit_fraction=a.inner_fit_fraction,
                 threads=a.threads, seed=a.seed, reps=a.bootstrap_reps)
    print(json.dumps({k: v["outer_models"]["nested_bias"]["game_equal_red_blue_equal"]
                      for k, v in report["horizons"].items()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
