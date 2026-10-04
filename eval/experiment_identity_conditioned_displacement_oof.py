"""Train-only grouped OOF test of behavior-conditioned per-robot motion.

The model predicts a unit's future displacement from its causal position,
same-ID recent velocity, deviation from its type formation, current functional
zone, side, role, type, and match phase. Complete games are held out so the
two side views cannot leak across folds. Targets are exact-ID positions and
only currently visible/alive mobile units are scored.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from lightgbm import LGBMRegressor
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.crossfit_validation_identity_trajectory import _collect, _zones  # noqa: E402
from forecast.entity import ROBOT_SPEED_MPS  # noqa: E402
from forecast.temporal import (ENTITY_TYPES, HORIZONS, MAP_X, MAP_Y,
                               MOBILE_TYPES, load_compact)  # noqa: E402
from worldstate.functional_zone import (  # noqa: E402
    FUNCTIONAL_ZONE_LABELS, assign_functional_zone,
)


def _features(sample, entity):
    p = np.asarray(entity["p"], dtype=np.float32)
    group = (entity["role"], entity["type"])
    center = np.asarray(sample["centroids"].get(group, p), dtype=np.float32)
    v = entity["v"]
    v = np.zeros(2, dtype=np.float32) if v is None else np.asarray(v, dtype=np.float32)
    group_v = np.asarray(sample["group_v"].get(group, v), dtype=np.float32)
    cap = max(float(ROBOT_SPEED_MPS.get(entity["type"], .35)), .05)
    type_onehot = [float(entity["type"] == value) for value in ENTITY_TYPES]
    role_onehot = [float(entity["role"] == "self"),
                   float(entity["role"] == "opponent")]
    side_onehot = [float(sample["side"] == "红"), float(sample["side"] == "蓝")]
    zone = assign_functional_zone(float(p[0]), float(p[1]))
    zone_onehot = [float(zone == value) for value in FUNCTIONAL_ZONE_LABELS]
    t = float(sample["issued"])
    return np.asarray([
        p[0] / MAP_X, p[1] / MAP_Y,
        v[0] / cap, v[1] / cap, float(np.linalg.norm(v)) / cap,
        (p[0] - center[0]) / MAP_X, (p[1] - center[1]) / MAP_Y,
        (v[0] - group_v[0]) / cap, (v[1] - group_v[1]) / cap,
        min(1.0, max(0.0, t / 420.0)),
        *type_onehot, *role_onehot, *side_onehot, *zone_onehot,
    ], dtype=np.float32)


def _prepare(npz_path, jsonl_path):
    compact = load_compact(npz_path)
    samples = _collect(jsonl_path, compact)
    entities = {"x": [], "p": [], "v": [], "games": [], "sides": [],
                "types": [], "ids": []}
    rows = {h: {"indices": [], "y": []} for h in HORIZONS}
    for sample in samples:
        for entity in sample["entities"]:
            if entity["type"] not in MOBILE_TYPES:
                continue
            features = _features(sample, entity)
            index = len(entities["x"])
            p = np.asarray(entity["p"], dtype=np.float32)
            velocity = (np.zeros(2, dtype=np.float32) if entity["v"] is None
                        else np.asarray(entity["v"], dtype=np.float32))
            entities["x"].append(features)
            entities["p"].append(p)
            entities["v"].append(velocity)
            entities["games"].append(int(sample["gid"]))
            entities["sides"].append(str(sample["side"]))
            entities["types"].append(str(entity["type"]))
            entities["ids"].append(str(entity["id"]))
            for h in HORIZONS:
                if not sample["future_valid"][h]:
                    continue
                truth = sample["future"][h].get(entity["id"])
                if truth is None:
                    continue
                r = rows[h]
                r["indices"].append(index)
                r["y"].append(np.asarray(truth, dtype=np.float32) - p)
    for key in ("x", "p", "v"):
        entities[key] = np.asarray(entities[key], dtype=np.float32)
    for key in ("games",):
        entities[key] = np.asarray(entities[key], dtype=np.int64)
    for key in ("sides", "types", "ids"):
        entities[key] = np.asarray(entities[key])
    for h, r in rows.items():
        r["indices"] = np.asarray(r["indices"], dtype=np.int64)
        r["y"] = np.asarray(r["y"], dtype=np.float32)
    return entities, rows


def _predict_oof(entities, rows, folds, seed):
    output = {}
    fold_metrics = []
    for hi, h in enumerate(HORIZONS):
        data = rows[h]
        idx = data["indices"]
        pred = np.zeros_like(data["y"])
        splitter = GroupKFold(n_splits=folds)
        split = list(splitter.split(np.zeros(len(idx)),
                                    groups=entities["games"][idx]))
        for fold, (tr_idx, te_idx) in enumerate(split):
            model_x = entities["x"][idx[tr_idx]]
            held_x = entities["x"][idx[te_idx]]
            for dim in range(2):
                model = LGBMRegressor(
                    objective="regression_l2", n_estimators=90,
                    learning_rate=.045, num_leaves=15,
                    min_child_samples=160, reg_lambda=8.0,
                    colsample_bytree=.9, subsample=.85,
                    subsample_freq=1, verbosity=-1, n_jobs=4,
                    random_state=seed + 100 * hi + 10 * fold + dim)
                model.fit(model_x, data["y"][tr_idx, dim])
                pred[te_idx, dim] = model.predict(held_x)
            fold_metrics.append({"horizon": h, "fold": fold,
                                 "train_games": int(len(np.unique(entities["games"][idx[tr_idx]]))),
                                 "heldout_games": int(len(np.unique(entities["games"][idx[te_idx]]))),
                                 "train_entities": int(len(tr_idx)),
                                 "heldout_entities": int(len(te_idx))})
            print(json.dumps({"horizon": h, "fold": fold,
                              "status": "heldout_predictions_ready",
                              "rows": len(te_idx)}, ensure_ascii=False), flush=True)
        output[h] = pred
    return output, fold_metrics


def _game_side_scores(data, prediction):
    pred_xy = np.clip(data["p"] + prediction, [0.0, 0.0], [MAP_X, MAP_Y])
    truth_xy = data["p"] + data["y"]
    zone_ids = _zones(np.concatenate([pred_xy, truth_xy], axis=0))
    pred_zone, truth_zone = np.split(zone_ids, 2)
    euclidean = np.linalg.norm(pred_xy - data["p"] - data["y"], axis=1)
    side_code = (data["sides"] == "蓝").astype(np.int64)
    pair_key = data["games"].astype(np.int64) * 2 + side_code
    unique_keys, inverse = np.unique(pair_key, return_inverse=True)
    counts = np.bincount(inverse, minlength=len(unique_keys)).astype(np.float64)
    coordinate_mae = np.mean(np.abs(pred_xy - truth_xy), axis=1)
    zone_hit = (pred_zone == truth_zone).astype(np.float64)
    mae = np.bincount(inverse, weights=coordinate_mae,
                      minlength=len(unique_keys)) / counts
    euclidean = np.bincount(inverse, weights=euclidean,
                            minlength=len(unique_keys)) / counts
    zone_hit = np.bincount(inverse, weights=zone_hit,
                           minlength=len(unique_keys)) / counts
    return {
        (int(key // 2), "蓝" if key % 2 else "红"): {
            "mae_m": float(mae[i]), "euclidean_m": float(euclidean[i]),
            "zone_hit": float(zone_hit[i]), "n": int(counts[i]),
        }
        for i, key in enumerate(unique_keys)
    }


def _aggregate(per, metric):
    games = sorted({g for g, _ in per})
    game_scores = []
    for game in games:
        sides = [per[(game, side)][metric] for side in ("红", "蓝")
                 if (game, side) in per]
        if sides:
            game_scores.append(float(np.mean(sides)))
    return float(np.mean(game_scores)) if game_scores else None


def _paired_ci(candidate, baseline, metric, *, reps=2000, seed=0):
    games = sorted({g for g, s in candidate
                    if (g, "红") in candidate and (g, "蓝") in candidate
                    and (g, "红") in baseline and (g, "蓝") in baseline})
    if not games:
        return {"delta": None, "ci95": [None, None], "games": 0}
    # Error metrics are candidate minus baseline; zone-hit metrics are also
    # candidate minus baseline, so positive is consistently better only for
    # zone_hit. Keep that sign explicit in the report metadata.
    values = []
    for game in games:
        values.append(float(np.mean([
            candidate[(game, side)][metric] - baseline[(game, side)][metric]
            for side in ("红", "蓝")])))
    arr = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = arr[rng.integers(0, len(arr), size=(reps, len(arr)))].mean(axis=1)
    return {"delta_candidate_minus_baseline": float(arr.mean()),
            "game_bootstrap_95_ci": [float(np.quantile(draws, .025)),
                                     float(np.quantile(draws, .975))],
            "n_games_red_blue_paired": len(games), "replicates": reps}


def _write_cache(path, entities, targets):
    arrays = {f"entity__{key}": value for key, value in entities.items()}
    for horizon, data in targets.items():
        arrays[f"h{horizon}__indices"] = data["indices"]
        arrays[f"h{horizon}__y"] = data["y"]
    np.savez(path, **arrays)


def _read_cache(path):
    with np.load(path, allow_pickle=False) as archive:
        entities = {key[len("entity__"):]: archive[key]
                    for key in archive.files if key.startswith("entity__")}
        targets = {h: {name: archive[f"h{h}__{name}"]
                       for name in ("indices", "y")}
                   for h in HORIZONS}
    return entities, targets


def run(args):
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    cache = out.with_name("training_features_cache.npz")
    if cache.exists() and not args.rebuild_cache:
        entities, targets = _read_cache(cache)
    else:
        entities, targets = _prepare(args.train_npz, args.train_jsonl)
        _write_cache(cache, entities, targets)
    pred, fold_rows = _predict_oof(entities, targets, args.folds, args.seed)
    report = {
        "schema": "identity-conditioned-displacement-game-oof-v1",
        "test_access": False,
        "validation_access": False,
        "split": "GroupKFold by complete game_id; both observing sides remain together",
        "features": ["current XY", "same-ID exact-3s velocity", "type-formation offset",
                     "type-relative velocity", "functional-zone one-hot", "team role",
                     "observing side", "elapsed game fraction"],
        "target": "future exact-ID XY minus current XY at exact 10/30/60/120 second horizons; future-alive matches only",
        "model": {"family": "LightGBM L2 regressor per coordinate and horizon",
                  "estimators": 90, "num_leaves": 15, "min_child_samples": 160},
        "zone_scoring": "5 cm raster lookup generated from the canonical functional-zone polygons",
        "folds": fold_rows,
        "horizons": {},
    }
    for hi, h in enumerate(HORIZONS):
        target = targets[h]
        idx = target["indices"]
        d = {"y": target["y"], **{
            key: entities[key][idx]
            for key in ("p", "v", "games", "sides", "types", "ids")
        }}
        persistence = np.zeros_like(d["y"])
        velocity = d["v"] * float(h)
        by_model = _game_side_scores(d, pred[h])
        by_persistence = _game_side_scores(d, persistence)
        by_velocity = _game_side_scores(d, velocity)
        variants = {"persistence": by_persistence,
                    "identity_velocity": by_velocity,
                    "behavior_conditioned": by_model}
        metrics = {}
        for name, per in variants.items():
            metrics[name] = {metric: _aggregate(per, metric)
                             for metric in ("mae_m", "euclidean_m", "zone_hit")}
        paired = {}
        for name in ("identity_velocity", "behavior_conditioned"):
            paired[name] = {}
            for mi, metric in enumerate(("mae_m", "euclidean_m", "zone_hit")):
                paired[name][metric] = _paired_ci(
                    variants[name], by_persistence, metric,
                    reps=args.bootstrap, seed=args.seed + 100 * hi + 10 * mi + (name == "identity_velocity"))
        metrics_by_type = {}
        for kind in sorted(set(d["types"].tolist())):
            mask = d["types"] == kind
            subset = {key: value[mask] for key, value in d.items()
                      if isinstance(value, np.ndarray) and len(value) == len(mask)}
            model_scores = _game_side_scores(subset, pred[h][mask])
            persist_scores = _game_side_scores(subset, np.zeros_like(subset["y"]))
            metrics_by_type[kind] = {
                "rows": int(mask.sum()),
                "behavior_conditioned": {metric: _aggregate(model_scores, metric)
                                         for metric in ("mae_m", "euclidean_m", "zone_hit")},
                "persistence": {metric: _aggregate(persist_scores, metric)
                                for metric in ("mae_m", "euclidean_m", "zone_hit")},
            }
        report["horizons"][str(h)] = {
            "samples": int(len(d["y"])),
            "games": int(len(np.unique(d["games"]))),
            "metrics_game_equal_red_blue_equal": metrics,
            "paired_game_bootstrap_delta_vs_persistence": paired,
            "metrics_by_robot_type": metrics_by_type,
        }
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    np.savez(out.with_suffix(".npz"), **{
        f"h{h}_{key}": value
        for h in HORIZONS for key, value in _prediction_arrays(
            entities, targets[h], pred[h], h).items()
    })
    print(json.dumps({"output": str(out), "horizons": report["horizons"]},
                     ensure_ascii=False), flush=True)


def _prediction_arrays(entities, target, prediction, horizon):
    idx = target["indices"]
    return {
        "games": entities["games"][idx], "sides": entities["sides"][idx],
        "types": entities["types"][idx], "ids": entities["ids"][idx],
        "truth_delta": target["y"],
        "persistence_delta": np.zeros_like(target["y"]),
        "identity_velocity_delta": entities["v"][idx] * float(horizon),
        "behavior_conditioned_delta": prediction,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-npz", default="out_m2/temporal_forecast_canonical_side/train.npz")
    parser.add_argument("--train-jsonl", default="out_m2/forecast_dataset/train.jsonl.gz")
    parser.add_argument("--output", default="out_m2/identity_conditioned_motion_oof/report.json")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260927)
    parser.add_argument("--rebuild-cache", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
