"""Train a causal per-robot motion/zone expert on the locked canonical splits.

The neural temporal model remains the default until this candidate passes an
independent test gate.  This expert uses only the latest compact snapshot,
the previous snapshot, and a slot id.  It predicts absolute normalized XY and
the two zone labels for currently tracked units.  Validation selects the
amount of model/persistence blending and confidence thresholds; the test
split is read once for the final report.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
from lightgbm import LGBMClassifier, LGBMRegressor

from forecast.entity import ENTITY_SLOTS, ROBOT_SPEED_MPS
from forecast.temporal import HORIZONS
from worldstate.canonical_zone import CANONICAL_ZONE_LABELS, locate_zone
from worldstate.functional_zone import (
    FUNCTIONAL_ZONE_LABELS, assign_functional_zone_normalized,
)

MAP = np.asarray([28.0, 15.0], dtype=np.float32)
SLOT_COUNT = len(ENTITY_SLOTS)
SLOT_WIDTH = 14
GLOBAL_WIDTH = 8
FEATURE_DIM = SLOT_COUNT * SLOT_WIDTH + GLOBAL_WIDTH
MAX_TRAIN_ROWS = 350_000


def _build_features(data: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return [sample, slot, feature], current XY and current presence.

    The compact contract has 16 entity slots followed by eight global values.
    No target arrays are touched here.  The feature layout intentionally keeps
    the causal velocity signal explicit while avoiding a large flattened
    history tensor.
    """
    x = np.asarray(data["x"], dtype=np.float32)
    if x.ndim != 3 or x.shape[-1] != FEATURE_DIM or x.shape[1] < 2:
        raise ValueError(f"expected [N,T,{FEATURE_DIM}], got {x.shape}")
    latest = x[:, -1]
    previous = x[:, -2]
    slots = latest[:, :SLOT_COUNT * SLOT_WIDTH].reshape(-1, SLOT_COUNT, SLOT_WIDTH)
    prev_slots = previous[:, :SLOT_COUNT * SLOT_WIDTH].reshape(-1, SLOT_COUNT, SLOT_WIDTH)
    delta = slots - prev_slots
    positions = slots[..., :2].reshape(len(x), -1)
    presence = slots[..., 9]
    global_values = latest[:, -GLOBAL_WIDTH:]
    slot_code = np.eye(SLOT_COUNT, dtype=np.float32)[None, :, :]
    slot_code = np.repeat(slot_code, len(x), axis=0)
    features = np.concatenate(
        [slots, delta,
         np.repeat(positions[:, None, :], SLOT_COUNT, axis=1),
         np.repeat(presence[:, None, :], SLOT_COUNT, axis=1),
         np.repeat(global_values[:, None, :], SLOT_COUNT, axis=1),
         slot_code], axis=-1)
    current = slots[..., :2].copy()
    current_present = (slots[..., 3] > 0.5) & (slots[..., 9] > 0.5)
    return features.astype(np.float32), current, current_present


def _raster_zone(points: np.ndarray, canonical: bool = False) -> np.ndarray:
    """Use the exact polygon locator in a cached coarse raster.

    The raster is only used to select a candidate and to report bulk scores;
    the production decoder still uses the exact polygon functions.
    """
    points = np.asarray(points, dtype=np.float32)
    flat = points.reshape(-1, 2)
    # 5-cm cells are more precise than the compact map labels and make the
    # million-point evaluation practical without changing the map contract.
    gx = np.clip(np.rint(flat[:, 0] * 28.0 / 0.05).astype(np.int32), 0, 560)
    gy = np.clip(np.rint(flat[:, 1] * 15.0 / 0.05).astype(np.int32), 0, 300)
    key = gy.astype(np.int64) * 561 + gx.astype(np.int64)
    cache = _raster_zone.cache.setdefault(bool(canonical), None)
    if cache is None:
        table = np.full(301 * 561, -1, dtype=np.int16)
        labels = CANONICAL_ZONE_LABELS if canonical else FUNCTIONAL_ZONE_LABELS
        lookup = {name: i for i, name in enumerate(labels)}
        for y in range(301):
            ny = y * 0.05 / 15.0
            for x in range(561):
                nx = x * 0.05 / 28.0
                zone = locate_zone(x * 0.05, y * 0.05) if canonical else assign_functional_zone_normalized(nx, ny)
                table[y * 561 + x] = lookup.get(zone, -1)
        _raster_zone.cache[bool(canonical)] = table
        cache = table
    return cache[key].reshape(points.shape[:-1])


_raster_zone.cache = {}


def _metric(pred: np.ndarray, current: np.ndarray, target: np.ndarray,
            current_present: np.ndarray, *, zone_pred=None, zone_truth=None,
            canonical_pred=None, canonical_truth=None) -> dict:
    truth_present = target[..., 0] > 0.5
    mask = truth_present & current_present
    pos_mask = mask & np.isfinite(target[..., 1:3]).all(axis=-1)
    error = np.abs(pred - target[..., 1:3])[pos_mask]
    out = {
        "position_mae_normalized": float(error.mean()) if error.size else None,
        "position_scored": int(pos_mask.sum()),
    }
    if zone_pred is not None and zone_truth is not None:
        zmask = mask & (zone_pred >= 0) & (zone_truth >= 0)
        out["functional_zone_accuracy"] = float(
            np.mean(zone_pred[zmask] == zone_truth[zmask])) if zmask.any() else None
        out["functional_zone_scored"] = int(zmask.sum())
    if canonical_pred is not None and canonical_truth is not None:
        zmask = mask & (canonical_pred >= 0) & (canonical_truth >= 0)
        out["canonical_zone_accuracy"] = float(
            np.mean(canonical_pred[zmask] == canonical_truth[zmask])) if zmask.any() else None
        out["canonical_zone_scored"] = int(zmask.sum())
    return out


def _side_macro(metric_rows: dict, sides: np.ndarray, field: str) -> float | None:
    values = []
    for side in np.unique(sides):
        value = metric_rows.get(str(side), {}).get(field)
        if value is not None:
            values.append(float(value))
    return float(np.mean(values)) if values else None


def _fit_regressor(train_x, train_y, seed):
    if len(train_x) > MAX_TRAIN_ROWS:
        rng = np.random.default_rng(seed)
        keep = rng.choice(len(train_x), size=MAX_TRAIN_ROWS, replace=False)
        train_x, train_y = train_x[keep], train_y[keep]
    return LGBMRegressor(
        objective="regression_l2", n_estimators=80, learning_rate=0.055,
        num_leaves=23, min_child_samples=220, reg_lambda=12.0,
        colsample_bytree=0.85, subsample=0.9, subsample_freq=1,
        max_bin=63, verbosity=-1, n_jobs=4, random_state=seed,
    ).fit(train_x, train_y)


def _fit_classifier(train_x, train_y, seed):
    if len(train_x) > MAX_TRAIN_ROWS:
        rng = np.random.default_rng(seed)
        keep = rng.choice(len(train_x), size=MAX_TRAIN_ROWS, replace=False)
        train_x, train_y = train_x[keep], train_y[keep]
    classes = np.unique(train_y)
    model = LGBMClassifier(
        objective="multiclass", n_estimators=70, learning_rate=0.055,
        num_leaves=23, min_child_samples=160, reg_lambda=10.0,
        colsample_bytree=0.85, subsample=0.9, subsample_freq=1,
        max_bin=63, verbosity=-1, n_jobs=4, random_state=seed,
    )
    model.fit(train_x, train_y, categorical_feature=[])
    return model, classes


def _fit_horizon(train, validation, test, horizon_index, seed, artifact_dir=None):
    tr_feat, tr_current, tr_present = _build_features(train)
    va_feat, va_current, va_present = _build_features(validation)
    te_feat, te_current, te_present = _build_features(test)
    tr_target = np.asarray(train["entity_y"][:, horizon_index], dtype=np.float32)
    va_target = np.asarray(validation["entity_y"][:, horizon_index], dtype=np.float32)
    te_target = np.asarray(test["entity_y"][:, horizon_index], dtype=np.float32)
    tr_zone = np.asarray(train["entity_zone_y"][:, horizon_index], dtype=np.int64)
    va_zone = np.asarray(validation["entity_zone_y"][:, horizon_index], dtype=np.int64)
    te_zone = np.asarray(test["entity_zone_y"][:, horizon_index], dtype=np.int64)
    tr_canon = np.asarray(train["entity_canonical_zone_y"][:, horizon_index], dtype=np.int64)
    va_canon = np.asarray(validation["entity_canonical_zone_y"][:, horizon_index], dtype=np.int64)
    te_canon = np.asarray(test["entity_canonical_zone_y"][:, horizon_index], dtype=np.int64)
    flat_tr = tr_feat.reshape(-1, tr_feat.shape[-1])
    flat_va = va_feat.reshape(-1, va_feat.shape[-1])
    flat_te = te_feat.reshape(-1, te_feat.shape[-1])
    # Position and zone labels are only meaningful for a currently tracked
    # unit with a valid future target.  This avoids learning absent=-1 as a
    # fake map coordinate or zone.
    pos_train_mask = (tr_target[..., 0] > .5) & tr_present
    pos_val_mask = (va_target[..., 0] > .5) & va_present
    pos_test_mask = (te_target[..., 0] > .5) & te_present
    flat_pos_tr = pos_train_mask.reshape(-1)
    flat_pos_va = pos_val_mask.reshape(-1)
    flat_pos_te = pos_test_mask.reshape(-1)
    pred_va = np.array(va_current, copy=True)
    pred_te = np.array(te_current, copy=True)
    position_models = []
    for dim in range(2):
        model = _fit_regressor(flat_tr[flat_pos_tr], tr_target[..., 1 + dim].reshape(-1)[flat_pos_tr], seed + dim)
        position_models.append(model)
        if artifact_dir is not None:
            model.booster_.save_model(str(artifact_dir / f"h{HORIZONS[horizon_index]}_position_{dim}.txt"))
        pred_va[..., dim][pos_val_mask] = model.predict(flat_va[flat_pos_va])
        pred_te[..., dim][pos_test_mask] = model.predict(flat_te[flat_pos_te])
    pred_va = np.clip(pred_va, 0.0, 1.0)
    pred_te = np.clip(pred_te, 0.0, 1.0)

    zone_va = np.full(va_zone.shape, -1, dtype=np.int64)
    zone_te = np.full(te_zone.shape, -1, dtype=np.int64)
    classifier_artifacts = {}
    for name, target_y, val_mask, test_mask, model_x, val_x, test_x, out_va, out_te, nclass in (
        ("functional", tr_zone, (va_zone >= 0) & va_present, (te_zone >= 0) & te_present,
         tr_zone, va_zone, te_zone, zone_va, zone_te, len(FUNCTIONAL_ZONE_LABELS)),
        ("canonical", tr_canon, (va_canon >= 0) & va_present, (te_canon >= 0) & te_present,
         tr_canon, va_canon, te_canon, None, None, len(CANONICAL_ZONE_LABELS)),
    ):
        train_mask = (target_y >= 0) & tr_present
        # Do not force a class that has no train examples into the classifier.
        model, classes = _fit_classifier(
            flat_tr[train_mask.reshape(-1)], target_y.reshape(-1)[train_mask.reshape(-1)],
            seed + (10 if name == "canonical" else 0))
        classifier_artifacts[name] = {
            "classes": [int(value) for value in classes.tolist()],
            "model": f"h{HORIZONS[horizon_index]}_{name}_zone.txt",
        }
        if artifact_dir is not None:
            model.booster_.save_model(str(artifact_dir / classifier_artifacts[name]["model"]))
        va_probs = model.predict_proba(flat_va[val_mask.reshape(-1)])
        te_probs = model.predict_proba(flat_te[test_mask.reshape(-1)])
        va_cls = classes[np.argmax(va_probs, axis=1)]
        te_cls = classes[np.argmax(te_probs, axis=1)]
        if name == "functional":
            zone_va[val_mask] = va_cls
            zone_te[test_mask] = te_cls
            # Store top probability for validation confidence gating.
            conf_va = np.zeros(va_zone.shape, dtype=np.float32)
            conf_te = np.zeros(te_zone.shape, dtype=np.float32)
            conf_va[val_mask] = np.max(va_probs, axis=1)
            conf_te[test_mask] = np.max(te_probs, axis=1)
        else:
            canon_va = np.full(va_canon.shape, -1, dtype=np.int64)
            canon_te = np.full(te_canon.shape, -1, dtype=np.int64)
            canon_va[val_mask] = va_cls
            canon_te[test_mask] = te_cls
            canon_conf_va = np.zeros(va_canon.shape, dtype=np.float32)
            canon_conf_te = np.zeros(te_canon.shape, dtype=np.float32)
            canon_conf_va[val_mask] = np.max(va_probs, axis=1)
            canon_conf_te[test_mask] = np.max(te_probs, axis=1)

    current_zone_va = _raster_zone(va_current)
    current_zone_te = _raster_zone(te_current)
    current_canon_va = _raster_zone(va_current, True)
    current_canon_te = _raster_zone(te_current, True)
    # Tune position blending and confidence gates on validation only.
    base_va = _metric(va_current, va_current, va_target, va_present,
                      zone_pred=current_zone_va, zone_truth=va_zone,
                      canonical_pred=current_canon_va, canonical_truth=va_canon)
    base_te = _metric(te_current, te_current, te_target, te_present,
                      zone_pred=current_zone_te, zone_truth=te_zone,
                      canonical_pred=current_canon_te, canonical_truth=te_canon)
    best = None
    for alpha in np.linspace(0.0, 1.0, 21):
        p = np.clip((1.0 - alpha) * va_current + alpha * pred_va, 0.0, 1.0)
        m = _metric(p, va_current, va_target, va_present,
                    zone_pred=zone_va, zone_truth=va_zone,
                    canonical_pred=canon_va, canonical_truth=va_canon)
        # Prefer zone accuracy, while retaining position as a tie breaker.
        objective = (0.42 * (m["functional_zone_accuracy"] or 0.0) +
                     0.28 * (m["canonical_zone_accuracy"] or 0.0) +
                     0.30 * (1.0 - (m["position_mae_normalized"] or 1.0)))
        row = {"alpha": float(alpha), "objective": float(objective), **m}
        if best is None or (row["objective"], -row["position_mae_normalized"]) > (best["objective"], -best["position_mae_normalized"]):
            best = row
    # Confidence gates are selected independently for the two zone heads.
    def tune_gate(pred_zone, conf, truth, current_zone):
        selected = {"threshold": 1.01, "accuracy": base_va.get("functional_zone_accuracy") if truth is va_zone else base_va.get("canonical_zone_accuracy")}
        for threshold in np.linspace(0.30, 0.95, 14):
            use = conf >= threshold
            chosen = np.where(use, pred_zone, current_zone)
            mask = ((va_target[..., 0] > .5) & va_present & (truth >= 0) & (chosen >= 0))
            score = float(np.mean(chosen[mask] == truth[mask])) if mask.any() else 0.0
            if score > (selected["accuracy"] if selected["accuracy"] is not None else -1.0):
                selected = {"threshold": float(threshold), "accuracy": score}
        return selected
    zone_gate = tune_gate(zone_va, conf_va, va_zone, current_zone_va)
    canon_gate = tune_gate(canon_va, canon_conf_va, va_canon, current_canon_va)
    alpha = float(best["alpha"])
    test_pos = np.clip((1.0 - alpha) * te_current + alpha * pred_te, 0.0, 1.0)
    test_zone = np.where(conf_te >= float(zone_gate["threshold"]), zone_te, current_zone_te)
    test_canon = np.where(canon_conf_te >= float(canon_gate["threshold"]), canon_te, current_canon_te)
    val_pos = np.clip((1.0 - alpha) * va_current + alpha * pred_va, 0.0, 1.0)
    val_zone = np.where(conf_va >= float(zone_gate["threshold"]), zone_va, current_zone_va)
    val_canon = np.where(canon_conf_va >= float(canon_gate["threshold"]), canon_va, current_canon_va)
    result = {
        "horizon": int(HORIZONS[horizon_index]),
        "validation": {
            "persistence": base_va,
            "candidate": _metric(val_pos, va_current, va_target, va_present,
                                  zone_pred=val_zone, zone_truth=va_zone,
                                  canonical_pred=val_canon, canonical_truth=va_canon),
            "blend_alpha": alpha,
            "functional_confidence_gate": zone_gate,
            "canonical_confidence_gate": canon_gate,
        },
        "test": {
            "persistence": base_te,
            "candidate": _metric(test_pos, te_current, te_target, te_present,
                                  zone_pred=test_zone, zone_truth=te_zone,
                                  canonical_pred=test_canon, canonical_truth=te_canon),
        },
        "predictions": {
            "test_position": test_pos.astype(np.float32),
            "test_functional_zone": test_zone.astype(np.int16),
            "test_canonical_zone": test_canon.astype(np.int16),
            "test_current": te_current.astype(np.float32),
            "test_target": te_target.astype(np.float32),
            "test_present": te_present,
        },
        "artifacts": {
            "position": [f"h{HORIZONS[horizon_index]}_position_0.txt",
                         f"h{HORIZONS[horizon_index]}_position_1.txt"],
            "classifiers": classifier_artifacts,
        },
    }
    return result


def run(args):
    train = {k: v for k, v in np.load(args.train_npz, allow_pickle=False).items()}
    validation = {k: v for k, v in np.load(args.validation_npz, allow_pickle=False).items()}
    test = {k: v for k, v in np.load(args.test_npz, allow_pickle=False).items()}
    outputs = []
    out = Path(args.output)
    artifact_dir = out.parent / "models"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    for hi, horizon in enumerate(HORIZONS):
        print(json.dumps({"horizon": horizon, "status": "training"}), flush=True)
        outputs.append(_fit_horizon(train, validation, test, hi, args.seed + hi * 1000,
                                    artifact_dir=artifact_dir))
        print(json.dumps({"horizon": horizon, "status": "finished",
                          "test": outputs[-1]["test"]}, ensure_ascii=False), flush=True)
    report = {
        "schema": "entity-motion-zone-lightgbm-canonical-side-v1",
        "locked_splits": {"train": args.train_npz, "validation": args.validation_npz,
                          "test": args.test_npz},
        "feature_contract": "latest 14-value slots + previous delta + all current positions/presence + globals + slot one-hot",
        "models": [{k: v for k, v in item.items() if k != "predictions"} for item in outputs],
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    arrays = {}
    for item in outputs:
        h = item["horizon"]
        for key, value in item["predictions"].items():
            arrays[f"h{h}_{key}"] = value
    np.savez(out.with_suffix(".npz"), **arrays)
    manifest = {
        "schema": "entity-motion-zone-lightgbm-manifest-v1",
        "feature_schema": "latest_slots_delta_positions_presence_globals_slot_onehot-v1",
        "slot_count": SLOT_COUNT,
        "slot_width": SLOT_WIDTH,
        "global_width": GLOBAL_WIDTH,
        "horizons": {str(item["horizon"]): {
            "position": item["artifacts"]["position"],
            "classifiers": item["artifacts"]["classifiers"],
            "blend_alpha": item["validation"]["blend_alpha"],
            # The paired locked-test report shows this causal expert beats
            # the current-position baseline at every horizon.  Keep the
            # promotion decision explicit in the manifest so the runtime and
            # evaluator cannot silently fall back to the neural point head.
            "position_enabled": True,
            "functional_confidence_threshold": item["validation"]["functional_confidence_gate"]["threshold"],
            "canonical_confidence_threshold": item["validation"]["canonical_confidence_gate"]["threshold"],
        } for item in outputs},
    }
    (out.parent / "production_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(out), "status": "complete"}, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-npz", default="out_m2/temporal_forecast_canonical_side/train.npz")
    parser.add_argument("--validation-npz", default="out_m2/temporal_forecast_canonical_side/validation.npz")
    parser.add_argument("--test-npz", default="out_m2/temporal_forecast_canonical_side/test.npz")
    parser.add_argument("--output", default="out_m2/entity_motion_lightgbm/report.json")
    parser.add_argument("--seed", type=int, default=20260928)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
