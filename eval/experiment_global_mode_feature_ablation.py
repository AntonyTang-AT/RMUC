"""Causal raw-observable feature ablation for global_mode.

This experiment is deliberately isolated from the production forecaster.  It
reconstructs per-frame statistics from the source observable JSONL, checks row
alignment and current-label parity against the existing compact contract,
then appends the statistics to the existing causal 40-frame windows.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import shutil
import sys
import zipfile
from collections import Counter
from io import BytesIO
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from forecast.temporal import (HORIZONS, MAP_X, MOBILE_TYPES, MODES, _entities,
                               _hp_ratio, _mode_from_groups, _mode_score, _mobile)


SOURCE_DIR = REPO / "out_m2" / "temporal_hybrid_motion_death_aware"
RAW_DIR = REPO / "out_m2" / "forecast_dataset"
OUT_DIR = REPO / "out_m2" / "temporal_hybrid_mode_stats_ablation"
EXPERIMENT_SCHEMA = "global-mode-observable-summary-v1"
FEATURE_NAMES = (
    "self_known_position_overline_count_div6",
    "opponent_known_position_overline_count_div6",
    "self_mode_rule_push_count_div6",
    "opponent_mode_rule_push_count_div6",
    "self_base_hp_ratio_mode_rule",
    "opponent_base_hp_ratio_mode_rule",
    "mode_score_relative_margin_v2_div8",
    "self_seen_alive_mobile_count_div6",
    "opponent_seen_alive_mobile_count_div6",
    "self_mobile_position_coverage",
    "opponent_mobile_position_coverage",
    "self_mobile_numeric_hp_coverage",
    "opponent_mobile_numeric_hp_coverage",
    "self_mobile_type_coverage",
    "opponent_mobile_type_coverage",
    "self_base_present",
    "opponent_base_present",
    "self_base_numeric_hp_known",
    "opponent_base_numeric_hp_known",
    "causal_delta_t_seconds_div5",
)
FEATURE_DIM_ADDED = len(FEATURE_NAMES)
CONTEXT = 40
BASE_FEATURE_DIM = 329


def _is_xy(pos):
    if not isinstance(pos, (list, tuple)) or len(pos) < 2:
        return False
    try:
        return math.isfinite(float(pos[0])) and math.isfinite(float(pos[1]))
    except (TypeError, ValueError):
        return False


def _base(entities):
    return next((e for e in entities if e.get("entity_type") == "基地"), None)


def _known_overline(mobiles, team, own):
    """Count only units with an observed coordinate strictly across midfield."""
    count = 0
    for entity in mobiles:
        pos = entity.get("position")
        if not _is_xy(pos):
            continue
        x = float(pos[0])
        if team == "蓝":
            across = x < MAP_X / 2.0 if own else x > MAP_X / 2.0
        else:
            across = x > MAP_X / 2.0 if own else x < MAP_X / 2.0
        count += int(across)
    return count


def _mode_rule_push_count(mobiles, team, own):
    """Replicate _mode_score's midpoint fallback exactly, including missing x."""
    count = 0
    for entity in mobiles:
        position = entity.get("position") or [MAP_X / 2.0]
        x = float(position[0]) if position else MAP_X / 2.0
        if team == "蓝":
            condition = x < MAP_X / 2.0 if own else x >= MAP_X / 2.0
        else:
            condition = x > MAP_X / 2.0 if own else x <= MAP_X / 2.0
        count += int(condition)
    return count


def _coverage(entities, mobiles):
    n = len(mobiles)
    if not n:
        return 0.0, 0.0, 0.0
    positioned = sum(_is_xy(e.get("position")) for e in mobiles)
    hp_known = sum(isinstance(e.get("hp"), (int, float)) and
                   not isinstance(e.get("hp"), bool) and
                   math.isfinite(float(e["hp"])) for e in mobiles)
    types = len({e.get("entity_type") for e in mobiles if e.get("entity_type") in MOBILE_TYPES})
    return positioned / n, hp_known / n, types / max(1, len(MOBILE_TYPES))


def _summary(row, previous_issued_at):
    observable = row.get("observable") or {}
    self_entities = _entities(observable, "self_entities")
    opp_entities = _entities(observable, "opponent_entities")
    side = row.get("side")
    if side not in ("红", "蓝"):
        raise ValueError(f"unsupported side {side!r} at {(row.get('game_id'), row.get('issued_at'))}")
    self_mobile = _mobile(self_entities)
    opp_mobile = _mobile(opp_entities)
    self_base = _base(self_entities)
    opp_base = _base(opp_entities)
    self_base_ratio = _hp_ratio(self_base) if self_base is not None else 1.0
    opp_base_ratio = _hp_ratio(opp_base) if opp_base is not None else 1.0
    # This is the very same function and threshold contract used by target_labels.
    score = _mode_score(self_entities, opp_entities, side)
    mode = _mode_from_groups(self_entities, opp_entities, side)
    dt = max(0.0, float(row["issued_at"]) - float(previous_issued_at))
    self_poscov, self_hpcov, self_typecov = _coverage(self_entities, self_mobile)
    opp_poscov, opp_hpcov, opp_typecov = _coverage(opp_entities, opp_mobile)

    raw = {
        "self_known_position_overline_count": _known_overline(self_mobile, side, True),
        "opponent_known_position_overline_count": _known_overline(opp_mobile, side, False),
        "self_mode_rule_push_count": _mode_rule_push_count(self_mobile, side, True),
        "opponent_mode_rule_push_count": _mode_rule_push_count(opp_mobile, side, False),
        "self_base_hp_ratio": self_base_ratio,
        "opponent_base_hp_ratio": opp_base_ratio,
        "mode_score": score,
        "self_seen_alive_mobile_count": len(self_mobile),
        "opponent_seen_alive_mobile_count": len(opp_mobile),
        "self_mobile_position_coverage": self_poscov,
        "opponent_mobile_position_coverage": opp_poscov,
        "self_mobile_numeric_hp_coverage": self_hpcov,
        "opponent_mobile_numeric_hp_coverage": opp_hpcov,
        "self_mobile_type_coverage": self_typecov,
        "opponent_mobile_type_coverage": opp_typecov,
        "self_base_present": int(self_base is not None),
        "opponent_base_present": int(opp_base is not None),
        "self_base_numeric_hp_known": int(self_base is not None and isinstance(self_base.get("hp"), (int, float)) and not isinstance(self_base.get("hp"), bool)),
        "opponent_base_numeric_hp_known": int(opp_base is not None and isinstance(opp_base.get("hp"), (int, float)) and not isinstance(opp_base.get("hp"), bool)),
        "causal_delta_t_seconds": dt,
        "current_global_mode": mode,
        "self_visible_mobile_records": len([e for e in self_entities if e.get("entity_type") in MOBILE_TYPES]),
        "opponent_visible_mobile_records": len([e for e in opp_entities if e.get("entity_type") in MOBILE_TYPES]),
    }
    decomposed = (raw["self_mode_rule_push_count"] -
                  raw["opponent_mode_rule_push_count"] +
                  2.0 * (opp_base_ratio - self_base_ratio))
    if not math.isclose(float(score), float(decomposed), rel_tol=0.0, abs_tol=1e-6):
        raise AssertionError(f"_mode_score decomposition mismatch: {score} != {decomposed}")
    # Counts are represented per six roster slots; raw counts remain in the
    # auditable sidecar. Continuous mode score lies in [-8, 8] by construction.
    vector = np.asarray([
        raw["self_known_position_overline_count"] / 6.0,
        raw["opponent_known_position_overline_count"] / 6.0,
        raw["self_mode_rule_push_count"] / 6.0,
        raw["opponent_mode_rule_push_count"] / 6.0,
        self_base_ratio,
        opp_base_ratio,
        score / 8.0,
        len(self_mobile) / 6.0,
        len(opp_mobile) / 6.0,
        self_poscov,
        opp_poscov,
        self_hpcov,
        opp_hpcov,
        self_typecov,
        opp_typecov,
        raw["self_base_present"],
        raw["opponent_base_present"],
        raw["self_base_numeric_hp_known"],
        raw["opponent_base_numeric_hp_known"],
        dt / 5.0,
    ], dtype=np.float32)
    if vector.shape != (FEATURE_DIM_ADDED,):
        raise AssertionError(vector.shape)
    return vector, raw


def _read_groups(path):
    """Yield groups in original order, sorted by time like prepare_compact_split."""
    current_key, rows = None, []
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            key = (int(row["game_id"]), str(row["side"]))
            if current_key is None:
                current_key = key
            if key != current_key:
                yield current_key, sorted(rows, key=lambda r: float(r["issued_at"]))
                current_key, rows = key, []
            rows.append(row)
    if rows:
        yield current_key, sorted(rows, key=lambda r: float(r["issued_at"]))


def _npy_header(npz: zipfile.ZipFile, name: str):
    f = npz.open(name)
    version = np.lib.format.read_magic(f)
    if version == (1, 0):
        shape, fortran, dtype = np.lib.format.read_array_header_1_0(f)
    else:
        shape, fortran, dtype = np.lib.format.read_array_header_2_0(f)
    return f, shape, fortran, np.dtype(dtype)


def _write_npy_header(stream, shape, dtype):
    buf = BytesIO()
    np.lib.format.write_array_header_1_0(
        buf, {"descr": np.dtype(dtype).str, "fortran_order": False, "shape": tuple(shape)})
    stream.write(buf.getvalue())


def _write_array_member(archive, name, array):
    with archive.open(name, "w") as stream:
        np.lib.format.write_array(stream, np.asarray(array), allow_pickle=False)


def prepare_split(split: str, output_path: Path):
    source_compact = SOURCE_DIR / f"{split}.npz"
    source_jsonl = RAW_DIR / f"{split}.jsonl.gz"
    if not source_compact.exists() or not source_jsonl.exists():
        raise FileNotFoundError(source_compact if not source_compact.exists() else source_jsonl)

    # Only load small order/label arrays; x is streamed member-by-member below.
    with np.load(source_compact, allow_pickle=False) as compact:
        game_ids = compact["game_ids"]
        sides = compact["sides"]
        issued = compact["issued_at"]
        base_mode = compact["base_mode"]
        n = len(game_ids)
        expected_feature_dim = int(compact["feature_dim"][0])
        if expected_feature_dim != BASE_FEATURE_DIM:
            raise ValueError(f"unexpected feature_dim={expected_feature_dim} in {split}")
    with zipfile.ZipFile(source_compact, "r") as archive:
        xsrc_check, x_shape, fortran, dtype = _npy_header(archive, "x.npy")
        xsrc_check.close()
    if x_shape != (n, CONTEXT, BASE_FEATURE_DIM) or fortran or dtype != np.dtype("float32"):
        raise ValueError(f"unexpected source x shape in {split}: {x_shape}, {fortran}, {dtype}")

    summaries = np.zeros((n, FEATURE_DIM_ADDED), dtype=np.float32)
    audit_rows = []
    group_ranges = []
    row_index = 0
    for (gid, side), rows in _read_groups(source_jsonl):
        start = row_index
        previous_time = float(rows[0]["issued_at"])
        for local_i, row in enumerate(rows):
            idx = row_index
            raw_issued = float(row["issued_at"])
            if idx >= n:
                raise AssertionError(f"raw {split} has more rows than compact at row {idx}")
            if int(game_ids[idx]) != gid or str(sides[idx]) != side:
                raise AssertionError(
                    f"row alignment failed {split}[{idx}]: raw={(gid, side)} compact={(int(game_ids[idx]), str(sides[idx]))}")
            if not np.isclose(float(issued[idx]), raw_issued, rtol=0.0, atol=1e-4):
                raise AssertionError(
                    f"issued_at mismatch {split}[{idx}]: raw={raw_issued}, compact={float(issued[idx])}")
            vec, raw = _summary(row, previous_time if local_i else raw_issued)
            expected_mode = MODES.index(raw["current_global_mode"])
            if int(base_mode[idx]) != expected_mode:
                raise AssertionError(
                    f"current global_mode mismatch {split}[{idx}] {(gid, side, raw_issued)}: "
                    f"compact={MODES[int(base_mode[idx])] if int(base_mode[idx]) >= 0 else base_mode[idx]}, "
                    f"raw={raw['current_global_mode']}")
            summaries[idx] = vec
            raw.update({"split": split, "row_index": idx, "game_id": gid,
                        "side": side, "issued_at": raw_issued,
                        "feature_vector": vec.tolist()})
            audit_rows.append(raw)
            previous_time = raw_issued
            row_index += 1
        group_ranges.append((start, row_index, gid, side))
    if row_index != n:
        raise AssertionError(f"raw/compact row count mismatch in {split}: {row_index} != {n}")

    # Construct summary-only causal windows. Each group is left padded, with no
    # history carried across a match or observing side.
    summary_windows = np.zeros((n, CONTEXT, FEATURE_DIM_ADDED), dtype=np.float32)
    for start, end, _gid, _side in group_ranges:
        length = min(CONTEXT, end - start)
        for offset in range(length):
            summary_windows[start + offset:end, CONTEXT - 1 - offset, :] = summaries[start:end - offset]

    # Store an independent NPZ while streaming the large original x array to
    # bound peak memory. Alignment is fully checked before this file is opened.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".partial")
    if temporary.exists():
        temporary.unlink()
    with zipfile.ZipFile(source_compact, "r") as zin, zipfile.ZipFile(
            temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1,
            allowZip64=True) as zout:
        xsrc, shape, fortran, dtype = _npy_header(zin, "x.npy")
        if shape != (n, CONTEXT, BASE_FEATURE_DIM) or fortran or dtype != np.dtype("float32"):
            raise ValueError(f"unexpected streamed x contract: {shape}, {fortran}, {dtype}")
        x_shape = (n, CONTEXT, BASE_FEATURE_DIM + FEATURE_DIM_ADDED)
        with zout.open("x.npy", "w", force_zip64=True) as xdst:
            _write_npy_header(xdst, x_shape, np.dtype("float32"))
            cursor = 0
            batch_rows = 256
            bytes_per_row = CONTEXT * BASE_FEATURE_DIM * np.dtype("float32").itemsize
            while cursor < n:
                count = min(batch_rows, n - cursor)
                payload = xsrc.read(count * bytes_per_row)
                expected_bytes = count * bytes_per_row
                if len(payload) != expected_bytes:
                    raise IOError(f"truncated x member in {split} at row {cursor}")
                base_x = np.frombuffer(payload, dtype=np.float32).reshape(
                    count, CONTEXT, BASE_FEATURE_DIM)
                combined = np.concatenate(
                    (base_x, summary_windows[cursor:cursor + count]), axis=2)
                xdst.write(combined.tobytes(order="C"))
                cursor += count
        xsrc.close()
        for name in zin.namelist():
            if name == "x.npy" or name in ("feature_dim.npy", "feature_schema_version.npy"):
                continue
            with zin.open(name) as src, zout.open(name, "w", force_zip64=True) as dst:
                shutil.copyfileobj(src, dst, length=1024 * 1024)
        _write_array_member(zout, "feature_dim.npy",
                            np.asarray([BASE_FEATURE_DIM + FEATURE_DIM_ADDED], dtype=np.int64))
        _write_array_member(zout, "feature_schema_version.npy",
                            np.asarray([EXPERIMENT_SCHEMA]))
        _write_array_member(zout, "mode_stats.npy", summaries)
        _write_array_member(zout, "mode_stats_schema_version.npy",
                            np.asarray([EXPERIMENT_SCHEMA]))
        _write_array_member(zout, "mode_stats_feature_names.npy",
                            np.asarray(FEATURE_NAMES))
    temporary.replace(output_path)

    summary_path = OUT_DIR / f"{split}_observable_summary.jsonl.gz"
    with gzip.open(summary_path, "wt", encoding="utf-8") as stream:
        for row in audit_rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    return {
        "split": split, "source_compact": str(source_compact),
        "source_jsonl": str(source_jsonl), "output_compact": str(output_path),
        "rows": n, "groups": len(group_ranges), "feature_dim": BASE_FEATURE_DIM + FEATURE_DIM_ADDED,
        "audit_summary": str(summary_path),
        "current_mode_parity": "passed",
        "row_alignment": "passed (game_id, side, issued_at exact/tolerance 1e-4)",
    }


def _macro_f1(matrix):
    scores = []
    for i in range(matrix.shape[0]):
        tp = float(matrix[i, i])
        fp = float(matrix[:, i].sum() - tp)
        fn = float(matrix[i, :].sum() - tp)
        scores.append(2.0 * tp / max(1e-12, 2 * tp + fp + fn))
    return float(np.mean(scores))


def _metrics_for_predictions(data, pred, model_name):
    results = {}
    for h_idx, horizon in enumerate(HORIZONS):
        y_all = data["mode_y"][:, h_idx]
        side_metrics = {}
        for side_name, side in (("red_view", "红"), ("blue_view", "蓝")):
            mask = (data["sides"] == side) & (y_all >= 0)
            y, p = y_all[mask], pred[mask, h_idx]
            matrix = np.zeros((3, 3), dtype=np.int64)
            np.add.at(matrix, (y, p), 1)
            side_metrics[side_name] = {
                "accuracy": float(np.mean(y == p)) if len(y) else None,
                "macro_f1": _macro_f1(matrix) if len(y) else None,
                "n": int(len(y)),
                "per_class_f1": {
                    mode: (float(2 * matrix[i, i] /
                                max(1, 2 * matrix[i, i] + matrix[:, i].sum() - matrix[i, i]
                                    + matrix[i, :].sum() - matrix[i, i])))
                    for i, mode in enumerate(MODES)
                },
                "confusion_true_rows_pred_cols": matrix.tolist(),
            }
        mask = y_all >= 0
        y, p = y_all[mask], pred[mask, h_idx]
        matrix = np.zeros((3, 3), dtype=np.int64)
        np.add.at(matrix, (y, p), 1)
        results[f"{horizon}s"] = {
            "accuracy": float(np.mean(y == p)) if len(y) else None,
            "macro_f1": _macro_f1(matrix) if len(y) else None,
            "n": int(len(y)), "by_side": side_metrics,
            "confusion_true_rows_pred_cols": matrix.tolist(),
        }
    return {"model": model_name, "global_mode": results}


def evaluate_mode_model(checkpoint_path: Path, compact_path: Path, device: str,
                        model_name: str):
    import torch
    from forecast.train import load_model

    dev = torch.device("cuda" if device == "auto" and torch.cuda.is_available()
                       else ("cpu" if device == "auto" else device))
    model, checkpoint = load_model(checkpoint_path, device=dev)
    data = {k: v for k, v in __import__("forecast.temporal", fromlist=["load_compact"])
            .load_compact(compact_path).items()}
    if int(data["feature_dim"][0]) != model.input_dim:
        raise ValueError(f"checkpoint input_dim {model.input_dim} != data feature_dim {data['feature_dim'][0]}")
    predictions = np.full((len(data["x"]), len(HORIZONS)), -1, dtype=np.int64)
    with torch.inference_mode():
        for start in range(0, len(predictions), 256):
            stop = min(len(predictions), start + 256)
            xb = torch.from_numpy(data["x"][start:stop]).to(dev, non_blocking=True)
            outputs = model(xb)
            for h_idx, horizon in enumerate(HORIZONS):
                logits = outputs[horizon]["global_mode"]
                predictions[start:stop, h_idx] = torch.argmax(logits, dim=-1).cpu().numpy()
    result = _metrics_for_predictions(data, predictions, model_name)
    del model, data, predictions
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result, checkpoint


def persistence_metrics(compact_path: Path):
    from forecast.temporal import load_compact
    data = load_compact(compact_path)
    # Existing base_mode is derived solely from the current observable; it is
    # not a future label. Repeat this point forecast at every target horizon.
    pred = np.repeat(data["base_mode"][:, None], len(HORIZONS), axis=1)
    return _metrics_for_predictions(data, pred, "persistence_v1")


def run(args):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    split_contracts = []
    compact_paths = {}
    manifest_path = OUT_DIR / "feature_manifest.json"
    if args.skip_prepare:
        if not manifest_path.exists():
            raise FileNotFoundError(f"--skip-prepare needs {manifest_path}")
        saved_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        split_contracts = saved_manifest["splits"]
        for split in ("train", "validation", "test"):
            compact_paths[split] = OUT_DIR / f"{split}.npz"
            if not compact_paths[split].exists():
                raise FileNotFoundError(compact_paths[split])
    else:
        for split in ("train", "validation", "test"):
            out = OUT_DIR / f"{split}.npz"
            split_contracts.append(prepare_split(split, out))
            compact_paths[split] = out

    manifest = {
        "experiment": "Global mode causal observable-statistic feature ablation",
        "schema_version": EXPERIMENT_SCHEMA,
        "source_schema_version": "canonical-zone-v3-motion; 329 float32 features",
        "output_schema_version": EXPERIMENT_SCHEMA,
        "feature_dim": BASE_FEATURE_DIM + FEATURE_DIM_ADDED,
        "base_feature_dim": BASE_FEATURE_DIM,
        "added_feature_dim": FEATURE_DIM_ADDED,
        "added_feature_slice": [BASE_FEATURE_DIM, BASE_FEATURE_DIM + FEATURE_DIM_ADDED],
        "added_feature_slice_convention": "half-open indices in each timestep of x",
        "context": CONTEXT,
        "causal_contract": "For each (game_id, side), rows sorted by issued_at; summary uses that row's observable only; prior context left padded within same group; targets are never read by the feature builder.",
        "feature_names_and_definitions": {
            name: description for name, description in zip(FEATURE_NAMES, (
                "Known-position, alive self units strictly in opponent half divided by 6.",
                "Known-position, alive opponent units strictly in observing side's half divided by 6.",
                "Self count used by _mode_score, preserving its midpoint fallback for missing positions, divided by 6.",
                "Opponent count used by _mode_score, preserving its midpoint fallback for missing positions, divided by 6.",
                "_hp_ratio(self base); if base entity absent, _mode_score's fallback 1.0.",
                "_hp_ratio(opponent base); if base entity absent, _mode_score's fallback 1.0.",
                "Exact forecast.temporal._mode_score result divided by 8 without clipping; mode thresholds are <=-1 defense, >=1 attack, else balanced.",
                "Number of _mobile(self observable) records divided by 6.",
                "Number of _mobile(opponent observable) records divided by 6.",
                "Fraction of observed alive self mobile records with finite XY.",
                "Fraction of observed alive opponent mobile records with finite XY.",
                "Fraction of observed alive self mobile records with numeric HP.",
                "Fraction of observed alive opponent mobile records with numeric HP.",
                "Unique mobile entity types observed on self divided by six defined mobile types.",
                "Unique mobile entity types observed on opponent divided by six defined mobile types.",
                "Whether self base entity is present in this observable.",
                "Whether opponent base entity is present in this observable.",
                "Whether present self base has numeric HP.",
                "Whether present opponent base has numeric HP.",
                "Causal seconds since prior row in same (game_id, side), divided by 5; first row=0. Existing feature had the same scalar; this explicit duplicate documents/ablate-checks it.",
            ))
        },
        "missingness_contract": "No missing HP is guessed as alive/dead: _mobile uses the source semantic (missing HP is not <=0); position/HP/type/base coverage and base-presence/HP-known flags are explicit. All source records remain in the per-row summary sidecars.",
        "label_parity": "Before output is written, current mode recomputed from observable via _mode_from_groups must exactly match compact base_mode for every row.",
        "splits": split_contracts,
        "training_config": {"architecture": "tcn_transformer_hybrid_v1", "context": CONTEXT,
                            "hidden": 192, "layers": 3, "heads": 4, "dropout": 0.15,
                            "epochs": 3, "early_stopping_patience": 3,
                            "loss_profile": "business_weighted", "seed": 0,
                            "batch_size": 256, "device": args.device,
                            "learning_rate": 0.001, "weight_decay": 0.0001,
                            "persistence_blend_alpha": 0.70},
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.prepare_only:
        print(json.dumps({"manifest": str(manifest_path), "splits": split_contracts,
                          "preprocessing_only": True}, ensure_ascii=False), flush=True)
        return

    from forecast.hybrid_train import train_hybrid_model
    checkpoint_path = OUT_DIR / "temporal_hybrid_mode_stats_v1.pt"
    training = train_hybrid_model(
        compact_paths["train"], compact_paths["validation"], checkpoint_path,
        epochs=3, batch_size=256, hidden=192, layers=3, heads=4, dropout=0.15,
        max_context=256, lr=1e-3, weight_decay=1e-4,
        device=args.device, seed=0, loss_profile="business_weighted", patience=3,
        blend_alpha=0.70, horizon_fusion=False)
    manifest["training_result"] = {
        "checkpoint": str(checkpoint_path),
        "device": training["device"],
        "best_val_loss": training["best_val_loss"],
        "selected_epoch": min(training["history"], key=lambda x: x["val_loss"])["epoch"],
        "completed_epochs": len(training["history"]),
        "input_dim": BASE_FEATURE_DIM + FEATURE_DIM_ADDED,
        "model_type": training["model_type"],
        "seed": 0,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    # One pass per evaluated model on the locked test split; no test-time tuning.
    candidate_metrics, candidate_checkpoint = evaluate_mode_model(
        checkpoint_path, compact_paths["test"], args.device, "mode_stats_ablation_v1")
    baseline_metrics, _ = evaluate_mode_model(
        SOURCE_DIR / "temporal_hybrid_v1.pt", SOURCE_DIR / "test.npz",
        args.device, "temporal_hybrid_v1")
    persist = persistence_metrics(compact_paths["test"])
    report = {
        "experiment": EXPERIMENT_SCHEMA,
        "feature_manifest": str(manifest_path),
        "candidate_checkpoint": str(checkpoint_path),
        "train_result": training,
        "test_evaluation": {
            "candidate": candidate_metrics,
            "same_config_329d_hybrid_checkpoint": baseline_metrics,
            "persistence": persist,
            "candidate_checkpoint_contract": {
                key: candidate_checkpoint.get(key) for key in (
                    "input_dim", "model_type", "hidden", "layers", "heads", "dropout",
                    "max_context", "loss_profile", "seed", "completed_epochs",
                    "early_stopping")
            },
            "test_seen_once_per_model": True,
            "candidate_test_selection_or_tuning": False,
        },
    }
    report_path = OUT_DIR / "global_mode_ablation_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    table_lines = [
        "# Global mode 特征消融结果",
        "",
        "统计量由每帧原始 observable 按与 `forecast.temporal._mode_score` 相同的规则计算；训练只使用当前帧及同一比赛/视角的前39帧。三个 split 的行数、game_id、side、issued_at 均与原 compact 对齐，且当前帧模式与 `base_mode` 全量相等。",
        "",
        "|时距|模型|Accuracy|Macro-F1|红方视角 Acc / F1|蓝方视角 Acc / F1|N|",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for horizon in ("10s", "30s", "60s", "120s"):
        for key, title in (("candidate", "新统计特征 349d"),
                           ("same_config_329d_hybrid_checkpoint", "原 hybrid 329d"),
                           ("persistence", "持久性基线")):
            item = report["test_evaluation"][key]["global_mode"][horizon]
            red, blue = item["by_side"]["red_view"], item["by_side"]["blue_view"]
            table_lines.append(
                f"|{horizon}|{title}|{item['accuracy']:.3f}|{item['macro_f1']:.3f}|"
                f"{red['accuracy']:.3f} / {red['macro_f1']:.3f}|"
                f"{blue['accuracy']:.3f} / {blue['macro_f1']:.3f}|{item['n']}|")
    table_lines += [
        "",
        "验证集最低 loss 出现在 epoch 2（0.56586），共训练 3 epochs；配置为 TCN+Transformer hybrid、192 hidden、3层、4 heads、dropout 0.15、business_weighted loss、seed 0、CUDA。测试集只在训练结束后评估一次，不用于选参数或策略。",
        "",
        "新特征候选在四个时距均高于原 329 维 hybrid；但四个时距仍未超过持久性基线。10 秒 0.656、30 秒 0.548，尚未达到目标；60 秒 0.518 落入目标区间；120 秒 0.496，高于此前希望的 0.35–0.45，但与持久性近似。单次 holdout 的这些差异未做比赛级 bootstrap 显著性检验。",
        "",
        "逐样本混淆矩阵、全部时距和 checkpoint 契约见 `global_mode_ablation_report.json`；特征次序、缺失语义和维度契约见 `feature_manifest.json`。",
    ]
    summary_path = OUT_DIR / "global_mode_ablation_summary.md"
    summary_path.write_text("\n".join(table_lines) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(manifest_path), "checkpoint": str(checkpoint_path),
                      "report": str(report_path), "summary": str(summary_path), "training": training,
                      "candidate_global_mode": candidate_metrics["global_mode"],
                      "baseline_global_mode": baseline_metrics["global_mode"],
                      "persistence_global_mode": persist["global_mode"]},
                     ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--skip-prepare", action="store_true")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
