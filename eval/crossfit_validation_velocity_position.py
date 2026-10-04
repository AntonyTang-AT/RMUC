"""Validation-only causal group-centroid velocity decoder ablation.

Never opens a test split. The neural checkpoint stays frozen. Speed is built
only from issue-time observable XY summaries at the current and exact t-3s
snapshot; no target, yaw, or future frame is consulted by the velocity builder.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from forecast.entity import ENTITY_SLOTS, ENTITY_TYPES, ROBOT_SPEED_MPS
from forecast.temporal import HORIZONS, MAP_X, MAP_Y, _entities
from forecast.train import load_model
from worldstate.functional_zone import FUNCTIONAL_ZONE_LABELS, ZONE_POLYGONS

ROLES = ("self", "opponent")
WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)  # weight on neural model; 1=raw model


def alive(e):
    hp = e.get("hp")
    return e.get("life_state") != "destroyed" and not (
        isinstance(hp, (int, float)) and not isinstance(hp, bool) and hp <= 0)


def point(pos):
    if not isinstance(pos, (list, tuple)) or len(pos) < 2:
        return None
    try:
        x, y = float(pos[0]), float(pos[1])
    except (TypeError, ValueError):
        return None
    if not np.isfinite(x + y):
        return None
    return np.array([x, y], dtype=np.float32)


def group_centroids(row):
    obs = row.get("observable") or {}
    result = {}
    for role, group_name in (("self", "self_entities"), ("opponent", "opponent_entities")):
        ents = _entities(obs, group_name)
        for typ in ENTITY_TYPES:
            pts = [p for e in ents if e.get("entity_type") == typ and alive(e)
                   if (p := point(e.get("position"))) is not None]
            result[(role, typ)] = np.mean(pts, axis=0) if pts else None
    return result


def load_velocity(validation_jsonl, expected):
    """Align each NPZ row; velocity uses current and exact previous 3s observable centroids."""
    n = len(expected["game_ids"])
    current = np.full((n, len(ENTITY_SLOTS), 2), np.nan, dtype=np.float32)
    velocity = np.zeros_like(current)
    has_velocity = np.zeros((n, len(ENTITY_SLOTS)), dtype=bool)
    last = {}  # (game, side, role, type) -> (issued_at, centroid)
    cursor = 0
    with gzip.open(validation_jsonl, "rt", encoding="utf-8") as src:
        for line in src:
            if not line.strip():
                continue
            row = json.loads(line)
            if cursor >= n:
                raise ValueError("validation JSONL contains extra rows")
            gid, side, issued = int(row["game_id"]), str(row["side"]), float(row["issued_at"])
            if (gid != int(expected["game_ids"][cursor]) or side != str(expected["sides"][cursor])
                    or abs(issued - float(expected["issued_at"][cursor])) > 1e-4):
                raise ValueError(f"JSONL/compact validation order mismatch at row {cursor}: {(gid, side, issued)}")
            centroids = group_centroids(row)
            for i, (role, typ) in enumerate(ENTITY_SLOTS):
                now = centroids[(role, typ)]
                key = (gid, side, role, typ)
                if now is not None:
                    current[cursor, i] = now
                    previous = last.get(key)
                    if previous is not None and abs(issued - previous[0] - 3.0) <= 1e-4:
                        delta = (now - previous[1]) / 3.0
                        # Suppress implausible group-centroid jumps using the per-type
                        # max-speed prior while preserving the measured direction.
                        speed = float(np.linalg.norm(delta))
                        cap = float(ROBOT_SPEED_MPS.get(typ, .35))
                        if speed > cap and speed > 1e-8:
                            delta = delta * (cap / speed)
                        velocity[cursor, i] = delta
                        has_velocity[cursor, i] = True
                    last[key] = (issued, now)
                else:
                    last.pop(key, None)
            cursor += 1
    if cursor != n:
        raise ValueError(f"validation JSONL row count {cursor} != compact {n}")
    return current, velocity, has_velocity


def predict_frozen(checkpoint, x, device, batch_size):
    model, _ = load_model(checkpoint, device=device)
    model.eval()
    out = np.zeros((len(x), len(HORIZONS), len(ENTITY_SLOTS), 2), dtype=np.float32)
    dev = torch.device(device)
    with torch.inference_mode():
        for start in range(0, len(x), batch_size):
            xb = torch.from_numpy(x[start:start + batch_size]).to(dev)
            pred = model(xb)
            for j, h in enumerate(HORIZONS):
                out[start:start + len(xb), j] = torch.sigmoid(
                    pred[h]["entity_state"][..., 1:3]).cpu().numpy()
    return out


def clipped_norm(xy):
    return np.clip(xy, np.array([0.0, 0.0]), np.array([1.0, 1.0]))


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


_ZONE_GRID = None


def _zone_grid():
    """Rasterize precedence-ordered functional polygons at 5 cm for fast scoring."""
    global _ZONE_GRID
    if _ZONE_GRID is not None:
        return _ZONE_GRID
    from PIL import Image, ImageDraw
    step = 0.05
    width, height = int(round(MAP_X / step)) + 1, int(round(MAP_Y / step)) + 1
    zmap = {z: i for i, z in enumerate(FUNCTIONAL_ZONE_LABELS)}
    unknown, opened = zmap["场地未知区"], zmap["场地开放区"]
    im = Image.new("I", (width, height), unknown)
    draw = ImageDraw.Draw(im)
    # Generic floor fallback, then exact project precedence: first listed is
    # most specific and must overwrite later entries.
    draw.rectangle((round(.35/step), round(.35/step), round((MAP_X-.35)/step), round((MAP_Y-.35)/step)), fill=opened)
    order = (
        '红基地', '蓝基地', '红停机坪', '蓝停机坪', '红补给区', '蓝补给区',
        '红能量机关区', '蓝能量机关区', '红上装配区', '红下装配区',
        '蓝上装配区', '蓝下装配区', '红前哨区', '蓝前哨区',
        '红梯形高地区', '蓝梯形高地区', '中央北隧道', '中央南隧道',
        '红上公路', '红下公路', '蓝上公路', '蓝下公路', '红中路', '蓝中路', '中央高地',
    )
    for name in reversed(order):
        polygon = [(round(float(x)/step), round(float(y)/step)) for x, y in ZONE_POLYGONS[name]]
        draw.polygon(polygon, fill=zmap[name])
    _ZONE_GRID = (np.asarray(im, dtype=np.int16), step, zmap["场地未知区"])
    return _ZONE_GRID


def zone_ids(xy_norm):
    grid, step, unknown = _zone_grid()
    flat = xy_norm.reshape(-1, 2)
    ids = np.full(len(flat), unknown, dtype=np.int16)
    good = np.isfinite(flat).all(axis=1) & (flat[:, 0] >= 0) & (flat[:, 0] <= 1) & (flat[:, 1] >= 0) & (flat[:, 1] <= 1)
    xs = np.clip(np.rint(flat[good, 0] * MAP_X / step).astype(int), 0, grid.shape[1] - 1)
    ys = np.clip(np.rint(flat[good, 1] * MAP_Y / step).astype(int), 0, grid.shape[0] - 1)
    ids[good] = grid[ys, xs]
    return ids.reshape(xy_norm.shape[:-1])


def per_game_scores(errors, zone_hit, yvalid, gids, sides, game_side_rows=None):
    # errors shape [row,slot], absent/missing positions are NaN.
    per_gs = {}
    if game_side_rows is None:
        game_side_rows = defaultdict(list)
        for i, (gid, side) in enumerate(zip(gids, sides)):
            game_side_rows[(int(gid), str(side))].append(i)
        game_side_rows = {k: np.asarray(v, dtype=np.int64) for k, v in game_side_rows.items()}
    for (gid, side), all_rows in game_side_rows.items():
            rows = all_rows[yvalid[all_rows]]
            if not len(rows):
                continue
            e = errors[rows]
            z = zone_hit[rows]
            vals = e[np.isfinite(e)]
            zv = z[np.isfinite(z)]
            if len(vals):
                per_gs[(int(gid), side)] = {
                    "mae": float(vals.mean()), "zone_hit": float(zv.mean()) if len(zv) else None,
                    "n": int(len(vals)), "zone_n": int(len(zv)),
                }
    return per_gs


def aggregate(per_gs):
    gids = sorted({g for g, _ in per_gs})
    complete = [g for g in gids if (g, "红") in per_gs and (g, "蓝") in per_gs]
    result = {}
    for metric in ("mae", "zone_hit"):
        values = []
        for gid in complete:
            v = [per_gs[(gid, s)][metric] for s in ("红", "蓝")]
            v = [x for x in v if x is not None]
            if v:
                values.append(float(np.mean(v)))
        result[metric] = float(np.mean(values)) if values else None
    result["complete_games"] = len(complete)
    result["rows_scored"] = int(sum(v["n"] for v in per_gs.values()))
    result["zone_rows_scored"] = int(sum(v["zone_n"] for v in per_gs.values()))
    return result


def bootstrap_delta(candidate, baseline, metric, reps, seed):
    common = sorted(g for g in set(g for g, _ in candidate) & set(g for g, _ in baseline)
                    if all((g, s) in candidate and (g, s) in baseline for s in ("红", "蓝")))
    vals = []
    for g in common:
        pair = []
        for side in ("红", "蓝"):
            a, b = candidate[(g, side)][metric], baseline[(g, side)][metric]
            if a is not None and b is not None:
                pair.append(a - b)
        if pair:
            vals.append(float(np.mean(pair)))
    vals = np.asarray(vals, dtype=float)
    if not len(vals):
        return {"mean_delta": None, "ci95": [None, None], "games": 0}
    rng = np.random.default_rng(seed)
    draws = vals[rng.integers(0, len(vals), size=(reps, len(vals)))].mean(axis=1)
    return {"mean_delta": float(vals.mean()),
            "ci95": [float(np.quantile(draws, .025)), float(np.quantile(draws, .975))],
            "games": int(len(vals)), "bootstrap_replicates": reps}


def run(args):
    data = {k: v for k, v in np.load(args.validation_npz, allow_pickle=False).items()}
    current_m, velocity_mps, has_velocity = load_velocity(args.validation_jsonl, data)
    model_norm = predict_frozen(args.checkpoint, data["x"], args.device, args.batch_size)
    truth = np.asarray(data["entity_y"])[..., 1:3]
    presence = np.asarray(data["entity_y"])[..., 0] > .5
    valid = presence & np.isfinite(current_m).all(axis=-1)[:, None, :]
    base_norm = current_m / np.array([MAP_X, MAP_Y], dtype=np.float32)
    vel_norm = (current_m[:, None, :, :] + velocity_mps[:, None, :, :] * np.asarray(HORIZONS)[None, :, None, None])
    vel_norm /= np.array([MAP_X, MAP_Y], dtype=np.float32)
    vel_norm = clipped_norm(vel_norm)
    model_norm = clipped_norm(model_norm)
    variants = {f"model_weight_{w:.2f}": clipped_norm(w * model_norm + (1-w) * vel_norm)
                for w in WEIGHTS}
    # Raw model is also kept under a clear name independent of the weight grid.
    variants["raw_model"] = model_norm
    # Every row is one side view of a game; fold membership is by game only.
    games = sorted(set(int(g) for g in data["game_ids"]))
    fold_for = {g: i % args.folds for i, g in enumerate(games)}
    fold = np.asarray([fold_for[int(g)] for g in data["game_ids"]])
    sides = np.asarray(data["sides"])
    gids = np.asarray(data["game_ids"], dtype=np.int64)
    game_side_rows = defaultdict(list)
    for i, (gid, side) in enumerate(zip(gids, sides)):
        game_side_rows[(int(gid), str(side))].append(i)
    game_side_rows = {k: np.asarray(v, dtype=np.int64) for k, v in game_side_rows.items()}
    report_h = {}
    for j, h in enumerate(HORIZONS):
        y = truth[:, j]
        vmask = valid[:, j] & np.isfinite(y).all(axis=-1)
        # Evaluate all fixed candidate weights OOF (no tuning needed), plus an
        # explicit 5-fold selected weight whose choice is based on training games.
        candidate_errors, candidate_zone = {}, {}
        truth_z = zone_ids(y[:, :, :])
        for name, pred in variants.items():
            d = np.mean((pred[:, j] - y) * np.array([MAP_X, MAP_Y], dtype=np.float32), axis=-1)
            # Existing project normalized L1 MAE: each coordinate scaled by its
            # field dimension, then averaged.
            err = np.mean(np.abs(pred[:, j] - y), axis=-1)
            err[~vmask] = np.nan
            pred_z = zone_ids(pred[:, j])
            zh = np.where(vmask & (truth_z >= 0) & (pred_z >= 0), pred_z == truth_z, np.nan)
            candidate_errors[name], candidate_zone[name] = err, zh
        selected_pred = np.full_like(model_norm[:, j], np.nan)
        fold_details = []
        grid_names = [f"model_weight_{w:.2f}" for w in WEIGHTS]
        for f in range(args.folds):
            train_rows = (fold != f) & vmask.any(axis=1)
            held_rows = (fold == f) & vmask.any(axis=1)
            scored = {}
            for name in grid_names:
                per = per_game_scores(candidate_errors[name], candidate_zone[name], train_rows, gids, sides, game_side_rows)
                scored[name] = aggregate(per)["mae"]
            best_name = min(grid_names, key=lambda name: (scored[name], -float(name.rsplit("_", 1)[1])))
            selected_pred[held_rows] = variants[best_name][held_rows, j]
            fold_details.append({"fold": f, "selected": best_name,
                                 "train_equal_red_blue_game_macro_normalized_l1_mae": scored[best_name],
                                 "heldout_rows": int(held_rows.sum())})
        selected_err = np.mean(np.abs(selected_pred - y), axis=-1)
        selected_err[~vmask] = np.nan
        selected_z = zone_ids(np.nan_to_num(selected_pred, nan=-1.0))
        selected_zone_hit = np.where(vmask & (truth_z >= 0) & (selected_z >= 0),
                                     selected_z == truth_z, np.nan)
        candidate_errors["crossfit_selected_weight"] = selected_err
        candidate_zone["crossfit_selected_weight"] = selected_zone_hit
        stats = {}
        for name in (*variants.keys(), "crossfit_selected_weight"):
            per = per_game_scores(candidate_errors[name], candidate_zone[name],
                                  np.ones(len(gids), dtype=bool), gids, sides, game_side_rows)
            by_side = {}
            for side in ("红", "蓝"):
                by_side[side] = {}
                side_keys = [(g, ss) for g, ss in per if ss == side]
                for metric in ("mae", "zone_hit"):
                    vals = [per[key][metric] for key in side_keys if per[key][metric] is not None]
                    by_side[side][metric] = float(np.mean(vals)) if vals else None
            stats[name] = {**aggregate(per), "by_side": by_side}
            ref_per = per_game_scores(candidate_errors["raw_model"], candidate_zone["raw_model"],
                                       np.ones(len(gids), dtype=bool), gids, sides, game_side_rows)
            for ref in ("raw_model",):
                if name != ref:
                    stats[name][f"paired_game_bootstrap_vs_{ref}"] = {
                        metric: bootstrap_delta(per, ref_per, metric, args.bootstrap,
                            9217 + j + int(round(float(name.rsplit("_", 1)[-1]) * 100)) if "weight_" in name else 9917 + j)
                        for metric in ("mae", "zone_hit")}
        report_h[f"{h}s"] = {
            "n_rows_with_position_truth_and_current_observation": int(vmask.sum()),
            "n_scored_entity_slot_samples": int(vmask.sum()),
            "velocity_valid_fraction_among_scored_slots": float(np.sum(has_velocity & vmask) / max(1, int(vmask.sum()))),
            "mean_velocity_speed_mps_when_available": float(np.mean(np.linalg.norm(
                velocity_mps, axis=-1)[vmask & has_velocity]))
                if np.any(vmask & has_velocity) else None,
            "metrics": stats, "crossfit_weight_selection": fold_details,
        }
    fixed_weights = {"10s": 0.25, "30s": 0.75, "60s": 0.75, "120s": 1.0}
    fixed_profile = {}
    for h, w in fixed_weights.items():
        row = report_h[h]["metrics"][f"model_weight_{w:.2f}"]
        fixed_profile[h] = {
            "model_weight": w,
            "normalized_l1_mae": row["mae"],
            "functional_zone_hit": row["zone_hit"],
            "by_side": row["by_side"],
            "paired_game_bootstrap_vs_raw_model": row.get("paired_game_bootstrap_vs_raw_model"),
        }
    report = {
        "schema": "validation-causal-velocity-position-ablation-v1",
        "checkpoint": args.checkpoint, "validation_npz": args.validation_npz,
        "validation_jsonl": args.validation_jsonl, "device": args.device,
        "input_sha256": {"validation_npz": file_sha256(args.validation_npz),
                         "validation_jsonl_gz": file_sha256(args.validation_jsonl),
                         "frozen_checkpoint": file_sha256(args.checkpoint)},
        "split_access": {"validation": True, "test": False},
        "row_alignment": "game_id/side/issued_at exact within 1e-4; every validation NPZ row matched once",
        "velocity_contract": "Each (game_id, observing_side, self/opponent role, entity_type) centroid is computed from alive entities with issue-time observable XY only. Velocity is (current group centroid - exact issued_at-3s group centroid)/3s; unavailable if either sample missing. Measured speed is capped by the project's type-level ROBOT_SPEED_MPS prior. No target, yaw, future frame, or hidden identity position is used by velocity estimation.",
        "target_contract": "Position truth is entity_y group-mean XY for the future horizon; score only future-present slots with current observable group centroid. Functional-zone hit is predicted-vs-truth GROUP-MEAN point assigned through assign_functional_zone (not first-unit zone label), so it is aligned to this position target but can differ from entity_zone_y for multi-unit groups.",
        "position_metric": "normalized L1 MAE = mean((abs(dx)/28 + abs(dy)/15)/2); lower is better. Clip predicted field positions to [0,28]×[0,15].",
        "zone_scoring_approximation": "Functional polygons use the repository zone order precedence rasterized at 5 cm resolution for efficient batched scoring; reported zone hit is therefore a 5 cm raster approximation.",
        "blend_contract": "model_weight_w means w*neural_position + (1-w)*velocity_extrapolation; model_weight_1.00 equals raw neural position, model_weight_0.00 equals pure velocity. Missing velocity falls back to current observed centroid, so the velocity variant degrades to persistence for that slot.",
        "fold_contract": "5 game_id folds. For crossfit_selected_weight, one model weight per horizon is selected on four fifths by minimizing equal-game, equal-red/blue validation normalized L1 MAE and applied to the held-out games. Fixed-weight variants are also reported on all validation rows.",
        "candidate_fixed_profile": fixed_profile,
        "horizons": report_h,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    brief = {h: {k: {m: round(v[m], 5) if isinstance(v.get(m), (int, float)) else v.get(m)
                      for m in ("mae", "zone_hit", "complete_games")}
                 for k, v in row["metrics"].items()} for h, row in report_h.items()}
    print(json.dumps({"output": str(out), "metrics": brief}, ensure_ascii=False))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="out_m2/temporal_forecast_canonical_side/temporal_gru_v1.pt")
    p.add_argument("--validation-npz", default="out_m2/temporal_forecast_canonical_side/validation.npz")
    p.add_argument("--validation-jsonl", default="out_m2/forecast_dataset/validation.jsonl.gz")
    p.add_argument("--output", default="out_m2/validation_velocity_position_ablation/report.json")
    p.add_argument("--device", default="auto")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--bootstrap", type=int, default=1000)
    args = p.parse_args()
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    run(args)


if __name__ == "__main__":
    main()
