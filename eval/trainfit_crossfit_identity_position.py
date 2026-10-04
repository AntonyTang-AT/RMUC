"""Fit identity-velocity/neural blend weights on game-crossfit train data.

The frozen full checkpoint is evaluated on validation once. No test split is
accepted or opened. Fold models train for a fixed single epoch so the held-out
fold does not select an epoch; they are used only to produce out-of-fold
neural offsets for train-only blend-weight fitting.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.crossfit_validation_identity_trajectory import _collect, _zones
from forecast.entity import ENTITY_SLOTS, ROBOT_SPEED_MPS
from forecast.temporal import HORIZONS, MAP_X, MAP_Y, MOBILE_TYPES, load_compact
from forecast.train import load_model, train_model

ALPHAS = tuple(np.linspace(0., 1., 21))


def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _save_subset(src, keep, path):
    n = len(src["game_ids"])
    arrays = {k: (v[keep] if getattr(v, "ndim", 0) and v.shape[0] == n else v)
              for k, v in src.items()}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def _predict(checkpoint, x, device, batch_size):
    model, _ = load_model(checkpoint, device=device)
    model.eval()
    out = np.zeros((len(x), len(HORIZONS), len(ENTITY_SLOTS), 2), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, len(x), batch_size):
            xb = torch.from_numpy(x[start:start+batch_size]).to(device)
            pred = model(xb)
            for j, h in enumerate(HORIZONS):
                out[start:start+len(xb), j] = torch.sigmoid(
                    pred[h]["entity_state"][..., 1:3]).cpu().numpy()
    return out * np.asarray([MAP_X, MAP_Y], dtype=np.float32)


def _records(samples, model_pred, rows=None):
    n_rows = len(samples)
    use_rows = range(n_rows) if rows is None else rows
    records = {h: [] for h in HORIZONS}
    eligible = {h: 0 for h in HORIZONS}
    for i in use_rows:
        sample = samples[int(i)]
        ents = [e for e in sample["entities"] if e["type"] in MOBILE_TYPES]
        for entity in ents:
            target_by_h = sample["future"]
            for hidx, h in enumerate(HORIZONS):
                if not sample["future_valid"][h]:
                    continue
                eligible[h] += 1
                truth = target_by_h[h].get(entity["id"])
                if truth is None:
                    continue
                group_key = (entity["role"], entity["type"])
                center = sample["centroids"].get(group_key, entity["p"])
                group_shift = model_pred[int(i), hidx, entity["slot"]] - center
                velocity = entity["v"]
                velocity_pred = entity["p"].copy()
                if velocity is not None:
                    velocity_pred += velocity * float(h)
                record = {
                    "gid": int(sample["gid"]), "side": sample["side"],
                    "entity_id": entity["id"], "type": entity["type"],
                    "persistence": entity["p"].copy(),
                    "identity_velocity": np.clip(velocity_pred, [0., 0.], [MAP_X, MAP_Y]),
                    "neural": np.clip(entity["p"] + group_shift, [0., 0.], [MAP_X, MAP_Y]),
                    "truth": truth, "velocity_available": velocity is not None,
                }
                records[h].append(record)
    return records, eligible


def _blend(records, alpha):
    blended = []
    for r in records:
        # Blend displacements, not absolute coordinates: the neural head
        # predicts a per-type group shift while identity CV predicts a unit path.
        p = r["persistence"]
        pred = (1.-alpha) * (r["neural"] - p) + alpha * (r["identity_velocity"] - p) + p
        blended.append({**r, "mixed": np.clip(pred, [0., 0.], [MAP_X, MAP_Y])})
    return blended


def _side_game_metrics(records, pred_name):
    by = {}
    for r in records:
        pred, truth = r[pred_name], r["truth"]
        dx = pred - truth
        zone = _zones(np.asarray([pred, truth]))
        key = (r["gid"], r["side"])
        acc = by.setdefault(key, {"abs_xy": [], "euclidean": [], "zone_hit": [],
                                  "n": 0, "velocity_n": 0})
        acc["abs_xy"].extend(np.abs(dx).tolist())
        acc["euclidean"].append(float(np.linalg.norm(dx)))
        acc["zone_hit"].append(float(zone[0] == zone[1]))
        acc["n"] += 1
        acc["velocity_n"] += int(r["velocity_available"])
    result = {}
    for key, v in by.items():
        result[key] = {
            "mae_xy_m": float(np.mean(v["abs_xy"])),
            "mean_euclidean_error_m": float(np.mean(v["euclidean"])),
            "zone_hit": float(np.mean(v["zone_hit"])),
            "n": int(v["n"]), "velocity_n": int(v["velocity_n"]),
        }
    return result


def _aggregate(per):
    common_games = sorted({g for g, s in per if (g, "红") in per and (g, "蓝") in per})
    metrics = ("mae_xy_m", "mean_euclidean_error_m", "zone_hit")
    out = {m: None for m in metrics}
    for m in metrics:
        per_game = []
        for g in common_games:
            vals = [per[g, s][m] for s in ("红", "蓝") if (g, s) in per]
            if len(vals) == 2:
                per_game.append(float(np.mean(vals)))
        out[m] = float(np.mean(per_game)) if per_game else None
    out["paired_games"] = len(common_games)
    out["entity_samples"] = int(sum(v["n"] for v in per.values()))
    out["id_velocity_availability"] = float(
        sum(v["velocity_n"] for v in per.values()) /
        max(1, sum(v["n"] for v in per.values())))
    return out


def _paired_delta_ci(per, baseline, metric, reps, seed):
    games = sorted(g for g in {g for g, _ in per} & {g for g, _ in baseline}
                   if all((g, side) in per and (g, side) in baseline
                          for side in ("红", "蓝")))
    deltas = []
    for game in games:
        per_side = [per[game, side][metric] - baseline[game, side][metric]
                    for side in ("红", "蓝")]
        deltas.append(float(np.mean(per_side)))
    if not deltas:
        return {"delta": None, "ci95": [None, None], "games": 0}
    values = np.asarray(deltas, dtype=float)
    rng = np.random.default_rng(seed)
    draw = values[rng.integers(0, len(values), size=(reps, len(values)))].mean(axis=1)
    return {"delta": float(values.mean()),
            "ci95": [float(np.quantile(draw, .025)), float(np.quantile(draw, .975))],
            "games": len(values), "replicates": reps}


def _fit_alphas(records_by_h):
    selected = {}
    table = {}
    for h in HORIZONS:
        rlist = records_by_h[h]
        p = np.asarray([r["persistence"] for r in rlist], dtype=np.float32)
        cv = np.asarray([r["identity_velocity"] for r in rlist], dtype=np.float32)
        neural = np.asarray([r["neural"] for r in rlist], dtype=np.float32)
        truth = np.asarray([r["truth"] for r in rlist], dtype=np.float32)
        games = np.asarray([r["gid"] for r in rlist], dtype=np.int64)
        sides = np.asarray([0 if r["side"] == "红" else 1 for r in rlist], dtype=np.int8)
        unique_games, game_ix = np.unique(games, return_inverse=True)
        pair_ix = game_ix * 2 + sides
        pair_count = np.bincount(pair_ix, minlength=len(unique_games) * 2)
        candidates = {}
        for alpha in ALPHAS:
            pred = p + (1. - float(alpha)) * (neural-p) + float(alpha) * (cv-p)
            row_mae = np.mean(np.abs(pred-truth), axis=1)
            pair_sum = np.bincount(pair_ix, weights=row_mae,
                                   minlength=len(unique_games) * 2)
            valid_pair = pair_count > 0
            pair_mae = np.zeros_like(pair_sum)
            pair_mae[valid_pair] = pair_sum[valid_pair] / pair_count[valid_pair]
            game_sum = pair_mae.reshape(-1, 2).sum(axis=1)
            game_count = valid_pair.reshape(-1, 2).sum(axis=1)
            game_mae = game_sum[game_count > 0] / game_count[game_count > 0]
            candidates[float(alpha)] = float(np.mean(game_mae)) if len(game_mae) else float("inf")
        best = min(candidates, key=lambda a: (candidates[a], a))
        selected[h] = float(best)
        table[str(h)] = {f"{a:.2f}": v for a, v in candidates.items()}
    return selected, table


def _score_validation(records_by_h, eligible, weights, bootstrap):
    report = {}
    names = ("persistence", "identity_velocity", "neural", "mixed")
    for hi, h in enumerate(HORIZONS):
        rlist = records_by_h[h]
        variants = {
            "persistence": [{**r, "pred": r["persistence"]} for r in rlist],
            "identity_velocity": [{**r, "pred": r["identity_velocity"]} for r in rlist],
            "neural": [{**r, "pred": r["neural"]} for r in rlist],
            "mixed": _blend(rlist, weights[h]),
        }
        for r in variants["mixed"]:
            r["pred"] = r["mixed"]
        per = {name: _side_game_metrics(rows, "pred") for name, rows in variants.items()}
        baseline = per["persistence"]
        horizon = {"alpha_identity_velocity": weights[h],
                   "eligible_current_observable_uncensored_robots": int(eligible[h]),
                   "exact_same_id_future_alive_matches": len(rlist),
                   "id_and_future_match_coverage": len(rlist)/max(1, eligible[h]),
                   "metrics": {name: _aggregate(per[name]) for name in names},
                   "paired_game_red_blue_bootstrap_delta_vs_persistence": {}}
        for mi, metric in enumerate(("mae_xy_m", "mean_euclidean_error_m", "zone_hit")):
            horizon["paired_game_red_blue_bootstrap_delta_vs_persistence"][metric] = {
                name: _paired_delta_ci(per[name], baseline, metric, bootstrap,
                                       42000 + hi*100 + mi*10 + ni)
                for ni, name in enumerate(("identity_velocity", "neural", "mixed"))}
        report[f"{h}s"] = horizon
    return report


def _fit_oof(train_npz, train_jsonl, checkpoint, out_dir, folds, batch_size,
             device, seed, reuse_fold_checkpoints=False):
    data = load_compact(train_npz)
    samples, _ = None, None
    samples = _collect(train_jsonl, data)
    games = sorted(set(int(g) for g in data["game_ids"]))
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(games)
    fold_for_game = {int(g): i % folds for i, g in enumerate(shuffled)}
    fold_rows = np.asarray([fold_for_game[int(g)] for g in data["game_ids"]])
    records = {h: [] for h in HORIZONS}
    fold_report = []
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frozen, config = load_model(checkpoint, device="cpu")
    hidden, layers = int(config["hidden"]), int(config["layers"])
    loss_profile = str(config.get("loss_profile", "business_weighted"))
    for fold in range(folds):
        held = fold_rows == fold
        tr_path = out_dir / f"fold_{fold}_train.npz"
        va_path = out_dir / f"fold_{fold}_heldout.npz"
        ckpt_path = out_dir / f"fold_{fold}_model.pt"
        if reuse_fold_checkpoints and ckpt_path.exists():
            fit = None
        else:
            _save_subset(data, ~held, tr_path)
            _save_subset(data, held, va_path)
            fit = train_model(tr_path, va_path, ckpt_path, epochs=1,
                batch_size=batch_size, hidden=hidden, layers=layers, lr=2e-3,
                weight_decay=1e-4, device=device, seed=0,
                loss_profile=loss_profile, architecture="gru")
        idx = np.flatnonzero(held)
        pred = _predict(ckpt_path, data["x"][idx], device, batch_size)
        # Build a local sample list aligned with this held-out tensor.
        local_samples = [samples[int(i)] for i in idx]
        local, _ = _records(local_samples, pred)
        for h in HORIZONS:
            records[h].extend(local[h])
        fold_row = {"fold": fold, "heldout_games": sorted(set(int(g) for g in data["game_ids"][held])),
                            "train_rows": int((~held).sum()), "heldout_rows": int(held.sum()),
                            "fixed_epochs": 1, "checkpoint_reused": fit is None}
        if fit is not None:
            fold_row.update({"train_loss": fit["history"][-1]["train_loss"],
                             "heldout_monitor_loss": fit["history"][-1]["val_loss"]})
        fold_report.append(fold_row)
        print(json.dumps({"fold": fold, "status": "oof_predictions_ready", "rows": int(held.sum())}), flush=True)
    return records, fold_report, len(games)


def _validation(checkpoint, validation_npz, validation_jsonl, batch_size, device):
    data = load_compact(validation_npz)
    samples = _collect(validation_jsonl, data)
    pred = _predict(checkpoint, data["x"], device, batch_size)
    records, eligible = _records(samples, pred)
    return records, eligible, len(set(int(g) for g in data["game_ids"]))


def run(a):
    if a.folds < 2:
        raise ValueError("folds must be >= 2")
    dev = torch.device("cuda" if a.device == "auto" and torch.cuda.is_available()
                       else ("cpu" if a.device == "auto" else a.device))
    train_records, fold_report, train_games = _fit_oof(
        a.train_npz, a.train_jsonl, a.checkpoint, a.out_dir, a.folds,
        a.batch_size, dev, a.seed,
        reuse_fold_checkpoints=a.reuse_fold_checkpoints)
    weights, grid = _fit_alphas(train_records)
    # Freeze the train-only alpha before loading validation targets for scoring.
    val_records, eligible, validation_games = _validation(
        a.checkpoint, a.validation_npz, a.validation_jsonl, a.batch_size, dev)
    validation_report = _score_validation(val_records, eligible, weights, a.bootstrap)
    report = {
        "schema": "train-only-crossfit-identity-velocity-blend-v1",
        "split_access": {"train": True, "validation": True, "test": False},
        "test_access": False,
        "validation_protocol": "all blend weights selected from 5-fold game-group out-of-fold train predictions; then one validation scoring pass; no validation tuning",
        "checkpoint": a.checkpoint, "device": str(dev),
        "train_games": train_games, "validation_games": validation_games,
        "train_fold_config": {"folds": a.folds, "epochs": 1,
            "reason_fixed_epoch": "prevents held-out fold target from selecting checkpoint epoch",
            "architecture": "GRU", "hidden": 128, "layers": 2,
            "batch_size": a.batch_size, "lr": 0.002, "weight_decay": 0.0001,
            "loss_profile": "business_weighted", "seed_each_fold": 0},
        "folds": fold_report,
        "label_and_track_policy": {
            "current": "only currently observable, alive mobile robots with nonempty stable entity_id and valid XY",
            "velocity": "same entity_id at exactly issued_at-3 seconds; type-specific speed cap; otherwise no velocity and CV falls back to persistence",
            "death_respawn": "future HP<=0/destroyed rows are excluded from position truth; missing/dead observations break the 3s track; a later live row with same ID starts with no velocity until one valid prior 3s live observation exists",
            "future_match": "exact horizon truth row and exact same entity_id; unmatched IDs lower coverage and are not imputed",
            "visibility": "self sensor and opponent radar observable groups only at issue time; no future data enters features/predictions",
        },
        "mix_definition": "p_now + (1-alpha)*(frozen neural type-slot predicted position - current same-role/type centroid) + alpha*(p_now + h*v_same_id - p_now); positions clipped to map; when velocity missing, that term equals persistence",
        "weight_fit": {"metric": "equal-game/equal-side mean absolute coordinate error in metres over 5-fold OOF train identities; argmin on alpha grid 0..1 step .05", "alpha_by_horizon": {str(h): weights[h] for h in HORIZONS},
                       "train_oof_grid_scores_mae_xy_m": grid},
        "validation_metrics": validation_report,
        "input_sha256": {name: _sha(getattr(a, name)) for name in
            ("train_npz", "train_jsonl", "validation_npz", "validation_jsonl", "checkpoint")},
    }
    out = Path(a.output); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"output": str(out), "weights": report["weight_fit"]["alpha_by_horizon"],
                      "validation": validation_report}, ensure_ascii=False))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="out_m2/temporal_forecast_canonical_side/temporal_gru_v1.pt")
    p.add_argument("--train-npz", default="out_m2/temporal_forecast_canonical_side/train.npz")
    p.add_argument("--train-jsonl", default="out_m2/forecast_dataset/train.jsonl.gz")
    p.add_argument("--validation-npz", default="out_m2/temporal_forecast_canonical_side/validation.npz")
    p.add_argument("--validation-jsonl", default="out_m2/forecast_dataset/validation.jsonl.gz")
    p.add_argument("--out-dir", default="out_m2/identity_trajectory_trainfit")
    p.add_argument("--output", default="out_m2/identity_trajectory_trainfit/report.json")
    p.add_argument("--device", default="auto")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--seed", type=int, default=20260927)
    p.add_argument("--reuse-fold-checkpoints", action="store_true",
                   help="reuse existing fold_*/fold_N_model.pt OOF checkpoints")
    run(p.parse_args())


if __name__ == "__main__":
    main()
