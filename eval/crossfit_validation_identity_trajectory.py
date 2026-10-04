"""Validation-only causal stable-entity trajectory versus group-centroid audit.

Scores exact future ``entity_id`` positions for currently observable robots.
Inputs use current and exact t-3s observable positions only. Targets are read
solely for scoring. The neural checkpoint remains frozen and test is never read.
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
from forecast.temporal import HORIZONS, MAP_X, MAP_Y
from forecast.train import load_model
from worldstate.functional_zone import FUNCTIONAL_ZONE_LABELS, ZONE_POLYGONS

WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)  # identity-CV weight; 0=neural formation shift


def _pos(e):
    xy = e.get("position")
    if not isinstance(xy, (list, tuple)) or len(xy) < 2:
        return None
    try:
        p = np.asarray([float(xy[0]), float(xy[1])], dtype=np.float32)
    except (ValueError, TypeError):
        return None
    return p if np.isfinite(p).all() else None


def _alive(e):
    hp = e.get("hp")
    return e.get("life_state") != "destroyed" and not (
        isinstance(hp, (int, float)) and not isinstance(hp, bool) and hp <= 0)


def _entities(value):
    if isinstance(value, dict):
        return list(value.values())
    return list(value or [])


def _clip(p):
    return np.clip(p, [0., 0.], [MAP_X, MAP_Y])


_ZONE_GRID = None


def _zones(points):
    """Fast 5 cm raster lookup using the repository functional-zone precedence."""
    global _ZONE_GRID
    if _ZONE_GRID is None:
        from PIL import Image, ImageDraw
        step = .05
        width, height = int(round(MAP_X/step))+1, int(round(MAP_Y/step))+1
        labels = {name: i for i, name in enumerate(FUNCTIONAL_ZONE_LABELS)}
        im = Image.new("I", (width, height), labels["场地未知区"])
        draw = ImageDraw.Draw(im)
        draw.rectangle((round(.35/step), round(.35/step), round((MAP_X-.35)/step),
                        round((MAP_Y-.35)/step)), fill=labels["场地开放区"])
        order = ('红基地', '蓝基地', '红停机坪', '蓝停机坪', '红补给区', '蓝补给区',
                 '红能量机关区', '蓝能量机关区', '红上装配区', '红下装配区',
                 '蓝上装配区', '蓝下装配区', '红前哨区', '蓝前哨区',
                 '红梯形高地区', '蓝梯形高地区', '中央北隧道', '中央南隧道',
                 '红上公路', '红下公路', '蓝上公路', '蓝下公路', '红中路', '蓝中路', '中央高地')
        for name in reversed(order):
            polygon = [(round(float(x)/step), round(float(y)/step)) for x, y in ZONE_POLYGONS[name]]
            draw.polygon(polygon, fill=labels[name])
        _ZONE_GRID = (np.asarray(im, dtype=np.int16), step, labels["场地未知区"])
    grid, step, unknown = _ZONE_GRID
    p = np.asarray(points, dtype=np.float32)
    flat = p.reshape(-1, 2)
    ids = np.full(len(flat), unknown, dtype=np.int16)
    good = np.isfinite(flat).all(axis=1) & (flat[:, 0] >= 0) & (flat[:, 0] <= MAP_X) & (flat[:, 1] >= 0) & (flat[:, 1] <= MAP_Y)
    xs = np.clip(np.rint(flat[good, 0]/step).astype(int), 0, grid.shape[1]-1)
    ys = np.clip(np.rint(flat[good, 1]/step).astype(int), 0, grid.shape[0]-1)
    ids[good] = grid[ys, xs]
    return ids.reshape(p.shape[:-1])


def _collect(path, expected, selected_games=None):
    n = len(expected["game_ids"])
    sample = []
    source_i = 0
    history = {}  # (game, side, entity id) -> exact issue time and point
    last_rows = {}  # prior issue row vectors for groups; per current row only
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            i = source_i
            source_i += 1
            if i >= n:
                raise ValueError("validation JSONL has more rows than NPZ")
            row = json.loads(line)
            gid, side, issued = int(row["game_id"]), str(row["side"]), float(row["issued_at"])
            if (gid != int(expected["game_ids"][i]) or side != str(expected["sides"][i])
                    or abs(issued-float(expected["issued_at"][i])) > 1e-4):
                raise ValueError(f"validation order mismatch at row {i}")
            if selected_games is not None and gid not in selected_games:
                continue
            obs = row.get("observable") or {}
            now_entities = []
            group_points = defaultdict(list)
            for role, group in (("self", "self_entities"), ("opponent", "opponent_entities")):
                for e in _entities(obs.get(group)):
                    p = _pos(e)
                    if p is None or not _alive(e) or e.get("entity_type") not in ENTITY_TYPES:
                        continue
                    eid = str(e.get("entity_id", ""))
                    if not eid:
                        continue
                    typ = e["entity_type"]
                    slot = ENTITY_SLOTS.index((role, typ))
                    key = (gid, side, eid)
                    prev = history.get(key)
                    iv = None
                    if prev is not None and abs(issued-prev[0]-3.) <= 1e-4:
                        iv = (p-prev[1])/3.
                        speed = float(np.linalg.norm(iv))
                        cap = float(ROBOT_SPEED_MPS.get(typ, .35))
                        if speed > cap and speed > 1e-8:
                            iv *= cap/speed
                    history[key] = (issued, p.copy())
                    group_points[(role, typ)].append(p)
                    now_entities.append({"id": eid, "role": role, "type": typ, "slot": slot,
                                         "p": p, "v": iv})
            centroids = {k: np.mean(v, axis=0) for k, v in group_points.items()}
            prev_centroids = last_rows.get((gid, side))
            cv = {}
            if prev_centroids is not None and abs(issued-prev_centroids[0]-3.) <= 1e-4:
                for k, p in centroids.items():
                    if k in prev_centroids[1]:
                        v = (p-prev_centroids[1][k])/3.
                        speed = float(np.linalg.norm(v))
                        cap = float(ROBOT_SPEED_MPS.get(k[1], .35))
                        if speed > cap and speed > 1e-8:
                            v *= cap/speed
                        cv[k] = v
            last_rows[(gid, side)] = (issued, centroids)
            # identity truth is scoring-only; censored horizons stay empty.
            future = {}
            future_valid = {}
            current_ids = {e["id"] for e in now_entities}
            for h in HORIZONS:
                target = (row.get("targets") or {}).get(str(h)) or {}
                if target.get("censored"):
                    future[h] = {}
                    future_valid[h] = False
                    continue
                future_valid[h] = True
                truth = ((target.get("truth") or {}).get("entities") or {})
                # Discard the verbose full future world immediately. Only keep
                # same-ID truth for units observable at issue time.
                wanted = current_ids
                future[h] = {}
                for e in truth.values():
                    eid = str(e.get("entity_id", ""))
                    if eid in wanted and _alive(e):
                        p = _pos(e)
                        if p is not None:
                            future[h][eid] = p
            sample.append({"gid": gid, "side": side, "issued": issued, "entities": now_entities,
                           "centroids": centroids, "group_v": cv, "future": future,
                           "future_valid": future_valid})
            # remove stale tracks so gaps cannot be mistaken for an adjacent sample
            for key, (t, _) in list(history.items()):
                if key[0] == gid and key[1] == side and issued-t > 3.0001:
                    del history[key]
    expected_selected = n if selected_games is None else int(np.isin(expected["game_ids"], list(selected_games)).sum())
    if source_i != n or len(sample) != expected_selected:
        raise ValueError(f"validation JSONL rows {source_i}, selected {len(sample)} != expected {n}/{expected_selected}")
    return sample


def _infer(checkpoint, x, device, batch_size):
    model, _ = load_model(checkpoint, device=device)
    model.eval()
    out = np.zeros((len(x), len(HORIZONS), len(ENTITY_SLOTS), 2), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, len(x), batch_size):
            xb = torch.from_numpy(x[start:start+batch_size]).to(device)
            pred = model(xb)
            for j, h in enumerate(HORIZONS):
                out[start:start+len(xb), j] = torch.sigmoid(pred[h]["entity_state"][..., 1:3]).cpu().numpy()
    return out * np.asarray([MAP_X, MAP_Y], dtype=np.float32)


def _game_metrics(records, pred_key, folds=None, test_fold=None):
    per = {}
    for r in records:
        if folds is not None and folds[r["row"]] == test_fold:
            continue
        pred = r[pred_key]
        err = np.mean(np.abs((pred-r["truth"])/[MAP_X, MAP_Y]))
        ph, th = _zones(np.asarray([pred, r["truth"]]))
        zhit = float(ph == th)
        per.setdefault((r["gid"], r["side"]), [[], []])
        per[(r["gid"], r["side"])][0].append(float(err))
        if np.isfinite(zhit):
            per[(r["gid"], r["side"])][1].append(zhit)
    return {k: {"mae": float(np.mean(v[0])),
                "zone_hit": float(np.mean(v[1])) if v[1] else None,
                "n": len(v[0]), "zone_n": len(v[1])} for k, v in per.items()}


def _aggregate(per):
    games = sorted({g for g, _ in per})
    both = [g for g in games if (g, "红") in per and (g, "蓝") in per]
    vals = {m: [] for m in ("mae", "zone_hit")}
    for g in both:
        for m in vals:
            v = [per[g, s][m] for s in ("红", "蓝") if per[g, s][m] is not None]
            if v:
                vals[m].append(float(np.mean(v)))
    return {m: float(np.mean(v)) if v else None for m, v in vals.items()} | {
        "paired_games": len(both), "entity_samples": sum(v["n"] for v in per.values()),
        "zone_samples": sum(v["zone_n"] for v in per.values())}


def _bootstrap(candidate, baseline, metric, reps, seed):
    games = sorted(g for g in {g for g, _ in candidate} & {g for g, _ in baseline}
                   if all((g, s) in candidate and (g, s) in baseline for s in ("红", "蓝")))
    d = []
    for g in games:
        pair = [candidate[g, s][metric]-baseline[g, s][metric]
                for s in ("红", "蓝") if candidate[g, s][metric] is not None and baseline[g, s][metric] is not None]
        if pair:
            d.append(float(np.mean(pair)))
    if not d:
        return {"delta": None, "ci95": [None, None], "games": 0}
    rng = np.random.default_rng(seed)
    a = np.asarray(d)
    draws = a[rng.integers(0, len(a), (reps, len(a)))].mean(axis=1)
    return {"delta": float(a.mean()), "ci95": [float(np.quantile(draws, .025)), float(np.quantile(draws, .975))],
            "games": len(a), "bootstrap_replicates": reps}


def sha(path):
    d = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1024*1024), b""):
            d.update(b)
    return d.hexdigest()


def run(args):
    data = dict(np.load(args.validation_npz, allow_pickle=False))
    all_game_ids = sorted(set(int(g) for g in data["game_ids"]))
    rng = np.random.default_rng(args.seed)
    if args.max_games and args.max_games < len(all_game_ids):
        selected_games = set(int(g) for g in rng.choice(all_game_ids, size=args.max_games, replace=False))
    else:
        selected_games = set(all_game_ids)
    keep = np.isin(data["game_ids"], list(selected_games))
    full_data = data
    data = {k: (v[keep] if getattr(v, "ndim", 0) and v.shape[0] == len(keep) else v)
            for k, v in full_data.items()}
    samples = _collect(args.validation_jsonl, full_data, selected_games)
    model = _infer(args.checkpoint, data["x"], args.device, args.batch_size)
    # Build one row per observable ID with exact future identity match.
    rec_by_h = {h: [] for h in HORIZONS}
    eligible = defaultdict(int)
    for i, row in enumerate(samples):
        for entity in row["entities"]:
            k = (entity["role"], entity["type"])
            slot = entity["slot"]
            c = row["centroids"].get(k)
            gv = row["group_v"].get(k)
            for hidx, h in enumerate(HORIZONS):
                # Denominator counts every current observed robot whenever its
                # future frame is available; an unmatched ID is real coverage loss.
                target_dict = row["future"][h]
                target = target_dict.get(entity["id"])
                # An empty dict can mean either censored or a valid future frame
                # with zero alive entities. Preserve that distinction from JSONL.
                if row["future_valid"][h]:
                    eligible[h] += 1
                else:
                    continue
                if target is None:
                    continue
                group_shift = model[i, hidx, slot] - c if c is not None else np.zeros(2, dtype=np.float32)
                neural_offset = _clip(entity["p"]+group_shift)
                group_cv = _clip(entity["p"]+(gv*h if gv is not None else np.zeros(2, dtype=np.float32)))
                individual_cv = _clip(entity["p"]+(entity["v"]*h if entity["v"] is not None else np.zeros(2, dtype=np.float32)))
                truth = target
                # If forecast slot says absent but exact identity exists, coordinate head still scored;
                # this audit isolates localization and intentionally does not score presence.
                rec_by_h[h].append({"row": i, "gid": row["gid"], "side": row["side"], "id": entity["id"],
                                    "slot": slot, "truth": truth, "persist": entity["p"].copy(),
                                    "identity_cv": individual_cv, "group_cv": group_cv,
                                    "neural_group_shift": neural_offset, "group_centroid": c.copy() if c is not None else entity["p"].copy(),
                                    "velocity_valid": entity["v"] is not None})
    games = sorted(set(int(g) for g in data["game_ids"]))
    fold_map = {g: ix % args.folds for ix, g in enumerate(games)}
    folds = np.asarray([fold_map[int(g)] for g in data["game_ids"]])
    result = {}
    for hi, h in enumerate(HORIZONS):
        records = rec_by_h[h]
        # For an apples-to-apples centroid check, compare known-entity average target
        # with raw neural group centroid and centroid persist/CV.
        for r in records:
            # Existing entity head is normalized group target; model decoder is group mean.
            r["neural_group"] = _clip(model[r["row"], hi, r["slot"]])
            sample = samples[r["row"]]
            k = ("self" if r["slot"] < len(ENTITY_TYPES) else "opponent",
                 ENTITY_TYPES[r["slot"] % len(ENTITY_TYPES)])
            center = sample["centroids"].get(k, r["persist"])
            vel = sample["group_v"].get(k)
            r["centroid_persist"] = center.copy()
            r["centroid_cv"] = _clip(center+(vel*h if vel is not None else np.zeros(2, dtype=np.float32)))
        # OOF blend: optimize identity-CV weight against neural group shift over training games.
        for w in WEIGHTS:
            for r in records:
                r[f"blend_{w:.2f}"] = _clip((1.-w)*r["neural_group_shift"]+w*r["identity_cv"])
        fold_id = np.asarray([folds[r["row"]] for r in records])
        selected = np.zeros((len(records), 2), dtype=np.float32)
        fold_selected = []
        for fold in range(args.folds):
            train_sub = [r for ix, r in enumerate(records) if fold_id[ix] != fold]
            scored = {}
            for w in WEIGHTS:
                vals = defaultdict(list)
                for r in train_sub:
                    vals[(r["gid"], r["side"])].append(float(np.mean(np.abs((r[f"blend_{w:.2f}"]-r["truth"])/[MAP_X, MAP_Y]))))
                pg = {k: {"mae": float(np.mean(v)), "zone_hit": None, "n": len(v), "zone_n": 0} for k, v in vals.items()}
                scored[w] = _aggregate(pg)["mae"]
            best = min(WEIGHTS, key=lambda w: (scored[w], w))
            fold_selected.append({"fold": fold, "identity_cv_weight": best, "train_mae": scored[best]})
            for ix, r in enumerate(records):
                if fold_id[ix] == fold:
                    selected[ix] = r[f"blend_{best:.2f}"]
        for ix, r in enumerate(records):
            r["crossfit_blend"] = selected[ix]
        names = ["persist", "identity_cv", "group_cv", "neural_group_shift", "neural_group",
                 "centroid_persist", "centroid_cv", *(f"blend_{w:.2f}" for w in WEIGHTS), "crossfit_blend"]
        stats = {}
        for name in names:
            # For individual-position assessment neural_group is offset to each robot; raw centroid
            # is reported separately under centroid_* metrics.
            key = name
            per = defaultdict(lambda: [[], []])
            for r in records:
                pred = r[key]
                err = float(np.mean(np.abs((pred-r["truth"])/[MAP_X, MAP_Y])))
                ph, th = _zones(np.asarray([pred, r["truth"]]))
                per[r["gid"], r["side"]][0].append(err)
                per[r["gid"], r["side"]][1].append(float(ph == th))
            per_scores = {k: {"mae": float(np.mean(v[0])), "zone_hit": float(np.mean(v[1])) if v[1] else None,
                              "n": len(v[0]), "zone_n": len(v[1])} for k, v in per.items()}
            stats[name] = _aggregate(per_scores)
            if name != "neural_group":
                # Each entity contributes equally within each game-side; then
                # red/blue are paired and games are the bootstrap unit.
                ref_per = {}
                for k in per_scores:
                    selected_rows = [r for r in records if (r["gid"], r["side"]) == k]
                    ref_per[k] = {"mae": float(np.mean([
                        np.mean(np.abs((r["neural_group_shift"]-r["truth"])/[MAP_X, MAP_Y]))
                        for r in selected_rows])),
                        "zone_hit": float(np.mean([
                            float(_zones(np.asarray([r["neural_group_shift"], r["truth"]]))[0] ==
                                  _zones(np.asarray([r["neural_group_shift"], r["truth"]]))[1])
                            for r in selected_rows]))}
                stats[name]["paired_bootstrap_vs_neural_group_shift"] = _bootstrap(
                    per_scores, ref_per, "mae", args.bootstrap, 8000+hi)
                stats[name]["paired_bootstrap_zone_hit_vs_neural_group_shift"] = _bootstrap(
                    per_scores, ref_per, "zone_hit", args.bootstrap, 9000+hi)
        result[f"{h}s"] = {"eligible_current_observable_and_exact_future_identity": len(records),
                           "coverage_of_current_entities_with_exact_future_match": len(records)/max(1, eligible[h]),
                           "identity_velocity_available_fraction": float(np.mean([r["velocity_valid"] for r in records])) if records else 0.,
                           "metrics": stats, "crossfit_weight_selection": fold_selected}
    report = {"schema": "validation-stable-identity-trajectory-crossfit-v1",
              "split_access": {"validation": True, "test": False},
              "row_alignment": "Each validation JSONL row exactly matched NPZ game_id/side/issued_at order.",
              "input_contract": "Only currently observable entity_id positions and exact issued_at-3s same-ID positions; group centroid alternatives use only current/prior observable XY. Speed is capped by type prior. No yaw or target enters prediction.",
              "target_contract": "Future truth is used only for exact-time scoring by same entity_id, alive target, and uncensored horizon. Current visibility required. Position field metrics are normalized L1 (mean(abs(dx)/28, abs(dy)/15)). Functional-zone scoring uses project assign_functional_zone.",
              "identity_coverage_contract": "Coverage denominator is eligible currently observable, alive entities with an uncensored exact future target row; entity_id match success reported as records/eligible.",
              "checkpoint": args.checkpoint, "input_sha256": {"validation_npz": sha(args.validation_npz), "validation_jsonl": sha(args.validation_jsonl), "checkpoint": sha(args.checkpoint)},
              "games": len(games), "sampled_games": sorted(selected_games), "sample_seed": args.seed,
              "folds_by_game_id": args.folds,
              "candidate_blend": "(1-w)*per-entity current position shifted by frozen neural predicted group displacement + w*same-entity capped constant-velocity position; one horizon-specific weight selected on other game folds.",
              "horizons": result}
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"output": str(out), "horizons": {h: {"n": v["eligible_current_observable_and_exact_future_identity"],
          "coverage": v["coverage_of_current_entities_with_exact_future_match"],
          "metrics": {m: {k: round(val, 5) if isinstance(val, float) else val for k, val in row.items() if k in ("mae", "zone_hit", "paired_games", "entity_samples")} for m, row in v["metrics"].items()}} for h,v in result.items()}}, ensure_ascii=False))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="out_m2/temporal_forecast_canonical_side/temporal_gru_v1.pt")
    p.add_argument("--validation-npz", default="out_m2/temporal_forecast_canonical_side/validation.npz")
    p.add_argument("--validation-jsonl", default="out_m2/forecast_dataset/validation.jsonl.gz")
    p.add_argument("--output", default="out_m2/validation_identity_trajectory/report.json")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--max-games", type=int, default=24, help="seeded game sample; 0 means all validation games")
    p.add_argument("--seed", type=int, default=20260927)
    run(p.parse_args())


if __name__ == "__main__":
    main()
