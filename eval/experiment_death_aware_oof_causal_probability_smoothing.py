"""Train-only, game-group OOF test of causal temporal probability smoothing.

The raw probabilities are the already cross-fitted zone+observable-score
candidate. This experiment does not refit a model or access validation/test.
For each (game, side, horizon), a candidate blends the current OOF vector with
the previous valid OOF vector only when the exact issue-time gap is 3 seconds.
The previous prediction is available at deployment time; no labels or future
rows are used to construct the smoothed probabilities.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from eval.experiment_death_aware_mode_lightgbm import scores, per_game_scores, paired_bootstrap


def exact_previous_indices(gids, sides, issued):
    """Return row index of exact t-3s sample; -1 means no usable predecessor."""
    out = np.full(len(gids), -1, dtype=np.int64)
    groups: dict[tuple[int, str], list[int]] = {}
    for i, (g, s) in enumerate(zip(gids, sides)):
        groups.setdefault((int(g), str(s)), []).append(i)
    for rows in groups.values():
        by_time = {round(float(issued[i]), 3): i for i in rows}
        for i in rows:
            prev = by_time.get(round(float(issued[i]) - 3.0, 3))
            if prev is not None:
                out[i] = prev
    return out


def smooth(prob, prev_idx, weight):
    q = np.asarray(prob, dtype=np.float64).copy()
    valid = np.isfinite(q).all(axis=1)
    rows = np.flatnonzero(valid & (prev_idx >= 0))
    pred = prev_idx[rows]
    usable = np.isfinite(prob[pred]).all(axis=1)
    rows, pred = rows[usable], pred[usable]
    q[rows] = (1.0 - weight) * prob[rows] + weight * prob[pred]
    sums = q.sum(axis=1, keepdims=True)
    good = np.isfinite(q).all(axis=1) & (sums[:, 0] > 0)
    q[good] /= sums[good]
    return q


def metric_bundle(y, p, gids, sides, valid, seed):
    yy = y[valid]
    pp = np.asarray(p[valid], dtype=np.float64)
    pp = pp / np.maximum(pp.sum(axis=1, keepdims=True), 1e-12)
    gs = per_game_scores(yy, pp, gids[valid], sides[valid], np.ones(valid.sum(), dtype=bool))
    return {
        "pooled": scores(yy, pp),
        "game_equal_red_blue_equal": {k: float(np.mean([x[k] for x in gs.values()]) if k != "n" else sum(x[k] for x in gs.values()) / len(gs)) for k in next(iter(gs.values()))},
        "game_scores": gs,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oof", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--bootstrap-reps", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=9027)
    a = ap.parse_args()

    d = np.load(a.oof, allow_pickle=False)
    required = {"game_ids", "sides", "issued_at", "mode_y", "base_mode", "fold_ids", "best_zone_score_oof"}
    missing = required - set(d.files)
    if missing:
        raise ValueError(f"OOF archive lacks required keys: {sorted(missing)}")
    gids = np.asarray(d["game_ids"], dtype=np.int64)
    sides = np.asarray(d["sides"]).astype(str)
    issued = np.asarray(d["issued_at"], dtype=np.float64)
    yall = np.asarray(d["mode_y"], dtype=np.int64)
    folds = np.asarray(d["fold_ids"], dtype=np.int64)
    probs = np.asarray(d["best_zone_score_oof"], dtype=np.float64)
    if len(np.unique(gids)) != 363 or probs.shape[:2] != yall.shape or probs.shape[2] != 3:
        raise ValueError(f"unexpected train OOF shapes/games: {probs.shape}, {yall.shape}, {len(np.unique(gids))}")
    if np.any(folds < 0) or len(np.unique(folds)) != 5:
        raise ValueError("expected complete five-fold game-group OOF assignments")
    # Paired red/blue group invariant and within-game side schedule parity.
    per_game_sides = {int(g): set(sides[gids == g]) for g in np.unique(gids)}
    if any(v != {"红", "蓝"} for v in per_game_sides.values()):
        raise ValueError("one or more games do not contain paired red and blue views")
    per_game_folds = {int(g): set(folds[gids == g]) for g in np.unique(gids)}
    if any(len(v) != 1 for v in per_game_folds.values()):
        raise ValueError("red/blue rows from the same game do not share exactly one OOF fold")

    prev = exact_previous_indices(gids, sides, issued)
    horizons = {}
    weights = (0.25, 0.50, 0.75)
    for j, horizon in enumerate((10, 30, 60, 120)):
        y = yall[:, j]
        p = probs[:, j]
        valid = (y >= 0) & np.isfinite(p).all(axis=1)
        raw_metrics = metric_bundle(y, p, gids, sides, valid, a.seed + j)
        candidates = {}
        for w in weights:
            q = smooth(p, prev, w)
            m = metric_bundle(y, q, gids, sides, valid, a.seed + j)
            delta = paired_bootstrap(m["game_scores"], raw_metrics["game_scores"], a.seed + 100 + j * 13 + int(w * 1000), reps=a.bootstrap_reps)
            # Normalize the helper's legacy field name for this explicitly named comparison.
            for metric in delta.values():
                if "paired_mean_delta_candidate_minus_persistence" in metric:
                    metric["paired_mean_delta_candidate_minus_raw"] = metric.pop("paired_mean_delta_candidate_minus_persistence")
            candidates[str(w)] = {"weight_previous_oof_prediction": w, **m, "paired_game_bootstrap_delta_vs_raw": delta}
        horizons[f"{horizon}s"] = {
            "valid_rows": int(valid.sum()),
            "raw": raw_metrics,
            "candidates": candidates,
            "predecessor_coverage": float(np.mean(prev[valid] >= 0)),
        }

    report = {
        "schema": "death-aware-oof-causal-probability-smoothing-v1",
        "access_contract": {"splits_opened": ["saved train OOF predictions and labels only"], "validation_access": False, "test_access": False},
        "source_oof": str(Path(a.oof).resolve()),
        "source_candidate": "zone + current-observable score candidate predictions, five-fold GroupKFold by full game_id; paired red/blue views in same held-out fold",
        "row_count": int(len(gids)), "unique_games": int(len(np.unique(gids))), "fold_count": int(len(np.unique(folds))),
        "smoothing_contract": "q_t=(1-w)*p_t+w*p_(t-3s); same game and same side only; exact 3s timestamp predecessor; no labels used for features; gaps/missing predictions leave p_t unchanged",
        "weights_tested": list(weights), "bootstrap": {"unit": "whole game, paired model-vs-raw; red/blue scores averaged within game", "replicates": a.bootstrap_reps},
        "horizons": horizons,
    }
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({h: {"raw": x["raw"]["game_equal_red_blue_equal"], "candidates": {w: {"metrics": y["game_equal_red_blue_equal"], "ci": y["paired_game_bootstrap_delta_vs_raw"]} for w,y in x["candidates"].items()}} for h,x in horizons.items()}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
