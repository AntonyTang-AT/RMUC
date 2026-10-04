"""Game-balanced proper-score audit for train-only global-mode OOF probabilities.

Reads only saved cross-fitted predictions and train labels. No validation or
test split is opened. ECE is a descriptive top-label, 10-bin estimate.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out_m2/death_aware_global_mode_experiments"
DIRECT = OUT / "visible_aggregate_with_obs_score_oof/visible_aggregate_oof.npz"
MOTION = OUT / "motion_zone_score_oof/motion_candidate_oof.npz"
REPORT = OUT / "oof_probability_quality/report.json"


def _scores(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-12, 1.0)
    y = np.asarray(y, dtype=np.int64)
    pred = p.argmax(axis=1)
    conf = p.max(axis=1)
    onehot = np.eye(3, dtype=np.float64)[y]
    ece = 0.0
    for b in range(10):
        lo, hi = b / 10.0, (b + 1) / 10.0
        mask = (conf >= lo) & ((conf < hi) if b < 9 else (conf <= hi))
        if mask.any():
            ece += float(mask.mean()) * abs(float((pred[mask] == y[mask]).mean()) - float(conf[mask].mean()))
    return {
        "brier_sum3": float(np.mean(np.sum((p - onehot) ** 2, axis=1))),
        "log_loss": float(np.mean(-np.log(p[np.arange(len(y)), y]))),
        "top_label_ece_10bin": ece,
    }


def _game_scores(y, p, gids, sides, horizon_index):
    out = {}
    valid = (y[:, horizon_index] >= 0) & np.isfinite(p[:, 0])
    for gid in np.unique(gids[valid]):
        by_side = []
        for side in ("红", "蓝"):
            mask = valid & (gids == gid) & (sides == side)
            if mask.any():
                by_side.append(_scores(y[mask, horizon_index], p[mask]))
        if len(by_side) == 2:
            out[int(gid)] = {key: float(np.mean([x[key] for x in by_side])) for key in by_side[0]}
    return out


def _paired_ci(candidate, reference, *, seed, reps=2000):
    ids = sorted(set(candidate) & set(reference))
    rng = np.random.default_rng(seed)
    sample_ix = rng.integers(0, len(ids), size=(reps, len(ids)))
    result = {"paired_games": len(ids), "replicates": reps, "lower_is_better": True, "metrics": {}}
    for metric in ("brier_sum3", "log_loss", "top_label_ece_10bin"):
        delta = np.asarray([candidate[gid][metric] - reference[gid][metric] for gid in ids])
        boot = delta[sample_ix].mean(axis=1)
        result["metrics"][metric] = {
            "candidate_minus_reference": float(delta.mean()),
            "paired_game_bootstrap_ci95": [float(np.quantile(boot, .025)), float(np.quantile(boot, .975))],
        }
    return result


def run():
    base = np.load(DIRECT, allow_pickle=False)
    motion = np.load(MOTION, allow_pickle=False)
    for key in ("game_ids", "sides", "issued_at", "mode_y", "base_mode"):
        if not np.array_equal(base[key], motion[key]):
            raise ValueError(f"OOF artifact alignment mismatch: {key}")
    y = np.asarray(base["mode_y"], dtype=np.int64)
    gids = np.asarray(base["game_ids"], dtype=np.int64)
    sides = np.asarray(base["sides"]).astype(str)
    predictions = {
        "direct_lightgbm": np.asarray(base["direct_lightgbm"], dtype=np.float64),
        "zone_score_candidate": np.asarray(base["aggregate_candidate"], dtype=np.float64),
        "motion_candidate": np.asarray(motion["motion_candidate"], dtype=np.float64),
        "persistence": np.asarray(base["persistence"], dtype=np.float64),
    }
    horizons = {}
    for j, horizon in enumerate((10, 30, 60, 120)):
        per_model = {name: _game_scores(y, prob[:, j], gids, sides, j)
                     for name, prob in predictions.items()}
        ids = sorted(set.intersection(*(set(v) for v in per_model.values())))
        summary = {
            name: {metric: float(np.mean([game[metric] for game in scores.values()]))
                   for metric in ("brier_sum3", "log_loss", "top_label_ece_10bin")}
            for name, scores in per_model.items()
        }
        horizons[f"{horizon}s"] = {
            "game_equal_red_blue_equal": summary,
            "paired_game_bootstrap": {
                "zone_score_vs_direct": _paired_ci(per_model["zone_score_candidate"], per_model["direct_lightgbm"], seed=20260927 + j),
                "motion_vs_zone_score": _paired_ci(per_model["motion_candidate"], per_model["zone_score_candidate"], seed=20261027 + j),
            },
            "paired_games": len(ids),
            "valid_train_rows": int(np.sum(y[:, j] >= 0)),
        }
    report = {
        "schema": "death-aware-train-oof-probability-quality-v1",
        "data_contract": {"opened_splits": ["train OOF predictions and labels"], "validation": False, "test": False,
                          "label_contract": "relative_margin_v2_death_aware_hp_le_zero",
                          "aggregation": "within each game compute red and blue metrics, average the two sides, then average games"},
        "metric_notes": {
            "brier_sum3": "mean sum of three squared class-probability errors; lower is better",
            "log_loss": "negative log probability assigned to the observed class; lower is better",
            "top_label_ece_10bin": "10 equal-width bins of max-class confidence; game/side ECE is averaged equally; descriptive, bin-sensitive, lower is better",
            "interpretation": "OOF probability quality is cross-fitted but these candidates were selected after inspecting prior OOF reports; treat comparisons as exploratory, not blind confirmation",
        },
        "horizons": horizons,
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))
