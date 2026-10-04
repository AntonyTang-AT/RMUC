"""Frozen train-to-validation confirmation of causal behavior-summary trend features.

This is a temporal holdout development check: fit every scaler, prototype and
predictor on train only, then score the existing validation split once. It
does not access test and does not tune hyperparameters on validation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from sklearn.cluster import KMeans

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from forecast.temporal import TRENDS, load_compact  # noqa: E402
from eval.experiment_death_aware_mode_lightgbm import (  # noqa: E402
    equal_game_summary,
    paired_bootstrap,
    per_game_scores,
)
from eval.experiment_style_conditioned_trend import (  # noqa: E402
    behavior_profile,
    make_model,
)


def _paired(per_game, candidate, comparator, seed):
    raw = paired_bootstrap(per_game[candidate], per_game[comparator],
                           seed=seed, reps=2000)
    return {
        metric: {
            "paired_mean_delta_candidate_minus_comparator": values[
                "paired_mean_delta_candidate_minus_persistence"],
            "game_bootstrap_95_ci": values["game_bootstrap_95_ci"],
            "n_games_red_blue_paired": values["n_games_red_blue_paired"],
        }
        for metric, values in raw.items()
    }


def run(train_path: Path, validation_path: Path, output: Path,
        threads=4, seed=101):
    train = load_compact(train_path)
    val = load_compact(validation_path)
    if str(train["global_mode_label_version"][0]) != str(val["global_mode_label_version"][0]):
        raise ValueError("train/validation label contracts differ")
    for key in ("x", "trend_y"):
        if train[key].shape[1:] != val[key].shape[1:]:
            raise ValueError(f"train/validation {key} schemas differ")
    yi, yv = 1, 1  # fixed opponent_trend@30s target
    train_y_all = np.asarray(train["trend_y"][:, yi], dtype=np.int64)
    val_y_all = np.asarray(val["trend_y"][:, yv], dtype=np.int64)
    train_ok = (train_y_all >= 0) & (train_y_all < len(TRENDS))
    val_ok = (val_y_all >= 0) & (val_y_all < len(TRENDS))
    train_x = np.asarray(train["x"], dtype=np.float32)
    val_x = np.asarray(val["x"], dtype=np.float32)
    train_times = np.asarray(train["issued_at"], dtype=np.float32)
    train_model_profile, train_route_profile, train_behavior = behavior_profile(train)
    val_model_profile, val_route_profile, val_behavior = behavior_profile(val)
    generic_x = train_x[:, -1][train_ok]
    val_generic_x = val_x[:, -1][val_ok]
    behavior_x = np.concatenate([train_x[:, -1], train_model_profile], axis=1)[train_ok]
    val_behavior_x = np.concatenate([val_x[:, -1], val_model_profile], axis=1)[val_ok]
    train_y = train_y_all[train_ok]
    val_y = val_y_all[val_ok]

    generic = make_model(seed, threads).fit(generic_x, train_y)
    behavior = make_model(seed + 1, threads).fit(behavior_x, train_y)
    pred = {
        "generic": generic.predict_proba(val_generic_x),
        "generic_plus_behavior": behavior.predict_proba(val_behavior_x),
    }
    for name, probabilities in pred.items():
        if probabilities.shape != (len(val_y), len(TRENDS)):
            raise ValueError(f"{name} returned unexpected probability shape {probabilities.shape}")
    train_route = train_route_profile[train_ok]
    val_route = val_route_profile[val_ok]
    coverage = val_behavior[val_ok, 6]
    for clusters in (2, 3):
        sampled = ((np.round(train_times[train_ok]).astype(int) - 1) % 15) == 0
        if sampled.sum() < 100:
            sampled[:] = True
        mean = train_route[sampled].mean(axis=0)
        scale = train_route[sampled].std(axis=0)
        scale[scale <= 1e-6] = 1.0
        tz, vz = (train_route - mean) / scale, (val_route - mean) / scale
        km = KMeans(n_clusters=clusters, n_init=20,
                    random_state=31 + clusters).fit(tz[sampled])
        td, vd = km.transform(tz), km.transform(vz)
        temperature = max(float(np.quantile(np.min(km.transform(tz[sampled]), axis=1), .75)), 1e-3)

        def membership(distance):
            logits = -0.5 * (distance / temperature) ** 2
            logits -= logits.max(axis=1, keepdims=True)
            p = np.exp(logits)
            return p / p.sum(axis=1, keepdims=True)

        trp, vap = membership(td), membership(vd)
        expert_probs = []
        for cluster_id in range(clusters):
            sample_weight = trp[:, cluster_id]
            sample_weight = sample_weight * len(sample_weight) / max(float(sample_weight.sum()), 1e-9)
            expert = make_model(seed + 100 * clusters + cluster_id, threads)
            expert.fit(behavior_x, train_y, sample_weight=sample_weight)
            expert_probs.append(expert.predict_proba(val_behavior_x))
        mixture = sum(vap[:, k, None] * expert_probs[k] for k in range(clusters))
        entropy = -np.sum(vap * np.log(np.maximum(vap, 1e-9)), axis=1) / np.log(clusters)
        strength = 0.5 * np.clip(1.0 - entropy, 0.0, 1.0) * np.clip(coverage, 0.0, 1.0)
        pred[f"soft_expert_K{clusters}"] = (
            (1.0 - strength[:, None]) * pred["generic_plus_behavior"]
            + strength[:, None] * mixture)

    gids = np.asarray(val["game_ids"])[val_ok]
    sides = np.asarray(val["sides"]).astype(str)[val_ok]
    mask = np.ones(len(val_y), dtype=bool)
    per_game = {
        name: per_game_scores(val_y, probabilities, gids, sides, mask)
        for name, probabilities in pred.items()}
    summary = {name: equal_game_summary(values) for name, values in per_game.items()}
    report = {
        "schema": "causal-behavior-trend-temporal-holdout-v1",
        "scope": "fit on train; single evaluation on existing validation; development evidence",
        "test_access": False,
        "task": "opponent_trend@30s",
        "label_contract": str(train["global_mode_label_version"][0]),
        "features": "all features at issue time or causal prefix; style router contains motion/visibility only",
        "fit_boundary": "models, KMeans prototypes, scaler and router temperatures fit on train only; no validation tuning",
        "aggregation": "per game and side, red/blue equal within complete game, games equal",
        "train_rows": int(train_ok.sum()),
        "validation_rows": int(val_ok.sum()),
        "train_games": int(len(np.unique(np.asarray(train["game_ids"])[train_ok]))),
        "validation_games": int(len(np.unique(gids))),
        "train_teams": "unavailable in compact split",
        "metrics_game_equal_red_blue_equal": summary,
        "paired_game_bootstrap_95_ci": {
            "generic_plus_behavior_minus_generic": _paired(per_game, "generic_plus_behavior", "generic", seed + 1),
            "soft_expert_K2_minus_generic_plus_behavior": _paired(per_game, "soft_expert_K2", "generic_plus_behavior", seed + 2),
            "soft_expert_K3_minus_generic_plus_behavior": _paired(per_game, "soft_expert_K3", "generic_plus_behavior", seed + 3),
        },
        "limitations": [
            "validation has been repeatedly used during prior development; this is not blind confirmation",
            "time holdout may contain teams also present in train; team identity was not retained",
            "only opponent trend at 30 seconds is evaluated",
            "LightGBM candidate, not a promotion or proof that the neural network improves",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    base = ROOT / "out_m2" / "temporal_forecast_death_aware_candidate"
    canonical = ROOT / "out_m2" / "temporal_forecast_death_aware_candidate"
    parser.add_argument("--train", type=Path, default=base / "train.npz")
    parser.add_argument("--validation", type=Path, default=canonical / "validation.npz")
    parser.add_argument("--output", type=Path, default=ROOT / "out_m2" / "style_experts_pilot" / "temporal_holdout_trend_30s.json")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    report = run(args.train, args.validation, args.output, args.threads)
    print(json.dumps({"output": str(args.output),
                      "metrics": report["metrics_game_equal_red_blue_equal"],
                      "paired_ci": report["paired_game_bootstrap_95_ci"]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
