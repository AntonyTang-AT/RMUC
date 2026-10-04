"""Train-fold-only early-match style experts for opponent trend@30s.

Each game/observer-side receives one soft style posterior derived from the
opponent's observable motion during the first 30 seconds. That posterior is
frozen for later forecasts in the same match. Whole games are held out by
GroupKFold; normalization and KMeans prototypes are fitted only on fold-train
games. No team identity, future target, or test split is read.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from sklearn.cluster import KMeans
from sklearn.model_selection import GroupKFold

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


def _posterior(train_profiles, predict_profiles, clusters, seed):
    mean = train_profiles.mean(axis=0)
    scale = train_profiles.std(axis=0)
    scale[scale <= 1e-6] = 1.0
    train_z = (train_profiles - mean) / scale
    predict_z = (predict_profiles - mean) / scale
    model = KMeans(n_clusters=clusters, n_init=20,
                   random_state=seed).fit(train_z)
    train_dist = model.transform(train_z)
    predict_dist = model.transform(predict_z)
    temperature = max(float(np.quantile(np.min(train_dist, axis=1), .75)), 1e-3)

    def soft(distance):
        logits = -0.5 * (distance / temperature) ** 2
        logits -= logits.max(axis=1, keepdims=True)
        probability = np.exp(logits)
        return probability / probability.sum(axis=1, keepdims=True)

    return soft(train_dist), soft(predict_dist), model, temperature


def _paired(per_game, candidate, comparator, seed):
    raw = paired_bootstrap(per_game[candidate], per_game[comparator],
                           seed=seed, reps=2000)
    return {
        metric: {
            "delta_candidate_minus_comparator": values[
                "paired_mean_delta_candidate_minus_persistence"],
            "game_bootstrap_95_ci": values["game_bootstrap_95_ci"],
            "n_games_red_blue_paired": values["n_games_red_blue_paired"],
        }
        for metric, values in raw.items()
    }


def run(train_path: Path, output: Path, folds=5, seed=211, threads=4):
    data = load_compact(train_path)
    if str(data["global_mode_label_version"][0]) != "relative_margin_v2_death_aware_hp_le_zero":
        raise ValueError("expected death-aware train label contract")

    target = np.asarray(data["trend_y"][:, 1], dtype=np.int64)
    issue_time = np.asarray(data["issued_at"], dtype=np.float32)
    game_id = np.asarray(data["game_ids"])
    sides = np.asarray(data["sides"]).astype(str)
    eligible = (target >= 0) & (target < len(TRENDS)) & (issue_time >= 30.0)
    rows = np.flatnonzero(eligible)
    y = target[rows]
    row_games = game_id[rows]
    row_sides = sides[rows]
    raw_x = np.asarray(data["x"], dtype=np.float32)[:, -1]
    model_profile, route_profile, behavior = behavior_profile(data)
    behavior_x = np.concatenate([raw_x, model_profile], axis=1)

    # One stable early-game profile per (game, observing side). No row after
    # second 30 can contribute to the profile used to route later predictions.
    unique_pairs = sorted(set(zip(game_id.tolist(), sides.tolist())),
                          key=lambda pair: (str(pair[0]), pair[1]))
    pair_to_index = {pair: i for i, pair in enumerate(unique_pairs)}
    row_pair_index = np.asarray([pair_to_index[(g, s)]
                                 for g, s in zip(game_id.tolist(), sides.tolist())])
    style_profiles = np.zeros((len(unique_pairs), route_profile.shape[1]),
                              dtype=np.float32)
    early_counts = np.zeros(len(unique_pairs), dtype=np.int64)
    for index in range(len(unique_pairs)):
        mask = (row_pair_index == index) & (issue_time <= 30.0)
        if not mask.any():
            raise ValueError(f"missing observable opening profile for pair {unique_pairs[index]}")
        style_profiles[index] = route_profile[mask].mean(axis=0)
        early_counts[index] = int(mask.sum())

    oof = {name: np.full((len(rows), len(TRENDS)), np.nan, dtype=np.float64)
           for name in ("snapshot", "snapshot_plus_causal_behavior",
                        "early_style_K2", "early_style_K3")}
    cv = GroupKFold(n_splits=folds)
    fold_details = []
    row_game_groups = row_games

    for fold, (train_pos, test_pos) in enumerate(
            cv.split(np.zeros(len(rows)), y, groups=row_game_groups)):
        train_rows, test_rows = rows[train_pos], rows[test_pos]
        train_pairs = np.unique(row_pair_index[train_rows])
        test_pairs = np.unique(row_pair_index[test_rows])
        if set(game_id[train_rows].tolist()) & set(game_id[test_rows].tolist()):
            raise AssertionError("whole-game split crossed fold boundary")

        # Give every training game-side equal total weight so long games do
        # not define the style experts merely by contributing more 3 s rows.
        _, inverse, counts = np.unique(row_pair_index[train_rows],
                                       return_inverse=True, return_counts=True)
        group_weights = 1.0 / counts[inverse]
        group_weights *= len(group_weights) / group_weights.sum()
        generic = make_model(seed + fold, threads)
        generic.fit(raw_x[train_rows], target[train_rows],
                    sample_weight=group_weights)
        behavior_model = make_model(seed + 10 + fold, threads)
        behavior_model.fit(behavior_x[train_rows], target[train_rows],
                           sample_weight=group_weights)
        oof["snapshot"][test_pos] = generic.predict_proba(raw_x[test_rows])
        oof["snapshot_plus_causal_behavior"][test_pos] = behavior_model.predict_proba(
            behavior_x[test_rows])

        for clusters in (2, 3):
            train_membership, test_membership, prototypes, temperature = _posterior(
                style_profiles[train_pairs], style_profiles[test_pairs],
                clusters, seed + fold * 100 + clusters)
            train_pair_to_local = {int(pair): j for j, pair in enumerate(train_pairs)}
            test_pair_to_local = {int(pair): j for j, pair in enumerate(test_pairs)}
            row_train_membership = np.stack([
                train_membership[train_pair_to_local[int(pair)]]
                for pair in row_pair_index[train_rows]], axis=0)
            row_test_membership = np.stack([
                test_membership[test_pair_to_local[int(pair)]]
                for pair in row_pair_index[test_rows]], axis=0)
            expert_probs = []
            for cluster in range(clusters):
                weights = group_weights * row_train_membership[:, cluster]
                weights *= len(weights) / max(float(weights.sum()), 1e-9)
                expert = make_model(seed + 1000 * clusters + fold * 10 + cluster,
                                    threads)
                expert.fit(behavior_x[train_rows], target[train_rows],
                           sample_weight=weights)
                expert_probs.append(expert.predict_proba(behavior_x[test_rows]))
            mixture = sum(row_test_membership[:, k, None] * expert_probs[k]
                          for k in range(clusters))
            entropy = -np.sum(row_test_membership * np.log(
                np.maximum(row_test_membership, 1e-9)), axis=1) / np.log(clusters)
            coverage = np.clip(behavior[test_rows, 6], 0.0, 1.0)
            strength = 0.5 * np.clip(1.0 - entropy, 0.0, 1.0) * coverage
            name = f"early_style_K{clusters}"
            oof[name][test_pos] = (
                (1.0 - strength[:, None]) * oof["snapshot_plus_causal_behavior"][test_pos]
                + strength[:, None] * mixture)
            fold_details.append({
                "fold": fold,
                "clusters": clusters,
                "train_games": int(len(np.unique(game_id[train_rows]))),
                "test_games": int(len(np.unique(game_id[test_rows]))),
                "train_game_side_profiles": int(len(train_pairs)),
                "test_game_side_profiles": int(len(test_pairs)),
                "cluster_profile_counts": np.bincount(prototypes.labels_,
                                                       minlength=clusters).tolist(),
                "mean_route_entropy": float(entropy.mean()),
                "mean_route_strength": float(strength.mean()),
                "profile_window_seconds": [0, 30],
                "prototype_temperature": temperature,
            })

    gids = row_games
    per_game = {
        name: per_game_scores(y, probability, gids, row_sides,
                              np.ones(len(y), dtype=bool))
        for name, probability in oof.items()
    }
    report = {
        "schema": "early-match-soft-style-experts-trend30-group-oof-v1",
        "scope": "train-only five-fold whole-game OOF; development evidence",
        "test_access": False,
        "label_contract": str(data["global_mode_label_version"][0]),
        "task": "opponent_trend@30s, forecast issues at or after second 30",
        "train_rows_scored": int(len(rows)),
        "train_games": int(len(np.unique(row_games))),
        "game_side_profiles": int(len(unique_pairs)),
        "mean_opening_observations_per_game_side": float(early_counts.mean()),
        "profile_source": "opponent motion/visibility summaries from causal inputs with issue_time <= 30s",
        "profile_freeze": "one opening profile per game and observing side; reused only at later issue times",
        "split": "GroupKFold by game_id, red and blue remain in same fold",
        "training_weighting": "each game-side has equal total weight; expert weights multiply soft early-style membership",
        "style_identity_limit": "no school/team identity exists in compact split; clusters are match-side behavior regimes, not persistent team labels",
        "metrics_game_equal_red_blue_equal": {
            name: equal_game_summary(values) for name, values in per_game.items()},
        "paired_game_bootstrap_95_ci": {
            "snapshot_plus_causal_behavior_minus_snapshot": _paired(
                per_game, "snapshot_plus_causal_behavior", "snapshot", seed + 1),
            "early_style_K2_minus_snapshot_plus_behavior": _paired(
                per_game, "early_style_K2", "snapshot_plus_causal_behavior", seed + 2),
            "early_style_K3_minus_snapshot_plus_behavior": _paired(
                per_game, "early_style_K3", "snapshot_plus_causal_behavior", seed + 3),
        },
        "fold_details": fold_details,
        "limitations": [
            "clusters describe observable opponent behavior in one match, not stable school tactics",
            "only 30-second opponent trend is evaluated",
            "opening-profile route is a fixed OOF experiment, not an online learner",
            "no untouched final split is used for hyperparameter selection or confirmation",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=ROOT / "out_m2" /
                        "temporal_forecast_death_aware_candidate" / "train.npz")
    parser.add_argument("--output", type=Path, default=ROOT / "out_m2" /
                        "style_experts_pilot" / "early_match_style_experts_trend_30s.json")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    report = run(args.train, args.output, folds=args.folds, threads=args.threads)
    print(json.dumps({
        "output": str(args.output),
        "metrics": report["metrics_game_equal_red_blue_equal"],
        "paired_ci": report["paired_game_bootstrap_95_ci"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
