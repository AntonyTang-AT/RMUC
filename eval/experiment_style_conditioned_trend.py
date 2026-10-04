"""Train-split OOF pilot for causal, soft style-conditioned opponent-trend experts.

This is an offline experiment only. It reads the death-aware train NPZ, uses
whole-game folds (red/blue together), and does not read validation or test.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.cluster import KMeans
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from forecast.temporal import TRENDS, load_compact  # noqa: E402
from experiment_death_aware_mode_lightgbm import (  # noqa: E402
    equal_game_summary,
    features as baseline_features,
    paired_bootstrap,
    per_game_scores,
    scores,
)


def behavior_profile(data: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Summarize only enemy motion through the current issue time.

    Absolute x/y coordinates are excluded from the style vector. Motion is
    side-oriented toward the observing team, with separate 3 s and 9 s rates.
    Per-slot visibility masks stay explicit, and tracks do not bridge missing
    positions because a velocity is emitted only when both endpoints exist.
    """
    x = np.asarray(data["x"], dtype=np.float32)
    sides = np.asarray(data["sides"])
    current = x[:, -1]
    previous_3 = x[:, -2]
    previous_9 = x[:, -4]
    toward = np.where(np.asarray([ord(str(side)) == 0x7EA2 for side in sides]), -1.0, 1.0)

    per_slot = []
    toward_speeds = []
    lateral_speeds = []
    observed_masks = []
    for slot in range(8, 14):
        offset = slot * 14
        pos = current[:, offset : offset + 4]
        prev3 = previous_3[:, offset : offset + 4]
        prev9 = previous_9[:, offset : offset + 4]
        visible = (pos[:, 3] > 0.5).astype(np.float32)
        mask3 = visible * (prev3[:, 3] > 0.5)
        mask9 = visible * (prev9[:, 3] > 0.5)
        toward3 = (pos[:, 0] - prev3[:, 0]) * toward * 28.0 / 3.0
        toward9 = (pos[:, 0] - prev9[:, 0]) * toward * 28.0 / 9.0
        lateral3 = (pos[:, 1] - prev3[:, 1]) * 15.0 / 3.0
        lateral9 = (pos[:, 1] - prev9[:, 1]) * 15.0 / 9.0
        speed3 = np.sqrt(toward3**2 + lateral3**2)
        acceleration = toward3 - toward9
        highlighted = current[:, offset + 6]
        slot_features = np.column_stack(
            [visible, mask3, mask9, toward3 * mask3, toward9 * mask9,
             lateral3 * mask3, lateral9 * mask9, speed3 * mask3,
             acceleration * (mask3 * mask9), highlighted * visible]
        )
        per_slot.append(slot_features)
        toward_speeds.append(toward3 * mask3)
        lateral_speeds.append(lateral3 * mask3)
        observed_masks.append(mask3)

    slots = np.stack(per_slot, axis=1)  # [N, 6, 10]
    toward_matrix = np.stack(toward_speeds, axis=1)
    lateral_matrix = np.stack(lateral_speeds, axis=1)
    observed = np.stack(observed_masks, axis=1)
    denominator = np.maximum(observed.sum(axis=1), 1.0)
    mean_toward = toward_matrix.sum(axis=1) / denominator
    mean_lateral = lateral_matrix.sum(axis=1) / denominator
    toward_std = np.sqrt(
        (((toward_matrix - mean_toward[:, None]) ** 2) * observed).sum(axis=1) / denominator
    )
    advancing_fraction = ((toward_matrix >= 0.10) * observed).sum(axis=1) / denominator
    retreating_fraction = ((toward_matrix <= -0.10) * observed).sum(axis=1) / denominator
    still_fraction = ((np.abs(toward_matrix) < 0.10) * observed).sum(axis=1) / denominator
    current_hp = np.stack([current[:, slot * 14 + 2] for slot in range(8, 14)], axis=1)
    visible_now = slots[:, :, 0]
    mean_hp = (current_hp * visible_now).sum(axis=1) / np.maximum(visible_now.sum(axis=1), 1.0)
    summary = np.column_stack(
        [
            mean_toward,
            mean_lateral,
            toward_std,
            advancing_fraction,
            retreating_fraction,
            still_fraction,
            observed.mean(axis=1),
            visible_now.mean(axis=1),
            mean_hp,
        ]
    )
    model_profile = np.concatenate([slots.reshape(len(x), -1), summary], axis=1)
    # The router excludes highlight and HP level; it groups motion patterns,
    # not damage state. Visibility masks remain so uncertainty can reduce the
    # expert contribution when little movement evidence is available.
    router_profile = np.concatenate([slots[:, :, :9].reshape(len(x), -1), summary[:, :8]], axis=1)
    return model_profile, router_profile, summary


def make_model(seed: int, threads: int) -> LGBMClassifier:
    return LGBMClassifier(
        objective="multiclass",
        num_class=len(TRENDS),
        n_estimators=140,
        learning_rate=0.055,
        num_leaves=9,
        min_child_samples=120,
        reg_lambda=5.0,
        colsample_bytree=0.82,
        subsample=0.85,
        subsample_freq=1,
        verbosity=-1,
        n_jobs=threads,
        random_state=seed,
    )


def style_posterior(train_profile: np.ndarray, predict_profile: np.ndarray, n_clusters: int, seed: int):
    """Fit train-only scaler/prototypes on 15-second game-balanced samples."""
    mean = train_profile.mean(axis=0)
    scale = train_profile.std(axis=0)
    scale[scale <= 1e-6] = 1.0
    train_z = (train_profile - mean) / scale
    predict_z = (predict_profile - mean) / scale
    km = KMeans(n_clusters=n_clusters, n_init=20, random_state=seed)
    km.fit(train_z)
    train_d = km.transform(train_z)
    predict_d = km.transform(predict_z)
    nearest = np.min(train_d, axis=1)
    temperature = float(np.quantile(nearest, 0.75))
    temperature = max(temperature, 1e-3)
    train_logits = -0.5 * (train_d / temperature) ** 2
    predict_logits = -0.5 * (predict_d / temperature) ** 2
    train_logits -= train_logits.max(axis=1, keepdims=True)
    predict_logits -= predict_logits.max(axis=1, keepdims=True)
    train_p = np.exp(train_logits)
    predict_p = np.exp(predict_logits)
    train_p /= train_p.sum(axis=1, keepdims=True)
    predict_p /= predict_p.sum(axis=1, keepdims=True)
    return km, train_p, predict_p, temperature


def run(train_path: Path, output: Path, folds: int = 5, seed: int = 31, threads: int = 4):
    data = load_compact(train_path)
    horizon_index = 1  # 30 s, primary tactical horizon for this first pilot.
    horizon_seconds = (10, 30, 60, 120)[horizon_index]
    y = np.asarray(data["trend_y"][:, horizon_index], dtype=np.int64)
    valid = (y >= 0) & (y < len(TRENDS))
    rows = np.flatnonzero(valid)
    y_valid = y[valid]
    groups = np.asarray(data["game_ids"])[valid]
    sides = np.asarray(data["sides"])[valid]
    issue_times = np.asarray(data["issued_at"])[valid]
    x = baseline_features(data)[valid]
    model_profile, router_profile, behavior_summary = behavior_profile(data)
    profile = model_profile[valid]
    routing = router_profile[valid]
    summary = behavior_summary[valid]
    x_with_behavior = np.concatenate([x, profile], axis=1)

    model_names = ["generic", "generic_plus_behavior", "soft_expert_K2", "soft_expert_K3", "velocity_rule"]
    oof = {name: np.full((len(rows), len(TRENDS)), np.nan, dtype=np.float64) for name in model_names}
    group_fold = GroupKFold(n_splits=folds)
    fold_details = []

    for fold, (train_pos, test_pos) in enumerate(group_fold.split(x, y_valid, groups)):
        train_global_rows = rows[train_pos]
        test_global_rows = rows[test_pos]
        # x/profile arrays are already filtered to valid target rows; raw row
        # indices are retained separately only for reading causal source arrays.
        train_x = x[train_pos]
        test_x = x[test_pos]
        train_behavior = x_with_behavior[train_pos]
        test_behavior = x_with_behavior[test_pos]
        y_train = y[train_global_rows]

        generic = make_model(seed + fold, threads)
        generic.fit(train_x, y_train)
        oof["generic"][test_pos] = generic.predict_proba(test_x)

        generic_behavior = make_model(seed + 100 + fold, threads)
        generic_behavior.fit(train_behavior, y_train)
        oof["generic_plus_behavior"][test_pos] = generic_behavior.predict_proba(test_behavior)

        # K-means sees one prefix per 15 s per game/side when fitting prototypes;
        # experts still use all causal rows, downweighted by their soft posterior.
        sampled = ((np.round(np.asarray(data["issued_at"])[train_global_rows]).astype(int) - 1) % 15) == 0
        if sampled.sum() < 100:
            sampled = np.ones(len(train_global_rows), dtype=bool)
        fold_cluster_sizes = {}
        for n_clusters in (2, 3):
            km, train_membership, test_membership, temperature = style_posterior(
                routing[train_pos][sampled],
                routing[test_pos],
                n_clusters,
                seed + 1000 * n_clusters + fold,
            )
            # Recompute posterior for every expert-training row using prototypes
            # fit on the balanced sample, keeping all scaler/prototype fitting
            # inside this outer training fold.
            mean = routing[train_pos][sampled].mean(axis=0)
            scale = routing[train_pos][sampled].std(axis=0)
            scale[scale <= 1e-6] = 1.0
            all_train_z = (routing[train_pos] - mean) / scale
            d_train = km.transform(all_train_z)
            logits = -0.5 * (d_train / temperature) ** 2
            logits -= logits.max(axis=1, keepdims=True)
            membership = np.exp(logits)
            membership /= membership.sum(axis=1, keepdims=True)
            # Update only the downstream prediction with explicit soft assignment.
            cluster_predictions = []
            for cluster_id in range(n_clusters):
                weight = membership[:, cluster_id]
                weight = weight * len(weight) / max(float(weight.sum()), 1e-9)
                expert = make_model(seed + 10000 * n_clusters + 100 * fold + cluster_id, threads)
                expert.fit(train_behavior, y_train, sample_weight=weight)
                cluster_predictions.append(expert.predict_proba(test_behavior))
            mixture = sum(
                test_membership[:, cluster_id, None] * cluster_predictions[cluster_id]
                for cluster_id in range(n_clusters)
            )
            # Fixed, predeclared shrinkage limits damage when a style posterior is
            # uncertain. It is identical for K=2 and K=3 and is not tuned on OOF.
            route_entropy = -np.sum(
                test_membership * np.log(np.maximum(test_membership, 1e-9)), axis=1
            ) / np.log(n_clusters)
            motion_coverage = np.clip(summary[test_pos, 6], 0.0, 1.0)
            route_strength = 0.5 * np.clip(1.0 - route_entropy, 0.0, 1.0) * motion_coverage
            routed = (
                (1.0 - route_strength[:, None]) * oof["generic_plus_behavior"][test_pos]
                + route_strength[:, None] * mixture
            )
            oof[f"soft_expert_K{n_clusters}"][test_pos] = routed
            fold_cluster_sizes[f"K{n_clusters}"] = {
                "fit_sample_count": int(sampled.sum()),
                "prototype_counts": np.bincount(km.labels_, minlength=n_clusters).tolist(),
                "mean_route_entropy": float(np.mean(route_entropy)),
                "mean_route_strength": float(np.mean(route_strength)),
            }

        # A causal physics baseline: extrapolate observed opponent centroid
        # velocity in the label's push/retreat direction for 30 seconds.
        direction = np.where(np.asarray([ord(str(s)) == 0x7EA2 for s in sides[test_pos]]), -1.0, 1.0)
        current = np.asarray(data["x"])[test_global_rows, -1]
        previous = np.asarray(data["x"])[test_global_rows, -2]
        velocities = []
        masks = []
        for slot in range(8, 14):
            offset = slot * 14
            now = current[:, offset : offset + 4]
            prev = previous[:, offset : offset + 4]
            mask = (now[:, 3] > 0.5) & (prev[:, 3] > 0.5)
            velocities.append((now[:, 0] - prev[:, 0]) * direction * 28.0 / 3.0)
            masks.append(mask)
        velocity = np.stack(velocities, axis=1)
        velocity_mask = np.stack(masks, axis=1)
        centroid_speed = (velocity * velocity_mask).sum(axis=1) / np.maximum(velocity_mask.sum(axis=1), 1)
        displacement = centroid_speed * horizon_seconds
        physics = np.full((len(test_pos), len(TRENDS)), 0.01, dtype=np.float64)
        physics[np.arange(len(test_pos)), np.where(displacement >= 0.8, 0, np.where(displacement <= -0.8, 2, 1))] = 0.98
        oof["velocity_rule"][test_pos] = physics
        fold_details.append({"fold": fold, "heldout_game_count": int(len(np.unique(groups[test_pos]))),
                             "train_game_count": int(len(np.unique(groups[train_pos]))),
                             "clusters": fold_cluster_sizes})

    # Group-equal paired metrics: red/blue first equal within a game, then games equal.
    per_model_games = {}
    for name, probability in oof.items():
        per_model_games[name] = per_game_scores(
            y_valid, probability, groups, sides, np.ones(len(y_valid), dtype=bool)
        )
    game_summary = {name: equal_game_summary(rows_by_game) for name, rows_by_game in per_model_games.items()}
    ci = {}
    for candidate in ("generic_plus_behavior", "soft_expert_K2", "soft_expert_K3"):
        ci[candidate] = {
            comparator: paired_bootstrap(
                per_model_games[candidate], per_model_games[comparator], seed=seed + 500 + i
            )
            for i, comparator in enumerate(("generic", "generic_plus_behavior", "velocity_rule"))
            if comparator != candidate
        }

    report = {
        "schema": "causal-style-conditioned-opponent-trend-oof-v1",
        "scope": "train-only development OOF",
        "test_access": False,
        "validation_access": False,
        "task": "opponent_trend",
        "horizon_seconds": horizon_seconds,
        "label_contract": str(data["global_mode_label_version"][0]),
        "groups": "GroupKFold by complete game_id; red/blue are always in the same fold",
        "style_features": "enemy visible mobile-slot movement at 3 s and 9 s, acceleration proxy, coordination, visibility masks, HP summary; no absolute x/y coordinates",
        "style_router": "fold-fitted KMeans K=2/3 on 15-second-sampled training prefixes, soft distance posterior, sample-weighted experts, fixed 0.5 shrinkage to generic behavior model",
        "metrics_game_equal_red_blue_equal": game_summary,
        "paired_game_bootstrap_95_ci": ci,
        "folds": fold_details,
        "n_scored_rows": int(len(y_valid)),
        "n_games": int(len(np.unique(groups))),
    }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=ROOT / "out_m2" / "temporal_forecast_death_aware_candidate" / "train.npz")
    parser.add_argument("--output", type=Path, default=ROOT / "out_m2" / "style_experts_pilot" / "opponent_trend_30s" / "report.json")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    report = run(args.train, args.output, args.folds, args.seed, args.threads)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "metrics": report["metrics_game_equal_red_blue_equal"], "ci": report["paired_game_bootstrap_95_ci"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
