"""Strict opponent-team-held-out OOF test for behavior and style experts.

Raw match metadata is used only to construct folds and equal-team reports.
No school/team identity enters a feature or predictor. Every held-out team's
matches are removed from model fitting, including rows where that team would
be the observing side. The target is the held-out opponent's future trend.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sqlite3
import sys

import numpy as np
from sklearn.cluster import KMeans

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from forecast.temporal import TRENDS, load_compact  # noqa: E402
from eval.experiment_death_aware_mode_lightgbm import (  # noqa: E402
    paired_bootstrap,
    per_game_scores,
    scores,
)
from eval.experiment_style_conditioned_trend import (  # noqa: E402
    behavior_profile,
    make_model,
    style_posterior,
)
from eval.experiment_death_aware_mode_lightgbm import features as baseline_features  # noqa: E402


def load_match_teams(db_path: Path) -> dict[int, tuple[str, str]]:
    """Read official red/blue school names for split construction only."""
    uri = f"file:{db_path.resolve().as_posix()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA query_only=ON")
        rows = conn.execute('SELECT game_id, "红方学校", "蓝方学校" FROM matches').fetchall()
    finally:
        conn.close()
    result = {}
    for row in rows:
        gid = int(row[0])
        red, blue = str(row[1] or "").strip(), str(row[2] or "").strip()
        if not red or not blue or red == blue:
            continue
        if gid in result and result[gid] != (red, blue):
            raise ValueError(f"conflicting school metadata for game {gid}")
        result[gid] = (red, blue)
    return result


def _team_metrics(labels: np.ndarray, probabilities: np.ndarray,
                  team_names: np.ndarray) -> dict[str, dict]:
    result = {}
    for team in sorted(set(team_names.tolist())):
        selected = team_names == team
        result[str(team)] = scores(labels[selected], probabilities[selected])
    return result


def _equal_team_summary(per_team: dict[str, dict]) -> dict:
    keys = ("accuracy", "macro_f1", "brier", "log_loss", "ece")
    return {key: float(np.mean([row[key] for row in per_team.values()]))
            for key in keys}


def _paired_team_ci(candidate: dict[str, dict], baseline: dict[str, dict],
                    seed: int, reps: int = 5000) -> dict:
    teams = sorted(set(candidate) & set(baseline))
    rng = np.random.default_rng(seed)
    result = {}
    for key in ("accuracy", "macro_f1", "brier", "log_loss", "ece"):
        delta = np.asarray([candidate[t][key] - baseline[t][key] for t in teams])
        sampled = rng.integers(0, len(delta), size=(reps, len(delta)))
        boot = delta[sampled].mean(axis=1)
        result[key] = {
            "delta_candidate_minus_baseline": float(delta.mean()),
            "team_bootstrap_95_ci": [float(np.quantile(boot, .025)),
                                      float(np.quantile(boot, .975))],
            "n_heldout_teams": len(teams),
        }
    return result


def run(train_path: Path, db_path: Path, output: Path, *, folds: int = 5,
        seed: int = 73, threads: int = 4, bootstrap_reps: int = 5000) -> dict:
    data = load_compact(train_path)
    contract = str(data["global_mode_label_version"][0])
    if contract != "relative_margin_v2_death_aware_hp_le_zero":
        raise ValueError(f"unexpected label contract: {contract}")
    y_all = np.asarray(data["trend_y"][:, 1], dtype=np.int64)  # 30 seconds
    valid = (y_all >= 0) & (y_all < len(TRENDS))
    rows = np.flatnonzero(valid)
    game_ids = np.asarray(data["game_ids"], dtype=np.int64)
    sides = np.asarray(data["sides"]).astype(str)
    metadata = load_match_teams(db_path)
    missing = sorted(set(game_ids.tolist()) - set(metadata))
    if missing:
        raise ValueError(f"{len(missing)} compact games lack official team metadata")

    pair_by_game = {gid: metadata[gid] for gid in np.unique(game_ids)}
    teams = sorted({team for pair in pair_by_game.values() for team in pair})
    if len(teams) < folds:
        raise ValueError(f"need at least {folds} teams, found {len(teams)}")
    # Enemy school is the forecast target's identity. The observing school's
    # name is never used as an input feature.
    enemy_team = np.asarray([
        pair_by_game[int(gid)][1] if side == "\u7ea2" else pair_by_game[int(gid)][0]
        for gid, side in zip(game_ids, sides)
    ], dtype=object)
    pair_teams = {int(g): set(pair_by_game[int(g)]) for g in np.unique(game_ids)}
    x = baseline_features(data)
    model_profile, router_profile, _summary = behavior_profile(data)
    x_behavior = np.concatenate([x, model_profile], axis=1)
    y = y_all[rows]
    row_games = game_ids[rows]
    row_sides = sides[rows]
    row_enemy = enemy_team[rows]

    rng = np.random.default_rng(seed)
    shuffled_teams = np.asarray(teams, dtype=object)
    rng.shuffle(shuffled_teams)
    team_fold = {str(team): index % folds
                 for index, team in enumerate(shuffled_teams)}
    predictions = {
        name: np.full((len(rows), len(TRENDS)), np.nan, dtype=np.float64)
        for name in ("generic", "generic_plus_behavior", "soft_expert_K2", "soft_expert_K3")
    }
    fold_details = []

    for fold_id in range(folds):
        heldout = {team for team, assigned in team_fold.items() if assigned == fold_id}
        train_game_mask = np.asarray([
            not (pair_teams[int(gid)] & heldout) for gid in row_games
        ], dtype=bool)
        test_mask = np.asarray([team in heldout for team in row_enemy], dtype=bool)
        train_pos = np.flatnonzero(train_game_mask)
        test_pos = np.flatnonzero(test_mask)
        if not len(train_pos) or not len(test_pos):
            raise ValueError(f"empty train/test in fold {fold_id}")
        train_games = set(row_games[train_pos].tolist())
        test_games = set(row_games[test_pos].tolist())
        leaked = set()
        for gid in train_games:
            leaked |= pair_teams[int(gid)] & heldout
        if leaked:
            raise AssertionError(f"heldout teams leaked into training: {sorted(leaked)}")

        generic = make_model(seed + fold_id, threads)
        generic.fit(x[rows[train_pos]], y[train_pos])
        predictions["generic"][test_pos] = generic.predict_proba(x[rows[test_pos]])

        behavior = make_model(seed + 100 + fold_id, threads)
        behavior.fit(x_behavior[rows[train_pos]], y[train_pos])
        base_test = behavior.predict_proba(x_behavior[rows[test_pos]])
        predictions["generic_plus_behavior"][test_pos] = base_test

        issue = np.asarray(data["issued_at"], dtype=np.float64)
        sampled_local = ((np.round(issue[rows[train_pos]]).astype(int) - 1) % 15) == 0
        if sampled_local.sum() < 100:
            sampled_local = np.ones(len(train_pos), dtype=bool)
        train_router_sample = router_profile[rows[train_pos]][sampled_local]
        cluster_info = {}
        for k in (2, 3):
            km, _train_membership_sample, test_membership, temperature = style_posterior(
                train_router_sample, router_profile[rows[test_pos]], k,
                seed + 1000 * k + fold_id,
            )
            mean = train_router_sample.mean(axis=0)
            scale = train_router_sample.std(axis=0)
            scale[scale <= 1e-6] = 1.0
            train_z = (router_profile[rows[train_pos]] - mean) / scale
            d_train = km.transform(train_z)
            logits = -0.5 * (d_train / temperature) ** 2
            logits -= logits.max(axis=1, keepdims=True)
            membership = np.exp(logits)
            membership /= membership.sum(axis=1, keepdims=True)
            expert_probs = []
            cluster_sizes = []
            for cluster_id in range(k):
                weights = membership[:, cluster_id]
                cluster_sizes.append(float(weights.sum()))
                expert = make_model(seed + 10000 * k + 100 * fold_id + cluster_id, threads)
                expert.fit(x_behavior[rows[train_pos]], y[train_pos], sample_weight=weights)
                expert_probs.append(expert.predict_proba(x_behavior[rows[test_pos]]))
            mixture = sum(test_membership[:, c, None] * expert_probs[c]
                          for c in range(k))
            entropy = -np.sum(test_membership * np.log(np.maximum(test_membership, 1e-9)), axis=1) / np.log(k)
            coverage = np.clip(_summary[rows[test_pos], 6], 0.0, 1.0)
            route_strength = 0.5 * np.clip(1.0 - entropy, 0.0, 1.0) * coverage
            routed = (1.0 - route_strength[:, None]) * base_test + route_strength[:, None] * mixture
            predictions[f"soft_expert_K{k}"][test_pos] = routed
            cluster_info[f"K{k}"] = {
                "fit_sample_count": int(sampled_local.sum()),
                "cluster_effective_weight": cluster_sizes,
                "mean_route_strength": float(route_strength.mean()),
            }
        fold_details.append({
            "fold": fold_id,
            "heldout_team_count": len(heldout),
            "heldout_teams": sorted(heldout),
            "train_match_count": len(train_games),
            "heldout_opponent_match_count": len(test_games),
            "train_row_count": len(train_pos),
            "scored_row_count": len(test_pos),
            "clusters": cluster_info,
        })

    if any(np.any(~np.isfinite(p).all(axis=1)) for p in predictions.values()):
        raise AssertionError("some valid target rows were not predicted")
    per_team = {name: _team_metrics(y, p, row_enemy) for name, p in predictions.items()}
    summaries = {name: _equal_team_summary(values) for name, values in per_team.items()}
    paired = {
        "generic_plus_behavior_minus_generic": _paired_team_ci(
            per_team["generic_plus_behavior"], per_team["generic"], seed + 11, bootstrap_reps),
        "soft_expert_K2_minus_generic_plus_behavior": _paired_team_ci(
            per_team["soft_expert_K2"], per_team["generic_plus_behavior"], seed + 12, bootstrap_reps),
        "soft_expert_K3_minus_generic_plus_behavior": _paired_team_ci(
            per_team["soft_expert_K3"], per_team["generic_plus_behavior"], seed + 13, bootstrap_reps),
    }
    game_scores = {
        name: per_game_scores(y, p, row_games, row_sides,
                              np.ones(len(rows), dtype=bool))
        for name, p in predictions.items()
    }
    game_summaries = {
        name: {metric: float(np.mean([v[metric] for v in values.values()]))
               for metric in ("accuracy", "macro_f1", "brier", "log_loss", "ece")}
        for name, values in game_scores.items()
    }
    game_ci = {
        candidate: paired_bootstrap(game_scores[candidate], game_scores[comparator],
                                    seed=seed + 30 + index, reps=bootstrap_reps)
        for index, (candidate, comparator) in enumerate((
            ("generic_plus_behavior", "generic"),
            ("soft_expert_K2", "generic_plus_behavior"),
            ("soft_expert_K3", "generic_plus_behavior"),
        ))
    }
    report = {
        "schema": "strict-opponent-team-heldout-style-trend-v1",
        "scope": "train split only; unseen opposing school teams; five outer folds",
        "test_access": False,
        "validation_access": False,
        "team_metadata_source": str(db_path),
        "team_metadata_use": "fold assignment and scoring only; school names excluded from all model features",
        "heldout_unit": "opponent school; all matches involving any fold-heldout school are excluded from training",
        "task": "opponent_trend@30s",
        "label_contract": contract,
        "feature_contract": "same generic and causal motion-summary feature sets as prior style OOF; no school/team ID or absolute XY in style router",
        "folds": fold_details,
        "scored_rows": len(rows),
        "scored_games": len(np.unique(row_games)),
        "scored_opponent_teams": len(np.unique(row_enemy)),
        "metrics_equal_heldout_team": summaries,
        "paired_team_bootstrap_95_ci": paired,
        "metrics_game_red_blue_equal": game_summaries,
        "paired_complete_game_bootstrap_95_ci": game_ci,
        "limitations": [
            "This tests transfer to unseen school teams in the observed train split, not every competition or unseen robot roster.",
            "Only the 30-second enemy trend field is tested; no claim is made for other targets or horizons.",
            "Team-equal estimates weight each school equally; game-equal summaries are provided as a second view.",
            "The official team field may aggregate multiple rosters/robots under one school name.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=ROOT / "out_m2" / "temporal_forecast_death_aware_candidate" / "train.npz")
    parser.add_argument("--db", type=Path, default=ROOT / "数据与资料" / "rmuc_2026_region_dataset.sqlite")
    parser.add_argument("--output", type=Path, default=ROOT / "out_m2" / "style_experts_pilot" / "team_heldout_opponent_trend_30s.json")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=73)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--bootstrap-reps", type=int, default=5000)
    args = parser.parse_args()
    report = run(args.train, args.db, args.output, folds=args.folds,
                 seed=args.seed, threads=args.threads,
                 bootstrap_reps=args.bootstrap_reps)
    print(json.dumps({"output": str(args.output),
                      "metrics_equal_heldout_team": report["metrics_equal_heldout_team"],
                      "paired_team_bootstrap_95_ci": report["paired_team_bootstrap_95_ci"],
                      "metrics_game_red_blue_equal": report["metrics_game_red_blue_equal"]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
