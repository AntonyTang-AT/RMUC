"""Causal OOF test of match-local, behavior-conditioned opponent calibration.

The supervised trend model and context priors are fitted inside whole-game
training folds. In each held-out game-side, observable enemy motion assigns a
coarse advance/hold/retreat context. Only one forecast per non-overlapping
10-second issue bucket is registered for online feedback. At maturity, the
outcome is recomputed from the current sensor/radar snapshot; it updates only
that match-side's context distribution and may calibrate later forecasts.
Serialized future targets are used for training/evaluation labels only, never
for the online update.
"""
from __future__ import annotations

import argparse
import gzip
import json
from collections import defaultdict, deque
from pathlib import Path
import sys

import numpy as np
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
from runtime.observable_feedback import feedback_labels_from_observable  # noqa: E402


CONTEXTS = ("advance", "hold_or_mixed", "retreat")
HORIZON = 10
PRIOR_STRENGTH = 5.0
MAX_LOCAL_OUTCOMES = 20


def motion_context(summary_row):
    """Map only current causal motion summaries to an interpretable regime."""
    toward_speed = float(summary_row[0])
    advancing = float(summary_row[3])
    retreating = float(summary_row[4])
    if toward_speed >= 0.10 and advancing >= retreating + 0.15:
        return 0
    if toward_speed <= -0.10 and retreating >= advancing + 0.15:
        return 2
    return 1


def _train_context_priors(train_rows, y, contexts, game_side, issue_time):
    """Fit fold-only conditional trend priors on separated issue buckets."""
    samples = []
    for group in np.unique(game_side[train_rows]):
        ix = train_rows[game_side[train_rows] == group]
        ix = ix[np.argsort(issue_time[ix])]
        if not len(ix):
            continue
        origin = float(issue_time[ix[0]])
        seen = set()
        for row in ix:
            if y[row] < 0:
                continue
            bucket = int(np.floor((float(issue_time[row]) - origin) / HORIZON + 1e-6))
            if bucket in seen:
                continue
            seen.add(bucket)
            samples.append(int(row))
    samples = np.asarray(samples, dtype=np.int64)
    global_counts = np.bincount(y[samples], minlength=len(TRENDS)).astype(np.float64)
    global_prior = (global_counts + 1.0) / (global_counts.sum() + len(TRENDS))
    priors = np.zeros((len(CONTEXTS), len(TRENDS)), dtype=np.float64)
    context_counts = np.zeros(len(CONTEXTS), dtype=np.int64)
    for context in range(len(CONTEXTS)):
        selected = samples[contexts[samples] == context]
        counts = np.bincount(y[selected], minlength=len(TRENDS)).astype(np.float64)
        context_counts[context] = len(selected)
        # Shrink sparse behavior contexts toward the whole-fold class prior.
        priors[context] = ((counts + PRIOR_STRENGTH * global_prior) /
                           (len(selected) + PRIOR_STRENGTH))
    return priors, global_prior, context_counts, len(samples)


def _score(y, p):
    y = np.asarray(y, dtype=np.int64)
    p = np.asarray(p, dtype=np.float64)
    p = np.clip(p, 1e-9, None)
    p /= p.sum(axis=1, keepdims=True)
    pred = p.argmax(axis=1)
    onehot = np.eye(len(TRENDS))[y]
    f1 = []
    for cls in range(len(TRENDS)):
        tp = np.sum((y == cls) & (pred == cls))
        fp = np.sum((y != cls) & (pred == cls))
        fn = np.sum((y == cls) & (pred != cls))
        f1.append(2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0)
    return {
        "n": int(len(y)),
        "accuracy": float(np.mean(pred == y)),
        "macro_f1": float(np.mean(f1)),
        "brier": float(np.mean(np.sum((p - onehot) ** 2, axis=1))),
        "log_loss": float(-np.mean(np.log(np.maximum(p[np.arange(len(y)), y], 1e-9)))),
    }


def _summary(game_scores):
    keys = ("accuracy", "macro_f1", "brier", "log_loss")
    if not game_scores:
        return {}
    return {key: float(np.mean([values[key] for values in game_scores.values()]))
            for key in keys}


def _paired(game_scores, candidate, baseline, seed, reps=2000):
    raw = paired_bootstrap(game_scores[candidate], game_scores[baseline],
                           seed=seed, reps=reps)
    return {
        metric: {
            "delta_candidate_minus_baseline": values[
                "paired_mean_delta_candidate_minus_persistence"],
            "game_bootstrap_95_ci": values["game_bootstrap_95_ci"],
            "n_games_red_blue_paired": values["n_games_red_blue_paired"],
        }
        for metric, values in raw.items()
    }


def run(train_npz: Path, train_jsonl: Path, output: Path,
        folds=5, seed=307, threads=4):
    data = load_compact(train_npz)
    contract = str(data["global_mode_label_version"][0])
    if contract != "relative_margin_v2_death_aware_hp_le_zero":
        raise ValueError(f"unexpected target contract: {contract}")
    n = len(data["x"])
    game_ids = np.asarray(data["game_ids"])
    sides = np.asarray(data["sides"]).astype(str)
    issue_time = np.asarray(data["issued_at"], dtype=np.float64)
    game_side = np.asarray([f"{g}:{s}" for g, s in zip(game_ids, sides)])
    y = np.asarray(data["trend_y"][:, 0], dtype=np.int64)
    score_mask = (y >= 0) & (y < len(TRENDS))
    score_rows = np.flatnonzero(score_mask)
    raw_x = np.asarray(data["x"], dtype=np.float32)[:, -1]
    model_profile, _router_profile, summary = behavior_profile(data)
    behavior_x = np.concatenate([raw_x, model_profile], axis=1)
    contexts = np.asarray([motion_context(row) for row in summary], dtype=np.int64)

    # Confirm row-by-row alignment and contiguous match-side blocks before the
    # sequential replay, so feedback never crosses a game or observing side.
    group_start = np.r_[True, game_side[1:] != game_side[:-1]]
    compact_groups = int(group_start.sum())
    fold = GroupKFold(n_splits=folds)
    oof_model = np.full((n, len(TRENDS)), np.nan, dtype=np.float64)
    row_fold = np.full(n, -1, dtype=np.int16)
    fold_priors = {}
    fold_details = []
    valid_rows = score_rows
    for fold_id, (train_pos, test_pos) in enumerate(
            fold.split(behavior_x[valid_rows], y[valid_rows],
                       groups=game_ids[valid_rows])):
        train_rows = valid_rows[train_pos]
        test_games = np.unique(game_ids[valid_rows[test_pos]])
        # Predict every row in held-out games: censored final rows may still
        # be valid online observations that mature earlier forecasts.
        all_test_rows = np.flatnonzero(np.isin(game_ids, test_games))
        if set(game_ids[train_rows].tolist()) & set(game_ids[all_test_rows].tolist()):
            raise AssertionError("whole game crossed train/test fold boundary")
        model = make_model(seed + fold_id, threads)
        model.fit(behavior_x[train_rows], y[train_rows])
        oof_model[all_test_rows] = model.predict_proba(behavior_x[all_test_rows])
        row_fold[all_test_rows] = fold_id
        priors, global_prior, counts, prior_rows = _train_context_priors(
            train_rows, y, contexts, game_side, issue_time)
        fold_priors[fold_id] = priors
        fold_details.append({
            "fold": fold_id,
            "train_games": int(len(np.unique(game_ids[train_rows]))),
            "test_games": int(len(test_games)),
            "training_label_rows": int(len(train_rows)),
            "context_prior_sample_count": int(prior_rows),
            "context_prior_counts": {CONTEXTS[i]: int(counts[i])
                                     for i in range(len(CONTEXTS))},
            "global_trend_prior": global_prior.tolist(),
        })

    if np.any(row_fold[score_rows] < 0) or np.any(~np.isfinite(oof_model[score_rows]).all(axis=1)):
        raise AssertionError("missing OOF model predictions for scored rows")

    oof_calibrated = np.full_like(oof_model, np.nan)
    oof_residual_calibrated = np.full_like(oof_model, np.nan)
    oof_hold_residual_calibrated = np.full_like(oof_model, np.nan)
    online_updates = 0
    online_records_issued = 0
    online_groups = 0
    feedback_label_counts = np.zeros(len(TRENDS), dtype=np.int64)
    stream_index = 0
    active_key = None
    pending = []
    local_outcomes = None
    local_residuals = None
    sampled_buckets = None
    origin = None

    def process_group():
        nonlocal online_updates, online_records_issued, online_groups
        if active_key is not None:
            online_groups += 1

    with gzip.open(train_jsonl, "rt", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            source = json.loads(line)
            if stream_index >= n:
                raise ValueError("JSONL has more rows than compact NPZ")
            gid = int(source["game_id"])
            side = str(source["side"])
            now = float(source["issued_at"])
            if (gid != int(game_ids[stream_index]) or side != str(sides[stream_index])
                    or abs(now - issue_time[stream_index]) > 1e-4):
                raise ValueError(f"JSONL/NPZ alignment mismatch at row {stream_index}")
            key = (gid, side)
            if key != active_key:
                process_group()
                active_key = key
                pending = []
                local_outcomes = {context: deque(maxlen=MAX_LOCAL_OUTCOMES)
                                  for context in range(len(CONTEXTS))}
                local_residuals = {
                    strategy: {context: deque(maxlen=MAX_LOCAL_OUTCOMES)
                               for context in range(len(CONTEXTS))}
                    for strategy in ("all", "hold_only")
                }
                sampled_buckets = set()
                origin = now
            visible = {"game_id": gid, "side": side, "issued_at": now,
                       "observable": source.get("observable") or {}}
            # Mature only outcomes whose 10-second target is at/before now;
            # this path never looks at source['targets'].
            still_pending = []
            for record in pending:
                if now + 1e-6 < record["issued_at"] + HORIZON:
                    still_pending.append(record)
                    continue
                labels = feedback_labels_from_observable(
                    record["snapshot"], visible, side, HORIZON)
                label = labels.get("opponent_trend")
                if label in TRENDS:
                    cls = TRENDS.index(label)
                    local_outcomes[record["context"]].append(cls)
                    target = np.eye(len(TRENDS), dtype=np.float64)[cls]
                    for strategy in ("all", "hold_only"):
                        local_residuals[strategy][record["context"]].append(
                            target - record["issued_probabilities"][strategy])
                    feedback_label_counts[cls] += 1
                    online_updates += 1
            pending = still_pending

            fold_id = int(row_fold[stream_index])
            if fold_id >= 0:
                base = oof_model[stream_index]
                context = int(contexts[stream_index])
                prior = fold_priors[fold_id][context]
                history = list(local_outcomes[context])
                counts = np.bincount(history, minlength=len(TRENDS)).astype(np.float64)
                n_history = len(history)
                posterior = ((PRIOR_STRENGTH * prior + counts) /
                             (PRIOR_STRENGTH + n_history))
                local_weight = n_history / (n_history + PRIOR_STRENGTH)
                calibrated = ((1.0 - local_weight) * base +
                              local_weight * posterior)
                calibrated = np.clip(calibrated, 1e-9, None)
                calibrated /= calibrated.sum()
                oof_calibrated[stream_index] = calibrated

                # Small additive logit correction from the local model's
                # matured residuals. This learns systematic over/under-prediction
                # within the current observable behavior context without
                # replacing the discriminative model by a local class prior.
                def residual_corrected(strategy):
                    residual_history = list(local_residuals[strategy][context])
                    residual_count = len(residual_history)
                    if not residual_count:
                        return base.copy()
                    residual_mean = np.mean(residual_history, axis=0)
                    residual_shrink = residual_count / (residual_count + 10.0)
                    corrected_logits = (np.log(np.clip(base, 1e-9, 1.0))
                                        + residual_shrink * residual_mean)
                    corrected_logits -= float(np.max(corrected_logits))
                    corrected = np.exp(corrected_logits)
                    return corrected / corrected.sum()

                corrected = residual_corrected("all")
                hold_corrected = (residual_corrected("hold_only")
                                  if context == 1 else base.copy())
                oof_residual_calibrated[stream_index] = corrected
                # Exploratory context gate: retain the residual correction only
                # in hold/mixed and use the fixed model in advance/retreat.
                oof_hold_residual_calibrated[stream_index] = hold_corrected

                bucket = int(np.floor((now - origin) / HORIZON + 1e-6))
                if bucket not in sampled_buckets:
                    sampled_buckets.add(bucket)
                    pending.append({
                        "issued_at": now,
                        "context": context,
                        "snapshot": visible,
                        "issued_probabilities": {
                            "all": corrected.copy(),
                            "hold_only": hold_corrected.copy(),
                        },
                    })
                    online_records_issued += 1
            stream_index += 1
    process_group()
    if stream_index != n:
        raise ValueError(f"JSONL rows {stream_index} != compact rows {n}")
    if (np.any(~np.isfinite(oof_calibrated[score_rows]).all(axis=1))
            or np.any(~np.isfinite(oof_residual_calibrated[score_rows]).all(axis=1))
            or np.any(~np.isfinite(oof_hold_residual_calibrated[score_rows]).all(axis=1))):
        raise AssertionError("missing online OOF calibrated predictions")

    gids = game_ids[score_rows]
    score_sides = sides[score_rows]
    labels = y[score_rows]
    persistence = np.full((len(score_rows), len(TRENDS)), 0.01, dtype=np.float64)
    persistence[:, TRENDS.index("稳定")] = 0.98
    model_scores = per_game_scores(labels, oof_model[score_rows], gids,
                                   score_sides, np.ones(len(score_rows), bool))
    calibrated_scores = per_game_scores(labels, oof_calibrated[score_rows], gids,
                                        score_sides, np.ones(len(score_rows), bool))
    residual_scores = per_game_scores(labels, oof_residual_calibrated[score_rows], gids,
                                      score_sides, np.ones(len(score_rows), bool))
    hold_residual_scores = per_game_scores(
        labels, oof_hold_residual_calibrated[score_rows], gids, score_sides,
        np.ones(len(score_rows), bool))
    persistence_scores = per_game_scores(labels, persistence, gids,
                                         score_sides, np.ones(len(score_rows), bool))
    per_game = {"fixed_oof_model": model_scores,
                "online_behavior_calibration": calibrated_scores,
                "online_residual_calibration": residual_scores,
                "online_hold_residual_calibration": hold_residual_scores,
                "persistence_stable": persistence_scores}
    metrics_by_context = {}
    context_rows = contexts[score_rows]
    for context_id, context_name in enumerate(CONTEXTS):
        selected = context_rows == context_id
        context_per_game = {
            "fixed_oof_model": per_game_scores(
                labels[selected], oof_model[score_rows][selected], gids[selected],
                score_sides[selected], np.ones(int(selected.sum()), bool)),
            "online_behavior_calibration": per_game_scores(
                labels[selected], oof_calibrated[score_rows][selected], gids[selected],
                score_sides[selected], np.ones(int(selected.sum()), bool)),
            "online_residual_calibration": per_game_scores(
                labels[selected], oof_residual_calibrated[score_rows][selected], gids[selected],
                score_sides[selected], np.ones(int(selected.sum()), bool)),
            "online_hold_residual_calibration": per_game_scores(
                labels[selected], oof_hold_residual_calibrated[score_rows][selected], gids[selected],
                score_sides[selected], np.ones(int(selected.sum()), bool)),
            "persistence_stable": per_game_scores(
                labels[selected], persistence[selected], gids[selected],
                score_sides[selected], np.ones(int(selected.sum()), bool)),
        }
        metrics_by_context[context_name] = {
            "rows": int(selected.sum()),
            "metrics_game_equal_red_blue_equal": {
                name: _summary(values) for name, values in context_per_game.items()},
            "paired_online_minus_fixed": _paired(
                context_per_game, "online_behavior_calibration", "fixed_oof_model",
                seed + 100 + context_id),
            "paired_residual_minus_fixed": _paired(
                context_per_game, "online_residual_calibration", "fixed_oof_model",
                seed + 200 + context_id),
        }
    report = {
        "schema": "observable-opponent-logic-online-calibration-group-oof-v1",
        "scope": "death-aware train only; five-fold group OOF; sequential sensor/radar feedback simulation",
        "test_access": False,
        "future_targets_read_for_online_feedback": False,
        "label_contract": contract,
        "task": "opponent_trend@10s",
        "behavior_regimes": {
            "advance": "mean enemy toward-speed >= 0.10 m/s and advancing fraction >= retreat fraction + 0.15",
            "hold_or_mixed": "otherwise",
            "retreat": "mean enemy toward-speed <= -0.10 m/s and retreat fraction >= advancing fraction + 0.15",
            "source": "current and causal-prefix enemy motion/visibility summaries only",
        },
        "online_update": {
            "issue_sampling": "one selected forecast per non-overlapping 10-second issue bucket and game-side",
            "maturity": "first sequential compact observation at or after issued_at + 10 seconds",
            "feedback": "recomputed opponent-trend class from issue/current observable enemy centroids; no JSONL targets read",
            "reset_scope": "each game and observing side has independent local outcome counts",
            "local_memory": MAX_LOCAL_OUTCOMES,
            "prior_strength": PRIOR_STRENGTH,
            "blend": "local_weight=n/(n+5); blend fixed OOF probabilities with fold-train behavior-context outcome posterior",
            "residual_correction": "context-local matured target-minus-issued-probability residual mean, max 20 outcomes, shrink n/(n+10), additive logit bias",
            "hold_residual_gate": "exploratory: apply residual correction only in hold_or_mixed; fixed probabilities in advance/retreat",
        },
        "train_rows_scored": int(len(score_rows)),
        "train_games_scored": int(len(np.unique(gids))),
        "game_side_blocks": compact_groups,
        "metrics_game_equal_red_blue_equal": {
            "fixed_oof_model": _summary(model_scores),
            "online_behavior_calibration": _summary(calibrated_scores),
            "online_residual_calibration": _summary(residual_scores),
            "online_hold_residual_calibration": _summary(hold_residual_scores),
            "persistence_stable": _summary(persistence_scores),
        },
        "paired_game_bootstrap_95_ci": {
            "online_behavior_calibration_minus_fixed_oof_model": _paired(
                per_game, "online_behavior_calibration", "fixed_oof_model", seed + 1),
            "online_residual_calibration_minus_fixed_oof_model": _paired(
                per_game, "online_residual_calibration", "fixed_oof_model", seed + 4),
            "online_hold_residual_calibration_minus_fixed_oof_model": _paired(
                per_game, "online_hold_residual_calibration", "fixed_oof_model", seed + 6),
            "online_residual_calibration_minus_behavior_frequency_calibration": _paired(
                per_game, "online_residual_calibration", "online_behavior_calibration", seed + 5),
            "fixed_oof_model_minus_persistence_stable": _paired(
                per_game, "fixed_oof_model", "persistence_stable", seed + 2),
            "online_behavior_calibration_minus_persistence_stable": _paired(
                per_game, "online_behavior_calibration", "persistence_stable", seed + 3),
        },
        "metrics_by_behavior_context": metrics_by_context,
        "online_counters": {
            "matured_observable_outcomes": int(online_updates),
            "selected_forecasts_issued": int(online_records_issued),
            "matured_trend_counts": {TRENDS[i]: int(feedback_label_counts[i])
                                      for i in range(len(TRENDS))},
        },
        "fold_details": fold_details,
        "limitations": [
            "motion regimes are coarse per-match observable behavior classes, not stable school identities",
            "only 10-second opponent trend is evaluated",
            "results are training OOF development evidence, not independent competition confirmation",
            "the same observable centroid label contract is used for training and online maturity; missing radar views are censored",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = ROOT / "out_m2" / "temporal_forecast_death_aware_candidate"
    parser.add_argument("--train-npz", type=Path, default=root / "train.npz")
    parser.add_argument("--train-jsonl", type=Path,
                        default=root / "train.death_aware.jsonl.gz")
    parser.add_argument("--output", type=Path, default=ROOT / "out_m2" /
                        "observable_online_calibration_audit" /
                        "opponent_logic_online_oof.json")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    report = run(args.train_npz, args.train_jsonl, args.output,
                 folds=args.folds, threads=args.threads)
    print(json.dumps({
        "output": str(args.output),
        "metrics": report["metrics_game_equal_red_blue_equal"],
        "paired_ci": report["paired_game_bootstrap_95_ci"],
        "online_counters": report["online_counters"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
