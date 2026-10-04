"""Causal replay of the match-local Brier router using only current observations.

Forecasts are issued in chronological order. A pending forecast is scored only
at the first current observable snapshot at or after its target horizon. Future
``targets`` from the JSONL are never read. This is an offline development audit,
not a production integration or blind test.
"""
from __future__ import annotations

import argparse
import gzip
import json
from collections import defaultdict, deque
from pathlib import Path
import sys

import numpy as np
import torch
from sklearn.metrics import f1_score, log_loss
from scipy.stats import t as student_t

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from forecast.calibration import calibrate_horizon_probabilities  # noqa: E402
from forecast.temporal import (HORIZONS, MODES, REGION_LABELS, TRENDS,
                               _centroid, _entities, _mobile, baseline_labels,
                               load_compact)  # noqa: E402
from forecast.train import load_model  # noqa: E402
from runtime.adaptive_strategy import AdaptiveStrategyRouter  # noqa: E402

TASK_LABELS = {
    "global_mode": MODES,
    "opponent_trend": TRENDS,
    "enemy_region": REGION_LABELS,
}
TASK_ARRAYS = {
    "global_mode": "mode_y",
    "opponent_trend": "trend_y",
    "enemy_region": "region_y",
}


def load_visible_states(jsonl_path: Path, compact: dict):
    """Read only visible current state; assert exact row alignment to model input."""
    n = len(compact["x"])
    labels = {task: np.empty(n, dtype=object) for task in TASK_LABELS}
    centers = np.full((n, 2), np.nan, dtype=np.float32)
    ids = np.empty(n, dtype=np.int64)
    sides = np.empty(n, dtype="U1")
    times = np.empty(n, dtype=np.float32)
    count = 0
    with gzip.open(jsonl_path, "rt", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            source = json.loads(line)
            if count >= n:
                raise ValueError("observable JSONL contains more rows than compact validation")
            game_id = int(source["game_id"])
            side = str(source["side"])
            issued = float(source["issued_at"])
            if (game_id != int(compact["game_ids"][count])
                    or side != str(compact["sides"][count])
                    or abs(issued - float(compact["issued_at"][count])) > 1e-4):
                raise ValueError(f"validation row alignment mismatch at row {count}")
            # Deliberately discard the serialized future targets before doing
            # any labels or feedback. Only observable + side + issue time pass.
            visible = {"observable": source.get("observable") or {},
                       "side": side, "issued_at": issued}
            current = baseline_labels(visible)
            for task in TASK_LABELS:
                labels[task][count] = current.get(task)
            opponent = _entities(visible["observable"], "opponent_entities")
            center = _centroid(_mobile(opponent))
            if center is not None:
                centers[count] = center
            ids[count] = game_id
            sides[count] = side
            times[count] = issued
            count += 1
    if count != n:
        raise ValueError(f"observable JSONL rows {count} != compact rows {n}")
    return ids, sides, times, labels, centers


def model_probabilities(checkpoint: Path, compact: dict, visible_labels: dict,
                        device: str, batch_size: int):
    dev = torch.device(device)
    model, metadata = load_model(checkpoint, device=dev)
    model.eval()
    n = len(compact["x"])
    raw = {task: {int(h): np.empty((n, len(labels)), dtype=np.float32)
                  for h in HORIZONS}
           for task, labels in TASK_LABELS.items()}
    x = torch.from_numpy(np.asarray(compact["x"], dtype=np.float32))
    with torch.inference_mode():
        for start in range(0, n, batch_size):
            out = model(x[start:start + batch_size].to(dev))
            for h in HORIZONS:
                for task in TASK_LABELS:
                    probs = torch.softmax(out[h][task], dim=-1).cpu().numpy()
                    raw[task][int(h)][start:start + len(probs)] = probs

    # Match runtime probability treatment: softmax -> fixed horizon
    # calibration -> checkpoint's explicit persistence blend.
    blend = metadata.get("persistence_blend") or {}
    alphas = blend.get("alphas") or {}
    result = {task: {int(h): np.empty_like(raw[task][int(h)])
                     for h in HORIZONS}
              for task in TASK_LABELS}
    for i in range(n):
        for task, labels in TASK_LABELS.items():
            calibrated = calibrate_horizon_probabilities(
                {int(h): raw[task][int(h)][i] for h in HORIZONS})
            for h in HORIZONS:
                p = np.asarray(calibrated[int(h)], dtype=np.float64)
                alpha = float(alphas.get(f"{task}@{int(h)}s", 1.0))
                current_label = visible_labels[task][i]
                if alpha < 1.0 and current_label in labels:
                    prior = np.full(len(labels), 0.02 / max(1, len(labels) - 1),
                                    dtype=np.float64)
                    prior[labels.index(str(current_label))] = 0.98
                    p = alpha * p + (1.0 - alpha) * prior
                result[task][int(h)][i] = p
    return result, metadata, str(dev)


def _one_hot(label, labels):
    if label is None:
        return None
    return {name: (1.0 if name == label else 0.0) for name in labels}


def _smoothed_persistence(label, labels, confidence):
    if label is None:
        return None
    residual = (1.0 - confidence) / max(1, len(labels) - 1)
    return {name: (confidence if name == label else residual) for name in labels}


def _persistence_vector(label, labels, confidence):
    values = _smoothed_persistence(label, labels, confidence)
    if values is None:
        values = {name: 1.0 / len(labels) for name in labels}
    return np.asarray([values[name] for name in labels], dtype=np.float64)


def _brier(p, y):
    target = np.zeros(len(p), dtype=np.float64)
    target[int(y)] = 1.0
    return float(np.square(np.asarray(p, dtype=np.float64) - target).sum())


def _score_rows(rows, n_classes):
    y = np.asarray([r[0] for r in rows], dtype=np.int64)
    p = np.asarray([r[1] for r in rows], dtype=np.float64)
    p = np.clip(p, 0.0, None)
    p /= np.maximum(p.sum(axis=1, keepdims=True), 1e-12)
    return {
        "n": int(len(y)),
        "accuracy": float(np.mean(np.argmax(p, axis=1) == y)),
        "macro_f1": float(f1_score(y, np.argmax(p, axis=1),
                                    labels=np.arange(n_classes), average="macro",
                                    zero_division=0)),
        "brier": float(np.mean(np.sum((p - np.eye(n_classes)[y]) ** 2, axis=1))),
        "log_loss": float(log_loss(y, p, labels=np.arange(n_classes))),
    }


def _select_online_alpha(history, *, penalty=0.02):
    """Fit a shrinkage weight using only matured feedback in the current match."""
    labels = np.asarray([row[2] for row in history], dtype=np.int64)
    model_p = np.asarray([row[0] for row in history], dtype=np.float64)
    baseline_p = np.asarray([row[1] for row in history], dtype=np.float64)
    one_hot = np.eye(model_p.shape[1])[labels]
    best = None
    for alpha in np.linspace(0.0, 1.0, 21):
        p = alpha * model_p + (1.0 - alpha) * baseline_p
        loss = float(np.mean(np.sum((p - one_hot) ** 2, axis=1)))
        objective = loss + penalty * float((alpha - 1.0) ** 2)
        key = (objective, -alpha)
        if best is None or key < best[0]:
            best = (key, float(alpha), loss)
    return best[1], best[2]


def _paired_ci(per_game, candidate, baseline, metric, rng, reps=2000):
    ids = sorted(set(per_game[candidate]) & set(per_game[baseline]))
    delta = np.asarray([per_game[candidate][g][metric] - per_game[baseline][g][metric]
                        for g in ids], dtype=np.float64)
    if not len(delta):
        return {"n_games": 0, "delta": None, "ci95": None}
    draws = rng.integers(0, len(delta), size=(reps, len(delta)))
    boot = delta[draws].mean(axis=1)
    return {"n_games": len(ids), "delta": float(delta.mean()),
            "ci95": [float(np.quantile(boot, .025)),
                     float(np.quantile(boot, .975))]}


def run(jsonl_path: Path, compact_path: Path, checkpoint: Path, output: Path,
        device="auto", batch_size=1024, seed=27, persistence_confidence=0.98):
    compact = load_compact(compact_path)
    ids, sides, times, visible_labels, centers = load_visible_states(jsonl_path, compact)
    probs, metadata, used_device = model_probabilities(
        checkpoint, compact, visible_labels, device, batch_size)
    rows_by_game = defaultdict(lambda: defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list)))))
    switch_counts = defaultdict(int)
    strategy_issue_counts = defaultdict(lambda: defaultdict(int))
    alpha_update_counts = defaultdict(int)
    alpha_selected_values = defaultdict(list)
    issued_count = defaultdict(int)
    matured_count = defaultdict(int)
    policy_names = ("runtime_fixed", "adaptive_router", "adaptive_smooth",
                    "block_router", "persistence")
    block_switches = defaultdict(int)
    block_strategy_issue_counts = defaultdict(lambda: defaultdict(int))

    for key in sorted(set(zip(ids.tolist(), sides.tolist()))):
        game_id, side = int(key[0]), str(key[1])
        index = np.flatnonzero((ids == game_id) & (sides == side))
        index = index[np.argsort(times[index])]
        router = AdaptiveStrategyRouter()
        pending = deque()
        last_strategy = {}
        alpha_state = {
            f"{task}@{int(h)}s": {"alpha": 1.0, "history": deque(maxlen=40),
                                  "since_update": 0}
            for task in TASK_LABELS for h in HORIZONS}
        block_state = {
            f"{task}@{int(h)}s": {"strategy": "model", "partial": [],
                                  "blocks": deque(maxlen=8)}
            for task in TASK_LABELS for h in HORIZONS}
        for row_idx in index:
            now = float(times[row_idx])
            # Mature prior forecasts only against the observable at this row.
            for record in list(pending):
                due = [h for h in HORIZONS
                       if h not in record["scored"] and record["issued"] + h <= now + 1e-6]
                if not due:
                    continue
                truth = {task: {} for task in TASK_LABELS}
                for h in due:
                    for task in TASK_LABELS:
                        if record["baseline"][task] is None:
                            continue
                        value = visible_labels[task][row_idx]
                        if task == "opponent_trend":
                            start = record["center"]
                            end = centers[row_idx]
                            if np.isfinite(start).all() and np.isfinite(end).all():
                                toward = 1.0 if side == "蓝" else -1.0
                                delta = float((end[0] - start[0]) * toward)
                                value = ("推进" if delta >= 0.8 else
                                         "回撤" if delta <= -0.8 else "稳定")
                            else:
                                value = None
                        if value is not None:
                            truth[task][str(h)] = str(value)
                            label_idx = TASK_LABELS[task].index(str(value))
                            key = f"{task}@{int(h)}s"
                            state = alpha_state[key]
                            state["history"].append((
                                record["runtime_fixed"][task][int(h)],
                                record["persistence"][task][int(h)], label_idx))
                            state["since_update"] += 1
                            if len(state["history"]) >= 20 and state["since_update"] >= 5:
                                state["alpha"], _ = _select_online_alpha(state["history"])
                                state["since_update"] = 0
                                alpha_update_counts[key] += 1
                                alpha_selected_values[key].append(state["alpha"])
                            block = block_state[key]
                            model_loss = _brier(
                                record["runtime_fixed"][task][int(h)], label_idx)
                            baseline_loss = _brier(
                                record["persistence"][task][int(h)], label_idx)
                            block["partial"].append(model_loss - baseline_loss)
                            if len(block["partial"]) == 5:
                                block["blocks"].append(float(np.mean(block["partial"])))
                                block["partial"].clear()
                                block_values = np.asarray(block["blocks"], dtype=np.float64)
                                if len(block_values) >= 4:
                                    mean_gap = float(block_values.mean())
                                    se = float(block_values.std(ddof=1) /
                                               np.sqrt(len(block_values)))
                                    critical = float(student_t.ppf(
                                        0.95, df=len(block_values) - 1))
                                    lower = mean_gap - critical * se
                                    upper = mean_gap + critical * se
                                    old_strategy = block["strategy"]
                                    if old_strategy == "model" and lower > 0.03:
                                        block["strategy"] = "persistence"
                                    elif old_strategy == "persistence" and upper < -0.03:
                                        block["strategy"] = "model"
                                    if block["strategy"] != old_strategy:
                                        block_switches[key] += 1
                            for policy in policy_names:
                                policy_p = record[policy][task][int(h)]
                                rows_by_game[game_id][side][task][f"{h}s"][policy].append(
                                    (label_idx, policy_p))
                            matured_count[f"{task}@{h}s"] += 1
                router.observe_outcome(record["forecast_id"], truth,
                                       matured_horizons=due)
                record["scored"].update(due)
                if len(record["scored"]) == len(HORIZONS):
                    pending.remove(record)

            baseline = {task: (str(visible_labels[task][row_idx])
                               if visible_labels[task][row_idx] is not None else None)
                        for task in TASK_LABELS}
            model_heads = {str(h): {
                task: {"probabilities": {
                    label: float(probs[task][int(h)][row_idx, j])
                    for j, label in enumerate(TASK_LABELS[task])}}
                for task in TASK_LABELS}
            for h in HORIZONS}
            baseline_map = {task: {str(h): _smoothed_persistence(
                                       baseline[task], TASK_LABELS[task],
                                       persistence_confidence)
                                   for h in HORIZONS}
                            for task in TASK_LABELS}
            prediction = {"issued_at": now,
                          "input_snapshot_hash": f"{game_id}:{side}:{now}",
                          "horizons": model_heads}
            forecast_id = f"{game_id}:{side}:{now:.3f}"
            routed = router.apply(prediction, baselines=baseline_map,
                                  forecast_id=forecast_id)
            for key, state in (routed.get("adaptive_strategy", {}).get(
                    "by_task_horizon") or {}).items():
                strategy = state.get("strategy", "model")
                strategy_issue_counts[key][strategy] += 1
                if key in last_strategy and last_strategy[key] != strategy:
                    switch_counts[key] += 1
                last_strategy[key] = strategy
            issued_count["forecasts"] += 1
            smooth_heads = {}
            block_heads = {}
            for h in HORIZONS:
                smooth_heads[str(h)] = {}
                block_heads[str(h)] = {}
                for task in TASK_LABELS:
                    alpha = alpha_state[f"{task}@{int(h)}s"]["alpha"]
                    fixed = probs[task][int(h)][row_idx]
                    q = _persistence_vector(
                        baseline[task], TASK_LABELS[task], persistence_confidence)
                    smooth = alpha * fixed + (1.0 - alpha) * q
                    smooth_heads[str(h)][task] = {"probabilities": {
                        label: float(smooth[j])
                        for j, label in enumerate(TASK_LABELS[task])}}
                    block = block_state[f"{task}@{int(h)}s"]
                    selected = (probs[task][int(h)][row_idx]
                                if block["strategy"] == "model" else q)
                    block_heads[str(h)][task] = {"probabilities": {
                        label: float(selected[j])
                        for j, label in enumerate(TASK_LABELS[task])}}
                    block_strategy_issue_counts[f"{task}@{int(h)}s"][
                        block["strategy"]] += 1
            policy_outputs = {"runtime_fixed": model_heads,
                              "adaptive_router": routed["horizons"],
                              "adaptive_smooth": smooth_heads}
            policy_outputs["block_router"] = block_heads
            record = {"forecast_id": forecast_id, "issued": now,
                      "center": centers[row_idx].copy(), "scored": set(),
                      "baseline": baseline.copy(),
                      **{name: {task: {int(h): np.asarray(
                          list((heads[str(h)][task]["probabilities"])[label]
                               for label in TASK_LABELS[task]), dtype=np.float64)
                          for h in HORIZONS} for task in TASK_LABELS}
                         for name, heads in policy_outputs.items() if heads is not None}}
            record["persistence"] = {
                task: {int(h): _persistence_vector(
                           baseline[task], TASK_LABELS[task], persistence_confidence)
                       for h in HORIZONS}
                for task in TASK_LABELS}
            pending.append(record)

    metrics = {}
    paired = {}
    rng = np.random.default_rng(seed)
    tasks = sorted({task for game in rows_by_game.values()
                    for side_data in game.values() for task in side_data})
    for task in tasks:
        metrics[task] = {}
        paired[task] = {}
        n_classes = len(TASK_LABELS[task])
        for horizon in sorted({h for game in rows_by_game.values()
                               for side_data in game.values()
                               for t, hs in side_data.items() if t == task for h in hs},
                              key=lambda z: int(z[:-1])):
            per_game = {name: {} for name in policy_names}
            for game_id, game in rows_by_game.items():
                for name in policy_names:
                    side_scores = []
                    for side in ("红", "蓝"):
                        values = game.get(side, {}).get(task, {}).get(horizon, {}).get(name, [])
                        if values:
                            side_scores.append(_score_rows(values, n_classes))
                    if len(side_scores) == 2:
                        per_game[name][game_id] = {
                            metric: float(np.mean([s[metric] for s in side_scores]))
                            for metric in ("accuracy", "macro_f1", "brier", "log_loss")}
                        per_game[name][game_id]["n"] = int(sum(s["n"] for s in side_scores))
            metrics[task][horizon] = {
                name: {metric: float(np.mean([g[metric] for g in games.values()]))
                       for metric in ("accuracy", "macro_f1", "brier", "log_loss")}
                | {"n_games": len(games), "n": int(sum(g["n"] for g in games.values()))}
                for name, games in per_game.items()}
            paired[task][horizon] = {
                f"{candidate}_minus_{baseline}": {
                    metric: _paired_ci(per_game, candidate, baseline, metric, rng)
                    for metric in ("accuracy", "macro_f1", "brier", "log_loss")}
                for candidate in ("adaptive_router", "adaptive_smooth")
                for baseline in ("runtime_fixed", "persistence")}
            for baseline in ("runtime_fixed", "persistence"):
                paired[task][horizon][f"block_router_minus_{baseline}"] = {
                    metric: _paired_ci(per_game, "block_router", baseline, metric, rng)
                    for metric in ("accuracy", "macro_f1", "brier", "log_loss")}

    report = {
        "schema": "observable-only-match-local-adaptation-audit-v1",
        "scope": "validation chronological replay; development evidence, not blind test",
        "test_access": False,
        "future_targets_read": False,
        "feedback_truth": "recomputed only from the current observable snapshot at the first issue tick >= issued_at+horizon; trend is current observable centroid displacement from issue-time observable centroid",
        "baseline": "current visible-state one-hot persistence; runtime_fixed additionally reproduces the checkpoint's .98-smoothed persistence mixture before adaptation; all policies are scored on the same samples where the issue-time baseline label is observable",
        "router_config": {"window": 40, "min_samples": 20, "patience": 3,
                          "recovery_patience": 4, "margin": 0.03,
                          "persistence_confidence": persistence_confidence},
        "checkpoint": str(checkpoint), "checkpoint_completed_epochs": metadata.get("completed_epochs"),
        "device": used_device, "validation_games": int(len(np.unique(ids))),
        "issued_game_side_sequences": int(len(set(zip(ids.tolist(), sides.tolist())))),
        "issued_forecasts": int(issued_count["forecasts"]),
        "matured_scored_labels": dict(matured_count),
        "maximum_feedback_delay_seconds": 3,
        "router_strategy_issue_counts": {
            task: dict(counts) for task, counts in strategy_issue_counts.items()},
        "router_switches_observed_by_task_horizon": dict(switch_counts),
        "online_smooth_alpha_updates": dict(alpha_update_counts),
        "online_smooth_alpha_selected_values": {
            task: values for task, values in alpha_selected_values.items()},
        "online_smooth_policy": {
            "initial_model_multiplier": 1.0,
            "history": "last 40 mature outcomes within current game and observing side",
            "minimum_feedback": 20,
            "update_block": 5,
            "selection": "grid over alpha 0..1 step .05; matured Brier plus .02*(alpha-1)^2 penalty",
            "test_access": False,
        },
        "block_router_strategy_issue_counts": {
            task: dict(counts) for task, counts in block_strategy_issue_counts.items()},
        "block_router_switches_by_task_horizon": dict(block_switches),
        "block_router_policy": {
            "loss": "paired per-prediction Brier gap, runtime_fixed minus current-observable persistence",
            "aggregation": "non-overlapping blocks of 5 matured 3-second forecasts; rolling last 8 blocks",
            "minimum_evidence": "4 blocks (20 matured examples)",
            "switch_rule": "switch only when one-sided 95% Student-t interval clears +/-0.03 Brier margin",
            "scope": "per game, observing side, task, and horizon; reset at each game-side boundary",
            "test_access": False,
        },
        "metrics_game_equal_red_blue_equal": metrics,
        "paired_game_bootstrap_95_ci": paired,
        "limitations": [
            "validation has been repeatedly used during development",
            "the run simulates feedback and source routing but does not integrate or persist runtime state",
            "visible-state labels can differ from the offline future-truth labeling contract",
            "no team identifiers are available for held-out-team evaluation",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    base = ROOT / "out_m2" / "temporal_forecast_death_aware_candidate"
    canonical = ROOT / "out_m2" / "temporal_forecast_canonical_side"
    parser.add_argument("--jsonl", type=Path, default=base / "validation.death_aware.jsonl.gz")
    parser.add_argument("--compact", type=Path, default=canonical / "validation.npz")
    parser.add_argument("--checkpoint", type=Path, default=canonical / "temporal_gru_v1.pt")
    parser.add_argument("--output", type=Path, default=ROOT / "out_m2" / "observable_online_calibration_audit" / "report_smoothed98.json")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--persistence-confidence", type=float, default=0.98)
    args = parser.parse_args()
    report = run(args.jsonl, args.compact, args.checkpoint, args.output,
                 args.device, args.batch_size,
                 persistence_confidence=args.persistence_confidence)
    compact_metrics = {}
    for task, by_h in report["metrics_game_equal_red_blue_equal"].items():
        compact_metrics[task] = {}
        for horizon, models in by_h.items():
            compact_metrics[task][horizon] = {
                name: {k: round(v[k], 4) for k in ("accuracy", "macro_f1", "brier")}
                for name, v in models.items()}
    print(json.dumps({"output": str(args.output), "metrics": compact_metrics,
                      "scored": report["matured_scored_labels"],
                      "switches": report["router_switches_observed_by_task_horizon"]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
