"""Chronological train-to-validation check for observable residual calibration.

Fit the 10-second opponent-trend model on train once. Validation predictions
are replayed in match order; only matured visible sensor/radar feedback can
update the match-local residual calibrator. Serialized validation targets are
never used for online updates and are accessed only through compact labels for
the final independent score.
"""
from __future__ import annotations

import argparse
import gzip
import json
from collections import deque
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from forecast.temporal import TRENDS, load_compact  # noqa: E402
from eval.experiment_death_aware_mode_lightgbm import (  # noqa: E402
    paired_bootstrap,
    per_game_scores,
)
from eval.experiment_style_conditioned_trend import (  # noqa: E402
    behavior_profile,
    make_model,
)
from eval.experiment_opponent_logic_online_oof import motion_context  # noqa: E402
from runtime.observable_feedback import feedback_labels_from_observable  # noqa: E402


CONTEXTS = ("advance", "hold_or_mixed", "retreat")
HORIZON = 10
MAX_MEMORY = 20


def summarize(per_game: dict) -> dict:
    keys = ("accuracy", "macro_f1", "brier", "log_loss", "ece")
    return {key: float(np.mean([row[key] for row in per_game.values()]))
            for key in keys}


def run(train_npz: Path, train_jsonl: Path, validation_npz: Path,
        validation_jsonl: Path, output: Path, threads: int = 4, seed: int = 419) -> dict:
    train = load_compact(train_npz)
    val = load_compact(validation_npz)
    if str(train["global_mode_label_version"][0]) != str(val["global_mode_label_version"][0]):
        raise ValueError("train/validation target contracts differ")
    contract = str(train["global_mode_label_version"][0])
    y_train_all = np.asarray(train["trend_y"][:, 0], dtype=np.int64)
    train_valid = (y_train_all >= 0) & (y_train_all < len(TRENDS))
    train_rows = np.flatnonzero(train_valid)
    y_val = np.asarray(val["trend_y"][:, 0], dtype=np.int64)
    val_valid = (y_val >= 0) & (y_val < len(TRENDS))
    score_rows = np.flatnonzero(val_valid)

    train_profile, _train_router, _ = behavior_profile(train)
    val_profile, _val_router, summary = behavior_profile(val)
    train_x = np.concatenate([train["x"][:, -1].astype(np.float32), train_profile], axis=1)
    val_x = np.concatenate([val["x"][:, -1].astype(np.float32), val_profile], axis=1)
    model = make_model(seed, threads)
    model.fit(train_x[train_rows], y_train_all[train_rows])
    base = model.predict_proba(val_x)
    contexts = np.asarray([motion_context(row) for row in summary], dtype=np.int64)

    game_ids = np.asarray(val["game_ids"], dtype=np.int64)
    sides = np.asarray(val["sides"]).astype(str)
    issued = np.asarray(val["issued_at"], dtype=np.float64)
    prediction_all = np.full_like(base, np.nan)
    prediction_hold = np.full_like(base, np.nan)
    pending = []
    local_residuals = None
    seen_buckets = None
    origin = None
    active_key = None
    mature_count = 0
    issued_count = 0
    stream_index = 0

    with gzip.open(validation_jsonl, "rt", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            source = json.loads(line)
            if stream_index >= len(game_ids):
                raise ValueError("validation JSONL has more rows than compact NPZ")
            gid = int(source["game_id"])
            side = str(source["side"])
            now = float(source["issued_at"])
            if (gid != int(game_ids[stream_index]) or side != sides[stream_index]
                    or abs(now - issued[stream_index]) > 1e-4):
                raise ValueError(f"validation row alignment mismatch at {stream_index}")
            key = (gid, side)
            if key != active_key:
                active_key = key
                pending = []
                local_residuals = {
                    strategy: {c: deque(maxlen=MAX_MEMORY) for c in range(len(CONTEXTS))}
                    for strategy in ("all", "hold_only")
                }
                seen_buckets = set()
                origin = now

            current = {"game_id": gid, "side": side, "issued_at": now,
                       "observable": source.get("observable") or {}}
            keep = []
            for record in pending:
                if now + 1e-6 < record["issued_at"] + HORIZON:
                    keep.append(record)
                    continue
                feedback = feedback_labels_from_observable(
                    record["snapshot"], current, side, HORIZON)
                label = feedback.get("opponent_trend")
                if label in TRENDS:
                    cls = TRENDS.index(label)
                    truth_vector = np.eye(len(TRENDS), dtype=np.float64)[cls]
                    for strategy in ("all", "hold_only"):
                        local_residuals[strategy][record["context"]].append(
                            truth_vector - record["issued_probabilities"][strategy])
                    mature_count += 1
            pending = keep

            context = int(contexts[stream_index])
            def residual_corrected(strategy):
                history = list(local_residuals[strategy][context])
                if not history:
                    return base[stream_index].copy()
                shrink = len(history) / (len(history) + 10.0)
                logits = (np.log(np.clip(base[stream_index], 1e-9, 1.0))
                          + shrink * np.mean(history, axis=0))
                logits -= logits.max()
                corrected = np.exp(logits)
                return corrected / corrected.sum()

            corrected = residual_corrected("all")
            hold_corrected = (residual_corrected("hold_only")
                              if context == 1 else base[stream_index].copy())
            prediction_all[stream_index] = corrected
            prediction_hold[stream_index] = hold_corrected

            bucket = int(np.floor((now - origin) / HORIZON + 1e-6))
            if bucket not in seen_buckets:
                seen_buckets.add(bucket)
                pending.append({"issued_at": now, "context": context,
                                "snapshot": current,
                                "issued_probabilities": {
                                    "all": corrected.copy(),
                                    "hold_only": hold_corrected.copy(),
                                }})
                issued_count += 1
            stream_index += 1
    if stream_index != len(game_ids):
        raise ValueError(f"validation JSONL rows {stream_index} != compact rows {len(game_ids)}")

    gids = game_ids[score_rows]
    score_sides = sides[score_rows]
    labels = y_val[score_rows]
    mask = np.ones(len(score_rows), dtype=bool)
    per_game = {
        "fixed_train_model": per_game_scores(labels, base[score_rows], gids, score_sides, mask),
        "online_residual_all_contexts": per_game_scores(
            labels, prediction_all[score_rows], gids, score_sides, mask),
        "online_residual_hold_only": per_game_scores(
            labels, prediction_hold[score_rows], gids, score_sides, mask),
    }
    report = {
        "schema": "observable-residual-calibration-train-validation-v1",
        "scope": "fit on train only; chronological validation prequential observable feedback",
        "test_access": False,
        "future_targets_read_for_online_feedback": False,
        "label_contract": contract,
        "task": "opponent_trend@10s",
        "validation_games": int(len(np.unique(gids))),
        "validation_scored_rows": int(len(score_rows)),
        "metrics_game_equal_red_blue_equal": {
            name: summarize(values) for name, values in per_game.items()},
        "paired_game_bootstrap_95_ci": {
            name: paired_bootstrap(per_game[name], per_game["fixed_train_model"],
                                   seed=seed + index, reps=5000)
            for index, name in enumerate(("online_residual_all_contexts",
                                          "online_residual_hold_only"))},
        "online_feedback": {
            "issued": issued_count,
            "matured_from_observable": mature_count,
            "reset": "each validation game and observing side",
            "update_rule": "residual mean from matured target-minus-issued-probability; max 20, shrink n/(n+10)",
            "gated_variant": "apply only to hold_or_mixed; all other contexts fixed",
        },
        "limitations": [
            "validation was used in earlier project development and is not a pristine blind test",
            "the hold_or_mixed gate was selected from train OOF context diagnostics before this run",
            "no online model-weight updates are performed",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    root = ROOT / "out_m2" / "temporal_forecast_death_aware_candidate"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-npz", type=Path, default=root / "train.npz")
    parser.add_argument("--train-jsonl", type=Path, default=root / "train.death_aware.jsonl.gz")
    parser.add_argument("--validation-npz", type=Path, default=root / "validation.npz")
    parser.add_argument("--validation-jsonl", type=Path, default=root / "validation.death_aware.jsonl.gz")
    parser.add_argument("--output", type=Path, default=ROOT / "out_m2" / "observable_online_calibration_audit" / "residual_validation.json")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    result = run(args.train_npz, args.train_jsonl, args.validation_npz,
                 args.validation_jsonl, args.output, threads=args.threads)
    print(json.dumps({"output": str(args.output),
                      "metrics": result["metrics_game_equal_red_blue_equal"],
                      "paired_ci": result["paired_game_bootstrap_95_ci"],
                      "feedback": result["online_feedback"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
