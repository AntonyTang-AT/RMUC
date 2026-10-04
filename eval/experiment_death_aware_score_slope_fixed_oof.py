"""Fixed lambda=1 score-slope extrapolation on death-aware train only.

This is a preregistered deterministic comparator, not a tuned model. It reads
only the death-aware train JSONL/NPZ and the existing full-game GroupKFold OOF
archive, reconstructs a causal observable score from issue-time sensors, and
compares fixed score extrapolation with persistence and the motion+zone+score
OOF predictor. No validation/test input is accepted.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from forecast.temporal import HORIZONS, MODES, _entities, _mode_score, load_compact

BASE = ROOT / "out_m2/death_aware_global_mode_experiments/motion_zone_score_oof/motion_candidate_oof.npz"
TRAIN_NPZ = ROOT / "out_m2/temporal_forecast_death_aware_candidate/train.npz"
TRAIN_JSONL = ROOT / "out_m2/temporal_forecast_death_aware_candidate/train.death_aware.jsonl.gz"
OUT = ROOT / "out_m2/death_aware_global_mode_experiments/fixed_score_slope_oof"
REPORT = ROOT / "docs/analysis/2026-09-27-death-aware-fixed-score-slope-oof.md"


def labels_from_score(score: np.ndarray) -> np.ndarray:
    """Exact death-aware mode thresholds; MODES order is attack/balance/defense."""
    score = np.asarray(score, dtype=np.float64)
    pred = np.full(score.shape, MODES.index("均衡"), dtype=np.int64)
    pred[score >= 1.0] = MODES.index("进攻")
    pred[score <= -1.0] = MODES.index("防守")
    return pred


def observable_score_history(npz_path: Path, jsonl_path: Path):
    """Compute causal score/slope and require exact issue row/mode parity."""
    data = load_compact(npz_path)
    gids = np.asarray(data["game_ids"], dtype=np.int64)
    sides = np.asarray(data["sides"]).astype(str)
    issued = np.asarray(data["issued_at"], dtype=np.float64)
    base_mode = np.asarray(data["base_mode"], dtype=np.int64)
    scores = np.full(len(gids), np.nan, dtype=np.float64)
    slopes = np.zeros(len(gids), dtype=np.float64)
    history: dict[tuple[int, str], deque] = defaultdict(lambda: deque(maxlen=5))
    rows = 0
    future_observation_violations = 0
    future_entity_timestamp_violations = 0
    rows_with_history = 0
    with gzip.open(jsonl_path, "rt", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("split") != "train":
                raise ValueError(f"non-train JSONL record at row {rows}")
            if rows >= len(gids):
                raise ValueError("train JSONL has extra rows")
            t = float(row["issued_at"])
            actual = (int(row["game_id"]), str(row["side"]), round(t, 5))
            expected = (int(gids[rows]), str(sides[rows]), round(float(issued[rows]), 5))
            if actual != expected:
                raise ValueError(f"NPZ/JSONL mismatch at row {rows}: {actual} != {expected}")
            max_seen = row.get("max_input_observed_at")
            if isinstance(max_seen, (int, float)) and not isinstance(max_seen, bool) and float(max_seen) > t + 1e-6:
                future_observation_violations += 1
            obs = row.get("observable") or {}
            own = _entities(obs, "self_entities")
            opp = _entities(obs, "opponent_entities")
            for entity in (*own, *opp):
                observed_at = entity.get("observed_at")
                if (isinstance(observed_at, (int, float)) and not isinstance(observed_at, bool)
                        and float(observed_at) > t + 1e-6):
                    future_entity_timestamp_violations += 1
            current = float(_mode_score(own, opp, str(row["side"])))
            if not np.isfinite(current):
                raise ValueError(f"non-finite observable score at row {rows}")
            scores[rows] = current
            key = (int(row["game_id"]), str(row["side"]))
            points = history[key]
            points.append((t, current))
            if len(points) >= 2:
                rows_with_history += 1
                x = np.asarray([p[0] for p in points], dtype=np.float64)
                y = np.asarray([p[1] for p in points], dtype=np.float64)
                x -= x[-1]
                denom = float(np.dot(x - x.mean(), x - x.mean()))
                slopes[rows] = float(np.dot(x - x.mean(), y - y.mean()) / denom) if denom > 0 else 0.0
            rows += 1
    if rows != len(gids):
        raise ValueError(f"JSONL row count {rows} != NPZ row count {len(gids)}")
    if future_entity_timestamp_violations:
        raise ValueError(f"future entity timestamps found: {future_entity_timestamp_violations}")
    reconstructed = labels_from_score(scores)
    valid_base = (base_mode >= 0) & (base_mode < 3)
    parity = valid_base & (reconstructed == base_mode)
    if not np.all(valid_base) or not np.all(parity):
        bad = np.flatnonzero(~parity)[:8].tolist()
        raise ValueError(f"current observable score/base_mode parity failed at rows {bad}")
    return data, scores, slopes, {
        "rows_aligned": int(rows),
        "current_observable_score_base_mode_exact_parity": bool(np.all(parity)),
        "parity_mismatches": int(np.sum(~parity)),
        "future_observation_violations": int(future_observation_violations),
        "future_entity_timestamp_violations": int(future_entity_timestamp_violations),
        "score_formula": "forecast.temporal._mode_score over issue-time observable entities; _mobile excludes destroyed and hp<=0 robots",
        "slope": "OLS score/sec over latest <=5 issue-time scores for same game_id+side; lambda=1 fixed; no tuning",
        "history_minimum": 2,
        "rows_with_at_least_two_scores": int(rows_with_history),
    }


def fixed_class_metrics(y: np.ndarray, pred: np.ndarray) -> dict:
    cm = np.zeros((3, 3), dtype=np.int64)
    np.add.at(cm, (y, pred), 1)
    recalls = []
    per_class = {}
    f1s = []
    for c, name in enumerate(MODES):
        tp = int(cm[c, c]); fp = int(cm[:, c].sum() - tp); fn = int(cm[c, :].sum() - tp)
        rec = tp / int(cm[c].sum()) if cm[c].sum() else 0.0
        f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
        recalls.append(rec); f1s.append(f1)
        per_class[name] = {"recall": rec, "f1": f1, "support": int(cm[c].sum())}
    return {"n": int(len(y)), "accuracy": float(np.mean(y == pred)),
            "macro_f1": float(np.mean(f1s)), "balanced_accuracy": float(np.mean(recalls)),
            "per_class": per_class, "confusion_true_rows_pred_cols": cm.tolist()}


def game_side_scores(y, pred, valid, gids, sides):
    by_game = {}
    for gid in np.unique(gids[valid]):
        side_rows = []
        for side in ("红", "蓝"):
            mask = valid & (gids == gid) & (sides == side)
            if mask.any():
                side_rows.append(fixed_class_metrics(y[mask], pred[mask]))
        if len(side_rows) == 2:
            keys = ("accuracy", "macro_f1", "balanced_accuracy")
            row = {k: float(np.mean([r[k] for r in side_rows])) for k in keys}
            for mode in MODES:
                row[f"recall_{mode}"] = float(np.mean([r["per_class"][mode]["recall"] for r in side_rows]))
            by_game[int(gid)] = row
    return by_game


def bootstrap_summary(per_game, seed=20260927, reps=2000):
    games = sorted(per_game)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(games), size=(reps, len(games)))
    names = ("accuracy", "macro_f1", "balanced_accuracy", *(f"recall_{m}" for m in MODES))
    result = {}
    for key in names:
        vals = np.asarray([per_game[g][key] for g in games], dtype=np.float64)
        sampled = vals[idx].mean(axis=1)
        result[key] = {"mean": float(vals.mean()), "ci95": [float(np.quantile(sampled, .025)), float(np.quantile(sampled, .975))]}
    return result


def paired_delta(candidate, reference, seed, reps=2000):
    games = sorted(set(candidate) & set(reference))
    if not games:
        raise ValueError("no paired complete red/blue games")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(games), size=(reps, len(games)))
    names = ("accuracy", "macro_f1", "balanced_accuracy", *(f"recall_{m}" for m in MODES))
    result = {"paired_games": len(games), "bootstrap_replicates": reps, "metrics": {}}
    for key in names:
        delta = np.asarray([candidate[g][key] - reference[g][key] for g in games])
        sampled = delta[idx].mean(axis=1)
        result["metrics"][key] = {"mean_delta": float(delta.mean()),
                                  "ci95": [float(np.quantile(sampled, .025)), float(np.quantile(sampled, .975))]}
    return result


def run(train_npz=TRAIN_NPZ, train_jsonl=TRAIN_JSONL, oof_path=BASE, output=OUT, reps=2000):
    data, score, slope, audit = observable_score_history(Path(train_npz), Path(train_jsonl))
    oof = np.load(oof_path, allow_pickle=False)
    for name in ("game_ids", "sides", "issued_at", "mode_y", "base_mode"):
        if not np.array_equal(np.asarray(data[name]), np.asarray(oof[name])):
            raise ValueError(f"OOF archive/train NPZ {name} mismatch")
    gids = np.asarray(data["game_ids"], dtype=np.int64)
    sides = np.asarray(data["sides"]).astype(str)
    issued = np.asarray(data["issued_at"], dtype=np.float64)
    y_all = np.asarray(data["mode_y"], dtype=np.int64)
    base = np.asarray(data["base_mode"], dtype=np.int64)
    fold_ids = np.asarray(oof["fold_ids"], dtype=np.int64)
    # Reconstruct the saved full-game GroupKFold assignment and prove both
    # sides and all rows from a match remain together.
    reconstructed_fold = np.full(len(gids), -1, dtype=np.int64)
    for fold, (_, held) in enumerate(GroupKFold(n_splits=5).split(np.zeros(len(gids)), groups=gids)):
        reconstructed_fold[held] = fold
    if not np.array_equal(fold_ids, reconstructed_fold):
        raise ValueError("saved OOF fold ids do not match deterministic 5-fold GroupKFold")
    for gid in np.unique(gids):
        gf = np.unique(fold_ids[gids == gid])
        if len(gf) != 1 or set(sides[gids == gid]) != {"红", "蓝"}:
            raise ValueError(f"game {gid} split integrity failed")
    base_mode = np.asarray(oof["base_mode"], dtype=np.int64)
    candidate = np.full((len(gids), len(HORIZONS)), -1, dtype=np.int64)
    for j, horizon in enumerate(HORIZONS):
        candidate[:, j] = labels_from_score(score + float(horizon) * slope)  # fixed lambda=1
    reports = {}
    for j, horizon in enumerate(HORIZONS):
        y = y_all[:, j]
        valid = y >= 0
        probs = np.asarray(oof["motion_candidate"][:, j, :], dtype=np.float64)
        motion_pred = np.argmax(probs, axis=1)
        preds = {"score_slope_lambda_1": candidate[:, j], "persistence": base_mode, "motion_zone_score_oof": motion_pred}
        per_game = {name: game_side_scores(y, pred, valid, gids, sides) for name, pred in preds.items()}
        metrics = {}
        for name, pred in preds.items():
            # Game-equal red/blue evaluation; pooled snapshot values included for context.
            mask = valid
            pooled = fixed_class_metrics(y[mask], pred[mask])
            gm = bootstrap_summary(per_game[name], seed=20260927 + j * 17 + len(name), reps=reps)
            metrics[name] = {"pooled_snapshot_metrics": pooled,
                             "game_equal_red_blue": gm,
                             "game_equal_red_blue_point": {k: v["mean"] for k, v in gm.items()}}
        candidate_scores = per_game["score_slope_lambda_1"]
        deltas = {ref: paired_delta(candidate_scores, per_game[ref], seed=20261001 + j * 31 + r, reps=reps)
                  for r, ref in enumerate(("persistence", "motion_zone_score_oof"))}
        truth_dist = {MODES[c]: int(np.sum(valid & (y == c))) for c in range(3)}
        reports[f"{horizon}s"] = {"valid_rows": int(valid.sum()), "label_distribution": truth_dist,
                                   "models": metrics, "paired_deltas_slope_minus_reference": deltas}
    result = {
        "schema": "death-aware-fixed-score-slope-lambda1-train-group-oof-v1",
        "access": {"opened_splits": ["train"], "validation": False, "test": False},
        "label_contract": str(np.asarray(data["global_mode_label_version"])[0]),
        "fixed_candidate": "current_observable_score + horizon_seconds * OLS_slope_per_second; lambda=1 fixed in advance; latest <=5 issue-time scores per game+side; no calibration or tuning",
        "comparators": ["death-aware persistence base_mode", "existing same-fold motion+zone+observable-score OOF"],
        "folds": "exact saved 5-fold GroupKFold by full game_id; both red/blue views remain together",
        "audit": audit,
        "train_games": int(len(np.unique(gids))), "train_rows": int(len(gids)),
        "fold_ids_exact_match_saved_motion_oof": True,
        "split_checks": {"all_games_have_both_sides": True, "red_blue_same_fold": True},
        "horizons": reports,
        "sha256": {"train_npz": hashlib.sha256(Path(train_npz).read_bytes()).hexdigest(),
                   "train_jsonl": hashlib.sha256(Path(train_jsonl).read_bytes()).hexdigest(),
                   "oof_archive": hashlib.sha256(Path(oof_path).read_bytes()).hexdigest()},
        "runtime_impact": "none; analysis-only, no runtime/default/model files changed",
    }
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    (output / "report.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    np.savez_compressed(output / "score_slope_oof.npz", game_ids=gids, sides=sides, issued_at=issued,
                        mode_y=y_all, base_mode=base_mode, fold_ids=fold_ids,
                        observable_score=score.astype(np.float32), slope_per_second=slope.astype(np.float32),
                        score_slope_lambda_1=candidate, motion_candidate=oof["motion_candidate"])
    return result


def markdown(report):
    lines = ["# Death-aware fixed score-slope OOF experiment", "",
             "Train-only 363-match experiment. Lambda=1 was fixed; no horizon-wise threshold, coefficient, or calibration was selected. Current score is reconstructed exclusively from each issue-time observable using the death-aware mode formula. Both red and blue sides from each full match remain in one GroupKFold fold.", "",
             "| Horizon | Model | Accuracy (game-equal) | Macro-F1 | Balanced accuracy | Attack recall | Balance recall | Defense recall |", "|---:|---|---:|---:|---:|---:|---:|---:|"]
    for h, row in report["horizons"].items():
        for name, model in row["models"].items():
            m = model["game_equal_red_blue"]
            vals = [m[k]["mean"] * 100 for k in ("accuracy", "macro_f1", "balanced_accuracy", "recall_进攻", "recall_均衡", "recall_防守")]
            lines.append(f"| {h} | {name} | " + " | ".join(f"{v:.2f}%" for v in vals) + " |")
    lines += ["", "## Paired whole-game bootstrap deltas", "", "Each delta is fixed-slope minus reference; 95% percentile intervals resample complete matches (both side views together), 2,000 replicates. A CI spanning zero is inconclusive.", ""]
    for h, row in report["horizons"].items():
        lines += [f"### {h}", "", "| Comparator | Metric | Delta | 95% CI |", "|---|---|---:|---:|"]
        for ref, result in row["paired_deltas_slope_minus_reference"].items():
            for metric, stat in result["metrics"].items():
                lo, hi = stat["ci95"]
                lines.append(f"| {ref} | {metric} | {stat['mean_delta'] * 100:+.2f} pp | [{lo * 100:+.2f}, {hi * 100:+.2f}] pp |")
    lines += ["", "## Data and parity audit", "",
              f"- Rows: {report['train_rows']:,}; full games: {report['train_games']}; label contract: `{report['label_contract']}`.",
              f"- Observable score reproduces current `base_mode` exactly: {report['audit']['current_observable_score_base_mode_exact_parity']} ({report['audit']['parity_mismatches']} mismatches).",
              f"- Future-observation violations: {report['audit']['future_observation_violations']}.",
              "- Persistence and motion+zone+score predictions are read from the existing full-game GroupKFold OOF artifact; its fold IDs are checked against a fresh deterministic GroupKFold reconstruction.",
              "- No validation/test data was opened. No runtime defaults or production model changed.", "",
              "## Reproduction", "", "```powershell", "python eval/experiment_death_aware_score_slope_fixed_oof.py", "```", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-npz", type=Path, default=TRAIN_NPZ)
    parser.add_argument("--train-jsonl", type=Path, default=TRAIN_JSONL)
    parser.add_argument("--oof", type=Path, default=BASE)
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--bootstrap-reps", type=int, default=2000)
    args = parser.parse_args()
    report = run(args.train_npz, args.train_jsonl, args.oof, args.output, args.bootstrap_reps)
    REPORT.write_text(markdown(report), encoding="utf-8")
    print(json.dumps({h: {n: m["game_equal_red_blue_point"]["accuracy"]
                         for n, m in x["models"].items()} for h, x in report["horizons"].items()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
