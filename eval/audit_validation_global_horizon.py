"""Exact-second, same-label-contract global-mode persistence audit.

Validation-only. Reconstructs issue-time labels from the aligned validation
JSONL and exact future labels from the raw replay DB, applying one label
contract to both sides of every comparison. No test path is accepted.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import DB_PATH
from forecast.temporal import (MAP_X, MODES, MOBILE_TYPES, _entities,
                               _hp_ratio, _mode_from_groups)

HORIZONS = (5, 10, 15, 20, 30, 45, 60, 90, 120)
MODE_INDEX = {name: i for i, name in enumerate(MODES)}
CONTRACTS = ("legacy", "death_aware")
HISTORICAL_REPORT = ROOT / "out_m2/temporal_forecast_canonical_side/global_horizon_validation.json"


def legacy_mode_from_groups(own, opp, side):
    """Reproduce committed pre-death-aware relative_margin_v2 semantics."""
    own_mobile = [e for e in own if e.get("entity_type") in MOBILE_TYPES]
    opp_mobile = [e for e in opp if e.get("entity_type") in MOBILE_TYPES]
    if side == "蓝":
        own_push = sum(1 for e in own_mobile
                       if (e.get("position") or [MAP_X / 2])[0] < MAP_X / 2)
        opp_push = sum(1 for e in opp_mobile
                       if (e.get("position") or [MAP_X / 2])[0] >= MAP_X / 2)
    else:
        own_push = sum(1 for e in own_mobile
                       if (e.get("position") or [MAP_X / 2])[0] > MAP_X / 2)
        opp_push = sum(1 for e in opp_mobile
                       if (e.get("position") or [MAP_X / 2])[0] <= MAP_X / 2)
    own_base = next((e for e in own if e.get("entity_type") == "基地"), None)
    opp_base = next((e for e in opp if e.get("entity_type") == "基地"), None)
    score = own_push - opp_push + 2 * (
        (_hp_ratio(opp_base) if opp_base else 1.) -
        (_hp_ratio(own_base) if own_base else 1.))
    return "防守" if score <= -1 else "进攻" if score >= 1 else "均衡"


def mode_index(own, opp, side, contract):
    if contract == "legacy":
        mode = legacy_mode_from_groups(own, opp, side)
    elif contract == "death_aware":
        mode = _mode_from_groups(own, opp, side)
    else:
        raise ValueError(f"Unknown label contract: {contract}")
    return MODE_INDEX[mode]


def contract_consistent_pair(current_own, current_opp, future_own, future_opp, side):
    """Return current/future class IDs under each identical rule version."""
    return {contract: (mode_index(current_own, current_opp, side, contract),
                       mode_index(future_own, future_opp, side, contract))
            for contract in CONTRACTS}


def exact_second(value, *, key):
    value = float(value)
    rounded = int(round(value))
    if abs(value - rounded) > 1e-8:
        raise ValueError(f"{key}={value} is not an exact integer second")
    return rounded


def assert_new_output_path(output):
    destination = Path(output).resolve()
    if destination == HISTORICAL_REPORT.resolve():
        raise ValueError("Refusing to overwrite the historical mixed-contract report")
    return destination


def entity_from_db(row):
    return {"entity_id": f"{row['阵营']}:{row['robot_id']}",
            "team": row["阵营"], "entity_type": row["机器人类型"],
            "position": [row["x"], row["y"]], "hp": row["当前血量"],
            "maxhp": row["最大血量"]}


def confusion_metrics(c):
    n = int(c.sum())
    if not n:
        return None
    f1s = []
    for k in range(len(MODES)):
        tp = float(c[k, k])
        fp = float(c[:, k].sum() - tp)
        fn = float(c[k, :].sum() - tp)
        f1s.append(2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.)
    return {"accuracy": float(np.trace(c) / n),
            "macro_f1": float(np.mean(f1s)),
            "transition_rate": float((n - np.trace(c)) / n), "n": n}


def bootstrap_game_mean(values, seed=20260927, reps=2000):
    ids = sorted(values)
    if not ids:
        return {"n_games": 0, "mean": None, "ci95": [None, None]}
    arr = np.asarray([values[g] for g in ids], dtype=np.float64)
    rng = np.random.default_rng(seed)
    draw = arr[rng.integers(0, len(arr), size=(reps, len(arr)))].mean(1)
    return {"n_games": len(ids), "mean": float(arr.mean()),
            "ci95": [float(np.quantile(draw, .025)),
                     float(np.quantile(draw, .975))],
            "bootstrap_replicates": reps}


def _score_confusions(conf, games, side_count):
    result = {}
    game_scores = {}
    for h in HORIZONS:
        side_scores = {}
        per_game = {}
        for side_idx, side in enumerate(("红", "蓝")):
            side_scores[side] = confusion_metrics(conf[h][:, side_idx].sum(0))
        for gi, gid in enumerate(games):
            r = confusion_metrics(conf[h][gi, 0])
            b = confusion_metrics(conf[h][gi, 1])
            if r and b:
                per_game[int(gid)] = {
                    metric: (r[metric] + b[metric]) / 2
                    for metric in ("accuracy", "macro_f1", "transition_rate")}
        one = {}
        for metric in ("accuracy", "macro_f1", "transition_rate"):
            one[f"{metric}_game_equal_red_blue_equal"] = (
                float(np.mean([row[metric] for row in per_game.values()]))
                if per_game else None)
        one.update({"by_side_pooled": side_scores,
                    "scored_count": int(conf[h].sum()),
                    "possible_count": side_count,
                    "coverage": float(conf[h].sum() / side_count) if side_count else None,
                    "complete_red_blue_games": len(per_game),
                    "accuracy_game_bootstrap_ci95": bootstrap_game_mean(
                        {g: row["accuracy"] for g, row in per_game.items()},
                        seed=20260927 + h)})
        result[str(h)] = one
        game_scores[h] = per_game
    adjacent = {}
    for h0, h1 in zip(HORIZONS[:-1], HORIZONS[1:]):
        common = set(game_scores[h0]) & set(game_scores[h1])
        adjacent[f"{h0}_to_{h1}s_accuracy_delta"] = bootstrap_game_mean(
            {g: game_scores[h1][g]["accuracy"] - game_scores[h0][g]["accuracy"]
             for g in common}, seed=20261000 + h1)
    return result, adjacent


def audit(db=DB_PATH,
          validation=ROOT / "out_m2/temporal_forecast_canonical_side/validation.npz",
          validation_jsonl=ROOT / "out_m2/forecast_dataset/validation.jsonl.gz",
          output=ROOT / "out_m2/validation_global_horizon_consistent/report.json"):
    output = assert_new_output_path(output)
    validation = Path(validation).resolve()
    validation_jsonl = Path(validation_jsonl).resolve()
    with np.load(validation, allow_pickle=False) as z:
        gids = z["game_ids"].astype(np.int64)
        sides = z["sides"].astype(str)
        issued = z["issued_at"].astype(np.float64)
        archived_base = z["base_mode"].astype(np.int64)
    n = len(gids)
    games = sorted(map(int, np.unique(gids)))
    game_index = {gid: i for i, gid in enumerate(games)}
    side_index = {"红": 0, "蓝": 1}
    current = {contract: np.full(n, -1, dtype=np.int8) for contract in CONTRACTS}
    current_entities = {}
    old_matches_archive = np.zeros(n, dtype=bool)
    new_matches_archive = np.zeros(n, dtype=bool)

    # Rebuild issue-time modes under both contracts from the exact aligned,
    # already radar-masked validation observable; do not trust old base_mode
    # for death-aware rows.
    seen = 0
    with gzip.open(validation_jsonl, "rt", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            if seen >= n:
                raise ValueError("validation JSONL has more rows than compact archive")
            expected = (int(gids[seen]), str(sides[seen]), float(issued[seen]))
            got = (int(row["game_id"]), str(row["side"]), float(row["issued_at"]))
            if got[:2] != expected[:2] or abs(got[2] - expected[2]) > 1e-8:
                raise ValueError(f"validation JSONL/NPZ row mismatch at {seen}: {got} != {expected}")
            obs = row.get("observable") or {}
            own = _entities(obs, "self_entities")
            opp = _entities(obs, "opponent_entities")
            current_entities[seen] = {"own": own, "opp": opp}
            side = str(row["side"])
            current["legacy"][seen] = mode_index(own, opp, side, "legacy")
            current["death_aware"][seen] = mode_index(own, opp, side, "death_aware")
            old_matches_archive[seen] = current["legacy"][seen] == archived_base[seen]
            new_matches_archive[seen] = current["death_aware"][seen] == archived_base[seen]
            seen += 1
    if seen != n:
        raise ValueError(f"validation JSONL rows={seen}, compact rows={n}")

    issue_by_game = defaultdict(list)
    for i, (gid, side, timestamp) in enumerate(zip(gids, sides, issued)):
        t = exact_second(timestamp, key=f"issued_at[{i}]")
        issue_by_game[int(gid)].append((i, str(side), t))

    confusion = {contract: {h: np.zeros((len(games), 2, 3, 3), dtype=np.uint32)
                            for h in HORIZONS}
                 for contract in CONTRACTS}
    exact_counts = {h: 0 for h in HORIZONS}
    raw_issue_count = 0
    con = sqlite3.connect(f"file:{Path(db).resolve().as_posix()}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    fields = 'game_id, "时刻秒", 阵营, robot_id, 机器人类型, x, y, 当前血量, 最大血量'
    try:
        for gid in games:
            states = defaultdict(list)
            for row in con.execute(
                    f'SELECT {fields} FROM timeseries WHERE game_id=? ORDER BY "时刻秒"',
                    (gid,)):
                t = exact_second(row["时刻秒"], key=f"timeseries[{gid}]")
                states[t].append(entity_from_db(row))
            raw_issue_count += sum(t in states for _, _, t in issue_by_game[gid])
            for row_i, side, t in issue_by_game[gid]:
                if t not in states:
                    continue
                for h in HORIZONS:
                    future_state = states.get(t + h)
                    if future_state is None:
                        continue
                    own_future = [e for e in future_state if e["team"] == side]
                    opp_future = [e for e in future_state if e["team"] != side]
                    # Current entities were built from the aligned observable
                    # at the issue time, never from the archive's old label.
                    # Retain their two contract-specific labels per row.
                    # JSONL row index is the compact sample index.
                    # Values are carried in `current_entities` below.
                    pair = contract_consistent_pair(
                        current_entities[row_i]["own"], current_entities[row_i]["opp"],
                        own_future, opp_future, side)
                    gi = game_index[gid]
                    si = side_index[side]
                    for contract in CONTRACTS:
                        pred, truth = pair[contract]
                        if pred < 0:
                            continue
                        confusion[contract][h][gi, si, pred, truth] += 1
                    exact_counts[h] += 1
    finally:
        con.close()

    metrics = {}
    adjacent = {}
    for contract in CONTRACTS:
        metrics[contract], adjacent[contract] = _score_confusions(
            confusion[contract], games, n)
    sha_npz = hashlib.sha256(validation.read_bytes()).hexdigest()
    sha_jsonl = hashlib.sha256(validation_jsonl.read_bytes()).hexdigest()
    report = {
        "schema": "validation-global-mode-exact-horizon-same-contract-v2",
        "test_access": False,
        "data_contract": {
            "validation_archive": str(validation.relative_to(ROOT)).replace("\\", "/"),
            "validation_jsonl": str(validation_jsonl.relative_to(ROOT)).replace("\\", "/"),
            "validation_npz_sha256": sha_npz,
            "validation_jsonl_sha256": sha_jsonl,
            "games": len(games), "issue_rows": n,
            "red_blue_views": True,
            "time_resolution_s": 1, "issue_step_s": 3,
            "target_rule": "exact integer issued_at + horizon second; no interpolation, nearest timestamp, or fill",
            "issue_label_source": "aligned validation JSONL observable, radar-masked and causal",
            "future_label_source": "raw match replay database exact timestamp state",
            "contracts": {
                "legacy": "committed pre-death-aware relative_margin_v2; mobile units count regardless of hp",
                "death_aware": "current _mode_from_groups; mobile units with hp<=0 or life_state=destroyed are excluded",
            },
            "aggregation": "per game and side; red/blue equal within each game; then games equally weighted; bootstrap resamples complete games",
        },
        "issue_parity": {
            "legacy_current_matches_archived_base_mode": float(old_matches_archive.mean()),
            "death_aware_current_matches_archived_base_mode": float(new_matches_archive.mean()),
            "raw_issue_seconds_found": int(raw_issue_count),
        },
        "metrics": metrics,
        "adjacent_horizon_paired_accuracy_deltas": adjacent,
        "scored_rows_by_horizon": {str(h): int(exact_counts[h]) for h in HORIZONS},
        "interpretation": {
            "scope": "same-contract current-to-future mode persistence predictability; not learned model performance",
            "historical_report_preserved": "out_m2/temporal_forecast_canonical_side/global_horizon_validation.json was not overwritten",
            "class_vocabulary": list(MODES),
            "macro_f1": "three fixed classes; undefined per-class score is zero",
            "coverage": "scored rows divided by all validation issue rows; post-game targets are censored, not imputed",
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=DB_PATH)
    parser.add_argument("--validation", default=str(ROOT / "out_m2/temporal_forecast_canonical_side/validation.npz"))
    parser.add_argument("--validation-jsonl", default=str(ROOT / "out_m2/forecast_dataset/validation.jsonl.gz"))
    parser.add_argument("--output", default=str(ROOT / "out_m2/validation_global_horizon_consistent/report.json"))
    args = parser.parse_args()
    report = audit(args.db, args.validation, args.validation_jsonl, args.output)
    summary = {contract: {
        h: {"accuracy": round(row["accuracy_game_equal_red_blue_equal"], 4),
            "macro_f1": round(row["macro_f1_game_equal_red_blue_equal"], 4),
            "coverage": round(row["coverage"], 4),
            "accuracy_ci95": row["accuracy_game_bootstrap_ci95"]["ci95"]}
        for h, row in values.items() if h in {"10", "30", "60", "120"}}
        for contract, values in report["metrics"].items()}
    print(json.dumps({"output": args.output, "test_access": False,
                      "summary": summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
