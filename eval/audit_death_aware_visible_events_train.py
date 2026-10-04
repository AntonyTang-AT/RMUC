"""Causal train-only audit and cache of visible combat event histories.

Uses the issued-at observable snapshot, not future labels or JSONL targets.
Own firing is inferred only from consecutive own cumulative ammo counters;
opponent firing is deliberately unavailable.  Enemy disappearance alone is
never interpreted as death.  A paired positive-HP to zero-HP observation is
the only loss event counted here.
"""
from __future__ import annotations

import gzip
import json
import sys
import time
from collections import Counter, deque
from pathlib import Path

import numpy as np
import orjson

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from forecast.temporal import MOBILE_TYPES
SOURCE = ROOT / "out_m2/temporal_forecast_death_aware_candidate/train.death_aware.jsonl.gz"
TRAIN = ROOT / "out_m2/temporal_forecast_death_aware_candidate/train.npz"
OUTPUT = ROOT / "out_m2/death_aware_global_mode_experiments/visible_events_train_audit"
WINDOWS = (3, 10, 30)
EVENT_FIELDS = (
    "own_firing_unit_count", "own_positive_ammo_delta", "own_hp_loss_unit_count",
    "own_hp_loss_fraction", "own_confirmed_death_count", "enemy_hp_loss_unit_count",
    "enemy_hp_loss_fraction", "enemy_confirmed_death_count",
    "own_ammo_pair_coverage", "own_hp_pair_coverage", "enemy_hp_pair_coverage",
)


def finite(v: object) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and np.isfinite(float(v))


def entities(value: object) -> list[dict]:
    seq = value.values() if isinstance(value, dict) else value if isinstance(value, list) else []
    return [item for item in seq if isinstance(item, dict)]


def current_mobile(value: object, issued_at: float, audit: Counter) -> list[dict]:
    result = []
    for item in entities(value):
        if item.get("entity_type") not in MOBILE_TYPES:
            continue
        if item.get("observability_state") not in (None, "current"):
            audit["noncurrent_mobile_excluded"] += 1
            continue
        if item.get("presence_state") not in (None, "present"):
            audit["nonpresent_mobile_excluded"] += 1
            continue
        observed_at = item.get("observed_at", issued_at)
        if not finite(observed_at) or float(observed_at) > issued_at + 1e-6:
            audit["future_or_invalid_observed_at_excluded"] += 1
            continue
        if item.get("entity_id") is None:
            audit["missing_entity_id_excluded"] += 1
            continue
        result.append(item)
    return result


def ammo_counters(item: dict) -> tuple[float | None, float | None]:
    return tuple(float(item[key]) if finite(item.get(key)) else None for key in ("ammo17", "ammo42"))


def event_row(current: list[dict], previous: dict[str, dict], issued_at: float,
              role: str, audit: Counter) -> tuple[np.ndarray, dict[str, dict]]:
    values = np.zeros(len(EVENT_FIELDS), dtype=np.float32)
    next_state: dict[str, dict] = {}
    paired_ammo = paired_hp = 0
    for item in current:
        eid = str(item["entity_id"])
        observed_at = float(item.get("observed_at", issued_at))
        hp = float(item["hp"]) if finite(item.get("hp")) else None
        maxhp = float(item["maxhp"]) if finite(item.get("maxhp")) and float(item["maxhp"]) > 0 else None
        ammo = ammo_counters(item) if role == "own" else (None, None)
        if role == "opponent" and any(key in item for key in ("ammo17", "ammo42", "ammo_delta", "firing_inferred")):
            audit["opponent_ammo_field_present"] += 1
        if role == "own" and item.get("ammo_delta") is not None:
            audit["own_ammo_delta_field_present"] += 1
        old = previous.get(eid)
        if old is not None:
            dt = observed_at - old["observed_at"]
            if 0.0 < dt <= 6.01:
                if hp is not None and old["hp"] is not None:
                    paired_hp += 1
                    audit[f"{role}_hp_pairs"] += 1
                    drop = max(0.0, old["hp"] - hp)
                    if drop > 0:
                        target = 2 if role == "own" else 5
                        values[target] += 1
                        denom = maxhp or old["maxhp"] or 500.0
                        values[target + 1] += min(drop / denom, 1.0)
                        audit[f"{role}_hp_drop_pairs"] += 1
                    if old["hp"] > 0 and hp <= 0:
                        values[4 if role == "own" else 7] += 1
                        audit[f"{role}_confirmed_deaths"] += 1
                if role == "own":
                    pairs = [(cur, prev) for cur, prev in zip(ammo, old["ammo"])
                             if cur is not None and prev is not None]
                    if pairs:
                        paired_ammo += 1
                        audit["own_ammo_pairs"] += 1
                        delta = sum(max(0.0, cur - prev) for cur, prev in pairs)
                        if delta > 0:
                            values[0] += 1
                            values[1] += min(delta, 100.0)
                            audit["own_firing_pair_events"] += 1
            else:
                audit[f"{role}_long_or_invalid_gap_excluded"] += 1
        next_state[eid] = {"observed_at": observed_at, "hp": hp, "maxhp": maxhp,
                           "ammo": ammo}
    if role == "own":
        values[8] = paired_ammo / max(1, len(current))
        values[9] = paired_hp / max(1, len(current))
    else:
        values[10] = paired_hp / max(1, len(current))
    audit[f"{role}_current_unit_observations"] += len(current)
    audit[f"{role}_issue_rows_with_current_units"] += int(bool(current))
    return values, next_state


def main() -> None:
    started = time.perf_counter()
    with np.load(TRAIN, allow_pickle=False) as compact:
        gids = compact["game_ids"].astype(np.int64)
        sides = compact["sides"].astype(str)
        times = compact["issued_at"].astype(np.float64)
    names = [f"{w}s_{field}" for w in WINDOWS for field in EVENT_FIELDS]
    x = np.zeros((len(gids), len(names)), dtype=np.float32)
    audit: Counter = Counter()
    previous: dict[tuple[int, str, str], dict[str, dict]] = {}
    histories: dict[tuple[int, str], deque] = {}
    last_issue: dict[tuple[int, str], float] = {}
    rows = 0
    with gzip.open(SOURCE, "rb") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = orjson.loads(line)
            if row.get("split") != "train":
                raise ValueError(f"non-train row at {rows}")
            if rows >= len(gids):
                raise ValueError("JSONL longer than train NPZ")
            game, side, t = int(row["game_id"]), str(row["side"]), float(row["issued_at"])
            if (game, side, round(t, 5)) != (int(gids[rows]), sides[rows], round(times[rows], 5)):
                raise ValueError(f"NPZ/JSONL alignment mismatch at {rows}")
            if finite(row.get("max_input_observed_at")) and float(row["max_input_observed_at"]) > t + 1e-6:
                raise ValueError(f"future max_input_observed_at at {rows}")
            key = (game, side)
            if key in last_issue and t <= last_issue[key]:
                raise ValueError(f"nonincreasing issue time for {key}")
            last_issue[key] = t
            obs = row.get("observable") or {}
            own = current_mobile(obs.get("self_entities"), t, audit)
            enemy = current_mobile(obs.get("opponent_entities"), t, audit)
            current_values = np.zeros(len(EVENT_FIELDS), dtype=np.float32)
            for role, group in (("own", own), ("opponent", enemy)):
                state_key = (game, side, role)
                part, next_state = event_row(group, previous.get(state_key, {}), t, role, audit)
                current_values += part
                # Missing radar observations break tracks; no death or damage
                # is imputed from disappearance.
                previous[state_key] = next_state
            history = histories.setdefault(key, deque())
            history.append((t, current_values))
            while history and history[0][0] <= t - max(WINDOWS):
                history.popleft()
            for j, window in enumerate(WINDOWS):
                included = [event for event_t, event in history if event_t > t - window]
                if included:
                    x[rows, j * len(EVENT_FIELDS):(j + 1) * len(EVENT_FIELDS)] = np.sum(included, axis=0)
            audit["issue_rows"] += 1
            audit["own_rows_with_firing"] += int(current_values[0] > 0)
            audit["enemy_rows_with_hp_drop"] += int(current_values[5] > 0)
            audit["own_rows_with_hp_drop"] += int(current_values[2] > 0)
            rows += 1
            if rows % 20000 == 0:
                print(json.dumps({"rows": rows, "elapsed_seconds": round(time.perf_counter() - started, 1)}), flush=True)
    if rows != len(gids):
        raise ValueError(f"row count mismatch: {rows} vs {len(gids)}")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUTPUT / "visible_event_features.npz", x=x, game_ids=gids,
                        sides=sides, issued_at=times.astype(np.float32), names=np.asarray(names))
    report = {
        "schema": "death-aware-causal-visible-event-train-audit-v1",
        "source": str(SOURCE), "opened_splits": ["train"], "validation_access": False,
        "test_access": False, "train_games": len(set(gids)), "train_rows": rows,
        "exact_npz_jsonl_alignment": True, "max_input_observed_at_future_violations": 0,
        "feature_source": "observable.self_entities and observable.opponent_entities only; no JSONL target keys read",
        "own_firing": "positive increase in consecutive current-observable own 17mm/42mm cumulative counters; gap at most 6.01s",
        "opponent_firing": "unknown; no feature inferred or fabricated",
        "hp_change": "positive-to-zero HP and HP decreases on consecutive current-observable same-ID snapshots; gap at most 6.01s; disappearance is unknown",
        "windows": "(issued_at-window, issued_at] with actual issued_at timestamps",
        "feature_names": names,
        "audit_counts": dict(audit), "runtime_seconds": time.perf_counter() - started,
    }
    (OUTPUT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"report": str(OUTPUT / "report.json"), "feature_cache": str(OUTPUT / "visible_event_features.npz"),
                      "runtime_seconds": report["runtime_seconds"], "audit_counts": dict(audit)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
