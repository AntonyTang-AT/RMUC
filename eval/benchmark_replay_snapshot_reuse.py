"""A/B benchmark the replay's duplicated radar-snapshot preparation."""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import eval.build_battle_replay_visual as replay
from data.db import connect
from data.forecast_dataset import build_game_forecast_examples as current_builder
from data.radar_visibility_replay import load_radar_visibility_snapshots


def _json_digest(payload):
    content = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _prediction_digest(payload):
    # Forecast IDs in adaptive_strategy identify records and can be generated
    # independently of the mathematical prediction. Compare all per-horizon
    # forecast, exact truth, entity output and evaluation fields separately.
    decisions = []
    for decision in payload.get("decisions", []):
        decisions.append({"t": decision.get("t"), "views": {
            side: view.get("horizons", {})
            for side, view in decision.get("views", {}).items()}})
    return _json_digest(decisions)


def _build(args, mode):
    if mode == "old_duplicate_snapshots":
        def old_builder(rows, game_id, *, horizons=(10, 30, 60, 120), step=3,
                        return_snapshots=False):
            examples = current_builder(rows, game_id, horizons=horizons, step=step)
            if not return_snapshots:
                return examples
            con = connect()
            try:
                snapshots = load_radar_visibility_snapshots(con, game_id)
            finally:
                con.close()
            return examples, snapshots
        replay.build_game_forecast_examples = old_builder
    else:
        replay.build_game_forecast_examples = current_builder
    started = time.perf_counter()
    payload = replay.build_payload(args.game_id, args.max_time, replay.DEFAULT_MAP,
        display_step=1.0, forecast_checkpoint=args.checkpoint,
        position_fusion_candidate=False)
    elapsed = time.perf_counter() - started
    return {"mode": mode, "seconds": elapsed,
            "frames": len(payload["frames"]),
            "decisions": len(payload.get("decisions", [])),
            "forecast_hash": _json_digest(payload),
            "prediction_hash": _prediction_digest(payload),
            "prediction_issues": sum(bool(f.get("decision_updated")) for f in payload["frames"])}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--game-id", type=int, default=None)
    p.add_argument("--max-time", type=float, default=420.0)
    p.add_argument("--repeat", type=int, default=2)
    p.add_argument("--checkpoint", default="out_m2/temporal_forecast_canonical_side/temporal_gru_v1.pt")
    p.add_argument("--output", default="out_m2/latency_profile/snapshot_reuse_ab.json")
    a = p.parse_args()
    if a.repeat < 1:
        raise ValueError("repeat must be positive")
    if a.game_id is None:
        con = connect()
        try:
            a.game_id = int(con.execute("SELECT game_id FROM matches ORDER BY game_id LIMIT 1").fetchone()[0])
        finally:
            con.close()
    rows = []
    for _ in range(a.repeat):
        rows.append(_build(a, "old_duplicate_snapshots"))
        rows.append(_build(a, "reuse_existing_snapshots"))
    old = [r["seconds"] for r in rows if r["mode"] == "old_duplicate_snapshots"]
    new = [r["seconds"] for r in rows if r["mode"] == "reuse_existing_snapshots"]
    paired = list(zip(rows[::2], rows[1::2]))
    report = {"schema": "replay-snapshot-reuse-ab-v1", "game_id": a.game_id,
        "max_time_seconds": a.max_time, "repeat": a.repeat, "checkpoint": a.checkpoint,
        "results": rows, "mean_old_seconds": statistics.mean(old),
        "mean_reuse_seconds": statistics.mean(new),
        "mean_saved_seconds": statistics.mean(old)-statistics.mean(new),
        "mean_speedup": statistics.mean(old)/statistics.mean(new),
        "each_pair_prediction_content_exactly_identical": all(
            old["prediction_hash"] == new["prediction_hash"] for old, new in paired),
        "each_pair_full_payload_exactly_identical": all(
            old["forecast_hash"] == new["forecast_hash"] for old, new in paired),
        "test_access": False,
        "scope": "offline replay payload generation only; no HTML browser/playback benchmark"}
    out = Path(a.output); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
