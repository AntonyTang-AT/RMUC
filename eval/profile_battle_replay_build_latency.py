"""Bounded timing probe for replay payload construction; never writes HTML."""
from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import eval.build_battle_replay_visual as replay
from data.db import connect


def distribution(values):
    if not values:
        return None
    values = sorted(float(x) for x in values)
    def q(p):
        # Linear interpolation, matching the usual percentile convention.
        pos = (len(values) - 1) * p
        lo, hi = math.floor(pos), math.ceil(pos)
        return values[lo] if lo == hi else values[lo] * (hi-pos) + values[hi] * (pos-lo)
    return {"calls": len(values), "mean_ms": statistics.mean(values),
            "p50_ms": q(.50), "p95_ms": q(.95), "max_ms": values[-1]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--game-id", type=int, default=None)
    parser.add_argument("--max-time", type=float, default=15.0,
                        help="Keep this small; only the visible prefix is rendered")
    parser.add_argument("--checkpoint", default="out_m2/temporal_forecast_canonical_side/temporal_gru_v1.pt")
    parser.add_argument("--output", default="out_m2/latency_profile/bounded_replay_build_latency.json")
    args = parser.parse_args()
    con = connect()
    game_id = args.game_id or int(con.execute("SELECT game_id FROM matches ORDER BY game_id LIMIT 1").fetchone()[0])
    con.close()

    call_rows = []
    original_resolve = replay.resolve_temporal_forecaster
    def resolve_timed(*pos, **kwargs):
        runtime = original_resolve(*pos, **kwargs)
        original_predict = runtime.predict
        side = getattr(runtime, "team", str(kwargs.get("team", "unknown")))
        def timed_predict(snap):
            start = time.perf_counter()
            result = original_predict(snap)
            call_rows.append({"side": side, "issued_at": float(snap.get("issued_at", 0.0)),
                              "elapsed_ms": (time.perf_counter() - start) * 1000.0})
            return result
        runtime.predict = timed_predict
        return runtime
    replay.resolve_temporal_forecaster = resolve_timed

    start = time.perf_counter()
    payload = replay.build_payload(game_id, args.max_time, replay.DEFAULT_MAP,
                                   display_step=1.0, forecast_checkpoint=args.checkpoint)
    build_s = time.perf_counter() - start
    start = time.perf_counter()
    raw = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    json_s = time.perf_counter() - start
    start = time.perf_counter()
    import gzip
    packed = gzip.compress(raw, compresslevel=6, mtime=0)
    gzip_s = time.perf_counter() - start
    seen_sides = set()
    for row in call_rows:
        row["call_type"] = "cold_first_call_for_side" if row["side"] not in seen_sides else "warm_call"
        seen_sides.add(row["side"])
    call_ms = [row["elapsed_ms"] for row in call_rows]
    cold = [row["elapsed_ms"] for row in call_rows if row["call_type"] == "cold_first_call_for_side"]
    warm = [row["elapsed_ms"] for row in call_rows if row["call_type"] == "warm_call"]
    by_side = {}
    for side in sorted({row["side"] for row in call_rows}):
        side_rows = [row for row in call_rows if row["side"] == side]
        side_warm = [row["elapsed_ms"] for row in side_rows if row["call_type"] == "warm_call"]
        by_side[side] = {"calls": len(side_rows),
                         "first_call_ms": side_rows[0]["elapsed_ms"],
                         "warm_calls": distribution(side_warm)}
    report = {
        "schema": "bounded-replay-build-latency-v1",
        "game_id": game_id,
        "max_time_seconds": args.max_time,
        "frames": len(payload["frames"]),
        "forecast_calls": len(call_rows),
        "build_payload_seconds": build_s,
        "summed_runtime_predict_seconds": sum(call_ms) / 1000.0,
        "all_predict_calls": distribution(call_ms),
        "cold_first_call_per_side": distribution(cold),
        "warm_calls_excluding_first_per_side": distribution(warm),
        "by_side": by_side,
        "per_call_ms": call_rows,
        "json_serialize_seconds": json_s,
        "gzip_compress_seconds": gzip_s,
        "payload_uncompressed_bytes": len(raw),
        "payload_compressed_bytes": len(packed),
        "total_build_and_package_seconds": build_s + json_s + gzip_s,
        "limitations": ["Bounded replay prefix; upstream data examples are still prepared for the full selected game.",
                        "The first call per observing side is called cold because it is that runtime's first predict call; this does not establish the cause of any outlier.",
                        "No HTML file is written and no DB rows are modified.",
                        "Full browser rendering and interactive playback are not measured."],
    }
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(out), **report}, ensure_ascii=False))


if __name__ == "__main__":
    main()
