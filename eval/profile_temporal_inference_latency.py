"""Small, validation-only runtime latency probe (does not modify production)."""
from __future__ import annotations

import argparse
import gzip
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runtime.temporal_forecast import resolve_temporal_forecaster
from worldstate.narrative import build_global_narrative


def percentile(values, q):
    import numpy as np
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def snapshot_from_row(row):
    entities = {}
    for group, role in (("self_entities", "self"), ("opponent_entities", "opponent")):
        values = (row.get("observable") or {}).get(group) or {}
        values = values.values() if isinstance(values, dict) else values
        for source in values:
            item = dict(source)
            key = str(item.get("entity_id") or f"{role}:{len(entities)}")
            item["role"] = role
            entities[key] = item
    return {"self_team": row["side"], "issued_at": float(row["issued_at"]), "entities": entities}


def measure_runtime(device, row, checkpoint, warmup, samples):
    import torch
    from time import perf_counter
    forecaster = resolve_temporal_forecaster(checkpoint, device=device, team=row["side"], context=12)
    # Include one-time model and calibration initialization separately.
    load_t0 = perf_counter()
    if not forecaster.available():
        raise RuntimeError("forecaster unavailable: " + str(forecaster._load_error))
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    load_seconds = perf_counter() - load_t0

    stage = {"feature_vector_s": [], "forward_s": []}
    original_feature = forecaster._feature_vector

    def timed_feature(*args, **kwargs):
        t0 = perf_counter()
        value = original_feature(*args, **kwargs)
        stage["feature_vector_s"].append(perf_counter() - t0)
        return value

    forecaster._feature_vector = timed_feature
    forward_start = []
    forward_end = []
    if device.startswith("cuda"):
        def pre_hook(_module, _inputs):
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            forward_start.append(event)
        def post_hook(_module, _inputs, _output):
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            forward_end.append(event)
    else:
        def pre_hook(_module, _inputs):
            forward_start.append(perf_counter())
        def post_hook(_module, _inputs, _output):
            forward_end.append(perf_counter())
    hook_a = forecaster._model.register_forward_pre_hook(pre_hook)
    hook_b = forecaster._model.register_forward_hook(post_hook)

    base = snapshot_from_row(row)
    total_s, feature_cursor, forward_cursor = [], 0, 0
    iters = warmup + samples
    for i in range(iters):
        snap = dict(base)
        snap["issued_at"] = float(base["issued_at"]) + 3.0 * i
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        t0 = perf_counter()
        result = forecaster.predict(snap)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        elapsed = perf_counter() - t0
        if i >= warmup:
            total_s.append(elapsed)
    # cuda events become queryable only after the synchronization above.
    for idx in range(warmup, iters):
        if device.startswith("cuda"):
            fwd = float(forward_start[idx].elapsed_time(forward_end[idx])) / 1000.0
        else:
            fwd = float(forward_end[idx] - forward_start[idx])
        stage["forward_s"].append(fwd)
    # Discard warmup feature-vector timings; there is one extraction per call.
    feature = stage["feature_vector_s"][warmup:]
    hook_a.remove(); hook_b.remove()
    forecaster.reset()

    remainder = [max(0.0, total_s[i] - feature[i] - stage["forward_s"][i])
                 for i in range(samples)]
    summary = {}
    for name, vals in (("end_to_end_predict", total_s),
                       ("preprocess_feature_vector", feature),
                       ("model_forward", stage["forward_s"]),
                       ("other_postprocess_window_tensor_decode", remainder)):
        summary[name] = {"mean_ms": statistics.mean(vals) * 1000.0,
                         "p50_ms": percentile(vals, 50) * 1000.0,
                         "p95_ms": percentile(vals, 95) * 1000.0}
    summary["model_and_runtime_load_ms"] = load_seconds * 1000.0
    summary["measured_samples"] = samples
    summary["warmup_calls"] = warmup
    summary["output_horizons"] = len((result or {}).get("horizons", {}))
    if device.startswith("cuda"):
        free_b, total_b = torch.cuda.mem_get_info()
        summary["gpu_memory"] = {"device": torch.cuda.get_device_name(),
                                  "allocated_after_run_mb": torch.cuda.memory_allocated() / (1024 ** 2),
                                  "reserved_after_run_mb": torch.cuda.memory_reserved() / (1024 ** 2),
                                  "free_after_run_mb": free_b / (1024 ** 2),
                                  "total_mb": total_b / (1024 ** 2)}
    return summary, result


def measure_narrative(result, row, repeats):
    from time import perf_counter
    entities = list(((row.get("observable") or {}).get("self_entities") or {}).values())
    enemy = list(((row.get("observable") or {}).get("opponent_entities") or {}).values())
    units = []
    for entity, relation in [(e, "己方") for e in entities] + [(e, "敌方") for e in enemy]:
        units.append({"entity_id": entity.get("entity_id"), "team": entity.get("team"),
                      "type": entity.get("entity_type"), "zone": "当前观测区",
                      "hp_ratio": None, "life_state": "active", "state": {}})
    predictions = []
    for horizon, forecast in (result or {}).get("horizons", {}).items():
        predictions.append({"horizon_seconds": int(horizon),
                             "global_mode": forecast.get("global_mode"),
                             "opponent_trend": forecast.get("opponent_trend"),
                             "enemy_region": forecast.get("enemy_region"),
                             "entities": list((forecast.get("entities") or {}).values()),
                             "outlook": forecast.get("global_outlook")})
    analysis = {"state": {"team": row["side"], "visible_unit_states": units,
                          "unit_locations": [], "control_posture": "unknown"},
                "current_mode": "均衡", "forecast_context": {"predictions": predictions},
                "recommendations": [], "detailed_posture": {},
                "risk_register": {}, "force_balance_guidance": {}}
    # Warmup to exclude interpreter/import cold costs.
    for _ in range(5):
        build_global_narrative(analysis)
    values = []
    for _ in range(repeats):
        t0 = perf_counter()
        build_global_narrative(analysis)
        values.append(perf_counter() - t0)
    return {"kind": "local rule-based text assembly (no remote LLM call)",
            "mean_ms": statistics.mean(values) * 1000,
            "p50_ms": percentile(values, 50) * 1000,
            "p95_ms": percentile(values, 95) * 1000,
            "repeats": repeats}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation-jsonl", default="out_m2/forecast_dataset/validation.jsonl.gz")
    parser.add_argument("--checkpoint", default="out_m2/temporal_forecast_canonical_side/temporal_gru_v1.pt")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--samples", type=int, default=40)
    parser.add_argument("--narrative-repeats", type=int, default=100)
    parser.add_argument("--output", default="out_m2/latency_profile/validation_runtime_latency.json")
    args = parser.parse_args()
    import torch
    if args.device == "auto":
        devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    else:
        devices = args.device.split(",")
    with gzip.open(args.validation_jsonl, "rt", encoding="utf-8") as stream:
        row = json.loads(next(line for line in stream if line.strip()))
    measurements = {}
    narrative = None
    for device in devices:
        stats, result = measure_runtime(device, row, args.checkpoint, args.warmup, args.samples)
        measurements[device] = stats
        if narrative is None:
            narrative = measure_narrative(result, row, args.narrative_repeats)
    output = {"schema": "validation-runtime-latency-profile-v1",
              "sample": {"game_id": row["game_id"], "side": row["side"],
                         "issued_at": row["issued_at"],
                         "entity_count": len(snapshot_from_row(row)["entities"])},
              "checkpoint": args.checkpoint,
              "devices": measurements,
              "narrative": narrative,
              "limitations": ["Single representative validation observation; runtime uses frozen canonical checkpoint.",
                              "GPU timing synchronizes each call to measure wall-clock latency.",
                              "Replay database extraction, full-game frame assembly, HTML serialization, and browser rendering are not included.",
                              "Narrative measurement is local deterministic text assembly, not any remote LLM inference."]}
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(out), "devices": measurements, "narrative": narrative}, ensure_ascii=False))


if __name__ == "__main__":
    main()
