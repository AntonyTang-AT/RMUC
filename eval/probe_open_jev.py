"""Run a small causal Open-Jev probe on the frozen RMUC forecast split.

This is an inference diagnostic, not a training script.  It sends only the
current and prior observable feature snapshots to Open-Jev, asks the same
four horizon questions, and scores the returned probability vectors against
the already frozen exact targets.  The probe is deliberately small because
the local Open-Jev service is a 2B model running on the competition GPU.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import sys
import urllib.error
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from forecast.temporal import MODES, feature_vector, target_labels  # noqa: E402

HORIZONS = (10, 30, 60, 120)


def _groups(path: Path):
    groups = {}
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            groups.setdefault((int(row["game_id"]), row["side"]), []).append(row)
    return groups


def _choose(groups, limit):
    """Choose a deterministic red/blue balanced sample from chronological groups."""
    ordered = sorted(groups.items(), key=lambda item: item[0])
    by_side = {side: [] for side in ("红", "蓝")}
    for (gid, side), rows in ordered:
        if side not in by_side or len(rows) < 2:
            continue
        # Avoid the first context warm-up points; then take evenly spaced rows.
        start = min(12, len(rows) - 1)
        indices = np.linspace(start, len(rows) - 1, num=min(4, len(rows) - start), dtype=int)
        for index in sorted(set(indices)):
            row = rows[int(index)]
            if all(target_labels(row, horizon) is not None for horizon in HORIZONS):
                by_side[side].append((gid, rows, int(index)))
    per_side = max(1, limit // 2)
    selected = []
    for side in ("红", "蓝"):
        selected.extend(by_side[side][:per_side])
    return selected


def _state(rows, index, context):
    history = []
    for row in rows[max(0, index - context + 1): index + 1]:
        vector = feature_vector(row)
        history.append({
            "t": round(float(row["issued_at"]), 3),
            "features": [round(float(value), 4) for value in vector],
        })
    row = rows[index]
    return {
        "task": "RMUC causal tactical forecast",
        "side": row["side"],
        "issued_at": round(float(row["issued_at"]), 3),
        "history_feature_schema": "forecast.temporal.FEATURE_DIM=167; observable-only",
        "history": history,
    }


def _metrics(rows):
    result = {}
    for horizon in HORIZONS:
        probs = np.asarray([row["probabilities"][str(horizon)] for row in rows], dtype=float)
        labels = np.asarray([MODES.index(row["label"][str(horizon)]) for row in rows], dtype=int)
        one_hot = np.eye(len(MODES), dtype=float)[labels]
        result[f"global_mode@{horizon}s"] = {
            "n": int(len(labels)),
            "accuracy": float(np.mean(probs.argmax(axis=1) == labels)),
            "brier": float(np.mean(np.sum((probs - one_hot) ** 2, axis=1))),
            "log_loss": float(-np.mean(np.log(np.clip(probs[np.arange(len(labels)), labels], 1e-12, 1.0)))),
        }
    if rows:
        result["side_macro_accuracy"] = float(np.mean([
            np.mean([int(np.argmax(r["probabilities"]["30"])) == MODES.index(r["label"]["30"])
                     for r in rows if r["side"] == side])
            for side in ("红", "蓝")
            if any(r["side"] == side for r in rows)
        ]))
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="out_m2/forecast_dataset/test.jsonl.gz")
    ap.add_argument("--output", default="out_m2/temporal_forecast/open_jev_probe.json")
    ap.add_argument("--endpoint", default="http://127.0.0.1:8791/v1/systemone")
    ap.add_argument("--max-examples", type=int, default=64)
    ap.add_argument("--context", type=int, default=6)
    args = ap.parse_args()

    from jev.client import Client

    groups = _groups(Path(args.input))
    selected = _choose(groups, args.max_examples)
    client = Client(endpoint=args.endpoint, timeout=300)
    outputs = []
    criteria = {
        "进攻": "the team advances, pressures, or takes offensive initiative",
        "均衡": "the global situation remains contested or broadly stable",
        "防守": "the team retreats, holds, or prioritizes defense",
    }
    for count, (gid, rows, index) in enumerate(selected, 1):
        current = rows[index]
        state = _state(rows, index, args.context)
        questions = {
            str(horizon): {
                "type": "choice",
                "instructions": f"Predict this team's global tactical mode {horizon} seconds after issued_at. Use only the causal history.",
                "criteria": criteria,
            }
            for horizon in HORIZONS
        }
        try:
            response = client.ask(state=state, questions=questions)
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Open-Jev request failed at sample {count}: HTTP {error.code}: {detail}") from error
        probabilities = {
            str(horizon): [float(response["answers"][str(horizon)]["probabilities"][mode]) for mode in MODES]
            for horizon in HORIZONS
        }
        outputs.append({
            "game_id": gid,
            "side": current["side"],
            "issued_at": current["issued_at"],
            "probabilities": probabilities,
            "label": {str(horizon): target_labels(current, horizon)["global_mode"] for horizon in HORIZONS},
        })
        if count % 8 == 0:
            print(f"probed {count}/{len(selected)}", flush=True)
    report = {
        "schema_version": 1,
        "model": "Open-Jev-2B local checkpoint",
        "checkpoint": "models_hf/Open-Jev-2B/package/checkpoint",
        "base_model": "Qwen/Qwen3.5-2B",
        "input_contract": "observable-only causal feature history",
        "n_examples": len(outputs),
        "red_n": sum(row["side"] == "红" for row in outputs),
        "blue_n": sum(row["side"] == "蓝" for row in outputs),
        "metrics": _metrics(outputs),
        "rows": outputs,
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(path), "n_examples": len(outputs), "metrics": report["metrics"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
