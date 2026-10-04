"""Fit and audit the causal enemy-region transition sidecar."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from forecast.temporal import REGION_LABELS


HORIZON_INDICES = {10: 0, 30: 1, 60: 2, 120: 3}
TIME_BUCKET_SECONDS = 60
LAPLACE_SMOOTHING = 10.0
ALPHA_GRID = np.linspace(0.0, 1.0, 41)


def load(root: Path, split: str):
    with np.load(root / f"{split}.npz", allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def _valid(data, horizon_index):
    y = data["region_y"][:, int(horizon_index)].astype(np.int64)
    base = data["base_region"].astype(np.int64)
    return y, base, (y >= 0) & (y < len(REGION_LABELS)) & (base >= 0) & (base < len(REGION_LABELS))


def _key(base_region, side, issued_at, base_mode):
    return json.dumps([
        int(base_region), str(side),
        int(max(0.0, float(issued_at)) // TIME_BUCKET_SECONDS),
        int(base_mode),
    ], ensure_ascii=False, separators=(",", ":"))


def fit_table(data, horizon_index):
    y, base, valid = _valid(data, horizon_index)
    sides = data["sides"].astype(str)
    table = {}
    for index in np.flatnonzero(valid):
        key = _key(base[index], sides[index], data["issued_at"][index],
                   data["base_mode"][index])
        counts = table.setdefault(key, np.full(len(REGION_LABELS), LAPLACE_SMOOTHING))
        counts[y[index]] += 1.0
    return {key: (counts / counts.sum()).round(12).tolist()
            for key, counts in table.items()}


def predict(data, table, alpha, horizon_index):
    y, base, valid = _valid(data, horizon_index)
    sides = data["sides"].astype(str)
    probabilities = np.zeros((len(y), len(REGION_LABELS)), dtype=np.float64)
    for index in range(len(y)):
        key = _key(base[index], sides[index], data["issued_at"][index],
                   data["base_mode"][index])
        transition = np.asarray(table.get(key, [1.0 / len(REGION_LABELS)] * len(REGION_LABELS)))
        persistence = np.full(len(REGION_LABELS), 0.02 / (len(REGION_LABELS) - 1))
        if 0 <= base[index] < len(REGION_LABELS):
            persistence[base[index]] = 0.98
        probabilities[index] = alpha * transition + (1.0 - alpha) * persistence
    return y, base, valid, probabilities


def metrics(y, pred, sides, valid):
    rows = []
    for side in ("红", "蓝"):
        mask = valid & (sides == side)
        rows.append({
            "accuracy": float(accuracy_score(y[mask], pred[mask])),
            "macro_f1": float(f1_score(y[mask], pred[mask], labels=np.arange(len(REGION_LABELS)),
                                        average="macro", zero_division=0)),
            "n": int(mask.sum()),
        })
    accuracy = float(np.mean([row["accuracy"] for row in rows]))
    macro_f1 = float(np.mean([row["macro_f1"] for row in rows]))
    return {"accuracy": accuracy, "macro_f1": macro_f1,
            "score": 0.6 * accuracy + 0.4 * macro_f1,
            "by_side": {side: row for side, row in zip(("红", "蓝"), rows)},
            "n": int(valid.sum())}


def run(root: Path, output: Path, horizon_seconds=30):
    horizon_index = HORIZON_INDICES[int(horizon_seconds)]
    train, validation, test = (load(root, split) for split in ("train", "validation", "test"))
    table = fit_table(train, horizon_index)
    yv, bv, valid_v, pv = predict(validation, table, 1.0, horizon_index)
    persistence_v = metrics(yv, bv, validation["sides"].astype(str), valid_v)
    candidates = []
    for alpha in ALPHA_GRID:
        # Recompute the blend for each frozen validation candidate.
        _, _, _, probabilities = predict(validation, table, float(alpha), horizon_index)
        candidate = metrics(yv, probabilities.argmax(axis=1), validation["sides"].astype(str), valid_v)
        candidate["alpha_transition"] = float(alpha)
        candidates.append(candidate)
    pareto = [row for row in candidates
              if row["accuracy"] >= persistence_v["accuracy"]
              and row["macro_f1"] >= persistence_v["macro_f1"]]
    chosen = max(pareto or candidates,
                 key=lambda row: (row["score"], row["macro_f1"], row["accuracy"],
                                  -row["alpha_transition"]))
    yt, bt, valid_t, pt = predict(test, table, chosen["alpha_transition"], horizon_index)
    test_candidate = metrics(yt, pt.argmax(axis=1), test["sides"].astype(str), valid_t)
    test_persistence = metrics(yt, bt, test["sides"].astype(str), valid_t)
    manifest = {
        "schema": "enemy-region-transition-manifest-v1",
        "feature_schema": "canonical-side-current-transition-v1",
        "horizon_seconds": int(horizon_seconds),
        "alpha_transition": chosen["alpha_transition"],
        "laplace_smoothing": LAPLACE_SMOOTHING,
        "time_bucket_seconds": TIME_BUCKET_SECONDS,
        "table": table,
        "enabled": bool(test_candidate["accuracy"] > test_persistence["accuracy"]
                         and test_candidate["macro_f1"] > test_persistence["macro_f1"]),
    }
    report = {
        "schema": "enemy-region-transition-v1",
        "validation": {"persistence": persistence_v, "chosen": chosen,
                        "pareto_count": len(pareto)},
        "test": {"candidate": test_candidate, "persistence": test_persistence,
                 "double_gate": manifest["enabled"]},
        "manifest": {key: value for key, value in manifest.items() if key != "table"},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (output.parent / "production_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def export_frozen(root: Path, output: Path, horizon_seconds: int,
                  alpha: float, promotion_evidence: dict):
    """Export a previously gated horizon without reopening its locked test split."""
    horizon_seconds = int(horizon_seconds)
    if horizon_seconds not in HORIZON_INDICES:
        raise ValueError(f"unsupported horizon: {horizon_seconds}")
    if not 0.0 <= float(alpha) <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    candidate = promotion_evidence["candidate"]
    persistence = promotion_evidence["persistence"]
    for split in ("validation", "locked_test"):
        if not (candidate[split]["accuracy"] > persistence[split]["accuracy"]
                and candidate[split]["macro_f1"] > persistence[split]["macro_f1"]):
            raise ValueError(f"{split} double-metric promotion gate failed")
    train = load(root, "train")
    table = fit_table(train, HORIZON_INDICES[horizon_seconds])
    manifest = {
        "schema": "enemy-region-transition-manifest-v1",
        "feature_schema": "canonical-side-current-transition-v1",
        "horizon_seconds": horizon_seconds,
        "alpha_transition": float(alpha),
        "laplace_smoothing": LAPLACE_SMOOTHING,
        "time_bucket_seconds": TIME_BUCKET_SECONDS,
        "enabled": True,
        "promotion_evidence": promotion_evidence,
        "table": table,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in manifest.items() if key != "table"},
                     ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="out_m2/temporal_forecast_canonical_side")
    parser.add_argument("--output", default="out_m2/enemy_region_transition/report.json")
    parser.add_argument("--horizon", type=int, choices=sorted(HORIZON_INDICES), default=30)
    parser.add_argument("--export-frozen", action="store_true",
                        help="export train-only table using an already frozen validation/test gate")
    parser.add_argument("--alpha", type=float)
    parser.add_argument("--promotion-evidence", default="",
                        help="JSON with candidate and persistence metrics for validation and locked_test")
    args = parser.parse_args()
    if args.export_frozen:
        if args.alpha is None or not args.promotion_evidence:
            parser.error("--export-frozen requires --alpha and --promotion-evidence")
        export_frozen(Path(args.root), Path(args.output), args.horizon, args.alpha,
                      json.loads(args.promotion_evidence))
    else:
        run(Path(args.root), Path(args.output), args.horizon)


if __name__ == "__main__":
    main()
