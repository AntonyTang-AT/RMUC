"""Evaluate the death-aware canonical GRU candidate on validation only."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from forecast.calibration import calibrate_horizon_probabilities
from forecast.temporal import HORIZONS, MODES, load_compact
from forecast.train import load_model


def _scores(y, p, n_classes):
    pred = p.argmax(axis=1)
    f1 = []
    for c in range(n_classes):
        tp = int(np.sum((y == c) & (pred == c)))
        fp = int(np.sum((y != c) & (pred == c)))
        fn = int(np.sum((y == c) & (pred != c)))
        den = 2 * tp + fp + fn
        f1.append((2 * tp / den) if den else 0.0)
    return {"n": int(len(y)), "accuracy": float(np.mean(pred == y)),
            "macro_f1": float(np.mean(f1))}


def evaluate(checkpoint, validation, output, *, device="auto", batch_size=512,
             bootstrap=2000, seed=0):
    data = load_compact(validation)
    dev = torch.device("cuda" if device == "auto" and torch.cuda.is_available()
                       else ("cpu" if device == "auto" else device))
    model, ckpt = load_model(checkpoint, device=dev)
    model.eval()
    loader = torch.utils.data.DataLoader(torch.from_numpy(data["x"]),
        batch_size=batch_size, shuffle=False, pin_memory=dev.type == "cuda")
    outputs = {h: [] for h in HORIZONS}
    with torch.no_grad():
        for xb in loader:
            logits = model(xb.to(dev, non_blocking=True))
            for h in HORIZONS:
                outputs[h].append(torch.softmax(logits[h]["global_mode"], -1).cpu().numpy())
    probs = {h: np.concatenate(outputs[h]) for h in HORIZONS}
    for i in range(len(data["x"])):
        calibrated = calibrate_horizon_probabilities({h: probs[h][i] for h in HORIZONS})
        for h in HORIZONS:
            probs[h][i] = calibrated[h]

    games = np.asarray(data["game_ids"], dtype=np.int64)
    sides = np.asarray(data["sides"])
    report = {
        "schema": "death-aware-global-mode-validation-v1",
        "validation_only": True, "test_access": False,
        "label_contract": str(data["global_mode_label_version"][0]),
        "feature_schema": str(data["feature_schema_version"][0]),
        "checkpoint": str(checkpoint), "device": str(dev),
        "train_config": {k: ckpt.get(k) for k in
                         ("input_dim", "hidden", "layers", "loss_profile", "seed", "completed_epochs")},
        "metric_population": "rows with exact future global_mode label; both red/blue observing sides",
        "horizons": {},
    }
    rng = np.random.default_rng(seed)
    unique_games = np.array(sorted(set(games.tolist())), dtype=np.int64)
    for j, h in enumerate(HORIZONS):
        y = data["mode_y"][:, j].astype(int)
        valid = y >= 0
        pred = probs[h].argmax(axis=1)
        base = data["base_mode"].astype(int)
        report["horizons"][f"{h}s"] = {
            "candidate": _scores(y[valid], probs[h][valid], len(MODES)),
            "death_aware_persistence": _scores(y[valid], np.eye(len(MODES))[base[valid]], len(MODES)),
            "by_side": {},
        }
        per_game = {"accuracy_delta": [], "macro_f1_delta": []}
        paired_games = []
        for gid in unique_games:
            side_metrics = []
            for side in ("红", "蓝"):
                mask = valid & (games == gid) & (sides == side)
                if not np.any(mask):
                    continue
                side_metrics.append((side, _scores(y[mask], probs[h][mask], len(MODES)),
                                     _scores(y[mask], np.eye(len(MODES))[base[mask]], len(MODES))))
                report["horizons"][f"{h}s"]["by_side"].setdefault(side, {"candidate": [], "death_aware_persistence": []})
                report["horizons"][f"{h}s"]["by_side"][side]["candidate"].append(side_metrics[-1][1])
                report["horizons"][f"{h}s"]["by_side"][side]["death_aware_persistence"].append(side_metrics[-1][2])
            if len(side_metrics) == 2:
                paired_games.append(int(gid))
                per_game["accuracy_delta"].append(float(np.mean([m[1]["accuracy"]-m[2]["accuracy"] for m in side_metrics])))
                per_game["macro_f1_delta"].append(float(np.mean([m[1]["macro_f1"]-m[2]["macro_f1"] for m in side_metrics])))
        for side, bundle in report["horizons"][f"{h}s"]["by_side"].items():
            for model_name, values in bundle.items():
                n = sum(v["n"] for v in values)
                bundle[model_name] = {metric: float(np.average([v[metric] for v in values], weights=[v["n"] for v in values]))
                                      for metric in ("accuracy", "macro_f1")}
                bundle[model_name]["games"] = len(values)
        paired = {}
        for metric, deltas in per_game.items():
            deltas = np.asarray(deltas, dtype=float)
            draws = rng.integers(0, len(deltas), size=(bootstrap, len(deltas)))
            boot = deltas[draws].mean(axis=1)
            paired[metric] = {"mean_red_blue_paired_delta": float(deltas.mean()),
                              "bootstrap_95_ci": [float(np.quantile(boot, .025)), float(np.quantile(boot, .975))],
                              "n_games_both_sides": len(deltas), "bootstrap": bootstrap}
        report["horizons"][f"{h}s"]["paired_per_game_red_blue_delta_vs_persistence"] = paired
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--validation", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="auto")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--bootstrap", type=int, default=2000)
    a = p.parse_args()
    r = evaluate(a.checkpoint, a.validation, a.output, device=a.device,
                 batch_size=a.batch_size, bootstrap=a.bootstrap)
    print(json.dumps({"output": a.output, "horizons": r["horizons"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
