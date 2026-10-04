"""Train and select a causal canonical-side mode candidate without reading test.

This experiment intentionally accepts only train/validation paths. It fits a
small GRU or TCN+Transformer hybrid using the existing compact canonical-side
contract, then writes validation-only per-horizon classification and
probability metrics. There is no test-path argument in this script.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from forecast.calibration import calibrate_horizon_probabilities
from forecast.temporal import HORIZONS, MODES, load_compact
from forecast.train import train_model, load_model


def _macro_f1(y, p, classes):
    pred = p.argmax(1)
    values = []
    for c in range(classes):
        tp = np.sum((y == c) & (pred == c))
        fp = np.sum((y != c) & (pred == c))
        fn = np.sum((y == c) & (pred != c))
        den = 2 * tp + fp + fn
        values.append(float(2 * tp / den) if den else 0.0)
    return float(np.mean(values))


def _probabilities(model_path, validation_path, device, batch_size):
    dev = torch.device("cuda" if device == "auto" and torch.cuda.is_available()
                       else ("cpu" if device == "auto" else device))
    model, _ = load_model(model_path, device=dev)
    model.eval()
    data = load_compact(validation_path)
    out = {h: [] for h in HORIZONS}
    loader = torch.utils.data.DataLoader(torch.from_numpy(data["x"]),
        batch_size=batch_size, shuffle=False, pin_memory=dev.type == "cuda")
    with torch.no_grad():
        for xb in loader:
            logits = model(xb.to(dev, non_blocking=True))
            for h in HORIZONS:
                out[h].append(torch.softmax(logits[h]["global_mode"], -1).cpu().numpy())
    probs = {h: np.concatenate(out[h]) for h in HORIZONS}
    for i in range(len(data["x"])):
        fixed = calibrate_horizon_probabilities({h: probs[h][i] for h in HORIZONS})
        for h in HORIZONS:
            probs[h][i] = fixed[h]
    return data, probs, str(dev)


def _blend(data, probs, alphas):
    rows = {}
    sides = np.asarray(data["sides"])
    for j, h in enumerate(HORIZONS):
        y = data["mode_y"][:, j].astype(int)
        valid = y >= 0
        p = probs[h].copy()
        idx = data["base_mode"].astype(int)
        prior = np.full_like(p, 0.02 / max(1, p.shape[1] - 1))
        legal = (idx >= 0) & (idx < p.shape[1])
        prior[np.arange(len(idx))[legal], idx[legal]] = 0.98
        alpha = float(alphas.get(h, 1.0))
        p = alpha * p + (1 - alpha) * prior
        def metrics(mask):
            yy, pp = y[mask], p[mask]
            return {"n": int(mask.sum()), "accuracy": float(np.mean(pp.argmax(1) == yy)),
                    "macro_f1": _macro_f1(yy, pp, len(MODES)),
                    "brier": float(np.mean(np.sum((pp - np.eye(len(MODES))[yy]) ** 2, axis=1)))}
        rows[f"{h}s"] = {"alpha": alpha, "pooled": metrics(valid),
                          "red": metrics(valid & (sides == "红")),
                          "blue": metrics(valid & (sides == "蓝"))}
    return rows


def _select_alpha(data, probs):
    selected = {}
    sides = np.asarray(data["sides"])
    for j, h in enumerate(HORIZONS):
        y = data["mode_y"][:, j].astype(int)
        valid = y >= 0
        idx = data["base_mode"].astype(int)
        prior = np.full((len(y), len(MODES)), 0.02 / (len(MODES) - 1), dtype=float)
        legal = (idx >= 0) & (idx < len(MODES))
        prior[np.arange(len(idx))[legal], idx[legal]] = 0.98
        best = None
        for alpha in np.linspace(0, 1, 21):
            p = alpha * probs[h] + (1 - alpha) * prior
            score_side = []
            for side in ("红", "蓝"):
                mask = valid & (sides == side)
                yy, pp = y[mask], p[mask]
                score_side.append(.6 * np.mean(pp.argmax(1) == yy) +
                                  .4 * _macro_f1(yy, pp, len(MODES)))
            score = float(np.mean(score_side))
            if best is None or score > best[0] + 1e-12:
                best = (score, float(alpha))
        selected[h] = best[1]
    return selected


def run(args):
    train_data = load_compact(args.train)
    val_data = load_compact(args.validation)
    if set(np.asarray(val_data["sides"]).tolist()) != {"红", "蓝"}:
        raise ValueError("validation must include both red and blue observing sides")
    if train_data["x"].shape[2] != val_data["x"].shape[2]:
        raise ValueError("train/validation feature dimensions differ")
    result = train_model(args.train, args.validation, args.model,
        epochs=args.epochs, batch_size=args.batch_size, hidden=args.hidden,
        layers=args.layers, lr=args.lr, weight_decay=args.weight_decay,
        device=args.device, seed=args.seed, loss_profile=args.loss_profile,
        architecture=args.architecture, nhead=args.heads, dropout=args.dropout,
        max_seq_len=train_data["x"].shape[1])
    data, probs, used_device = _probabilities(args.model, args.validation,
                                               args.device, args.batch_size)
    alphas = _select_alpha(data, probs)
    report = {
        "schema": "canonical-mode-candidate-validation-v1",
        "model": str(args.model), "architecture": args.architecture,
        "device": used_device, "seed": args.seed,
        "training": {k: result.get(k) for k in ("best_val_loss", "history", "completed_epochs")},
        "validation_split": str(args.validation),
        "validation_game_ids": sorted(set(int(x) for x in data["game_ids"])),
        "selected_alpha_by_horizon": {f"{h}s": a for h, a in alphas.items()},
        "metrics": _blend(data, probs, alphas),
        "selection": "validation-only; per-horizon equal-red-blue mean(0.6*accuracy+0.4*macro_f1); alpha grid 0.00..1.00 step 0.05",
        "test_access": False,
    }
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"model": args.model, "report": args.report,
                      "val_metrics": report["metrics"], "alphas": report["selected_alpha_by_horizon"]},
                     ensure_ascii=False))
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train", required=True)
    p.add_argument("--validation", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--report", required=True)
    p.add_argument("--architecture", default="hybrid", choices=("hybrid", "gru"))
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.15)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=41)
    p.add_argument("--loss-profile", default="business_weighted", choices=("baseline", "business_weighted"))
    run(p.parse_args())


if __name__ == "__main__":
    main()
