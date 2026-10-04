"""Cross-fit a confidence-gated global-mode override using validation only.

The frozen canonical-side checkpoint's probabilities are scored by game-group
cross-fitting: thresholds are selected on four fifths of validation games and
applied to the held-out fifth. Test files are intentionally not accepted.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from eval.train_canonical_mode_candidate import _probabilities
from forecast.temporal import HORIZONS, MODES
from forecast.train import load_model


def metrics(y, pred):
    cm = np.zeros((len(MODES), len(MODES)), dtype=np.int64)
    np.add.at(cm, (y, pred), 1)
    f1 = []
    for c in range(len(MODES)):
        tp = cm[c, c]
        den = 2 * tp + (cm[:, c].sum() - tp) + (cm[c, :].sum() - tp)
        f1.append(float(2 * tp / den) if den else 0.0)
    return {"n": int(len(y)), "accuracy": float(np.mean(y == pred)),
            "macro_f1": float(np.mean(f1)), "confusion": cm.tolist()}


def side_balanced_score(y, pred, sides):
    scores = []
    for side in ("红", "蓝"):
        m = sides == side
        if np.any(m):
            scores.append(metrics(y[m], pred[m])["accuracy"])
    return float(np.mean(scores))


def bootstrap_game_delta(y, pred, baseline, sides, gids, n_boot=2000, seed=7301):
    """Paired red+blue complete-game bootstrap for accuracy difference."""
    unique = np.asarray(sorted(set(int(g) for g in gids)), dtype=np.int64)
    game_delta = []
    for gid in unique:
        ds = []
        for side in ("红", "蓝"):
            m = (gids == gid) & (sides == side)
            if np.any(m):
                ds.append(float(np.mean(pred[m] == y[m]) - np.mean(baseline[m] == y[m])))
        if len(ds) == 2:
            game_delta.append(float(np.mean(ds)))
    vals = np.asarray(game_delta, dtype=float)
    rng = np.random.default_rng(seed)
    samples = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        samples[i] = vals[rng.integers(0, len(vals), len(vals))].mean()
    return {"mean_delta_accuracy_pp": float(vals.mean() * 100),
            "ci95_pp": [float(x * 100) for x in np.quantile(samples, [.025, .975])],
            "complete_games": int(len(vals)), "bootstrap_replicates": n_boot}


def run(args):
    data, probs, device = _probabilities(args.checkpoint, args.validation, args.device,
                                          args.batch_size)
    # Read frozen alphas from the checkpoint, exactly as runtime does.
    _, ckpt = load_model(args.checkpoint, device="cpu")
    blend = ckpt.get("persistence_blend") or {}
    alpha = {h: float(blend.get("alphas", {}).get(f"global_mode@{h}s", 1.0))
             for h in HORIZONS}
    y_all = np.asarray(data["mode_y"], dtype=int)
    base = np.asarray(data["base_mode"], dtype=int)
    sides = np.asarray(data["sides"])
    gids = np.asarray(data["game_ids"], dtype=np.int64)
    games = sorted(set(int(g) for g in gids))
    fold_for_game = {g: i % args.folds for i, g in enumerate(games)}
    folds = np.asarray([fold_for_game[int(g)] for g in gids])
    # 98/1.0 smooth one-hot prior matches existing alpha-selection scripts.
    prior = np.full((len(base), len(MODES)), 0.01, dtype=np.float64)
    ok = (base >= 0) & (base < len(MODES))
    prior[np.arange(len(base))[ok], base[ok]] = 0.98
    rows = {}
    thresholds = np.round(np.arange(.34, .991, .01), 2)
    for j, h in enumerate(HORIZONS):
        y = y_all[:, j]
        valid = (y >= 0) & (base >= 0) & (base < len(MODES))
        persistence = base.copy()
        model_pred = probs[h].argmax(axis=1)
        confidence = probs[h].max(axis=1)
        auto_p = alpha[h] * probs[h] + (1 - alpha[h]) * prior
        auto_pred = auto_p.argmax(axis=1)
        gate_pred = persistence.copy()
        fold_details = []
        for fold in range(args.folds):
            train_mask = valid & (folds != fold)
            held = valid & (folds == fold)
            best = None
            for threshold in thresholds:
                proposal = persistence.copy()
                use = (confidence >= threshold) & (model_pred != persistence)
                proposal[use] = model_pred[use]
                score = side_balanced_score(y[train_mask], proposal[train_mask], sides[train_mask])
                # Keep higher threshold on exact ties to limit unnecessary overrides.
                key = (score, float(threshold))
                if best is None or key > best:
                    best = key
            threshold = best[1]
            override = (confidence >= threshold) & (model_pred != persistence)
            gate_pred[held] = persistence[held]
            gate_pred[held & override] = model_pred[held & override]
            fold_details.append({"fold": fold, "train_games": len(set(gids[train_mask])),
                                 "heldout_games": len(set(gids[held])),
                                 "selected_threshold": threshold,
                                 "train_side_balanced_accuracy": best[0],
                                 "heldout_override_rate": float(np.mean(override[held]))})
        rows[f"{h}s"] = {
            "fold_thresholds": fold_details,
            "validation_oof": {
                "persistence": metrics(y[valid], persistence[valid]),
                "canonical_auto_frozen_alpha": metrics(y[valid], auto_pred[valid]),
                "crossfit_confidence_gate": metrics(y[valid], gate_pred[valid]),
                "by_side": {
                    side: {
                        "persistence": metrics(y[valid & (sides == side)], persistence[valid & (sides == side)]),
                        "canonical_auto_frozen_alpha": metrics(y[valid & (sides == side)], auto_pred[valid & (sides == side)]),
                        "crossfit_confidence_gate": metrics(y[valid & (sides == side)], gate_pred[valid & (sides == side)]),
                    } for side in ("红", "蓝")
                },
                "persistence_side_balanced_accuracy": side_balanced_score(y[valid], persistence[valid], sides[valid]),
                "auto_side_balanced_accuracy": side_balanced_score(y[valid], auto_pred[valid], sides[valid]),
                "gate_side_balanced_accuracy": side_balanced_score(y[valid], gate_pred[valid], sides[valid]),
                "gate_vs_persistence_paired_game_bootstrap": bootstrap_game_delta(y[valid], gate_pred[valid], persistence[valid], sides[valid], gids[valid], args.bootstrap),
                "gate_vs_auto_paired_game_bootstrap": bootstrap_game_delta(y[valid], gate_pred[valid], auto_pred[valid], sides[valid], gids[valid], args.bootstrap, 7302),
                "override_rate": float(np.mean(gate_pred[valid] != persistence[valid])),
            },
            "frozen_checkpoint_alpha": alpha[h],
        }
    report = {
        "schema": "validation-crossfit-mode-confidence-gate-v1",
        "checkpoint": args.checkpoint, "validation_split": args.validation,
        "device": device, "games": len(games), "folds": args.folds,
        "split_access": {"train": False, "validation": True, "test": False},
        "policy": "On each held-out game fold, start from current base_mode persistence. Replace it with frozen GRU argmax only when raw calibrated GRU max probability >= a threshold selected on the other game folds to maximize equal-red-blue accuracy. Threshold grid .34.. .99 step .01.",
        "limitation": "This is validation cross-fitting of a lightweight override rule, not an independent final test. Repeated experimentation on this validation corpus may still cause selection bias.",
        "horizons": rows,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({h: {k: round(v, 5) if isinstance(v, float) else v for k, v in row["validation_oof"].items() if k in ("persistence_side_balanced_accuracy", "auto_side_balanced_accuracy", "gate_side_balanced_accuracy", "override_rate")} for h, row in rows.items()}, ensure_ascii=False))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="out_m2/temporal_forecast_canonical_side/temporal_gru_v1.pt")
    p.add_argument("--validation", default="out_m2/temporal_forecast_canonical_side/validation.npz")
    p.add_argument("--output", default="out_m2/validation_crossfit_mode_gate/report.json")
    p.add_argument("--device", default="auto")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--bootstrap", type=int, default=2000)
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
