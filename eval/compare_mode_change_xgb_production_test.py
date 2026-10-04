"""One-shot, no-tuning paired comparison of the frozen 30 s XGB vs deployment.

Uses the existing locked test split only to compare two already-frozen models.
No thresholds, weights, or model parameters are selected here.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score

from eval.train_mode_change_xgb import load_split
from forecast.calibration import calibrate_horizon_probabilities
from forecast.mode_change_xgb import ModeChangeXGBPredictor
from forecast.temporal import HORIZONS, MODES
from forecast.train import load_compact, load_model


ROOT = Path(__file__).resolve().parents[1]
DEPLOYED = ROOT / "out_m2/temporal_hybrid_motion_death_aware/temporal_hybrid_deployment_fused_v2.pt"
TEST_NPZ = ROOT / "out_m2/temporal_hybrid_motion_death_aware/test.npz"
XGB_DATA = ROOT / "out_m2/temporal_hybrid_mode_stats_ablation"
XGB_MODEL = ROOT / "out_m2/mode_change_xgb_probe/report.model.ubj"
OUT = ROOT / "out_m2/mode_change_xgb_production_comparison/locked_test_comparison.json"


def _metrics(y, pred, side):
    result = {}
    for label in ("红", "蓝"):
        mask = (side == label) & (y >= 0)
        result[label] = {
            "accuracy": float(accuracy_score(y[mask], pred[mask])),
            "macro_f1": float(f1_score(y[mask], pred[mask], labels=np.arange(3),
                                       average="macro", zero_division=0)),
            "n": int(mask.sum()),
        }
    result["equal_side_accuracy"] = float(np.mean([result[s]["accuracy"] for s in ("红", "蓝")]))
    result["equal_side_macro_f1"] = float(np.mean([result[s]["macro_f1"] for s in ("红", "蓝")]))
    return result


def _game_rows(y, candidate, deployed, side, games):
    rows = {}
    for game in np.unique(games):
        rows[int(game)] = {}
        for team in ("红", "蓝"):
            mask = (games == game) & (side == team) & (y >= 0)
            rows[int(game)][team] = (
                int(np.sum(candidate[mask] == y[mask])),
                int(np.sum(deployed[mask] == y[mask])),
                int(mask.sum()),
            )
    return rows


def _paired_game_bootstrap(rows, reps=5000, seed=20260927):
    ids = np.asarray(list(rows), dtype=np.int64)
    rng = np.random.default_rng(seed)
    deltas, candidate_acc = [], []
    for _ in range(reps):
        draw = rng.choice(ids, len(ids), replace=True)
        side_delta, side_acc = [], []
        for team in ("红", "蓝"):
            c = sum(rows[int(game)][team][0] for game in draw)
            d = sum(rows[int(game)][team][1] for game in draw)
            n = sum(rows[int(game)][team][2] for game in draw)
            if n:
                side_delta.append((c - d) / n)
                side_acc.append(c / n)
        deltas.append(float(np.mean(side_delta)))
        candidate_acc.append(float(np.mean(side_acc)))
    return {
        "unit_of_resampling": "game_id; red and blue views kept paired",
        "replicates": reps,
        "candidate_minus_deployed_accuracy_pp_95ci":
            (100 * np.quantile(deltas, [.025, .975])).tolist(),
        "candidate_accuracy_95ci": np.quantile(candidate_acc, [.025, .975]).tolist(),
    }


def run():
    candidate_x, candidate_y, current, candidate_side, candidate_games = load_split(XGB_DATA, "test")
    data = load_compact(TEST_NPZ)
    for key in ("game_ids", "sides", "issued_at", "mode_y", "base_mode"):
        expected = {"mode_y": candidate_y, "base_mode": current,
                    "game_ids": candidate_games, "sides": candidate_side}.get(key)
        actual = data[key][:, HORIZONS.index(30)] if key == "mode_y" else data[key]
        if expected is not None and not np.array_equal(actual, expected):
            raise ValueError(f"locked test row mismatch in {key}")
    # The XGB sidecar stores a float32 issue time; enforce exact key parity
    # without relying on NPZ row ordering alone.
    with np.load(XGB_DATA / "test.npz", allow_pickle=False) as xgb_npz:
        if not np.array_equal(data["issued_at"], xgb_npz["issued_at"]):
            raise ValueError("locked test issue times differ")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_model(DEPLOYED, device=device)
    model.eval()
    baseline_probs = np.zeros((len(candidate_y), len(MODES)), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(candidate_y), 256):
            end = min(len(candidate_y), start + 256)
            x = torch.from_numpy(data["x"][start:end]).to(device)
            outputs = model(x)
            for local in range(end - start):
                by_h = {}
                for horizon in HORIZONS:
                    by_h[horizon] = torch.softmax(
                        outputs[horizon]["global_mode"][local], dim=-1).cpu().numpy()
                calibrated = calibrate_horizon_probabilities(by_h)[30]
                baseline_probs[start + local] = calibrated

    blend = checkpoint.get("persistence_blend") or {}
    alpha = float((blend.get("alphas") or {}).get("global_mode@30s", 1.0))
    deployed_probs = baseline_probs.copy()
    for row in np.flatnonzero(candidate_y >= 0):
        prior = np.full(3, .02 / 2, dtype=np.float32)
        prior[int(current[row])] = .98
        deployed_probs[row] = alpha * baseline_probs[row] + (1 - alpha) * prior
    deployed_pred = deployed_probs.argmax(axis=1)

    xgb = ModeChangeXGBPredictor(XGB_MODEL, persistence_alpha=.95)
    candidate_probs = xgb.predict_proba(candidate_x, current)
    candidate_pred = candidate_probs.argmax(axis=1)
    valid = candidate_y >= 0
    report = {
        "purpose": "frozen_model_comparison_only; no hyperparameter or threshold tuning",
        "test_split_previously_used_for_candidate_report": True,
        "candidate_alpha_frozen": .95,
        "deployed_alpha_from_checkpoint": alpha,
        "n_games": int(len(np.unique(candidate_games[valid]))),
        "candidate": _metrics(candidate_y, candidate_pred, candidate_side),
        "deployed": _metrics(candidate_y, deployed_pred, candidate_side),
        "paired_game_bootstrap": _paired_game_bootstrap(
            _game_rows(candidate_y, candidate_pred, deployed_pred,
                       candidate_side, candidate_games)),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run()
