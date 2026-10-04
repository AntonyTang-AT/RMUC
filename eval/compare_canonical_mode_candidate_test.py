"""One-shot paired locked-test comparison for a frozen canonical mode model.

Run only after train/validation model and blend choices have been frozen. This
script compares identical labels, calibration, split, and per-checkpoint causal
persistence blends; bootstrap resampling is paired by game and balances the
red/blue observer sides.
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

from eval.train_canonical_mode_candidate import _blend, _probabilities
from forecast.train import load_model


def _pred_metrics(y, p):
    pred = p.argmax(1)
    f1s = []
    for c in range(p.shape[1]):
        tp = np.sum((y == c) & (pred == c))
        fp = np.sum((y != c) & (pred == c))
        fn = np.sum((y == c) & (pred != c))
        den = 2 * tp + fp + fn
        f1s.append(float(2 * tp / den) if den else 0.0)
    brier = np.mean(np.sum((p - np.eye(p.shape[1])[y]) ** 2, axis=1))
    return np.array([np.mean(y == pred), np.mean(f1s), brier], dtype=float)


def _score(data, probs, alphas):
    return _blend(data, probs, alphas)


def _bootstrap(data, candidate_probs, candidate_alphas, baseline_probs, baseline_alphas,
               *, n_boot=3000, seed=20260927):
    rng = np.random.default_rng(seed)
    game = np.asarray(data["game_ids"]).astype(int)
    side = np.asarray(data["sides"])
    ids = np.unique(game)
    samples = {h: [] for h in (10, 30, 60, 120)}
    for _ in range(n_boot):
        draw = rng.choice(ids, len(ids), replace=True)
        # Build a row-index vector that retains both side views of each drawn game.
        ix_parts = []
        for k, gid in enumerate(draw):
            ix_parts.append(np.flatnonzero(game == gid))
        ix = np.concatenate(ix_parts) if ix_parts else np.array([], dtype=int)
        for j, h in enumerate((10, 30, 60, 120)):
            y_all = data["mode_y"][ix, j].astype(int)
            valid = y_all >= 0
            cur = []
            for team in ("红", "蓝"):
                mask = valid & (side[ix] == team)
                if not mask.any():
                    continue
                def blend(probs, alphas):
                    p = probs[h][ix][mask].copy()
                    base = data["base_mode"][ix][mask].astype(int)
                    prior = np.full_like(p, .02 / (p.shape[1] - 1))
                    ok = (base >= 0) & (base < p.shape[1])
                    prior[np.arange(len(base))[ok], base[ok]] = .98
                    return float(alphas[h]) * p + (1 - float(alphas[h])) * prior
                y = y_all[mask]
                cm = _pred_metrics(y, blend(candidate_probs, candidate_alphas))
                bm = _pred_metrics(y, blend(baseline_probs, baseline_alphas))
                cur.append(cm - bm)
            if cur:
                samples[h].append(np.mean(cur, axis=0))
    ci = {}
    for h, values in samples.items():
        a = np.asarray(values)
        ci[f"{h}s"] = {metric: {"low": float(np.quantile(a[:, i], .025)),
                                  "high": float(np.quantile(a[:, i], .975))}
                         for i, metric in enumerate(("accuracy", "macro_f1", "brier"))}
    return ci


def run(args):
    frozen = json.loads(Path(args.validation_report).read_text(encoding="utf-8"))
    candidate_alpha = {int(k[:-1]): float(v) for k, v in frozen["selected_alpha_by_horizon"].items()}
    baseline_ckpt = load_model(args.auto_checkpoint, "cpu")[1]
    baseline_alpha = {h: float(baseline_ckpt["persistence_blend"]["alphas"][f"global_mode@{h}s"])
                      for h in (10, 30, 60, 120)}
    data, cp, device = _probabilities(args.candidate, args.test, args.device, args.batch_size)
    same_data, bp, _ = _probabilities(args.auto_checkpoint, args.test, args.device, args.batch_size)
    for key in ("game_ids", "sides", "mode_y", "base_mode"):
        if not np.array_equal(data[key], same_data[key]):
            raise ValueError(f"candidate and auto test row contract differs: {key}")
    cand = _score(data, cp, candidate_alpha)
    base = _score(data, bp, baseline_alpha)
    delta = {}
    for h in (10, 30, 60, 120):
        key = f"{h}s"
        delta[key] = {}
        for metric in ("accuracy", "macro_f1", "brier"):
            delta[key][metric] = round(float(cand[key]["pooled"][metric] - base[key]["pooled"][metric]), 8)
    report = {
        "schema": "canonical-mode-candidate-locked-test-paired-v1",
        "candidate_checkpoint": str(args.candidate), "auto_checkpoint": str(args.auto_checkpoint),
        "candidate_validation_report": str(args.validation_report),
        "test_split": str(args.test), "test_access": "single final comparison after candidate frozen on train/validation",
        "device": device, "n_games": int(len(np.unique(data["game_ids"]))),
        "candidate_alpha": {f"{h}s": candidate_alpha[h] for h in (10, 30, 60, 120)},
        "auto_alpha": {f"{h}s": baseline_alpha[h] for h in (10, 30, 60, 120)},
        "candidate": cand, "auto": base, "delta_candidate_minus_auto": delta,
        "paired_game_bootstrap_95ci": _bootstrap(data, cp, candidate_alpha, bp, baseline_alpha),
        "comparison_contract": {"same_labels": True, "same_calibration": "calibrate_horizon_probabilities",
            "same_test_rows": True, "side_weighting": "equal red/blue observer-side mean",
            "bootstrap_unit": "game; each sampled game includes both observer sides"},
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": args.output, "candidate": cand, "auto": base,
                      "delta": delta, "ci": report["paired_game_bootstrap_95ci"]}, ensure_ascii=False))
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--candidate", required=True)
    p.add_argument("--auto-checkpoint", required=True)
    p.add_argument("--validation-report", required=True)
    p.add_argument("--test", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="auto")
    p.add_argument("--batch-size", type=int, default=512)
    run(p.parse_args())


if __name__ == "__main__":
    main()
