"""Validation-only score-trend expert for canonical global-mode forecasting.

The only optimized value is a per-horizon shrinkage of an online slope of the
current observable-derived mode score. It is selected by game-group CV within
validation; the test split is not accepted or opened.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
from collections import defaultdict, deque
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.train_canonical_mode_candidate import _probabilities
from forecast.temporal import HORIZONS, MODES, MOBILE_TYPES, MAP_X, _entities, _hp_ratio, load_compact
from forecast.train import load_model


def score_history(jsonl, data):
    n = len(data["game_ids"])
    score = np.full(n, np.nan, dtype=np.float32)
    slope = np.zeros(n, dtype=np.float32)
    recent = defaultdict(lambda: deque(maxlen=5))
    parity = np.zeros(n, dtype=bool)
    cursor = 0
    with gzip.open(jsonl, "rt", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            if cursor >= n:
                raise ValueError("validation JSONL has extra rows")
            key = (int(row["game_id"]), str(row["side"]), float(row["issued_at"]))
            expected = (int(data["game_ids"][cursor]), str(data["sides"][cursor]),
                        float(data["issued_at"][cursor]))
            if key[:2] != expected[:2] or abs(key[2] - expected[2]) > 1e-4:
                raise ValueError(f"validation JSONL compact row mismatch at {cursor}: {key} != {expected}")
            obs = row.get("observable") or {}
            own, opp = _entities(obs, "self_entities"), _entities(obs, "opponent_entities")
            side = row["side"]
            # The frozen checkpoint/data use the pre-death-aware contract.
            # Reconstruct that exact rule from issue-time observable so trend
            # and persistence are compared on the same label semantics.
            own_mobile = [e for e in own if e.get("entity_type") in MOBILE_TYPES]
            opp_mobile = [e for e in opp if e.get("entity_type") in MOBILE_TYPES]
            if side == "蓝":
                own_push = sum(1 for e in own_mobile if (e.get("position") or [MAP_X / 2])[0] < MAP_X / 2)
                opp_push = sum(1 for e in opp_mobile if (e.get("position") or [MAP_X / 2])[0] >= MAP_X / 2)
            else:
                own_push = sum(1 for e in own_mobile if (e.get("position") or [MAP_X / 2])[0] > MAP_X / 2)
                opp_push = sum(1 for e in opp_mobile if (e.get("position") or [MAP_X / 2])[0] <= MAP_X / 2)
            own_base = next((e for e in own if e.get("entity_type") == "基地"), None)
            opp_base = next((e for e in opp if e.get("entity_type") == "基地"), None)
            current = float(own_push - opp_push + 2.0 *
                            ((_hp_ratio(opp_base) if opp_base else 1.0) -
                             (_hp_ratio(own_base) if own_base else 1.0)))
            parity[cursor] = labels_from_score(current) == int(data["base_mode"][cursor])
            group = recent[(key[0], key[1])]
            group.append((key[2], current))
            score[cursor] = current
            # OLS slope over at most the five latest 3-second issue samples.
            # It uses only current/past observable-derived scores.
            if len(group) >= 2:
                points = np.asarray(group, dtype=np.float64)
                t = points[:, 0] - points[-1, 0]
                slope[cursor] = float(np.polyfit(t, points[:, 1], 1)[0])
            cursor += 1
    if cursor != n:
        raise ValueError(f"validation JSONL row count {cursor} != compact {n}")
    return score, slope, parity


def labels_from_score(score):
    # Match _mode_from_groups' fixed score thresholds: defense <= -1,
    # attack >= 1, balanced otherwise. MODES index order is attack/balance/defense.
    pred = np.ones(np.asarray(score).shape, dtype=np.int64)
    pred[np.asarray(score) >= 1.0] = 0
    pred[np.asarray(score) <= -1.0] = 2
    return pred


def confusion_metrics(y, pred):
    cm = np.zeros((3, 3), dtype=np.int64)
    np.add.at(cm, (y, pred), 1)
    per_class = {}
    f1 = []
    for c, name in enumerate(MODES):
        tp = int(cm[c, c]); fp = int(cm[:, c].sum() - tp); fn = int(cm[c, :].sum() - tp)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        value = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1.append(value)
        per_class[name] = {"precision": precision, "recall": recall, "f1": value,
                           "support": int(cm[c, :].sum())}
    return {"n": int(len(y)), "accuracy": float(np.mean(y == pred)),
            "macro_f1": float(np.mean(f1)), "per_class": per_class,
            "confusion_true_rows_pred_cols": cm.tolist()}


def per_game_side(y, pred, valid, gids, sides):
    out = {}
    indices = defaultdict(list)
    for i, (g, s) in enumerate(zip(gids, sides)):
        indices[(int(g), str(s))].append(i)
    for key, values in indices.items():
        idx = np.asarray(values, dtype=np.int64)
        idx = idx[valid[idx]]
        if len(idx):
            out[key] = confusion_metrics(y[idx], pred[idx])
    return out


def balanced_game_score(gs, metric):
    games = sorted(set(g for g, _ in gs if (g, "红") in gs and (g, "蓝") in gs))
    vals = [(gs[(g, "红")][metric] + gs[(g, "蓝")][metric]) / 2 for g in games]
    return float(np.mean(vals)) if vals else None


def paired_game_bootstrap(gs_a, gs_b, metric, reps, seed):
    games = sorted(g for g in set(g for g, _ in gs_a) & set(g for g, _ in gs_b)
                   if all((g, s) in gs_a and (g, s) in gs_b for s in ("红", "蓝")))
    deltas = np.asarray([
        np.mean([gs_a[(g, s)][metric] - gs_b[(g, s)][metric] for s in ("红", "蓝")])
        for g in games], dtype=float)
    if not len(deltas):
        return {"mean_delta": None, "ci95": [None, None], "games": 0}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(deltas), size=(reps, len(deltas)))
    means = deltas[idx].mean(axis=1)
    return {"mean_delta": float(deltas.mean()),
            "ci95": [float(np.quantile(means, .025)), float(np.quantile(means, .975))],
            "games": len(deltas), "bootstrap_replicates": reps}


def run(args):
    data = load_compact(args.validation_npz)
    score, slope, base_mode_parity = score_history(args.validation_jsonl, data)
    model, ckpt = load_model(args.checkpoint, device="cpu")
    data, probs, device = _probabilities(args.checkpoint, args.validation_npz,
                                          args.device, args.batch_size)
    alphas = (ckpt.get("persistence_blend") or {}).get("alphas", {})
    current = np.asarray(data["base_mode"], dtype=np.int64)
    y_all = np.asarray(data["mode_y"], dtype=np.int64)
    gids = np.asarray(data["game_ids"], dtype=np.int64)
    sides = np.asarray(data["sides"])
    games = sorted(set(int(g) for g in gids))
    fold_map = {g: i % args.folds for i, g in enumerate(games)}
    fold = np.asarray([fold_map[int(g)] for g in gids])
    # Frozen deployment auto; copy contract: alpha*calibrated GRU + (1-alpha)*smoothed persistence.
    prior = np.full((len(current), 3), .01, dtype=np.float64)
    legal = (current >= 0) & (current < 3)
    prior[np.arange(len(current))[legal], current[legal]] = .98
    auto_pred = np.zeros((len(current), len(HORIZONS)), dtype=np.int64)
    alpha_by_h = {}
    for j, h in enumerate(HORIZONS):
        alpha = float(alphas.get(f"global_mode@{h}s", 1.0))
        alpha_by_h[f"{h}s"] = alpha
        p = alpha * probs[h] + (1-alpha) * prior
        auto_pred[:, j] = p.argmax(axis=1)

    lambdas = np.round(np.arange(0.0, 2.001, .05), 2)
    horizons_report = {}
    for j, h in enumerate(HORIZONS):
        y = y_all[:, j]
        valid = y >= 0
        persistence = current.copy()
        auto = auto_pred[:, j]
        # Candidate is just a current-score trajectory expert. Shrinkage lambda
        # scales the causal OLS score slope before applying the official class thresholds.
        candidate_pred = np.full(len(y), -1, dtype=np.int64)
        fold_report = []
        for f in range(args.folds):
            train = valid & (fold != f)
            held = valid & (fold == f)
            best = None
            for lam in lambdas:
                pred = labels_from_score(score + slope * float(h) * float(lam))
                per = per_game_side(y, pred, train, gids, sides)
                objective = balanced_game_score(per, "accuracy")
                # Smallest correction wins exact ties.
                if best is None or objective > best[0] + 1e-12:
                    best = (objective, float(lam), pred)
            candidate_pred[held] = best[2][held]
            fold_report.append({"fold": f, "heldout_games": len(set(gids[held])),
                                "selected_slope_shrinkage": best[1],
                                "train_equal_red_blue_game_accuracy": best[0]})
        # Error decomposition on the crossfit predictions and both references.
        result = {}
        predictions = {"persistence": persistence, "canonical_auto": auto,
                       "crossfit_score_trend": candidate_pred}
        game_scores = {}
        for name, pred in predictions.items():
            gs = per_game_side(y, pred, valid, gids, sides)
            game_scores[name] = gs
            result[name] = {
                "pooled": confusion_metrics(y[valid], pred[valid]),
                "red_view": confusion_metrics(y[valid & (sides == "红")], pred[valid & (sides == "红")]),
                "blue_view": confusion_metrics(y[valid & (sides == "蓝")], pred[valid & (sides == "蓝")]),
                "paired_game_equal_red_blue_accuracy": balanced_game_score(gs, "accuracy"),
                "paired_game_equal_red_blue_macro_f1": balanced_game_score(gs, "macro_f1"),
            }
        deltas = {}
        for ref in ("persistence", "canonical_auto"):
            deltas[ref] = {
                metric: paired_game_bootstrap(game_scores["crossfit_score_trend"], game_scores[ref],
                                              metric, args.bootstrap, 7771 + j + (0 if ref == "persistence" else 30))
                for metric in ("accuracy", "macro_f1")
            }
        current_transitions = (current != y) & valid
        truth_counts = np.bincount(y[valid], minlength=3)
        # Difference groups by current->future transition to distinguish
        # persistent accuracy from actual transitions.
        transition = {f"{MODES[a]}→{MODES[b]}": int(np.sum(valid & (current == a) & (y == b)))
                      for a in range(3) for b in range(3) if np.any(valid & (current == a) & (y == b))}
        horizons_report[f"{h}s"] = {
            "n": int(valid.sum()), "alpha_canonical_auto": alpha_by_h[f"{h}s"],
            "label_distribution": {MODES[c]: int(truth_counts[c]) for c in range(3)},
            "persistence_transition_rate": float(np.mean(current_transitions[valid])),
            "current_to_future_transition_counts": transition,
            "models": result, "crossfit_deltas": deltas,
            "folds": fold_report,
        }
    report = {
        "schema": "validation-global-mode-score-trend-crossfit-v1",
        "checkpoint": args.checkpoint, "validation_npz": args.validation_npz,
        "validation_jsonl": args.validation_jsonl, "device": device,
        "sha256": {"validation_npz": hashlib.sha256(Path(args.validation_npz).read_bytes()).hexdigest(),
                    "validation_jsonl": hashlib.sha256(Path(args.validation_jsonl).read_bytes()).hexdigest(),
                    "checkpoint": hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest()},
        "split_access": {"train": False, "validation": True, "test": False},
        "current_score": "pre-death-aware relative_margin_v2 score reconstructed from issue-time observable only; exact parity with frozen compact base_mode is reported",
        "validation_input_consistency": {
            "raw_observable_mode_vs_compact_base_mode_agreement": float(np.mean(base_mode_parity)),
            "mismatch_rows": int(np.sum(~base_mode_parity)),
            "rows": int(len(base_mode_parity)),
            "warning": "The current worktree's death-aware mode differs on some rows; this experiment intentionally reconstructs the checkpoint/archive pre-death-aware rule to maintain a comparable persistence reference. See the label-contract audit before promotion.",
        },
        "trend": "OLS slope/sec over at most the five latest issue-time current-mode scores under frozen pre-death-aware contract, within the same game and observing side; current plus lambda*horizon*slope; no target/future fields used to form features",
        "selection": "Per horizon and heldout game fold, select lambda in [0,2] step .05 on other validation games to maximize equal-game, equal-red/blue accuracy. 5 folds. OOF candidate is compared with frozen canonical auto and persistence.",
        "caveat": "This experiment is a validation-only development estimate. The hand-designed score expert and shrinkage grid were chosen after inspecting the available validation reports; independent new-match validation is required before promotion.",
        "horizons": horizons_report,
    }
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    brief = {h: {name: {"accuracy": row["paired_game_equal_red_blue_accuracy"],
                        "macro_f1": row["paired_game_equal_red_blue_macro_f1"]}
                 for name, row in v["models"].items()} for h, v in horizons_report.items()}
    print(json.dumps({"output": str(out), "metrics": brief,
                      "lambdas": {h: [f["selected_slope_shrinkage"] for f in v["folds"]]
                                  for h, v in horizons_report.items()}}, ensure_ascii=False))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="out_m2/temporal_forecast_canonical_side/temporal_gru_v1.pt")
    p.add_argument("--validation-npz", default="out_m2/temporal_forecast_canonical_side/validation.npz")
    p.add_argument("--validation-jsonl", default="out_m2/forecast_dataset/validation.jsonl.gz")
    p.add_argument("--output", default="out_m2/validation_global_mode_score_trend/report.json")
    p.add_argument("--device", default="auto")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--bootstrap", type=int, default=2000)
    args = p.parse_args()
    if args.device == "auto":
        import torch
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    run(args)


if __name__ == "__main__":
    main()
