"""Validation-only audit for the live/death-aware global-mode contract drift.

Reads only the canonical validation NPZ and its aligned validation JSONL. It
reconstructs issue-time and exact future-time modes under the compact archive's
legacy alive/dead treatment and current death-aware treatment, then scores the
same frozen canonical model under both label contracts. No training or test
split paths are accepted.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.crossfit_global_mode_score_trend import (
    balanced_game_score, confusion_metrics, paired_game_bootstrap, per_game_side,
)
from eval.train_canonical_mode_candidate import _probabilities
from forecast.temporal import HORIZONS, MODES, _entities, _mode_from_groups, _mode_score, _target, load_compact
from forecast.train import load_model


def old_mode(own, opp, side):
    """Reproduce the pre-death-aware _mobile contract in committed HEAD."""
    from forecast.temporal import MOBILE_TYPES, MAP_X, _hp_ratio
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
    score = own_push - opp_push + 2 * ((_hp_ratio(opp_base) if opp_base else 1.0)
                                       - (_hp_ratio(own_base) if own_base else 1.0))
    return "防守" if score <= -1 else "进攻" if score >= 1 else "均衡"


def index_mode(label):
    return MODES.index(label)


def run(args):
    data = load_compact(args.validation_npz)
    model_data, probs, device = _probabilities(args.checkpoint, args.validation_npz,
                                                args.device, args.batch_size)
    if not np.array_equal(data["game_ids"], model_data["game_ids"]):
        raise ValueError("validation archive changed while loading")
    n = len(data["game_ids"])
    old_current = np.asarray(data["base_mode"], dtype=np.int64)
    new_current = np.full(n, -1, dtype=np.int64)
    old_future = np.full_like(np.asarray(data["mode_y"], dtype=np.int64), -1)
    new_future = np.full_like(old_future, -1)
    current_old_match = np.zeros(n, dtype=bool)
    future_old_match = np.zeros_like(old_future, dtype=bool)
    cursor = 0
    with gzip.open(args.validation_jsonl, "rt", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            key = (int(row["game_id"]), str(row["side"]), float(row["issued_at"]))
            expected = (int(data["game_ids"][cursor]), str(data["sides"][cursor]), float(data["issued_at"][cursor]))
            if key[:2] != expected[:2] or abs(key[2] - expected[2]) > 1e-4:
                raise ValueError(f"validation JSONL compact row mismatch at {cursor}")
            obs = row.get("observable") or {}
            own = _entities(obs, "self_entities")
            opp = _entities(obs, "opponent_entities")
            old_label = old_mode(own, opp, row["side"])
            new_label = _mode_from_groups(own, opp, row["side"])
            current_old_match[cursor] = index_mode(old_label) == old_current[cursor]
            new_current[cursor] = index_mode(new_label)
            for j, h in enumerate(HORIZONS):
                truth = _target(row, h)
                if truth is None:
                    continue
                entities = list((truth.get("entities") or {}).values())
                side = row["side"]
                future_own = [e for e in entities if e.get("team") == side]
                future_opp = [e for e in entities if e.get("team") != side]
                old_future[cursor, j] = index_mode(old_mode(future_own, future_opp, side))
                new_future[cursor, j] = index_mode(_mode_from_groups(future_own, future_opp, side))
                future_old_match[cursor, j] = old_future[cursor, j] == data["mode_y"][cursor, j]
            cursor += 1
    if cursor != n:
        raise ValueError(f"JSONL rows={cursor}, compact={n}")

    alphas = (load_model(args.checkpoint, device="cpu")[1].get("persistence_blend") or {}).get("alphas", {})
    prior_old = np.full((n, 3), .01, dtype=np.float64)
    prior_new = np.full((n, 3), .01, dtype=np.float64)
    valid_old_idx = (old_current >= 0) & (old_current < 3)
    valid_new_idx = (new_current >= 0) & (new_current < 3)
    prior_old[np.arange(n)[valid_old_idx], old_current[valid_old_idx]] = .98
    prior_new[np.arange(n)[valid_new_idx], new_current[valid_new_idx]] = .98
    h_report = {}
    gids = np.asarray(data["game_ids"], dtype=np.int64)
    sides = np.asarray(data["sides"])
    for j, h in enumerate(HORIZONS):
        y_old = np.asarray(data["mode_y"][:, j], dtype=np.int64)
        valid = (y_old >= 0) & (new_future[:, j] >= 0)
        y_new = new_future[:, j]
        alpha = float(alphas.get(f"global_mode@{h}s", 1.0))
        raw_pred = probs[h].argmax(axis=1)
        pred_auto_old_prior = (alpha * probs[h] + (1-alpha) * prior_old).argmax(axis=1)
        pred_auto_new_prior = (alpha * probs[h] + (1-alpha) * prior_new).argmax(axis=1)
        result = {}
        for contract, y in (("legacy", y_old), ("death_aware", y_new)):
            preds = {"frozen_persistence_prior": old_current if contract == "legacy" else new_current,
                     "model_raw_argmax": raw_pred,
                     "frozen_auto_legacy_prior": pred_auto_old_prior,
                     "auto_recomputed_death_aware_prior": pred_auto_new_prior}
            game_scores = {}
            metrics = {}
            for name, pred in preds.items():
                gs = per_game_side(y, pred, valid, gids, sides)
                game_scores[name] = gs
                metrics[name] = {"accuracy": balanced_game_score(gs, "accuracy"),
                                 "macro_f1": balanced_game_score(gs, "macro_f1"),
                                 "pooled": confusion_metrics(y[valid], pred[valid]),
                                 "red_view": confusion_metrics(y[valid & (sides == "红")], pred[valid & (sides == "红")]),
                                 "blue_view": confusion_metrics(y[valid & (sides == "蓝")], pred[valid & (sides == "蓝")])}
            deltas = {}
            for name in preds:
                if name == "frozen_persistence_prior":
                    continue
                deltas[name] = {m: paired_game_bootstrap(game_scores[name],
                                                           game_scores["frozen_persistence_prior"], m,
                                                           args.bootstrap, 812 + j + (0 if m == "accuracy" else 100))
                                for m in ("accuracy", "macro_f1")}
            result[contract] = {"models": metrics, "paired_delta_vs_persistence": deltas,
                                "target_distribution": {MODES[c]: int(np.sum(valid & (y == c))) for c in range(3)}}
        h_report[f"{h}s"] = {"n": int(valid.sum()), "frozen_auto_alpha": alpha,
                             "legacy_target_old_rule_match_rate": float(future_old_match[valid, j].mean()),
                             "legacy_vs_death_aware_target_disagreement": float(np.mean(y_old[valid] != y_new[valid])),
                             "current_to_future_disagreement_rate": {
                                 "legacy_contract": float(np.mean(old_current[valid] != y_old[valid])),
                                 "death_aware_contract": float(np.mean(new_current[valid] != y_new[valid])),
                             },
                             "legacy_to_death_aware_target_transition_counts": {
                                 f"{MODES[a]}→{MODES[b]}": int(np.sum(valid & (y_old == a) & (y_new == b)))
                                 for a in range(3) for b in range(3) if np.any(valid & (y_old == a) & (y_new == b))},
                             "contracts": result}

    report = {
        "schema": "validation-mode-label-contract-audit-v1",
        "split_access": {"validation": True, "train": False, "test": False},
        "validation_npz": args.validation_npz,
        "validation_jsonl": args.validation_jsonl,
        "checkpoint": args.checkpoint,
        "device": device,
        "sha256": {"validation_npz": hashlib.sha256(Path(args.validation_npz).read_bytes()).hexdigest(),
                    "validation_jsonl": hashlib.sha256(Path(args.validation_jsonl).read_bytes()).hexdigest(),
                    "checkpoint": hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest()},
        "contract_trace": {
            "compact_base_mode_matches_pre_death_aware_rule": float(current_old_match.mean()),
            "compact_base_mode_mismatches_pre_death_aware_rule": int(np.sum(~current_old_match)),
            "compact_base_mode_matches_worktree_death_aware_rule": float(np.mean(new_current == old_current)),
            "interpretation": "The compact validation labels reproduce the committed pre-death-aware rule. Worktree _mobile now excludes hp<=0 and life_state=destroyed; that implementation change alters issue and target mode labels without regenerating the archived validation labels/checkpoint. This is a label-contract/version drift, not row alignment failure.",
        },
        "horizons": h_report,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(out), "parity_old_current": report["contract_trace"]["compact_base_mode_matches_pre_death_aware_rule"],
                      "horizons": {h: {c: {m: round(v["accuracy"], 4) for m, v in vals["models"].items()}
                                        for c, vals in row["contracts"].items()} for h, row in h_report.items()}}, ensure_ascii=False))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="out_m2/temporal_forecast_canonical_side/temporal_gru_v1.pt")
    p.add_argument("--validation-npz", default="out_m2/temporal_forecast_canonical_side/validation.npz")
    p.add_argument("--validation-jsonl", default="out_m2/forecast_dataset/validation.jsonl.gz")
    p.add_argument("--output", default="out_m2/validation_mode_label_contract/report.json")
    p.add_argument("--device", default="auto")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--bootstrap", type=int, default=2000)
    args = p.parse_args()
    if args.device == "auto":
        import torch
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    run(args)


if __name__ == "__main__":
    main()
