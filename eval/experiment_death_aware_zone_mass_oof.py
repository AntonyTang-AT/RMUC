"""Train-only GroupKFold OOF test of visible-state aggregate features.

Only the death-aware training compact NPZ and its aligned training JSONL are
opened. New features summarize each issued-at observable snapshot: all visible
mobile robots (not one representative per class), per-class alive counts,
canonical-zone occupancy, team-side health condition, and self-side resources.
Targets are read only from the compact NPZ for training/scoring; the feature
extractor never accesses JSONL targets. Validation/test inputs are not accepted.
"""
from __future__ import annotations

import argparse
import gzip
import json
import time
from pathlib import Path
import sys

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from forecast.temporal import (HORIZONS, MODES, MOBILE_TYPES, _entities, _mobile,
                               _mode_from_groups, _mode_score, _hp_ratio, load_compact)
from eval.experiment_death_aware_mode_lightgbm import features as direct_features
from worldstate.canonical_zone import CANONICAL_ZONE_LABELS, locate_zone, zone_index

TYPE_ORDER = tuple(sorted(MOBILE_TYPES))
ZONE_DIM = len(CANONICAL_ZONE_LABELS)


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and np.isfinite(float(value))


def observable_aggregate(row):
    """Encode only values carried in this issue-time observable snapshot."""
    obs = row.get("observable") or {}
    own_all = _entities(obs, "self_entities")
    opp_all = _entities(obs, "opponent_entities")
    own = _mobile(own_all)
    opp = _mobile(opp_all)
    side = str(row["side"])
    vec = []
    # Aggregate canonical-zone population rather than exposing one scalar zone
    # index per class; only measured positions present at issue time contribute.
    zone_counts = []
    for entities in (own, opp):
        z = np.zeros(ZONE_DIM, dtype=np.float32)
        for e in entities:
            p = e.get("position")
            if isinstance(p, (list, tuple)) and len(p) >= 2 and _finite(p[0]) and _finite(p[1]):
                zi = zone_index(locate_zone(float(p[0]), float(p[1])))
                if zi >= 0:
                    z[zi] += 1.0
        zone_counts.append(z / 6.0)
    vec.extend(zone_counts[0]); vec.extend(zone_counts[1])

    # Per-class alive counts preserve duplicates if future rules/data contain
    # more than one unit of an archetype. They are based on current observables.
    for entities in (own, opp):
        for typ in TYPE_ORDER:
            vec.append(min(1.0, sum(e.get("entity_type") == typ for e in entities)))

    # Health features use own reported max-HP, also as the same-class reference
    # for radar-reported enemy HP. Unknown HP contributes to explicit coverage,
    # never to a guessed health value.
    own_caps = {}
    for e in own_all:
        typ, mhp = e.get("entity_type"), e.get("maxhp")
        if typ in MOBILE_TYPES and _finite(mhp) and float(mhp) > 0:
            own_caps.setdefault(typ, float(mhp))
    hp_ratios = []
    hp_coverage = []
    low_hp_fraction = []
    for entities, is_own in ((own, True), (opp, False)):
        ratios = []
        known = 0
        for e in entities:
            hp = e.get("hp")
            denom = e.get("maxhp") if is_own else own_caps.get(e.get("entity_type"))
            if _finite(hp) and _finite(denom) and float(denom) > 0:
                ratios.append(float(np.clip(float(hp) / float(denom), 0, 1)))
                known += 1
        hp_ratios.append(ratios)
        hp_coverage.append(known / max(1, len(entities)))
        low_hp_fraction.append(sum(r < .3 for r in ratios) / max(1, len(ratios)))
    for ratios in hp_ratios:
        vec.extend([float(np.mean(ratios)) if ratios else 0.0,
                    float(np.min(ratios)) if ratios else 0.0])
    vec.extend(hp_coverage)
    vec.extend(low_hp_fraction)

    # Map-relative centroid and dispersion encode formation shape with only
    # current measured positions. Missing tracks have explicit coverage bits.
    for entities in (own, opp):
        pts = []
        for e in entities:
            p = e.get("position")
            if isinstance(p, (list, tuple)) and len(p) >= 2 and _finite(p[0]) and _finite(p[1]):
                pts.append((float(p[0]) / 28.0, float(p[1]) / 15.0))
        if pts:
            arr = np.asarray(pts, dtype=np.float32)
            vec.extend([float(arr[:, 0].mean()), float(arr[:, 1].mean()),
                        float(arr[:, 0].std()), float(arr[:, 1].std()), len(pts) / 6.0])
        else:
            vec.extend([0.0] * 5)

    # Resource telemetry is taken only from the observer's own sensor-side
    # entities. Enemy resource fields are intentionally excluded.
    coins_total = [float(e["team_coins_total"]) for e in own_all if _finite(e.get("team_coins_total"))]
    coins_left = [float(e["team_coins_left"]) for e in own_all if _finite(e.get("team_coins_left"))]
    vec.extend([float(np.mean(coins_total)) / 500.0 if coins_total else 0.0,
                float(np.mean(coins_left)) / 500.0 if coins_left else 0.0,
                float(bool(coins_total)), float(bool(coins_left)),
                1.0 if side == "蓝" else 0.0])
    # The label recipe thresholds an issue-time observable score proxy at
    # +/-1. Expose the observable-derived value and decomposition explicitly;
    # historical evolution is added causally in extract_train_aggregates.
    own_base = next((e for e in own_all if e.get("entity_type") == "基地"), None)
    opp_base = next((e for e in opp_all if e.get("entity_type") == "基地"), None)
    score = _mode_score(own_all, opp_all, side)
    # Keep the side-relative decomposition aligned with forecast._mode_score.
    own_push = sum(1 for e in own if ((e.get("position") or [14.])[0] < 14. if side == "蓝" else (e.get("position") or [14.])[0] > 14.))
    opp_push = sum(1 for e in opp if ((e.get("position") or [14.])[0] >= 14. if side == "蓝" else (e.get("position") or [14.])[0] <= 14.))
    vec.extend([own_push / 6.0, opp_push / 6.0,
                _hp_ratio(own_base) if own_base is not None else 1.0,
                _hp_ratio(opp_base) if opp_base is not None else 1.0,
                score / 8.0])
    return np.asarray(vec, dtype=np.float32)


def extract_train_aggregates(npz_path, jsonl_path, expected_split="train"):
    data = load_compact(npz_path)
    gids = np.asarray(data["game_ids"], dtype=np.int64)
    sides = np.asarray(data["sides"]).astype(str)
    issued = np.asarray(data["issued_at"], dtype=np.float32)
    base = np.asarray(data["base_mode"], dtype=np.int64)
    base_dim = ZONE_DIM * 2 + len(TYPE_ORDER) * 2 + 4 + 4 + 10 + 10
    x = np.zeros((len(gids), base_dim + 4), np.float32)
    score_history = {}
    duplicate_types = {"own": 0, "opponent": 0}
    max_mobile_count = {"own": 0, "opponent": 0}
    rows = 0
    with gzip.open(jsonl_path, "rt", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("split") != expected_split:
                raise ValueError(f"expected {expected_split} row at JSONL index {rows}: split={row.get('split')!r}")
            if rows >= len(gids):
                raise ValueError("train JSONL has more rows than train NPZ")
            actual = (int(row["game_id"]), str(row["side"]), round(float(row["issued_at"]), 5))
            expected = (int(gids[rows]), str(sides[rows]), round(float(issued[rows]), 5))
            if actual != expected:
                raise ValueError(f"train JSONL/NPZ order mismatch at {rows}: {actual} != {expected}")
            obs = row.get("observable") or {}
            for role, key in (("own", "self_entities"), ("opponent", "opponent_entities")):
                entities = _mobile(_entities(obs, key))
                max_mobile_count[role] = max(max_mobile_count[role], len(entities))
                counts = {}
                for e in entities:
                    counts[e.get("entity_type")] = counts.get(e.get("entity_type"), 0) + 1
                duplicate_types[role] += int(any(n > 1 for n in counts.values()))
            if rows % 10000 == 0:
                print(json.dumps({"parsed_train_rows": rows}, ensure_ascii=False), flush=True)
            vec = observable_aggregate(row)
            key = (int(row["game_id"]), str(row["side"]))
            hist = score_history.setdefault(key, [])
            score = float(vec[-1])
            prior3 = hist[-3:]
            prior6 = hist[-6:]
            trailing = prior6 + [score]
            mean3 = float(np.mean(prior3[-2:] + [score])) if prior3 else score
            mean6 = float(np.mean(trailing))
            d1 = score - hist[-1] if hist else 0.0
            d3 = score - hist[-3] if len(hist) >= 3 else 0.0
            hist.append(score)
            vec = np.concatenate([vec, np.asarray([mean3, mean6, d1, d3], np.float32)])
            x[rows] = vec
            current = _mode_from_groups(_entities(obs, "self_entities"),
                                        _entities(obs, "opponent_entities"), str(row["side"]))
            if MODES.index(current) != base[rows]:
                raise ValueError(f"current mode mismatch against death-aware NPZ at row {rows}")
            rows += 1
    if rows != len(gids):
        raise ValueError(f"train JSONL rows={rows} != train NPZ rows={len(gids)}")
    return data, x, {"rows_aligned": rows, "current_mode_parity": "exact", "split":expected_split,
                    "duplicate_mobile_archetype_row_count": duplicate_types,
                    "max_observed_mobile_count": max_mobile_count,
                    "added_feature_dim": int(x.shape[1]),
                    "added_feature_names": [
                        *(f"self_zone_mass:{z}" for z in CANONICAL_ZONE_LABELS),
                        *(f"opponent_zone_mass:{z}" for z in CANONICAL_ZONE_LABELS),
                        *(f"self_alive_count:{t}" for t in TYPE_ORDER),
                        *(f"opponent_visible_alive_count:{t}" for t in TYPE_ORDER),
                        "self_mean_hp_ratio", "self_min_hp_ratio", "opponent_mean_hp_ratio",
                        "opponent_min_hp_ratio", "self_hp_known_coverage", "opponent_hp_known_coverage",
                        "self_low_hp_fraction_lt_0.3", "opponent_low_hp_fraction_lt_0.3",
                        "self_centroid_x", "self_centroid_y", "self_position_spread_x", "self_position_spread_y", "self_position_coverage",
                        "opponent_centroid_x", "opponent_centroid_y", "opponent_position_spread_x", "opponent_position_spread_y", "opponent_position_coverage",
                        "self_team_coins_total_div500", "self_team_coins_left_div500", "self_coins_total_known", "self_coins_left_known", "side_is_blue",
                        "observable_mode_score_proxy_div8", "observable_own_push_count_div6", "observable_opponent_push_count_div6",
                        "observable_self_base_hp_ratio", "observable_opponent_base_hp_ratio_proxy",
                        "observable_score_mean_trailing3_observations", "observable_score_mean_trailing7_observations", "observable_score_delta1", "observable_score_delta3"]}


def metrics(y, p):
    pred = p.argmax(1)
    cm = np.zeros((3, 3), dtype=np.int64)
    np.add.at(cm, (y, pred), 1)
    per = {}
    f1s = []
    for c, name in enumerate(MODES):
        tp = int(cm[c, c]); fp = int(cm[:, c].sum() - tp); fn = int(cm[c, :].sum() - tp)
        r = tp / int(cm[c].sum()) if cm[c].sum() else 0.0
        f = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
        f1s.append(f)
        per[name] = {"support": int(cm[c].sum()), "recall": r, "f1": f}
    return {"n": int(len(y)), "accuracy": float(np.mean(pred == y)),
            "macro_f1": float(np.mean(f1s)), "per_class": per,
            "confusion_true_rows_pred_cols": cm.tolist()}


def game_scores(y, p, gids, sides, valid):
    out = {}
    for gid in np.unique(gids[valid]):
        side_results = []
        for side in ("红", "蓝"):
            m = valid & (gids == gid) & (sides == side)
            if m.any(): side_results.append(metrics(y[m], p[m]))
        if len(side_results) == 2:
            row = {k: float(np.mean([s[k] for s in side_results])) for k in ("accuracy", "macro_f1")}
            for mode in MODES:
                row[f"recall_{mode}"] = float(np.mean([s["per_class"][mode]["recall"] for s in side_results]))
            out[int(gid)] = row
    return out


def bootstrap_delta(cand, base, seed=20260927, reps=2000):
    ids = sorted(set(cand) & set(base)); rng = np.random.default_rng(seed)
    ix = rng.integers(0, len(ids), size=(reps, len(ids)))
    out = {"paired_games": len(ids), "replicates": reps, "delta_candidate_minus_direct_lgbm": {}}
    keys = ["accuracy", "macro_f1"] + [f"recall_{m}" for m in MODES]
    for key in keys:
        delta = np.asarray([cand[g][key] - base[g][key] for g in ids]); b = delta[ix].mean(1)
        out["delta_candidate_minus_direct_lgbm"][key] = {"mean": float(delta.mean()),
            "ci95": [float(np.quantile(b, .025)), float(np.quantile(b, .975))]}
    return out


def run(train_npz, train_jsonl, baseline_oof, output, *, folds=5, threads=8, seed=1927, reps=2000):
    started = time.perf_counter()
    d, extra, audit = extract_train_aggregates(train_npz, train_jsonl)
    prior = np.load(baseline_oof, allow_pickle=False)
    for name in ("game_ids", "sides", "issued_at", "mode_y", "base_mode"):
        if not np.array_equal(np.asarray(d[name]), np.asarray(prior[name])):
            raise ValueError(f"baseline OOF/training NPZ {name} mismatch")
    gids = np.asarray(d["game_ids"], dtype=np.int64)
    sides = np.asarray(d["sides"]).astype(str)
    yall = np.asarray(d["mode_y"], dtype=np.int64)
    baseall = np.asarray(d["base_mode"], dtype=np.int64)
    X = np.concatenate([direct_features(d), extra], axis=1)
    baseline = np.asarray(prior["lightgbm"], dtype=np.float64)
    candidate = np.full_like(baseline, np.nan)
    folds_per_row = np.full(len(gids), -1, np.int8)
    fold_info = []
    cv = GroupKFold(n_splits=folds)
    for fold, (tr, held) in enumerate(cv.split(X, groups=gids)):
        folds_per_row[held] = fold
        hor = {}
        for j, h in enumerate(HORIZONS):
            train_ok = (yall[tr, j] >= 0) & (baseall[tr] >= 0) & (baseall[tr] < 3)
            held_ok = (yall[held, j] >= 0) & (baseall[held] >= 0) & (baseall[held] < 3)
            trr, hrr = tr[train_ok], held[held_ok]
            if len(np.unique(yall[trr, j])) != 3:
                raise ValueError(f"fold {fold} horizon {h}: train fold lacks a class")
            model = LGBMClassifier(objective="multiclass", num_class=3, n_estimators=260,
                learning_rate=.045, num_leaves=15, max_depth=-1, min_child_samples=90,
                reg_lambda=4., colsample_bytree=.8, subsample=.85, subsample_freq=1,
                class_weight=None, verbosity=-1, n_jobs=threads, random_state=seed + fold)
            model.fit(X[trr], yall[trr, j])
            ph = np.zeros((len(hrr), 3), dtype=np.float32)
            for col, cls in enumerate(model.classes_): ph[:, int(cls)] = model.predict_proba(X[hrr])[:, col]
            candidate[hrr, j] = ph
            hor[f"{h}s"] = {"train_rows": int(len(trr)), "heldout_rows": int(len(hrr))}
        fold_info.append({"fold": fold, "train_games": int(len(np.unique(gids[tr]))),
                          "heldout_games": int(len(np.unique(gids[held]))), "horizons": hor})
        print(json.dumps({"fold": fold, "horizons": hor}, ensure_ascii=False), flush=True)
    results = {}
    for j, h in enumerate(HORIZONS):
        valid = (yall[:, j] >= 0) & np.isfinite(candidate[:, j, 0])
        yy, pp, bp = yall[valid, j], candidate[valid, j], baseline[valid, j]
        prior_p = np.full((valid.sum(), 3), .01, dtype=np.float64)
        bv = baseall[valid]; prior_p[np.arange(len(bv)), bv] = .98
        gs_c = game_scores(yall[:, j], candidate[:, j], gids, sides, valid)
        gs_b = game_scores(yall[:, j], baseline[:, j], gids, sides, valid)
        gs_p = game_scores(yall[:, j], np.where(valid[:, None], np.nan_to_num(prior["persistence"][:, j]), 0.), gids, sides, valid)
        keys = next(iter(gs_c.values())).keys()
        results[f"{h}s"] = {"valid_rows": int(valid.sum()),
            "pooled": {"aggregate_candidate": metrics(yy, pp), "direct_lightgbm_oof": metrics(yy, bp), "persistence": metrics(yy, prior_p)},
            "game_equal_red_blue_equal": {
                "aggregate_candidate": {k: float(np.mean([v[k] for v in gs_c.values()])) for k in keys},
                "direct_lightgbm_oof": {k: float(np.mean([v[k] for v in gs_b.values()])) for k in keys},
                "persistence": {k: float(np.mean([v[k] for v in gs_p.values()])) for k in keys}},
            "paired_game_bootstrap_vs_direct_lgbm": bootstrap_delta(gs_c, gs_b, seed + j, reps)}
    out = Path(output); out.mkdir(parents=True, exist_ok=True)
    report = {"schema": "death-aware-visible-zone-mass-train-group-oof-v1",
        "validation_access": False, "test_access": False, "opened_data_splits": ["train"],
        "train_npz": str(Path(train_npz).resolve()), "train_jsonl": str(Path(train_jsonl).resolve()),
        "baseline_oof": str(Path(baseline_oof).resolve()), "label_contract": str(d["global_mode_label_version"][0]),
        "grouping": "GroupKFold by full game_id; red and blue views remain in the same fold",
        "candidate_features": "direct LightGBM causal snapshot/trailing means/deltas plus all-current-observable alive mobile zone mass per 59 canonical polygons and side, per-archetype alive counts, current HP ratios/coverage/low-HP fraction, formation centroid/spread/coverage, own team funds only, issue-time observable_mode_score_proxy and its same-side causal trailing 3/7-observation means plus 1/3-observation deltas",
        "leakage_contract": "extra features read only JSONL observable; no targets keys, validation/test files, enemy coins, or unobserved statuses",
        "feature_audit": audit, "train_rows": int(len(gids)), "train_games": int(len(np.unique(gids))),
        "folds": fold_info, "horizons": results, "runtime_seconds": time.perf_counter() - started}
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    np.savez_compressed(out / "visible_aggregate_oof.npz", game_ids=gids, sides=sides,
        issued_at=np.asarray(d["issued_at"]), mode_y=yall, base_mode=baseall,
        fold_ids=folds_per_row, aggregate_candidate=candidate.astype(np.float32),
        direct_lightgbm=baseline.astype(np.float32), persistence=np.asarray(prior["persistence"]))
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train-npz", required=True); p.add_argument("--train-jsonl", required=True)
    p.add_argument("--baseline-oof", required=True); p.add_argument("--output", required=True)
    p.add_argument("--folds", type=int, default=5); p.add_argument("--threads", type=int, default=8)
    p.add_argument("--seed", type=int, default=1927); p.add_argument("--bootstrap-reps", type=int, default=2000)
    a = p.parse_args()
    r = run(a.train_npz, a.train_jsonl, a.baseline_oof, a.output, folds=a.folds,
            threads=a.threads, seed=a.seed, reps=a.bootstrap_reps)
    print(json.dumps({h: v["game_equal_red_blue_equal"] for h, v in r["horizons"].items()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
