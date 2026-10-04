"""Validation-only diagnostics and locked selective-risk evaluation for mode@30s.

No fitting/training occurs here.  The trained 329-D hybrid is frozen; all
conditional alpha analysis and the one confidence threshold are selected on
validation.  Test is read once afterward, only to describe that frozen policy.
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from forecast.calibration import calibrate_horizon_probabilities
from forecast.temporal import HORIZONS, MODES, load_compact
from forecast.train import load_model

MODEL = REPO / "out_m2/temporal_hybrid_motion_death_aware/temporal_hybrid_v1.pt"
DATA_DIR = REPO / "out_m2/temporal_hybrid_motion_death_aware"
FUSION_PATH = DATA_DIR / "persistence_blend_tuned.json"
SUMMARY_DIR = REPO / "out_m2/temporal_hybrid_mode_stats_ablation"
OUT_DIR = REPO / "out_m2/global_mode30_validation_diagnostic"
HORIZON_INDEX = HORIZONS.index(30)
ALPHA_GRID = np.linspace(0.0, 1.0, 11)


def _load_summary(split: str, data: dict):
    path = SUMMARY_DIR / f"{split}_observable_summary.jsonl.gz"
    rows = []
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                rows.append(json.loads(line))
    if len(rows) != len(data["game_ids"]):
        raise ValueError(f"{split}: summary length {len(rows)} != compact {len(data['game_ids'])}")
    for i, item in enumerate(rows):
        if (int(item["game_id"]) != int(data["game_ids"][i]) or
                str(item["side"]) != str(data["sides"][i]) or
                not np.isclose(float(item["issued_at"]), float(data["issued_at"][i]), atol=1e-4, rtol=0)):
            raise ValueError(f"{split} summary/compact alignment failed at row {i}")
    return rows


def _predict_probabilities(model_path: Path, compact_path: Path, device: str):
    import torch
    model_device = torch.device("cuda" if device == "auto" and torch.cuda.is_available()
                                else ("cpu" if device == "auto" else device))
    model, ckpt = load_model(model_path, device=model_device)
    data = load_compact(compact_path)
    if int(ckpt["input_dim"]) != int(data["feature_dim"][0]):
        raise ValueError(f"checkpoint/data dimension mismatch: {ckpt['input_dim']} vs {data['feature_dim'][0]}")
    n = len(data["x"])
    raw = np.zeros((n, len(HORIZONS), len(MODES)), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, n, 512):
            stop = min(n, start + 512)
            x = torch.from_numpy(data["x"][start:stop]).to(model_device, non_blocking=True)
            outputs = model(x)
            for h_idx, horizon in enumerate(HORIZONS):
                raw[start:stop, h_idx] = torch.softmax(
                    outputs[horizon]["global_mode"], dim=-1).cpu().numpy()
            if start and start % 8192 == 0:
                print(f"inference {compact_path.name}: {start}/{n}", flush=True)
    calibrated_30 = np.zeros((n, len(MODES)), dtype=np.float32)
    for i in range(n):
        calibrated = calibrate_horizon_probabilities({
            h: raw[i, h_idx] for h_idx, h in enumerate(HORIZONS)})
        calibrated_30[i] = calibrated[30]
    del raw, model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return data, calibrated_30, str(model_device)


def _fuse(model_p, current_mode, alpha):
    n_classes = len(MODES)
    prior = np.full(n_classes, 0.02 / (n_classes - 1), dtype=np.float32)
    if 0 <= int(current_mode) < n_classes:
        prior[int(current_mode)] = 0.98
    else:
        prior.fill(1.0 / n_classes)
    return float(alpha) * model_p + (1.0 - float(alpha)) * prior


def _fused_predictions(probabilities, current_mode, alpha):
    fused = np.vstack([_fuse(probabilities[i], current_mode[i], alpha)
                       for i in range(len(current_mode))])
    return fused, np.argmax(fused, axis=1)


def _balanced_metric(y, pred, sides, mask=None):
    mask = np.ones(len(y), dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    output = {}
    scores = []
    for label, side in (("red_view", "红"), ("blue_view", "蓝")):
        keep = mask & (sides == side) & (y >= 0)
        yt, yp = y[keep], pred[keep]
        cm = np.zeros((len(MODES), len(MODES)), dtype=np.int64)
        if len(yt):
            np.add.at(cm, (yt, yp), 1)
            acc = float(np.mean(yt == yp))
        else:
            acc = None
        f1s = []
        for k in range(len(MODES)):
            tp = int(cm[k, k])
            fp = int(cm[:, k].sum() - tp)
            fn = int(cm[k, :].sum() - tp)
            f1s.append(2.0 * tp / max(1, 2 * tp + fp + fn))
        f1 = float(np.mean(f1s)) if len(yt) else None
        output[label] = {"accuracy": acc, "macro_f1": f1,
                         "n": int(len(yt)), "per_class_f1": f1s}
        if acc is not None:
            scores.append(acc)
    return {"balanced_accuracy": float(np.mean(scores)) if len(scores) == 2 else None,
            "by_side": output}


def _mode_threshold_score(score):
    return min(abs(float(score) + 1.0), abs(float(score) - 1.0))


def _strata(data, summary):
    n = len(summary)
    current = data["base_mode"].astype(np.int64)
    target = data["mode_y"][:, HORIZON_INDEX].astype(np.int64)
    scores = np.asarray([float(r["mode_score"]) for r in summary], dtype=np.float32)
    distance = np.asarray([_mode_threshold_score(x) for x in scores], dtype=np.float32)
    boundary = np.where(distance <= 0.25, "at_or_near_0.25",
               np.where(distance <= 0.75, "near_0.25_to_0.75", "farther_than_0.75"))
    quality_value = np.asarray([
        min(float(r["self_mobile_type_coverage"]), float(r["opponent_mobile_type_coverage"]),
            float(r["self_mobile_position_coverage"]), float(r["opponent_mobile_position_coverage"]))
        for r in summary], dtype=np.float32)
    quality = np.where(quality_value < 0.50, "low_lt_0.50",
              np.where(quality_value < 0.80, "medium_0.50_to_0.80", "high_ge_0.80"))
    flip = np.where((current >= 0) & (target >= 0),
                    np.where(current != target, "true_flip", "no_flip"), "unscored")
    values = {
        "current_mode": np.asarray([MODES[x] if 0 <= x < len(MODES) else "unknown" for x in current]),
        "true_flip": flip,
        "mode_score_boundary_distance": boundary,
        "observation_quality": quality,
        "opponent_base_visible": np.asarray(["visible" if r["opponent_base_present"] else "not_visible" for r in summary]),
        "game_id": data["game_ids"].astype(np.int64),
    }
    return current, target, scores, distance, quality_value, values


def _best_alpha(model_p, current, y, sides, train_mask, eligible=None):
    eligible = np.ones(len(y), dtype=bool) if eligible is None else np.asarray(eligible, dtype=bool)
    train = train_mask & eligible & (y >= 0)
    if sum(train) < 100:
        return 0.6
    best = None
    for alpha in ALPHA_GRID:
        _, pred = _fused_predictions(model_p[train], current[train], alpha)
        metric = _balanced_metric(y[train], pred, sides[train])
        value = metric["balanced_accuracy"]
        # Macro-F1 is included as the tie breaker, but alpha choice remains
        # accuracy focused for the requested 60% selective-risk target.
        f1_values = [x["macro_f1"] for x in metric["by_side"].values() if x["macro_f1"] is not None]
        f1 = float(np.mean(f1_values)) if f1_values else 0.0
        objective = 0.8 * value + 0.2 * f1
        if best is None or objective > best[0] + 1e-12:
            best = (objective, float(alpha))
    return best[1]


def _crossfit_policies(data, p30, strata, n_folds=5, seed=20260927):
    current, y, _scores, _distance, _quality, groups = strata
    sides = data["sides"].astype(str)
    valid = y >= 0
    unique_games = np.unique(data["game_ids"])
    rng = np.random.default_rng(seed)
    shuffled = unique_games.copy()
    rng.shuffle(shuffled)
    folds = [np.asarray(x, dtype=np.int64) for x in np.array_split(shuffled, n_folds)]
    policies = {"global_oof": np.full(len(y), -1, dtype=np.int64),
                "current_mode_oof": np.full(len(y), -1, dtype=np.int64),
                "boundary_oof": np.full(len(y), -1, dtype=np.int64),
                "quality_oof": np.full(len(y), -1, dtype=np.int64)}
    fold_notes = []
    specs = {
        "current_mode_oof": groups["current_mode"],
        "boundary_oof": groups["mode_score_boundary_distance"],
        "quality_oof": groups["observation_quality"],
    }
    for fold_idx, held_games in enumerate(folds):
        held = np.isin(data["game_ids"], held_games) & valid
        train = ~np.isin(data["game_ids"], held_games) & valid
        global_alpha = _best_alpha(p30, current, y, sides, train)
        _, policies["global_oof"][held] = _fused_predictions(
            p30[held], current[held], global_alpha)
        note = {"fold": fold_idx, "held_out_games": int(len(held_games)),
                "global_alpha": global_alpha, "stratified_alphas": {}}
        for policy_name, labels in specs.items():
            values = np.unique(labels[held])
            for group in values:
                gmask = labels == group
                alpha = _best_alpha(p30, current, y, sides, train, gmask)
                choose = held & gmask
                _, policies[policy_name][choose] = _fused_predictions(
                    p30[choose], current[choose], alpha)
                note["stratified_alphas"][f"{policy_name}:{group}"] = alpha
        fold_notes.append(note)
    # Fill invalid target rows with the frozen current-state prior; they are not
    # included in any forecast score or OOF holdout metric.
    for key, pred in policies.items():
        missing = pred < 0
        pred[missing] = current[missing]
    return policies, fold_notes


def _bootstrap_diff(data, pred_a, pred_b, mask, n_boot=1000, seed=20260927):
    """Paired, game-cluster bootstrap of equal-red/blue accuracy difference."""
    y = data["mode_y"][:, HORIZON_INDEX]
    sides = data["sides"].astype(str)
    games = np.unique(data["game_ids"])
    values = []
    rng = np.random.default_rng(seed)
    game_indices = {g: np.flatnonzero((data["game_ids"] == g) & mask & (y >= 0)) for g in games}
    for _ in range(n_boot):
        draw = rng.choice(games, size=len(games), replace=True)
        side_acc = []
        for side in ("红", "蓝"):
            ids = np.concatenate([game_indices[g] for g in draw if len(game_indices[g])]) if any(len(game_indices[g]) for g in draw) else np.asarray([],dtype=np.int64)
            ids = ids[sides[ids] == side]
            if len(ids):
                side_acc.append((float(np.mean(pred_a[ids] == y[ids])),
                                 float(np.mean(pred_b[ids] == y[ids]))) )
        if len(side_acc) == 2:
            values.append(np.mean([a - b for a, b in side_acc]))
    if not values:
        return {"delta": None, "ci95": None, "n_boot_valid": 0}
    return {"delta": float(np.mean(values)),
            "ci95": [float(np.quantile(values, .025)), float(np.quantile(values, .975))],
            "n_boot_valid": len(values), "unit": "game; paired red/blue views retained"}


def _group_table(data, predictions, strata):
    current, y, _scores, _distance, _quality, groups = strata
    sides = data["sides"].astype(str)
    valid = y >= 0
    _, pred_model = _fused_predictions(predictions, current, 1.0)
    _, pred_persist = _fused_predictions(predictions, current, 0.0)
    _, pred_current = _fused_predictions(predictions, current, 0.6)
    definitions = {
        **{f"current_mode={v}": groups["current_mode"] == v for v in MODES},
        **{f"true_flip={v}": groups["true_flip"] == v for v in ("true_flip", "no_flip")},
        **{f"boundary={v}": groups["mode_score_boundary_distance"] == v for v in np.unique(groups["mode_score_boundary_distance"])},
        **{f"quality={v}": groups["observation_quality"] == v for v in np.unique(groups["observation_quality"])},
        **{f"opponent_base={v}": groups["opponent_base_visible"] == v for v in ("visible", "not_visible")},
    }
    result = {}
    for name, group_mask in definitions.items():
        mask = valid & group_mask
        result[name] = {
            "n": int(mask.sum()),
            "games": int(len(np.unique(data["game_ids"][mask]))),
            "model": _balanced_metric(y, pred_model, sides, mask),
            "fixed_fusion_alpha_0_6": _balanced_metric(y, pred_current, sides, mask),
            "persistence": _balanced_metric(y, pred_persist, sides, mask),
            "fixed_fusion_minus_persistence_accuracy_ci95": _bootstrap_diff(
                data, pred_current, pred_persist, mask),
        }
    return result


def _risk_metrics_at_threshold(y, pred, confidence, sides, threshold):
    valid = y >= 0
    accepted = valid & (confidence >= threshold)
    by_side = {}
    side_accuracies, side_coverages = [], []
    for label, side in (("red_view", "红"), ("blue_view", "蓝")):
        side_all = valid & (sides == side)
        side_accept = accepted & (sides == side)
        coverage = float(side_accept.sum() / max(1, side_all.sum()))
        accuracy = float(np.mean(pred[side_accept] == y[side_accept])) if side_accept.any() else None
        by_side[label] = {"coverage": coverage, "accuracy": accuracy,
                          "accepted": int(side_accept.sum()), "total": int(side_all.sum())}
        side_coverages.append(coverage)
        if accuracy is not None:
            side_accuracies.append(accuracy)
    return {"threshold": float(threshold),
            "balanced_coverage": float(np.mean(side_coverages)),
            "balanced_accuracy": float(np.mean(side_accuracies)) if len(side_accuracies) == 2 else None,
            "n_accepted": int(accepted.sum()), "by_side": by_side}


def _risk_curve(y, pred, confidence, sides, target_accuracy=0.60):
    """Exact confidence sweep in O(N log N), retaining a compact display curve."""
    valid_ids = np.flatnonzero(y >= 0)
    order = valid_ids[np.argsort(-confidence[valid_ids], kind="mergesort")]
    totals = {side: int(np.sum((y >= 0) & (sides == side))) for side in ("红", "蓝")}
    accepted_n = {side: 0 for side in totals}
    correct_n = {side: 0 for side in totals}
    display = []
    best = None
    n = len(order)
    cursor = 0
    last_display_bucket = None
    while cursor < n:
        end = cursor + 1
        threshold = float(confidence[order[cursor]])
        while end < n and float(confidence[order[end]]) == threshold:
            end += 1
        ids = order[cursor:end]
        for side in totals:
            keep = sides[ids] == side
            accepted_n[side] += int(keep.sum())
            correct_n[side] += int(np.sum(keep & (pred[ids] == y[ids])))
        coverages = [accepted_n[s] / max(1, totals[s]) for s in totals]
        accuracies = [correct_n[s] / accepted_n[s] for s in totals if accepted_n[s] > 0]
        record = {
            "threshold": threshold,
            "balanced_coverage": float(np.mean(coverages)),
            "balanced_accuracy": float(np.mean(accuracies)) if len(accuracies) == 2 else None,
            "n_accepted": int(end),
            "by_side": {
                "red_view": {"coverage": coverages[0],
                             "accuracy": correct_n["红"] / accepted_n["红"] if accepted_n["红"] else None,
                             "accepted": accepted_n["红"], "total": totals["红"]},
                "blue_view": {"coverage": coverages[1],
                              "accuracy": correct_n["蓝"] / accepted_n["蓝"] if accepted_n["蓝"] else None,
                              "accepted": accepted_n["蓝"], "total": totals["蓝"]},
            },
        }
        if record["balanced_accuracy"] is not None and record["balanced_accuracy"] >= target_accuracy:
            if best is None or record["balanced_coverage"] > best["balanced_coverage"] + 1e-12:
                best = record
        # At most ~101 points for a readable coverage-risk curve.
        bucket = int(round(record["balanced_coverage"] * 100))
        if bucket != last_display_bucket:
            display.append(record)
            last_display_bucket = bucket
        cursor = end
    return {"sampled_curve": display, "selected_threshold": best,
            "target_accuracy": float(target_accuracy), "exact_unique_confidence_thresholds_scanned": int(n)}


def _select_threshold(curve, target_accuracy=0.60):
    selected = curve.get("selected_threshold")
    if selected is None or selected.get("balanced_accuracy", 0.0) < target_accuracy:
        return None
    return selected


def _test_risk_bootstrap(data, y, pred, conf, threshold, n_boot=2000, seed=20260927):
    sides = data["sides"].astype(str)
    game_ids = data["game_ids"].astype(np.int64)
    valid = y >= 0
    accept = valid & (conf >= threshold)
    point = _risk_metrics_at_threshold(y, pred, conf, sides, threshold)
    games = np.unique(game_ids)
    rng = np.random.default_rng(seed)
    stats = []
    indices = {g: np.flatnonzero((game_ids == g) & valid) for g in games}
    for _ in range(n_boot):
        draw = rng.choice(games, len(games), replace=True)
        side_acc, side_cov = [], []
        ok = True
        for side in ("红", "蓝"):
            ids = np.concatenate([indices[g] for g in draw if len(indices[g])]) if any(len(indices[g]) for g in draw) else np.asarray([],dtype=np.int64)
            ids = ids[sides[ids] == side]
            a = ids[conf[ids] >= threshold]
            if not len(ids) or not len(a):
                ok = False
                break
            side_cov.append(len(a) / len(ids))
            side_acc.append(float(np.mean(pred[a] == y[a])))
        if ok:
            stats.append((float(np.mean(side_cov)), float(np.mean(side_acc)),
                          side_cov[0], side_cov[1], side_acc[0], side_acc[1]))
    arr = np.asarray(stats, dtype=np.float64)
    def ci(column):
        return [float(np.quantile(arr[:, column], .025)),
                float(np.quantile(arr[:, column], .975))]
    return {
        "threshold_frozen_from_validation": float(threshold),
        "point_estimate": point,
        "game_bootstrap_95ci": {
            "balanced_coverage": ci(0), "balanced_conditional_accuracy": ci(1),
            "red_coverage": ci(2), "blue_coverage": ci(3),
            "red_conditional_accuracy": ci(4), "blue_conditional_accuracy": ci(5),
        },
        "n_boot": int(n_boot), "n_boot_valid": int(len(stats)),
        "n_games": int(len(games)),
        "unit": "game; red/blue observations resampled together within game",
    }


def _format_report(report):
    def fmt(value, digits=3):
        return "n/a" if value is None else f"{value:.{digits}f}"

    lines = [
        "# global_mode@30s 验证集分层诊断与选择性置信策略",
        "",
        "固定模型：现有329维 hybrid checkpoint；不训练、不调模型；分组/OOF alpha 只使用 validation。置信阈值也只在 validation 选一次，然后冻结并在 test 仅评估一次。",
        "",
        "## 30秒当前态势融合诊断",
        "",
        f"按比赛分组的 {report['validation']['crossfit']['n_folds']}-fold OOF 等权红蓝准确率：持久性 {fmt(report['validation']['crossfit']['persistence']['balanced_accuracy'])}，固定alpha=0.6融合 {fmt(report['validation']['crossfit']['fixed_alpha_0_6']['balanced_accuracy'])}，折内全局alpha {fmt(report['validation']['crossfit']['global_oof']['balanced_accuracy'])}，按当前mode选择alpha {fmt(report['validation']['crossfit']['current_mode_oof']['balanced_accuracy'])}，按分数边界选择alpha {fmt(report['validation']['crossfit']['boundary_oof']['balanced_accuracy'])}，按观测质量选择alpha {fmt(report['validation']['crossfit']['quality_oof']['balanced_accuracy'])}。",
        "",
        "分层表的模型列是纯 hybrid；融合列用现有alpha=0.6；差值区间按game成簇并保留红蓝配对。真实翻转只用于回顾分层，线上不可用来选alpha。",
        "",
        "|分层|n|games|纯hybrid 红/蓝Acc|alpha=.6 红/蓝Acc|持久性 红/蓝Acc|融合-持久性差及game-bootstrap 95%CI|",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, item in report["validation"]["stratified_metrics"].items():
        diff = item["fixed_fusion_minus_persistence_accuracy_ci95"]
        ci = diff.get("ci95")
        lines.append(
            f"|{name}|{item['n']}|{item['games']}|"
            f"{fmt(item['model']['by_side']['red_view']['accuracy'])} / {fmt(item['model']['by_side']['blue_view']['accuracy'])}|"
            f"{fmt(item['fixed_fusion_alpha_0_6']['by_side']['red_view']['accuracy'])} / {fmt(item['fixed_fusion_alpha_0_6']['by_side']['blue_view']['accuracy'])}|"
            f"{fmt(item['persistence']['by_side']['red_view']['accuracy'])} / {fmt(item['persistence']['by_side']['blue_view']['accuracy'])}|"
            f"{fmt(diff['delta'],3)} [{fmt(ci[0] if ci else None)}, {fmt(ci[1] if ci else None)}]|")
    lines += ["", "## 置信阈值与锁定测试描述", "",
              "阈值按 validation 融合概率的最大类别概率选择；约束为红蓝等权 conditional accuracy ≥0.60，取满足条件时 coverage 最大者。测试集未参与阈值选择。",
              "Selective conditional accuracy 仅针对通过阈值的预测；覆盖率不足1时不能当作总体准确率。本次validation选出的阈值覆盖率为1，因此它没有筛除任何预测，条件准确率恰等于总体准确率，也没有产生实际的选择性拒答收益。",
              ""]
    chosen = report["selective_risk"]["validation_selected_threshold"]
    if chosen is None:
        lines.append("Validation 没有任何阈值能使红蓝等权准确率达到0.60；测试阈值评估跳过。")
    else:
        lines.append(f"Validation 选定阈值 `{chosen['threshold']:.6f}`：等权 coverage `{chosen['balanced_coverage']:.3f}`，等权 conditional accuracy `{chosen['balanced_accuracy']:.3f}`，红/蓝 coverage `{chosen['by_side']['red_view']['coverage']:.3f}/{chosen['by_side']['blue_view']['coverage']:.3f}`，红/蓝 accuracy `{chosen['by_side']['red_view']['accuracy']:.3f}/{chosen['by_side']['blue_view']['accuracy']:.3f}`。")
        test = report["selective_risk"].get("locked_test")
        if test:
            p = test["point_estimate"]
            ci = test["game_bootstrap_95ci"]
            lines.append(f"冻结后 test：coverage `{p['balanced_coverage']:.3f}`，conditional accuracy `{p['balanced_accuracy']:.3f}`，game-bootstrap 95%CI `{ci['balanced_conditional_accuracy'][0]:.3f}–{ci['balanced_conditional_accuracy'][1]:.3f}`；红/蓝条件准确率 `{p['by_side']['red_view']['accuracy']:.3f}/{p['by_side']['blue_view']['accuracy']:.3f}`，各自95%CI `{ci['red_conditional_accuracy'][0]:.3f}–{ci['red_conditional_accuracy'][1]:.3f}` / `{ci['blue_conditional_accuracy'][0]:.3f}–{ci['blue_conditional_accuracy'][1]:.3f}`。冻结阈值覆盖率为1，这些条件准确率也就是总体准确率。")
            lines.append(f"部署判据为等权95%CI下界≥0.60、两侧点估计均≥0.60；当前结论：**{'满足' if report['selective_risk']['robust_for_deployment'] else '不满足，不建议部署为决策门控'}**。可作为界面置信提示的候选，仍需监控覆盖率与跨赛季漂移。")
        elif report["selective_risk"].get("test_evaluation_note"):
            lines.append(report["selective_risk"]["test_evaluation_note"])
    lines += ["", "细节与fold参数见同目录JSON报告。"]
    return "\n".join(lines) + "\n"


def run_locked_test_once(args):
    """Run only the already validation-locked test policy, without retuning."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    threshold_path = OUT_DIR / "validation_selected_confidence_threshold.json"
    validation_report_path = OUT_DIR / "global_mode30_validation_diagnostic.json"
    if not threshold_path.exists() or not validation_report_path.exists():
        raise FileNotFoundError("validation diagnostics and frozen threshold must exist before locked test")
    threshold_info = json.loads(threshold_path.read_text(encoding="utf-8"))
    validation_report = json.loads(validation_report_path.read_text(encoding="utf-8"))
    if threshold_info.get("split") != "validation" or threshold_info.get("test_used_for_selection") is not False:
        raise ValueError("threshold artifact is not explicitly validation-only")
    threshold = float(threshold_info["threshold"])
    alpha = float(threshold_info["alpha"])
    if not (0.0 <= threshold <= 1.0 and 0.0 <= alpha <= 1.0):
        raise ValueError("invalid frozen threshold or alpha")
    if str(threshold_info.get("model_checkpoint")) != str(MODEL):
        raise ValueError("frozen threshold checkpoint differs from current model checkpoint")

    # Exactly one deterministic test inference; threshold/alpha are read-only.
    test_data, test_raw_p, test_device = _predict_probabilities(
        MODEL, DATA_DIR / "test.npz", args.device)
    test_summary = _load_summary("test", test_data)
    test_current, test_y, *_ = _strata(test_data, test_summary)
    test_fused, test_pred = _fused_predictions(test_raw_p, test_current, alpha)
    test_conf = np.max(test_fused, axis=1)
    test_eval = _test_risk_bootstrap(
        test_data, test_y, test_pred, test_conf, threshold,
        n_boot=args.bootstrap, seed=args.seed)
    red_ci = test_eval["game_bootstrap_95ci"]["red_conditional_accuracy"]
    blue_ci = test_eval["game_bootstrap_95ci"]["blue_conditional_accuracy"]
    lower_balanced = test_eval["game_bootstrap_95ci"]["balanced_conditional_accuracy"][0]
    red_acc = test_eval["point_estimate"]["by_side"]["red_view"]["accuracy"]
    blue_acc = test_eval["point_estimate"]["by_side"]["blue_view"]["accuracy"]
    robust = lower_balanced >= 0.60 and red_acc >= 0.60 and blue_acc >= 0.60

    # Preserve all validation evidence, adding only this locked test description.
    report = validation_report
    report["selective_risk"]["locked_test"] = test_eval
    report["selective_risk"]["robust_for_deployment"] = bool(robust)
    report["selective_risk"]["robustness_rule"] = (
        "balanced game-bootstrap 95% CI lower bound >=0.60 and each side point accuracy >=0.60"
    )
    report["selective_risk"]["test_device"] = test_device
    report["selective_risk"]["red_conditional_accuracy_95ci"] = red_ci
    report["selective_risk"]["blue_conditional_accuracy_95ci"] = blue_ci
    report["selective_risk"]["test_evaluation_note"] = (
        "Single locked test inference at validation-selected threshold and alpha; no test tuning. "
        "Selective conditional accuracy applies only to accepted predictions and is not overall accuracy."
    )
    report["selective_risk"]["locked_test_evaluation_count"] = 1

    report_path = OUT_DIR / "global_mode30_validation_diagnostic.json"
    temp_path = report_path.with_suffix(".json.tmp")
    temp_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    temp_path.replace(report_path)
    summary_path = OUT_DIR / "global_mode30_validation_diagnostic.md"
    summary_path.write_text(_format_report(report), encoding="utf-8")
    print(json.dumps({
        "test_evaluations": 1, "threshold_frozen_from_validation": threshold,
        "alpha_frozen_from_validation": alpha, "test": test_eval,
        "robust_for_deployment": bool(robust), "report": str(report_path),
        "summary": str(summary_path),
    }, ensure_ascii=False), flush=True)


def run(args):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fusion = json.loads(FUSION_PATH.read_text(encoding="utf-8"))
    alpha = float(fusion["alphas"]["global_mode@30s"])
    if not 0 <= alpha <= 1:
        raise ValueError(f"invalid validation-selected fusion alpha {alpha}")

    # Validation only: prediction, strata, game-grouped OOF alpha comparisons,
    # and confidence-threshold selection all happen before test is loaded.
    val_data, val_raw_p, used_device = _predict_probabilities(
        MODEL, DATA_DIR / "validation.npz", args.device)
    val_summary = _load_summary("validation", val_data)
    val_strata = _strata(val_data, val_summary)
    val_current, val_y = val_strata[0], val_strata[1]
    val_fused, val_fused_pred = _fused_predictions(val_raw_p, val_current, alpha)
    val_prior_pred = val_current.copy()
    val_model_pred = np.argmax(val_raw_p, axis=1)
    fold_policies, fold_notes = _crossfit_policies(val_data, val_raw_p, val_strata)
    val_sides = val_data["sides"].astype(str)
    cv_metrics = {
        "n_folds": 5,
        "n_games": int(len(np.unique(val_data["game_ids"]))),
        "persistence": _balanced_metric(val_y, val_prior_pred, val_sides),
        "fixed_alpha_0_6": _balanced_metric(val_y, val_fused_pred, val_sides),
        "global_oof": _balanced_metric(val_y, fold_policies["global_oof"], val_sides),
        "current_mode_oof": _balanced_metric(val_y, fold_policies["current_mode_oof"], val_sides),
        "boundary_oof": _balanced_metric(val_y, fold_policies["boundary_oof"], val_sides),
        "quality_oof": _balanced_metric(val_y, fold_policies["quality_oof"], val_sides),
    }
    stratified = _group_table(val_data, val_raw_p, val_strata)

    # Confidence policy: search unique validation confidences only. A single
    # scalar threshold is then frozen for the sole locked test evaluation.
    confidence = np.max(val_fused, axis=1)
    thresholds = np.unique(np.concatenate(([0.0], confidence)))
    val_curve = _risk_curve(val_y, val_fused_pred, confidence, val_sides, target_accuracy=0.60)
    selected = _select_threshold(val_curve, target_accuracy=0.60)
    if selected is not None:
        chosen_threshold = selected["threshold"]
        locked_test = True
    else:
        chosen_threshold = None
        locked_test = False

    report = {
        "schema_version": "global-mode30-validation-diagnostic-v1",
        "model_checkpoint": str(MODEL),
        "model_input_dim": 329,
        "device": used_device,
        "split_policy": "all tuning/strata/threshold selection on validation; test loaded only after threshold freeze",
        "fusion": {"source": str(FUSION_PATH), "current_validation_selected_alpha": alpha,
                   "formula": "alpha*calibrated_hybrid_probability+(1-alpha)*smoothed_current_mode_prior",
                   "current_mode_prior_confidence": 0.98},
        "validation": {
            "split": "validation",
            "n_rows": int(len(val_y)),
            "n_scored": int(np.sum(val_y >= 0)),
            "n_games": int(len(np.unique(val_data["game_ids"]))),
            "crossfit": cv_metrics,
            "fold_alphas": fold_notes,
            "stratification_definitions": {
                "current_mode": list(MODES),
                "mode_score_boundary_distance": "min(|score+1|,|score-1|): <=0.25, (0.25,0.75], >0.75",
                "true_flip": "30s target mode != current base_mode; retrospective only",
                "observation_quality": "minimum of both-side mobile type coverage and position coverage: <0.50 low, [0.50,0.80) medium, >=0.80 high",
                "bootstrap": "paired game-cluster; red/blue views resampled together",
            },
            "stratified_metrics": stratified,
        },
        "selective_risk": {
            "threshold_constraint": "maximize mean(red coverage, blue coverage) subject to mean(red conditional accuracy, blue conditional accuracy) >=0.60",
            "validation_curve_sampled": val_curve["sampled_curve"],
            "validation_curve_scan": {k: val_curve[k] for k in ("target_accuracy", "exact_unique_confidence_thresholds_scanned")},
            "validation_selected_threshold": selected,
            "locked_test_evaluated": bool(locked_test),
            "test_threshold_selection_used": False,
        },
    }

    # Test is not read until the validation-selected threshold is already fixed.
    if locked_test and not args.validation_only:
        test_data, test_raw_p, test_device = _predict_probabilities(
            MODEL, DATA_DIR / "test.npz", args.device)
        test_summary = _load_summary("test", test_data)
        test_current, test_y, *_ = _strata(test_data, test_summary)
        test_fused, test_pred = _fused_predictions(test_raw_p, test_current, alpha)
        test_conf = np.max(test_fused, axis=1)
        test_eval = _test_risk_bootstrap(
            test_data, test_y, test_pred, test_conf, chosen_threshold,
            n_boot=args.bootstrap, seed=args.seed)
        red_ci = test_eval["game_bootstrap_95ci"]["red_conditional_accuracy"]
        blue_ci = test_eval["game_bootstrap_95ci"]["blue_conditional_accuracy"]
        lower_balanced = test_eval["game_bootstrap_95ci"]["balanced_conditional_accuracy"][0]
        red_acc = test_eval["point_estimate"]["by_side"]["red_view"]["accuracy"]
        blue_acc = test_eval["point_estimate"]["by_side"]["blue_view"]["accuracy"]
        robust = lower_balanced >= 0.60 and red_acc >= 0.60 and blue_acc >= 0.60
        report["selective_risk"]["locked_test"] = test_eval
        report["selective_risk"]["robust_for_deployment"] = bool(robust)
        report["selective_risk"]["robustness_rule"] = "balanced game-bootstrap 95% CI lower bound >=0.60 and each side point accuracy >=0.60"
        report["selective_risk"]["test_device"] = test_device
        report["selective_risk"]["red_conditional_accuracy_95ci"] = red_ci
        report["selective_risk"]["blue_conditional_accuracy_95ci"] = blue_ci

    if args.validation_only and locked_test:
        report["selective_risk"]["test_evaluation_note"] = (
            "验证集已选出阈值；本次按上级要求只交付validation诊断，不重复锁定test评估。"
        )
    elif args.validation_only and not locked_test:
        report["selective_risk"]["test_evaluation_note"] = (
            "验证集未找到达到等权准确率0.60的阈值，因此没有可冻结到test的阈值。"
        )

    report_path = OUT_DIR / "global_mode30_validation_diagnostic.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    summary_path = OUT_DIR / "global_mode30_validation_diagnostic.md"
    summary_path.write_text(_format_report(report), encoding="utf-8")
    if selected is not None:
        threshold_path = OUT_DIR / "validation_selected_confidence_threshold.json"
        threshold_path.write_text(json.dumps({
            "split": "validation", "alpha": alpha,
            "threshold": chosen_threshold, "validation_metrics": selected,
            "model_checkpoint": str(MODEL), "fusion_report": str(FUSION_PATH),
            "test_used_for_selection": False,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"report": str(report_path), "summary": str(summary_path),
                      "validation_alpha": alpha,
                      "validation_oof": cv_metrics,
                      "validation_selected_threshold": selected,
                      "locked_test": report["selective_risk"].get("locked_test"),
                      "robust_for_deployment": report["selective_risk"].get("robust_for_deployment")},
                     ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260927)
    parser.add_argument("--validation-only", action="store_true",
                        help="do not load test; emit only validation diagnostics and frozen threshold")
    parser.add_argument("--locked-test-once", action="store_true",
                        help="run exactly one test inference using the saved validation threshold; no validation rerun")
    args = parser.parse_args()
    if args.locked_test_once:
        run_locked_test_once(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
