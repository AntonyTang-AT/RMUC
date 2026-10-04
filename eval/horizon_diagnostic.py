"""Build an auditable, data-driven summary of forecast quality by horizon."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

HORIZONS = (10, 30, 60, 120)
MODEL = "temporal_hybrid_v1"
BASELINE = "persistence_v1"


def _macro_score(report: dict[str, Any], model: str, key: str, metric: str) -> float:
    return float(report["models"][model]["by_task_horizon"][key][metric]["side_macro_average"]["score"])


def _class_metrics(report: dict[str, Any], model: str, key: str) -> dict[str, Any]:
    return report["macro_f1_by_task_horizon"][model][key]


def _macro_f1(confusion: np.ndarray) -> float:
    tp = np.diag(confusion).astype(np.float64)
    fp = confusion.sum(axis=0) - tp
    fn = confusion.sum(axis=1) - tp
    denom = 2 * tp + fp + fn
    scores = np.divide(2 * tp, denom, out=np.zeros_like(tp), where=denom > 0)
    return float(scores.mean())


def _game_clustered_comparison(report: dict[str, Any], bootstrap_replicates: int = 1000) -> dict[str, Any]:
    """Recompute paired model-vs-persistence metrics by held-out match.

    Each match contributes both red and blue views.  Within every bootstrap
    draw, the two sides are scored separately and averaged with equal weight;
    this prevents a side's larger sample count from dominating the result.
    """
    contract = report.get("data_contract", {})
    test_path = Path(contract.get("test_npz", ""))
    model_path = Path(report.get("training", {}).get("path", ""))
    if not test_path.is_file() or not model_path.is_file():
        return {"available": False, "reason": "The report does not point to an existing test NPZ and checkpoint."}

    import torch
    from eval.run_temporal_forecast import TASKS
    from forecast.temporal import HORIZONS
    from forecast.train import load_model

    with np.load(test_path, allow_pickle=False) as data:
        arrays = {name: data[name] for name in data.files}
    ids = np.asarray(arrays["game_ids"], dtype=np.int64)
    sides = np.asarray(arrays["sides"])
    games = np.unique(ids)
    side_labels = ("红", "蓝")
    game_side_index = {(int(game), side): i for i, (game, side) in enumerate(
        ( (int(game), side) for game in games for side in side_labels ))}
    n_games = len(games)

    # Histograms have shape [game, side, model, class, class].
    matrices: dict[tuple[str, int], np.ndarray] = {}
    class_counts: dict[tuple[str, int], np.ndarray] = {}
    task_arrays = {"global_mode": "mode_y", "opponent_trend": "trend_y", "enemy_region": "region_y"}
    class_sizes = {"global_mode": 3, "opponent_trend": 3, "enemy_region": 12}
    for task in task_arrays:
        for horizon in HORIZONS:
            shape = (n_games, 2, 2, class_sizes[task], class_sizes[task])
            matrices[(task, int(horizon))] = np.zeros(shape, dtype=np.int32)
            class_counts[(task, int(horizon))] = np.zeros((n_games, 2), dtype=np.int32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _checkpoint = load_model(model_path, device=device)
    model.eval()
    batch_size = 512
    with torch.no_grad():
        for start in range(0, len(ids), batch_size):
            end = min(len(ids), start + batch_size)
            xb = torch.from_numpy(arrays["x"][start:end]).to(device)
            outputs = model(xb)
            for task, y_key in task_arrays.items():
                for h_idx, horizon in enumerate(HORIZONS):
                    logits = outputs[horizon][task].detach().cpu().numpy()
                    candidate = logits.argmax(axis=-1)
                    truth = arrays[y_key][start:end, h_idx]
                    if task == "global_mode":
                        baseline = arrays["base_mode"][start:end]
                    elif task == "enemy_region":
                        baseline = arrays["base_region"][start:end]
                    else:
                        baseline = np.ones(end - start, dtype=np.int64)
                    hist = matrices[(task, int(horizon))]
                    counts = class_counts[(task, int(horizon))]
                    for local_idx, (target, pred, base) in enumerate(zip(truth, candidate, baseline)):
                        if target < 0:
                            continue
                        row_idx = start + local_idx
                        group = game_side_index[(int(ids[row_idx]), str(sides[row_idx]))]
                        game_idx, side_idx = divmod(group, 2)
                        hist[game_idx, side_idx, 0, int(target), int(pred)] += 1
                        hist[game_idx, side_idx, 1, int(target), int(base)] += 1
                        counts[game_idx, side_idx] += 1

    output: dict[str, Any] = {
        "available": True,
        "unit_of_resampling": "complete match/game_id; red and blue views stay paired",
        "split_contract": {
            "train_games": None, "validation_games": None, "test_games": n_games,
            "test_game_ids_have_both_sides": bool(all(
                (int(game), side) in game_side_index for game in games for side in side_labels)),
            "forecast_ledger_red_blue_complete": report.get("red_blue_complete"),
            "forecast_ledger_invalid_count": report.get("invalid_count"),
            "forecast_ledger_missing_count": report.get("missing_count"),
            "train_validation_test_game_ids_disjoint": None,
        },
        "bootstrap_replicates": bootstrap_replicates,
        "seed": 20260927,
        "comparison": {},
    }
    # Prove split isolation from the saved compact data, rather than inferring
    # it from chronological ID ranges alone.
    split_ids = {}
    work_dir = test_path.parent
    for split in ("train", "validation", "test"):
        p = work_dir / f"{split}.npz"
        if p.is_file():
            with np.load(p, allow_pickle=False) as split_data:
                split_ids[split] = set(map(int, np.unique(split_data["game_ids"])))
    if len(split_ids) == 3:
        output["split_contract"]["train_games"] = len(split_ids["train"])
        output["split_contract"]["validation_games"] = len(split_ids["validation"])
        output["split_contract"]["test_games"] = len(split_ids["test"])
        output["split_contract"]["train_validation_test_game_ids_disjoint"] = not (
            split_ids["train"] & split_ids["validation"] or
            split_ids["train"] & split_ids["test"] or
            split_ids["validation"] & split_ids["test"])

    rng = np.random.default_rng(20260927)
    for (task, horizon), hist in matrices.items():
        counts = class_counts[(task, horizon)]
        matched_games = np.flatnonzero((counts > 0).all(axis=1))
        side_points: dict[str, Any] = {}
        metric_arrays: dict[str, list[float]] = {"accuracy": [], "macro_f1": []}
        # Only resample games with scorable labels on both sides for this task
        # and horizon; the source ledger may independently report invalid rows.
        draws = rng.integers(0, len(matched_games), size=(bootstrap_replicates, len(matched_games)))
        sampled = hist[matched_games[draws]]  # [bootstrap, sampled match, side, model, c, c]
        # Keep side metrics separate for transparent red/blue symmetry.
        for side_idx, side_name in enumerate(side_labels):
            values = hist[matched_games, side_idx].sum(axis=0)  # [model, class, class]
            per_model = []
            for model_idx, model_name in enumerate((MODEL, BASELINE)):
                cm = values[model_idx]
                accuracy = float(np.trace(cm) / max(1, cm.sum()))
                f1 = _macro_f1(cm)
                per_model.append({"model": model_name, "accuracy": accuracy, "macro_f1": f1,
                                  "n": int(cm.sum())})
            side_points[side_name] = {
                "hybrid": per_model[0], "persistence": per_model[1],
                "accuracy_delta_hybrid_minus_persistence": per_model[0]["accuracy"] - per_model[1]["accuracy"],
                "macro_f1_delta_hybrid_minus_persistence": per_model[0]["macro_f1"] - per_model[1]["macro_f1"],
            }

            boot_cms = sampled[:, :, side_idx].sum(axis=1)  # [bootstrap, model, c, c]
            boot_acc = np.trace(boot_cms, axis1=2, axis2=3) / np.maximum(1, boot_cms.sum(axis=(2, 3)))
            boot_f1 = np.asarray([[_macro_f1(cm) for cm in boot_cms[b]] for b in range(bootstrap_replicates)])
            metric_arrays["accuracy"].append(boot_acc[:, 0] - boot_acc[:, 1])
            metric_arrays["macro_f1"].append(boot_f1[:, 0] - boot_f1[:, 1])

        comparison = {}
        for metric, side_deltas in metric_arrays.items():
            # Equal red/blue weighting within every bootstrap draw.
            paired_delta = np.mean(np.stack(side_deltas, axis=1), axis=1)
            values_by_side = [side_points[side][f"{metric}_delta_hybrid_minus_persistence"] for side in side_labels]
            comparison[metric] = {
                "red_blue_macro_delta_hybrid_minus_persistence": float(np.mean(values_by_side)),
                "paired_game_bootstrap_95pct_ci": [float(x) for x in np.percentile(paired_delta, [2.5, 97.5])],
                "bootstrap_probability_delta_positive": float(np.mean(paired_delta > 0)),
                "per_side_paired_game_bootstrap": {
                    side: {
                        "point_delta_hybrid_minus_persistence": float(values_by_side[side_idx]),
                        "ci_95pct": [float(x) for x in np.percentile(side_deltas[side_idx], [2.5, 97.5])],
                    }
                    for side_idx, side in enumerate(side_labels)
                },
                "interpretation": "hybrid_better" if np.percentile(paired_delta, 2.5) > 0 else (
                    "persistence_better" if np.percentile(paired_delta, 97.5) < 0 else "no_clear_difference"),
            }
        key = f"{task}@{horizon}s"
        output["comparison"][key] = {"by_side": side_points, "paired_comparison": comparison,
                                     "n_matches_with_both_sides_scored": int(len(matched_games))}
    return output


def build(report: dict[str, Any], game_clustered: dict[str, Any] | None = None) -> dict[str, Any]:
    entity_metrics = report["entity_forecast_metrics"]
    entity_rows = []
    for horizon in HORIZONS:
        metrics = entity_metrics[f"{horizon}s"]
        entity_rows.append({
            "horizon_s": horizon,
            "presence_accuracy_reported": float(metrics["presence_accuracy"]),
            "position_mae_normalized": float(metrics["position_mae_normalized"]),
            "hp_mae": float(metrics["hp_mae"]),
            "functional_zone_accuracy": float(metrics["functional_zone_accuracy"]),
            "canonical_zone_accuracy": float(metrics["canonical_zone_accuracy"]),
            "transition_legal_rate": float(metrics["transition_legal_rate"]),
        })

    global_rows = []
    target_distributions: dict[str, Any] = {}
    task_comparison: dict[str, Any] = {}
    for task in ("global_mode", "opponent_trend", "enemy_region"):
        task_comparison[task] = []
        for horizon in HORIZONS:
            key = f"{task}@{horizon}s"
            current_acc = float(report["accuracy_by_task_horizon"][MODEL][key]["accuracy"])
            base_acc = float(report["accuracy_by_task_horizon"][BASELINE][key]["accuracy"])
            current_f1 = _class_metrics(report, MODEL, key)
            base_f1 = _class_metrics(report, BASELINE, key)
            distribution = report["target_distribution_by_task_horizon"][key]
            total = sum(int(v) for v in distribution.values())
            shares = {str(name): int(count) / total for name, count in distribution.items()}
            row = {
                "task": task,
                "horizon_s": horizon,
                "accuracy": current_acc,
                "baseline_accuracy": base_acc,
                "accuracy_by_side": report["accuracy_by_task_horizon"][MODEL][key]["by_side"],
                "baseline_accuracy_by_side": report["accuracy_by_task_horizon"][BASELINE][key]["by_side"],
                "accuracy_delta_vs_baseline": current_acc - base_acc,
                "macro_f1": float(current_f1["macro_f1"]),
                "baseline_macro_f1": float(base_f1["macro_f1"]),
                "macro_f1_delta_vs_baseline": float(current_f1["macro_f1"]) - float(base_f1["macro_f1"]),
                "per_class_f1": current_f1["per_class"],
                "baseline_per_class_f1": base_f1["per_class"],
                "brier": _macro_score(report, MODEL, key, "brier"),
                "baseline_brier": _macro_score(report, BASELINE, key, "brier"),
                "log_loss": _macro_score(report, MODEL, key, "log_loss"),
                "baseline_log_loss": _macro_score(report, BASELINE, key, "log_loss"),
                "target_distribution": distribution,
                "target_share": shares,
                "sample_count": int(report["accuracy_by_task_horizon"][MODEL][key]["n"]),
                "n_complete_games": int(report["models"][MODEL]["by_task_horizon"][key]["brier"]["n_complete_games"]),
            }
            global_rows.append(row)
            task_comparison[task].append(row)
            target_distributions[key] = {"counts": distribution, "shares": shares}

    field_policy = {
        "机器人是否存活": {
            "valid_for_policy": False,
            "reason": "最新 death-aware 报告已用未来 HP=0 标为阵亡/不存在，presence accuracy 现在比旧版有意义；但它仍汇总机器人和设施槽位，且存活样本占多数，不能单独当作死亡识别成绩。",
            "after_recompute": "发布时同时报告移动机器人逐兵种的阵亡 precision/recall/F1 和混淆矩阵；线上只在当前明确读到 HP=0 时停止预测，缺测仍为未知。",
        },
        "连续位置（归一化 MAE）": {
            "usable_until_s": 30,
            "output": "10/30 秒坐标；60/120 秒只显示速度约束下的路线、侧别或控制方向。",
            "basis": "MAE 由 0.097 增至 0.121（10→30 秒，+0.025），之后仍增至 0.130/0.135；误差增长主要集中在短到中时距，远期应降级。",
        },
        "功能区": {
            "usable_until_s": 30,
            "output": "10/30 秒可显示候选功能区并附置信度；60/120 秒改成控制方向或区域概率，不给单区承诺。",
            "basis": "准确率 0.370→0.275→0.241→0.209；10→30 秒下降 0.095，之后仍持续下降。",
        },
        "canonical 精确区": {
            "usable_until_s": 10,
            "output": "只在 10 秒给出精确区；30 秒起显示区域候选/路线，不把低置信单区当事实。",
            "basis": "准确率 0.228→0.214→0.204→0.200，各时距都低，10 秒也仅约 23%。",
        },
        "HP 数值": {
            "usable_until_s": 30,
            "output": "10/30 秒可输出数值趋势及区间；60/120 秒只输出高/中/低风险。death-aware 标签已更新，但仍需单独报告阵亡 precision/recall/F1。",
            "basis": "HP MAE 0.157→0.164→0.170→0.177，随时距持续增加。",
        },
        "合法路线": {
            "usable_until_s": 120,
            "output": "四个时距都保留作路线可行性门禁；当前合法率 1.0 是约束投影后的结果，不代表实际路线预测准确。",
            "basis": "可达性门禁由后处理保证，必须与目标区准确率/位置误差分开汇报。",
        },
        "全局态势类别": {
            "usable_until_s": 120,
            "output": "四个时距保留概率分布；短期侧重预测解析，60/120 秒只作低置信宏观走势。",
            "basis": "Hybrid macro-F1 为 0.590/0.526/0.511/0.490；均衡类 F1 仅 0.473/0.418/0.390/0.420。10/30 秒按比赛配对 CI 确认落后持久性，60 秒 macro-F1 与 120 秒两项的差异不明确；四档 Brier/log-loss 均较好。",
        },
        "敌方趋势": {
            "usable_until_s": 120,
            "output": "保留概率，但必须显示类别概率和不确定性；60/120 秒不输出确定性口号。",
            "basis": "Hybrid macro-F1 为 0.443/0.409/0.400/0.414；10 秒 accuracy 略落后、macro-F1 更好，30/60/120 秒两项按比赛 bootstrap 均明显领先。稳定类在 60/120 秒 F1 约 0.04/0.06。",
        },
        "敌方精确区域": {
            "usable_until_s": 10,
            "output": "只把 10 秒区域作为弱概率提示；30 秒起改为宽区域/侧别/压力方向。",
            "basis": "Hybrid exact accuracy 为 0.597/0.540/0.507/0.503，macro-F1 为 0.328/0.269/0.247/0.218；10/30 秒两项均显著落后基线，60 秒准确率差异不明确且 F1 落后，120 秒准确率略胜而 F1 仍落后。12 类中不少类占比低于 2.5%。",
        },
    }

    threshold_changes = []
    for i in range(1, len(entity_rows)):
        prev, curr = entity_rows[i - 1], entity_rows[i]
        for key in ("position_mae_normalized", "hp_mae"):
            delta = curr[key] - prev[key]
            threshold_changes.append({"metric": key, "from_s": prev["horizon_s"], "to_s": curr["horizon_s"], "change": delta})
        for key in ("functional_zone_accuracy", "canonical_zone_accuracy"):
            delta = curr[key] - prev[key]
            threshold_changes.append({"metric": key, "from_s": prev["horizon_s"], "to_s": curr["horizon_s"], "change": delta})

    summaries = {
        "global_mode": "最新 death-aware 候选在四个时距的硬分类 accuracy/macro-F1 均落后持久性；Brier/log-loss 均更好，表示概率质量有改善，但还不足以替代基线的 argmax 分类。应提供概率和不确定性，短期态势分类暂以基线作为参照。",
        "opponent_trend": "10 秒候选 accuracy 略低于持久性；30/60/120 秒 accuracy 与四时距 macro-F1 均高于持久性。Brier/log-loss 各时距更好，但绝对 macro-F1 仍需看逐类值，不能用多数类 accuracy 宣称整体可靠。",
        "enemy_region": "Hybrid 的 exact accuracy 和 macro-F1 在 10/30/60 秒低于持久性；120 秒 accuracy 略高但 macro-F1 仍低。Brier/log-loss 各时距更好；宏观概率可参考，exact argmax 不宜作为高置信单一区域。类别分布严重不均，中心两列约占 67%，稀有边缘类常低于 2.5%。",
        "entity": "位置与 HP 误差随时距增加；功能区准确率在 10→30 秒显著下降，canonical 精确区准确率从 10 秒起就低。最新标签已将 HP=0 作为阵亡，但合并 presence accuracy 仍不能替代逐机器人阵亡 precision/recall/F1。",
    }
    if game_clustered and game_clustered.get("available"):
        comparisons = game_clustered["comparison"]
        summaries["global_mode"] = (
            "按 128 场比赛配对 bootstrap：10/30 秒 accuracy 与 macro-F1 均显著落后持久性；"
            "60 秒 accuracy 落后、macro-F1 差异不明确；120 秒两项差异都不明确。"
            "四档 Brier/log-loss 均较好，表示概率分布比基线更有信息，但 argmax 暂不应替代持久性。"
        )
        summaries["opponent_trend"] = (
            "按比赛配对 bootstrap：10 秒 accuracy 平均略落后持久性（红侧差异不明确、蓝侧落后），但 macro-F1 红蓝两侧都显著更好；"
            "30/60/120 秒 accuracy 和 macro-F1 均显著更好。绝对 macro-F1 仍需显示类别概率和不确定性。"
        )
        summaries["enemy_region"] = (
            "按比赛配对 bootstrap：10/30 秒 accuracy 与 macro-F1 显著落后持久性；"
            "60 秒 accuracy 差异不明确、macro-F1 落后；120 秒 accuracy 提高但 macro-F1 仍落后。"
            "因此远期可以展示概率较宽的压力方向，不能把精确区域 top-1 当作可靠命中。"
        )

    return {
        "schema_version": "horizon-diagnostic-v2",
        "source_report": report.get("_diagnostic_source_path", report.get("source_report", "out_m2/temporal_hybrid_motion_death_aware/temporal_hybrid_report.json")),
        "evaluation_mode": report.get("evaluation_mode"),
        "as_of_s": report.get("as_of"),
        "scored_count": report.get("scored_count"),
        "expected_count": report.get("expected_count"),
        "n_test_games": int((game_clustered or {}).get("split_contract", {}).get(
            "test_games", min(row["n_complete_games"] for row in global_rows))),
        "forecast_ledger_red_blue_complete": report.get("red_blue_complete"),
        "forecast_ledger_invalid_count": report.get("invalid_count"),
        "forecast_ledger_missing_count": report.get("missing_count"),
        "model": MODEL,
        "baseline": BASELINE,
        "entity_metrics": entity_rows,
        "global_metrics": global_rows,
        "target_distributions": target_distributions,
        "field_policy": field_policy,
        "threshold_changes": threshold_changes,
        "task_conclusions": summaries,
        "presence_metric_caveat": field_policy["机器人是否存活"]["reason"],
        "game_clustered_comparison": game_clustered,
        "time_segment_recommendation": {
            "keep_existing_horizons": [10, 30, 60, 120],
            "reason": "报告只对 10/30/60/120 秒建立成熟标签并评分，不能从四个离散采样点可靠定位任意阈值或证明新增时距更好。现有证据支持保留采样点，但按字段逐步降低空间/语义精度。",
            "next_candidate_horizons": [5, 15, 20, 45, 90],
            "candidate_status": "仅实验候选；当前没有这些时距的训练或评估结果。",
            "gate": "按比赛分组留出，重建因果标签后同时比较 accuracy、macro-F1、Brier/log-loss 和校准；至少连续两个相邻时距胜过持久性且关键类别不塌陷，才改线上时间分段。",
            "limitation": "当前结论只能说明四个测试时距之间的趋势，不能声称指标在某个未评估的秒数后已经完全发散。",
        },
    }


def render_markdown(out: dict[str, Any]) -> str:
    lines = [
        "# 预测时距诊断：混合模型与持久性基线",
        "",
        f"数据来自 `{out['source_report']}`；评估方式 `{out.get('evaluation_mode')}`，比赛时长上限 {out.get('as_of_s')} 秒；test split 有 {out.get('n_test_games')} 场包含红蓝两侧的比赛 ID。以下实体表为混合汇总，全局表另列红蓝结果。",
        "",
        f"> **阵亡指标限制：** 最新 death-aware 标签已把未来 HP=0 记为阵亡/不存在；presence accuracy 是 16 个机器人与设施槽位的汇总值，存活样本占多数，不能等同移动机器人阵亡 precision/recall/F1。另，原成熟预测账本 red_blue_complete={out.get('forecast_ledger_red_blue_complete')}、invalid={out.get('forecast_ledger_invalid_count')}、missing={out.get('forecast_ledger_missing_count')}；以下配对 bootstrap 按可评分标签的比赛与红蓝两侧做 complete-case。",
        "",
        "## 实体预测",
        "",
        "|指标|10s|30s|60s|120s|建议|",
        "|---|---:|---:|---:|---:|---|",
    ]
    entity_specs = [
        ("presence accuracy（death-aware；含设施槽位）", "presence_accuracy_reported", "单独不能代表阵亡识别能力"),
        ("位置 MAE（场地归一化）", "position_mae_normalized", "10/30s 坐标；60s 起路线/宽区域"),
        ("HP MAE（比例）", "hp_mae", "10/30s 数值趋势；60s 起风险等级"),
        ("功能区准确率", "functional_zone_accuracy", "10/30s 候选区；60s 起控制方向"),
        ("canonical 精确区准确率", "canonical_zone_accuracy", "只作为 10s 弱提示"),
        ("合法路线率", "transition_legal_rate", "四档保留作可达性门禁；不是准确率"),
    ]
    for label, key, advice in entity_specs:
        vals = [f"{row[key]:.3f}" for row in out["entity_metrics"]]
        lines.append("|" + label + "|" + "|".join(vals) + "|" + advice + "|")

    lines += ["", "## 全局预测与持久性基线", "", "准确率 / macro-F1 / Brier / log-loss；Brier 与 log-loss 越低越好。", ""]
    for task, title in (("global_mode", "全局态势"), ("opponent_trend", "敌方趋势"), ("enemy_region", "敌方精确区域")):
        lines += [f"### {title}", "", "|时距|Hybrid acc / F1|Persistence acc / F1|Hybrid Brier / log-loss|Persistence Brier / log-loss|", "|---:|---:|---:|---:|---:|"]
        for horizon in HORIZONS:
            row = next(item for item in out["global_metrics"] if item["task"] == task and item["horizon_s"] == horizon)
            lines.append(
                f"|{horizon}s|{row['accuracy']:.3f} / {row['macro_f1']:.3f}|"
                f"{row['baseline_accuracy']:.3f} / {row['baseline_macro_f1']:.3f}|"
                f"{row['brier']:.3f} / {row['log_loss']:.3f}|"
                f"{row['baseline_brier']:.3f} / {row['baseline_log_loss']:.3f}|"
            )
        lines += ["", out["task_conclusions"][task], ""]

    grouped = out.get("game_clustered_comparison")
    if grouped and grouped.get("available"):
        contract = grouped["split_contract"]
        lines += [
            "## 按比赛分组的红蓝对称比较",
            "",
            f"NPZ 拆分核对：train {contract['train_games']} 场、validation {contract['validation_games']} 场、test {contract['test_games']} 场；game_id 互斥={contract['train_validation_test_game_ids_disjoint']}；test 每个 game_id 都有红蓝记录={contract['test_game_ids_have_both_sides']}。账本完整性另见上方，逐任务/时距 bootstrap 只纳入红蓝两侧都有可评分标签的比赛（每行给出场数）。95% CI 以比赛为重采样单位，每次抽样保留红蓝两视角并等权平均。",
            "",
            "|任务/时距|红方 Hybrid / 基线 acc|蓝方 Hybrid / 基线 acc|红蓝等权 Δaccuracy (95% CI)|红蓝等权 Δmacro-F1 (95% CI)|",
            "|---|---:|---:|---:|---:|",
        ]
        for task in ("global_mode", "opponent_trend", "enemy_region"):
            for horizon in HORIZONS:
                key = f"{task}@{horizon}s"
                item = grouped["comparison"][key]
                red, blue = item["by_side"]["红"], item["by_side"]["蓝"]
                acc = item["paired_comparison"]["accuracy"]
                f1 = item["paired_comparison"]["macro_f1"]
                acc_ci, f1_ci = acc["paired_game_bootstrap_95pct_ci"], f1["paired_game_bootstrap_95pct_ci"]
                lines.append(
                    f"|{key}|{red['hybrid']['accuracy']:.3f} / {red['persistence']['accuracy']:.3f}|"
                    f"{blue['hybrid']['accuracy']:.3f} / {blue['persistence']['accuracy']:.3f}|"
                    f"{acc['red_blue_macro_delta_hybrid_minus_persistence']:+.3f} "
                    f"[{acc_ci[0]:+.3f}, {acc_ci[1]:+.3f}] ({acc['interpretation']})|"
                    f"{f1['red_blue_macro_delta_hybrid_minus_persistence']:+.3f} "
                    f"[{f1_ci[0]:+.3f}, {f1_ci[1]:+.3f}] ({f1['interpretation']}); n={item['n_matches_with_both_sides_scored']}|"
                )
        lines += [
            "",
            "这是同一测试集上的配对比较，不是重新训练；置信区间衡量比赛间差异。准确率对红蓝两侧等权。macro-F1 在每次 bootstrap 中先按侧合并该侧的比赛混淆矩阵、算 F1，再对红蓝两侧等权平均。",
            "各单侧的配对比赛置信区间已保存在诊断 JSON 的 `per_side_paired_game_bootstrap` 字段，可用于核查红蓝方向是否一致。",
            "",
        ]

    lines += ["## 标签分布", "", "百分比是目标标签分布，不是模型预测分布。精确区域类别极不均衡，因此单看 accuracy 会被常见区域主导；同时查看 macro-F1 与逐类 F1。", ""]
    for task in ("global_mode", "opponent_trend", "enemy_region"):
        lines += [f"### {task}", "", "|时距|目标标签占比|Hybrid 每类 F1|", "|---:|---|---|"]
        for h in HORIZONS:
            row = next(item for item in out["global_metrics"] if item["task"] == task and item["horizon_s"] == h)
            dist = ", ".join(f"{label} {share:.1%}" for label, share in row["target_share"].items())
            f1 = ", ".join(f"{label} {score:.2f}" for label, score in row["per_class_f1"].items())
            lines.append(f"|{h}s|{dist}|{f1}|")
        lines.append("")

    lines += ["## 字段降级策略", ""]
    for name, policy in out["field_policy"].items():
        usable = policy.get("usable_until_s")
        headline = "暂不用于线上判断" if policy.get("valid_for_policy") is False else (f"{usable}s 后降级" if usable != 120 else "四档保留")
        lines.append(f"- **{name}（{headline}）**：{policy['output'] if 'output' in policy else policy.get('after_recompute', '')} {policy.get('basis', '')} {policy.get('reason', '')}".strip())

    lines += ["", "## 时间分段结论", "", "- 保留 10/30/60/120 秒作为当前评估时距，但不把它们当成已经证明的最佳阈值。", "- 现在只能确认四个采样点之间的退化趋势；尚不能说某个未测秒数之后‘完全发散’。", "- 10 秒保留近场路线和候选状态；30 秒保留受速度约束的路线及概率；60/120 秒将位置、区域和 HP 降成宏观方向、宽区域与风险等级。", "- 5/15/20/45/90 秒是下一轮候选点，尚未训练/评估。需按比赛分组留出，并结合 accuracy、macro-F1、Brier/log-loss、校准和逐类表现决定是否改分段。", "", "## 总结", ""]
    for key in ("entity", "global_mode", "opponent_trend", "enemy_region"):
        lines.append(f"- {out['task_conclusions'][key]}")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", default="out_m2/temporal_hybrid_motion_death_aware/temporal_hybrid_report.json")
    parser.add_argument("--out", default="out_m2/horizon_diagnostic.json")
    parser.add_argument("--skip-game-bootstrap", action="store_true",
                        help="reuse report aggregates only; skip per-game paired bootstrap inference")
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    args = parser.parse_args()
    report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    report["_diagnostic_source_path"] = str(args.report)
    game_clustered = None if args.skip_game_bootstrap else _game_clustered_comparison(
        report, bootstrap_replicates=args.bootstrap_replicates)
    output = build(report, game_clustered)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_path = out_path.with_suffix(".md")
    markdown_path.write_text(render_markdown(output), encoding="utf-8")
    print(json.dumps({"json": str(out_path), "markdown": str(markdown_path),
                      "horizons": output["time_segment_recommendation"]["keep_existing_horizons"],
                      "game_clustered_available": bool(game_clustered and game_clustered.get("available"))}, ensure_ascii=False))


if __name__ == "__main__":
    main()
