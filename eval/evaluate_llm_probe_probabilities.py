"""Row-level audit for the frozen paired Qwen/Open-JEV RM mode probe.

This script evaluates only the saved global-mode probability forecasts.  It
does not evaluate language-model action advice, fit calibration/thresholds, or
touch training data.  Paired percentile intervals resample whole games so that
both sides and repeated timestamps from a match remain clustered together.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from forecast.temporal import HORIZONS, MODES, baseline_labels  # noqa: E402

DEFAULT_QWEN = ROOT / "out_m2/temporal_forecast/qwen_probe_full_128_context2.json"
DEFAULT_JEV = ROOT / "out_m2/temporal_forecast/open_jev_probe_full_128_context2_cache.json"
DEFAULT_DATA = ROOT / "out_m2/forecast_dataset/test.jsonl.gz"
DEFAULT_JSON = ROOT / "out_m2/temporal_forecast/llm_probe_probability_audit.json"
DEFAULT_MD = ROOT / "docs/analysis/2026-09-27-llm-probe-probability-audit.md"
SIDES = ("红", "蓝")
METHODS = ("persistence", "qwen", "open_jev")
SCORE_NAMES = ("accuracy", "balanced_accuracy", "macro_f1", "brier", "log_loss", "top_label_ece")


def _sample_key(row):
    return (int(row["game_id"]), str(row["side"]), float(row["issued_at"]))


def _read_probe(path: Path, model_name: str):
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise ValueError(f"{path}: expected a row list")
    by_key = {}
    for row in rows:
        key = _sample_key(row)
        if key in by_key:
            raise ValueError(f"{path}: duplicate sample key {key}")
        if row.get("side") not in SIDES:
            raise ValueError(f"{path}: unsupported side for {key}: {row.get('side')!r}")
        probs_by_h = row.get("probabilities")
        labels_by_h = row.get("label")
        if not isinstance(probs_by_h, dict) or not isinstance(labels_by_h, dict):
            raise ValueError(f"{path}: missing probabilities/labels for {key}")
        for horizon in HORIZONS:
            p = np.asarray(probs_by_h.get(str(horizon)), dtype=float)
            label = labels_by_h.get(str(horizon))
            if p.shape != (len(MODES),) or not np.isfinite(p).all():
                raise ValueError(f"{path}: invalid {horizon}s probability vector for {key}")
            if np.any(p < 0.0) or np.any(p > 1.0) or not np.isclose(p.sum(), 1.0, atol=1e-5):
                raise ValueError(f"{path}: probabilities out of range or not normalized for {key}/{horizon}")
            if label not in MODES:
                raise ValueError(f"{path}: unknown label {label!r} for {key}/{horizon}")
        by_key[key] = row
    return payload, by_key


def _read_current_modes(path: Path):
    """Get persistence-class forecasts from the exact observable test rows."""
    import gzip

    current = {}
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            row = json.loads(line)
            key = _sample_key(row)
            if key in current:
                raise ValueError(f"{path}:{line_no}: duplicate current-observation key {key}")
            current[key] = baseline_labels(row)["global_mode"]
    return current


def _metrics(rows, horizon: int):
    if not rows:
        return None
    y = np.asarray([MODES.index(r["label"][str(horizon)]) for r in rows], dtype=int)
    probs = np.asarray([r["probabilities"][str(horizon)] for r in rows], dtype=float)
    pred = probs.argmax(axis=1)
    n_class = len(MODES)
    cm = np.zeros((n_class, n_class), dtype=np.int64)
    np.add.at(cm, (y, pred), 1)
    support = cm.sum(axis=1)
    predicted = cm.sum(axis=0)
    tp = np.diag(cm).astype(float)
    recall = np.divide(tp, support, out=np.zeros(n_class), where=support > 0)
    precision = np.divide(tp, predicted, out=np.zeros(n_class), where=predicted > 0)
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros(n_class), where=(precision + recall) > 0)
    one_hot = np.eye(n_class, dtype=float)[y]
    correct = pred == y
    confidence = probs.max(axis=1)

    # Equal-width top-label ECE. Full bin payload is emitted for inspection.
    bins = []
    ece = 0.0
    for bin_idx in range(10):
        lo, hi = bin_idx / 10.0, (bin_idx + 1) / 10.0
        mask = (confidence >= lo) & ((confidence < hi) if bin_idx < 9 else (confidence <= hi))
        count = int(mask.sum())
        if count:
            bin_acc = float(correct[mask].mean())
            bin_conf = float(confidence[mask].mean())
            ece += count / len(rows) * abs(bin_acc - bin_conf)
        else:
            bin_acc = bin_conf = None
        bins.append({"lower": lo, "upper": hi, "count": count,
                     "mean_confidence": bin_conf, "accuracy": bin_acc})

    return {
        "n": int(len(rows)),
        "accuracy": float(correct.mean()),
        "balanced_accuracy": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "per_class": {
            MODES[i]: {"support": int(support[i]), "predicted": int(predicted[i]),
                       "precision": float(precision[i]), "recall": float(recall[i]),
                       "f1": float(f1[i])}
            for i in range(n_class)
        },
        "confusion_matrix_truth_rows_predicted_columns": cm.tolist(),
        "brier": float(np.mean(np.sum((probs - one_hot) ** 2, axis=1))),
        "log_loss": float(-np.mean(np.log(np.clip(probs[np.arange(len(y)), y], 1e-15, 1.0)))),
        "top_label_ece": float(ece),
        "mean_top_confidence": float(confidence.mean()),
        "reliability_bins": bins,
    }


def _all_views(rows_by_key, horizon: int):
    rows = list(rows_by_key.values())
    out = {"pooled": _metrics(rows, horizon)}
    for side in SIDES:
        out[side] = _metrics([r for r in rows if r["side"] == side], horizon)
    out["side_macro"] = {
        name: float(np.mean([out[side][name] for side in SIDES]))
        for name in SCORE_NAMES
    }
    return out


def _bootstrap(rows_by_key, horizon: int, game_draws):
    game_rows = {}
    for row in rows_by_key.values():
        game_rows.setdefault(int(row["game_id"]), []).append(row)
    draws = {"pooled": {m: [] for m in SCORE_NAMES},
             "side_macro": {m: [] for m in SCORE_NAMES},
             **{side: {m: [] for m in SCORE_NAMES} for side in SIDES}}
    for draw in game_draws:
        sampled = [row for gid in draw for row in game_rows[int(gid)]]
        views = {"pooled": _metrics(sampled, horizon)}
        for side in SIDES:
            views[side] = _metrics([r for r in sampled if r["side"] == side], horizon)
        if any(views[s] is None for s in SIDES):
            continue
        for name in SCORE_NAMES:
            views["side_macro"] = views.get("side_macro", {})
            views["side_macro"][name] = float(np.mean([views[s][name] for s in SIDES]))
        for view in draws:
            for metric in SCORE_NAMES:
                draws[view][metric].append(views[view][metric])
    result = {}
    for view, metrics in draws.items():
        result[view] = {}
        for name, vals in metrics.items():
            if not vals:
                result[view][name] = {"ci95": [None, None], "valid_replicates": 0}
            else:
                result[view][name] = {
                    "ci95": [float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))],
                    "valid_replicates": len(vals),
                }
    return result


def _paired_delta(rows_a, rows_b, horizon: int, game_draws):
    """A-minus-B paired metric deltas using identical resampled game clusters."""
    ids = set(rows_a) & set(rows_b)
    if ids != set(rows_a) or ids != set(rows_b):
        raise ValueError("paired delta requires exact matching sample-key sets")
    point_a, point_b = _all_views(rows_a, horizon), _all_views(rows_b, horizon)
    result = {}
    game_ids = set(key[0] for key in ids)
    # Keep row-wise matching inside the bootstrap replicate. We pass the same
    # duplicated game draw to both systems, preserving pairing and clustering.
    by_game_a, by_game_b = {}, {}
    for key in ids:
        by_game_a.setdefault(key[0], []).append(rows_a[key])
        by_game_b.setdefault(key[0], []).append(rows_b[key])
    samples = {view: {metric: [] for metric in SCORE_NAMES}
               for view in ("pooled", "side_macro", *SIDES)}
    for draw in game_draws:
        arows = [r for gid in draw for r in by_game_a.get(int(gid), [])]
        brows = [r for gid in draw for r in by_game_b.get(int(gid), [])]
        a, b = {"pooled": _metrics(arows, horizon)}, {"pooled": _metrics(brows, horizon)}
        for side in SIDES:
            a[side] = _metrics([r for r in arows if r["side"] == side], horizon)
            b[side] = _metrics([r for r in brows if r["side"] == side], horizon)
        if any(a[s] is None or b[s] is None for s in SIDES):
            continue
        a["side_macro"], b["side_macro"] = {}, {}
        for metric in SCORE_NAMES:
            a["side_macro"][metric] = float(np.mean([a[s][metric] for s in SIDES]))
            b["side_macro"][metric] = float(np.mean([b[s][metric] for s in SIDES]))
        for view in samples:
            for metric in SCORE_NAMES:
                samples[view][metric].append(a[view][metric] - b[view][metric])
    for view in samples:
        result[view] = {}
        for metric in SCORE_NAMES:
            vals = samples[view][metric]
            result[view][metric] = {
                "delta_a_minus_b": float(point_a[view][metric] - point_b[view][metric]),
                "ci95": ([float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))]
                         if vals else [None, None]),
                "valid_replicates": len(vals),
                "better_direction": "higher" if metric in ("accuracy", "balanced_accuracy", "macro_f1") else "lower",
            }
    return result


def _add_persistence_rows(qwen, jev, current_modes):
    output = {}
    for method, source in (("qwen", qwen), ("open_jev", jev)):
        cloned = {}
        for key, row in source.items():
            cloned[key] = dict(row)
        output[method] = cloned
    persistence = {}
    for key, qrow in qwen.items():
        mode = current_modes.get(key)
        if mode is None:
            continue
        persistence[key] = {
            "game_id": qrow["game_id"], "side": qrow["side"], "issued_at": qrow["issued_at"],
            "probabilities": {str(h): [1.0 if m == mode else 0.0 for m in MODES] for h in HORIZONS},
            "label": qrow["label"],
        }
    output["persistence"] = persistence
    return output


def _summarize_class_collapse(rows, horizon):
    by_label = Counter(r["label"][str(horizon)] for r in rows.values())
    by_prediction = Counter(MODES[int(np.argmax(r["probabilities"][str(horizon)]))]
                            for r in rows.values())
    return {
        "truth_counts": dict(by_label), "prediction_counts": dict(by_prediction),
        "single_class_collapse": len(by_prediction) == 1,
        "prediction_entropy_nats": float(-sum((n / len(rows)) * math.log(n / len(rows))
                                              for n in by_prediction.values())),
    }


def _report_markdown(report):
    lines = [
        "# Qwen / Open-JEV 逐行概率评估",
        "",
        "本报告只评估全局态势三分类概率，不评估动作建议质量、控制器执行或因果收益。128 个冻结配对样本来自项目 test 子集的选择器；它不是外部测试集。没有拟合校准器或阈值。置信区间按整场比赛聚类 bootstrap，红蓝两侧在每场内一起重采样。",
        "",
        f"样本：{report['n_rows']} 行、{report['n_games']} 场、红蓝各 {report['side_counts'].get('红', 0)}/{report['side_counts'].get('蓝', 0)} 行。类别顺序：" + "、".join(MODES) + "。",
        "",
        "## 等权红蓝主指标",
        "",
        "| 时距 | 模型 | Acc [95% CI] | Balanced Acc | Macro-F1 | Brier | LogLoss | Top-label ECE |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for h in HORIZONS:
        for method in METHODS:
            view = report["horizons"][str(h)]["methods"][method]["point_estimate"]["side_macro"]
            ci = report["horizons"][str(h)]["methods"][method]["game_bootstrap_95ci"]["side_macro"]["accuracy"]["ci95"]
            lines.append(f"| {h}s | {method} | {view['accuracy']:.3f} [{ci[0]:.3f}, {ci[1]:.3f}] | {view['balanced_accuracy']:.3f} | {view['macro_f1']:.3f} | {view['brier']:.3f} | {view['log_loss']:.3f} | {view['top_label_ece']:.3f} |")
        lines.append("")
    lines += [
        "## 对持久性基线的配对差值",
        "",
        "差值为模型减 persistence；正准确率差更好，负 Brier 差更好。95% CI 跨 0 表示该样本上差异不稳定。完整六项指标（含红蓝分别值）的 Qwen/Jev、Qwen/persistence、Jev/persistence 配对差都在 JSON。",
        "",
        "| 时距 | 模型 | Acc 差 [95% CI] | Macro-F1 差 [95% CI] | Brier 差 [95% CI] |",
        "|---:|---|---:|---:|---:|",
    ]
    for h in HORIZONS:
        for method in ("qwen", "open_jev"):
            d = report["horizons"][str(h)]["paired_deltas"][f"{method}_minus_persistence"]["side_macro"]
            cells = []
            for metric in ("accuracy", "macro_f1", "brier"):
                x = d[metric]
                cells.append(f"{x['delta_a_minus_b']:+.3f} [{x['ci95'][0]:+.3f}, {x['ci95'][1]:+.3f}]")
            lines.append(f"| {h}s | {method} | {cells[0]} | {cells[1]} | {cells[2]} |")
        lines.append("")
    lines += [
        "Qwen 与 Open-JEV 的逐行类别 argmax 完全相同，四个时距的 Acc / Macro-F1 / Balanced Acc 差均为 0；Open-JEV 的 Brier / LogLoss / ECE 明显优于 Qwen，主要说明概率较平缓，分类没有变好。Persistence 是硬 one-hot，LogLoss 对其错例惩罚很重，不把该项与软概率模型直接解释为排序证据。",
    ]
    lines += [
        "## 红蓝表现与预测类别",
        "",
        "| 时距 | 模型 | 红方 Acc | 蓝方 Acc | 红蓝等权 Acc | 预测类别 |",
        "|---:|---|---:|---:|---:|---|",
    ]
    for h in HORIZONS:
        for method in METHODS:
            v = report["horizons"][str(h)]["methods"][method]["point_estimate"]
            pred_counts = report["horizons"][str(h)]["methods"][method]["class_counts"]["prediction_counts"]
            pred_summary = ", ".join(f"{label}:{count}" for label, count in pred_counts.items())
            lines.append(f"| {h}s | {method} | {v['红']['accuracy']:.3f} | {v['蓝']['accuracy']:.3f} | {v['side_macro']['accuracy']:.3f} | {pred_summary} |")
        lines.append("")
    lines += [
        "以上模型概率逐行检查发现，Qwen 与 Open-JEV 在四个 horizon 都把全部 128 行判成‘进攻’；因此两模型的 macro-F1 / balanced accuracy 约为随机三类水平，准确率只是各时距‘进攻’真值占比。两模型红蓝准确率差异很大，且蓝方远低于红方；不能只看等权总准确率。每个 horizon/model 的 truth/prediction counts、预测熵、混淆矩阵及 10 个等宽 top-label ECE 可靠性分箱均保存在 JSON。可靠性和红蓝差异的置信区间需考虑样本仅 22 场的限制。",
        "",
        "## 限制",
        "",
        "- 128 行来自 22 场，点估计虽然可复算，但比赛聚类 CI 仍较宽；不把逐行数当作 128 个独立比赛。",
        "- Persistence 概率是把当前可观测状态按同一 `baseline_labels` 三态规则变成 one-hot，并假定未来保持该态；同标签版本 `relative_margin_v2`。这是概率基线的确定性极限，LogLoss 使用 1e-15 clip，因此错例会产生很大惩罚。",
        "- Accuracy/类别指标衡量项目定义的未来全局态势标签，不衡量建议的动作是否正确。测试集分数不能用于再调模型或校准器。",
        "- ECE 分箱不稳定且依赖分箱边界；同时报告 Brier、LogLoss 和分箱表，不单独以 ECE 排名。",
    ]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qwen", type=Path, default=DEFAULT_QWEN)
    parser.add_argument("--open-jev", type=Path, default=DEFAULT_JEV)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--markdown-out", type=Path, default=DEFAULT_MD)
    parser.add_argument("--bootstrap-replicates", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=20260927)
    args = parser.parse_args()
    if args.bootstrap_replicates < 100:
        raise ValueError("use at least 100 game bootstrap replicates")

    qmeta, qwen = _read_probe(args.qwen, "qwen")
    jmeta, jev = _read_probe(args.open_jev, "open_jev")
    qkeys, jkeys = set(qwen), set(jev)
    if qkeys != jkeys:
        raise ValueError(f"Qwen/Open-JEV keys differ: qwen-only={len(qkeys-jkeys)}, jev-only={len(jkeys-qkeys)}")
    for key in sorted(qkeys):
        for horizon in HORIZONS:
            if qwen[key]["label"][str(horizon)] != jev[key]["label"][str(horizon)]:
                raise ValueError(f"future label mismatch at {key}/{horizon}s")

    current = _read_current_modes(args.dataset)
    if not qkeys <= set(current):
        raise ValueError(f"{len(qkeys-set(current))} selected rows do not have exact current-observation keys")
    qwen, jev = ({k: v for k, v in qwen.items()}, {k: v for k, v in jev.items()})
    methods = _add_persistence_rows(qwen, jev, current)
    game_ids = sorted({int(k[0]) for k in qkeys})
    rng = np.random.default_rng(args.seed)
    game_draws = [rng.choice(game_ids, size=len(game_ids), replace=True).tolist()
                  for _ in range(args.bootstrap_replicates)]

    results = {}
    for horizon in HORIZONS:
        horizon_result = {"methods": {}, "paired_deltas": {}}
        for method in METHODS:
            mrows = methods[method]
            horizon_result["methods"][method] = {
                "point_estimate": _all_views(mrows, horizon),
                "game_bootstrap_95ci": _bootstrap(mrows, horizon, game_draws),
                "class_counts": _summarize_class_collapse(mrows, horizon),
            }
        for a, b in (("qwen", "persistence"), ("open_jev", "persistence"), ("open_jev", "qwen")):
            horizon_result["paired_deltas"][f"{a}_minus_{b}"] = _paired_delta(
                methods[a], methods[b], horizon, game_draws)
        results[str(horizon)] = horizon_result

    report = {
        "schema": "llm-probe-probability-audit-v1",
        "task": "global_mode_forecast_only",
        "created_by": "eval/evaluate_llm_probe_probabilities.py",
        "sources": {"qwen": str(args.qwen), "open_jev": str(args.open_jev), "dataset": str(args.dataset)},
        "input_claim": "saved paired rows selected from this project's test split; not an external test set",
        "n_rows": len(qkeys), "n_games": len(game_ids),
        "side_counts": dict(Counter(qwen[k]["side"] for k in qkeys)),
        "game_ids": game_ids,
        "sample_keys_exactly_paired": True,
        "future_labels_exactly_paired_for_all_horizons": True,
        "current_baseline_exact_key_coverage": len(qkeys & set(current)) / len(qkeys),
        "class_order": list(MODES),
        "horizons_seconds": list(HORIZONS),
        "persistence_definition": "one-hot current observable global_mode from forecast.temporal.baseline_labels; same relative_margin_v2 class semantics, predicts persistence at every horizon",
        "probability_validation": {"all_vectors_finite": True, "range_0_1": True, "sum_tolerance": 1e-5,
                                   "qwen_rows": len(qmeta["rows"]), "open_jev_rows": len(jmeta["rows"])},
        "bootstrap": {"cluster": "game_id (both sides and all sampled timestamps together)",
                      "replicates": args.bootstrap_replicates, "seed": args.seed,
                      "ci": "percentile 95%", "red_blue_point_estimates": "reported separately and equally averaged as side_macro"},
        "no_test_set_fitting": True,
        "horizons": results,
        "limitations": [
            "Metrics evaluate only global-mode probabilities against the forecast task labels, not natural-language or single-unit action recommendations.",
            "The 128-row probe is a selected subset of the project's test split and is not an independent external test.",
            "No calibration or decision threshold was fitted on these rows.",
            "There are only 22 game clusters; percentile intervals may be unstable and should not replace a larger game-level holdout.",
            "Persistence uses a deterministic one-hot distribution; clipped log loss strongly penalizes its errors.",
            "Ten-bin ECE is a sample-sensitive diagnostic; inspect reliability bins and Brier/LogLoss together.",
        ],
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_out.write_text(_report_markdown(report), encoding="utf-8")
    print(json.dumps({"json": str(args.json_out), "markdown": str(args.markdown_out),
                      "n_rows": report["n_rows"], "n_games": report["n_games"],
                      "side_counts": report["side_counts"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
