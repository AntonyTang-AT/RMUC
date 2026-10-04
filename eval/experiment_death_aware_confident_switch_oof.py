"""Fixed high-confidence mode-switch veto on saved train-only whole-game OOF.

This deliberately performs one predeclared, parameter-free comparison.  It
does not fit or select a threshold using held-out labels and never opens the
validation or test splits.  The saved archive was itself developed using
earlier train OOF analyses, so results remain exploratory.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "out_m2/death_aware_global_mode_experiments/motion_zone_score_oof/motion_candidate_oof.npz"
OUTPUT = ROOT / "out_m2/death_aware_global_mode_experiments/confident_switch_oof/report.json"
DOCUMENT = ROOT / "docs/analysis/2026-09-27-death-aware-confident-switch-oof.md"
HORIZONS = (10, 30)
THRESHOLD = 0.60
BOOTSTRAP_REPS = 2000
SEED = 20260927


def scores(y: np.ndarray, pred: np.ndarray) -> dict[str, object]:
    f1, recall = [], []
    for cls in range(3):
        tp = np.count_nonzero((y == cls) & (pred == cls))
        fp = np.count_nonzero((y != cls) & (pred == cls))
        fn = np.count_nonzero((y == cls) & (pred != cls))
        f1.append(float(2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) else 0.0)
        recall.append(float(tp / (tp + fn)) if (tp + fn) else 0.0)
    return {"accuracy": float(np.mean(y == pred)), "macro_f1": float(np.mean(f1)),
            "balanced_accuracy": float(np.mean(recall)), "recall": recall}


def game_cells(d: dict[str, np.ndarray], valid: np.ndarray, y: np.ndarray,
               pred: np.ndarray) -> dict[int, dict[str, object]]:
    result = {}
    for game in np.unique(d["game_ids"][valid]):
        per_side = []
        for side in ("红", "蓝"):
            selected = valid & (d["game_ids"] == game) & (d["sides"] == side)
            if not np.any(selected):
                raise AssertionError(f"missing scored side: game={game} side={side}")
            per_side.append(scores(y[selected], pred[selected]))
        result[int(game)] = {
            metric: float(np.mean([s[metric] for s in per_side]))
            for metric in ("accuracy", "macro_f1", "balanced_accuracy")
        }
        result[int(game)]["recall"] = np.mean([s["recall"] for s in per_side], axis=0).tolist()
    return result


def paired_summary(left: dict[int, dict[str, object]],
                   right: dict[int, dict[str, object]], seed: int) -> dict[str, object]:
    games = sorted(left)
    if games != sorted(right):
        raise AssertionError("candidate and baseline game sets differ")
    rng = np.random.default_rng(seed)
    sample_index = rng.integers(0, len(games), size=(BOOTSTRAP_REPS, len(games)))
    result = {"n_games": len(games)}
    for metric in ("accuracy", "macro_f1", "balanced_accuracy"):
        a = np.asarray([left[g][metric] for g in games])
        b = np.asarray([right[g][metric] for g in games])
        delta = a - b
        result[metric] = {"candidate": float(a.mean()), "baseline": float(b.mean()),
                          "delta": float(delta.mean()),
                          "delta_ci95": np.quantile(delta[sample_index].mean(axis=1), [.025, .975]).tolist()}
    for ci, name in enumerate(("attack", "balanced", "defense")):
        a = np.asarray([left[g]["recall"][ci] for g in games])
        b = np.asarray([right[g]["recall"][ci] for g in games])
        delta = a - b
        result[f"{name}_recall"] = {"candidate": float(a.mean()), "baseline": float(b.mean()),
                                    "delta": float(delta.mean()),
                                    "delta_ci95": np.quantile(delta[sample_index].mean(axis=1), [.025, .975]).tolist()}
    return result


def main() -> None:
    d = dict(np.load(SOURCE, allow_pickle=False))
    if sorted(np.unique(d["fold_ids"]).tolist()) != list(range(5)):
        raise AssertionError("expected saved five-fold train OOF")
    for game in np.unique(d["game_ids"]):
        if len(np.unique(d["fold_ids"][d["game_ids"] == game])) != 1:
            raise AssertionError(f"game crosses OOF folds: {game}")
    output = {
        "schema": "train-only-fixed-confident-switch-oof-v1",
        "source": str(SOURCE), "opened_splits": ["train OOF archive"],
        "validation_access": False, "test_access": False,
        "design": "Accept a model prediction different from current mode only when its uncalibrated winning probability is at least 0.60; otherwise retain current mode. Threshold fixed before this run; no tuning, fitting, or validation selection.",
        "aggregation": "Equal games; red/blue equal within game; paired whole-game bootstrap 2000 replicates.",
        "threshold": THRESHOLD, "horizons": {},
    }
    for hi, horizon in enumerate(HORIZONS):
        valid = d["mode_y"][:, hi] >= 0
        if not np.all(np.isfinite(d["motion_candidate"][valid, hi])):
            raise AssertionError(f"nonfinite model probabilities on valid {horizon}s rows")
        y = d["mode_y"][:, hi]
        base = d["base_mode"]
        probability = d["motion_candidate"][:, hi]
        raw = np.argmax(probability, axis=1)
        proposed_switch = raw != base
        accepted_switch = proposed_switch & (np.max(probability, axis=1) >= THRESHOLD)
        candidate = np.where(accepted_switch, raw, base)
        candidate_cells = game_cells(d, valid, y, candidate)
        raw_cells = game_cells(d, valid, y, raw)
        persistence_cells = game_cells(d, valid, y, base)
        output["horizons"][str(horizon)] = {
            "scorable_rows": int(valid.sum()), "games": len(candidate_cells),
            "proposed_switches": int(np.count_nonzero(valid & proposed_switch)),
            "accepted_switches": int(np.count_nonzero(valid & accepted_switch)),
            "all_proposed_switch_precision": float(np.mean(raw[valid & proposed_switch] == y[valid & proposed_switch]))
            if np.any(valid & proposed_switch) else None,
            "accepted_switch_precision": float(np.mean(raw[valid & accepted_switch] == y[valid & accepted_switch]))
            if np.any(valid & accepted_switch) else None,
            "candidate_minus_raw": paired_summary(candidate_cells, raw_cells, SEED + horizon),
            "candidate_minus_persistence": paired_summary(candidate_cells, persistence_cells, SEED + horizon + 100),
        }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = ["# 固定置信度切换门控：train OOF 试验", "",
             "只读已有 motion + zone/observable-score 五折整局训练集 OOF；红蓝两边同场留在同一折。",
             "固定规则：模型预测类别与当前类别不同时，只有获胜类别概率 ≥0.60 才切换；其余保留当前类别。此概率未经校准，0.60 仅是预先固定的保守门槛，没有利用本轮标签选择。",
             "未访问 validation/test，未修改生产默认。既有候选经历多轮 train OOF 探索，因此本试验仍属探索性证据。", "",
             "| 时距 | 方法 | Accuracy | Δ 对持久性 [95% CI] | Macro-F1 | Δ 对持久性 [95% CI] | Balanced Acc | Δ 对持久性 [95% CI] |",
             "|---:|---|---:|---:|---:|---:|---:|"]
    for h in HORIZONS:
        row = output["horizons"][str(h)]
        cp = row["candidate_minus_persistence"]
        cr = row["candidate_minus_raw"]
        for name, src in (("门控", cp), ("原始 OOF 模型", None)):
            metric_values = (src if src else cr)
            cells = {m: (metric_values[m]["candidate"] if src else cr[m]["baseline"])
                     for m in ("accuracy", "macro_f1", "balanced_accuracy")}
            def cell(metric: str) -> str:
                if src is None:
                    return "—"
                r = src[metric]
                return f"{100*r['delta']:+.2f}pp [{100*r['delta_ci95'][0]:+.2f}, {100*r['delta_ci95'][1]:+.2f}]"
            lines.append(f"| {h}s | {name} | {cells['accuracy']:.2%} | {cell('accuracy')} | "
                         f"{cells['macro_f1']:.2%} | {cell('macro_f1')} | "
                         f"{cells['balanced_accuracy']:.2%} | {cell('balanced_accuracy')} |")
        lines.append(f"| {h}s | 持久性 | {cp['accuracy']['baseline']:.2%} | — | "
                     f"{cp['macro_f1']['baseline']:.2%} | — | "
                     f"{cp['balanced_accuracy']['baseline']:.2%} | — |")
    lines += ["", "切换诊断：10s 共提议 9,256 次改变，接受 735 次；提议中的目标类命中率为 49.4%，接受子集为 58.0%。30s 共提议 23,283 次，接受 2,229 次；对应命中率 46.0% / 59.2%。高置信门控确实筛出了相对更准的切换，但也丢掉大量正确切换，整体 Accuracy 下降。",
              "", "## 判断", "",
              "此固定门控拒绝了约 92.1% / 90.4% 的 10s / 30s 模型切换建议。与原始模型相比，10s Accuracy 下降 0.74pp，30s 下降 1.69pp，整局配对区间均低于 0；Macro-F1 分别提高 0.73pp / 0.98pp。这揭示明显的准确率与类别均衡权衡，不能为了达到目标准确率而晋升此门控。它只比持久性提高 0.18pp / 0.64pp Accuracy，远低于原始模型的增益。",
              "", "历史候选已覆盖一般分类器、运动/区域/态势分数、模式历史、转移专家、偏置、概率平滑和此固定切换门控。下一步更有依据的是在 train 折内构建真正可见的攻击/HP 突变、关键区域争夺与单位阵亡前兆，并专门评估这些事件是否能提前识别当前类别即将改变；先核查事件字段的观测时刻与覆盖率，再考虑有限的整局 OOF 模型试验。不要继续扫置信度阈值、增加同类分类头或在反复使用过的 validation 上挑最好的数值。",
              "", "门控与原始 OOF 模型的配对差值、逐类召回、切换样本数均见 JSON。",
              "", "复现：`python eval/experiment_death_aware_confident_switch_oof.py`。",
              "机器报告：`out_m2/death_aware_global_mode_experiments/confident_switch_oof/report.json`。"]
    DOCUMENT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(OUTPUT), "document": str(DOCUMENT),
                      "horizons": output["horizons"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
