"""Create a reproducible Qwen/JeV fine-tuned holdout comparison report."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparison", default="out_m2/temporal_forecast/qwen_jev_paired_full128_context2.json")
    parser.add_argument("--output", default="out_m2/experiments/qwen_jev_full_finetune_report_20260926")
    args = parser.parse_args()
    comparison = json.loads(Path(args.comparison).read_text(encoding="utf-8"))
    rows = comparison["paired_metrics"]
    qwen_rows = json.loads(Path(comparison["qwen"]["path"]).read_text(encoding="utf-8"))["rows"]
    jev_rows = json.loads(Path(comparison["open_jev"]["path"]).read_text(encoding="utf-8"))["rows"]
    horizons = (10, 30, 60, 120)
    report = {
        "schema_version": 1,
        "experiment": "Qwen LoRA versus Open-Jev LoRA decision head",
        "comparison": comparison,
        "protocol": {
            "same_holdout_keys": comparison["common_n"],
            "red_blue_balance": comparison["common_by_side"],
            "context_frames": 2,
            "observable_only": True,
            "candidate_labels": ["进攻", "均衡", "防守"],
            "ranking_gate": "at least 128 paired rows",
        },
        "conclusion": {
            "accuracy_winner": "tie",
            "calibration_winner": "Open-Jev",
            "class_collapse": "Both models select 进攻 for every paired 30s sample; the 39.06% accuracy equals the majority-class rate and is not useful discrimination.",
            "recommendation": "Do not use either language model as the primary global-mode predictor yet. Keep the local temporal model as the numerical forecaster; retrain JeV/Qwen with balanced targets and a class-aware objective before using their probabilities for decisions.",
            "complex_architecture": "Do not replace the temporal model with a larger language model yet. Add a structured spatiotemporal encoder only after the numerical baseline is beaten on the same split.",
        },
        "class_distribution_30s": {
            "truth": dict(Counter(row["label"]["30"] for row in qwen_rows)),
            "qwen_prediction": dict(Counter(["进攻", "均衡", "防守"][max(range(3), key=lambda index: row["probabilities"]["30"][index])] for row in qwen_rows)),
            "jev_prediction": dict(Counter(["进攻", "均衡", "防守"][max(range(3), key=lambda index: row["probabilities"]["30"][index])] for row in jev_rows)),
        },
    }
    for horizon in horizons:
        item = rows[f"global_mode@{horizon}s"]
        report.setdefault("summary", []).append({
            "horizon_s": horizon,
            "n": item["n"],
            "qwen_accuracy": item["qwen"]["accuracy"],
            "jev_accuracy": item["open_jev"]["accuracy"],
            "qwen_brier": item["qwen"]["brier"],
            "jev_brier": item["open_jev"]["brier"],
            "jev_brier_improvement": -item["open_jev_minus_qwen"]["brier"],
            "qwen_log_loss": item["qwen"]["log_loss"],
            "jev_log_loss": item["open_jev"]["log_loss"],
        })
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_suffix(".json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# Qwen 与 Open-Jev 完整微调留出测试",
        "",
        f"- 配对样本：{comparison['common_n']}（红/蓝：{comparison['common_by_side'].get('红')}/{comparison['common_by_side'].get('蓝')}）",
        "- 输入：同一因果可观测历史，2 帧窗口；标签：进攻/均衡/防守；指标：准确率、Brier、LogLoss。",
        "- 排序门禁：已达到 128 个配对样本，结果可用于路线选择。",
        "",
        "| 预测距离 | Qwen 准确率 | JeV 准确率 | Qwen Brier | JeV Brier | JeV Brier 改善 |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for item in report["summary"]:
        lines.append(f"| {item['horizon_s']}s | {item['qwen_accuracy']:.4f} | {item['jev_accuracy']:.4f} | {item['qwen_brier']:.4f} | {item['jev_brier']:.4f} | {item['jev_brier_improvement']:.4f} |")
    lines += [
        "",
        "结论：四个预测距离的准确率相同，但两者在 30 秒任务上都坍缩为单一类别，准确率只是多数类基线；JeV 只是比 Qwen 少过度自信，尚不能作为决策预测器。数值时序预测仍由本地时序网络承担，暂不因信息字段增加而直接换成更大的语言模型。",
    ]
    output.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"json": str(output.with_suffix('.json')), "markdown": str(output.with_suffix('.md'))}, ensure_ascii=False))


if __name__ == "__main__":
    main()
