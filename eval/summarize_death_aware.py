"""Write a concise, reproducible summary of the death-aware candidate report."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def build(report: dict) -> str:
    horizons = (10, 30, 60, 120)
    lines = [
        "# HP=0 阵亡标签修正后的预测评估",
        "",
        f"评估来自 `{report.get('data_contract', {}).get('test_npz', 'test.npz')}`，候选 checkpoint 使用同一 329 维 motion 特征、context=40、hidden=192、3 层、4 heads、dropout=0.15、3 epochs、CUDA。总评分只作辅助，逐字段结果优先。",
        "",
        "> HP=0 的移动单位在未来标签中标为阵亡/不存在；此处 presence 汇总 16 个机器人和设施槽位，因此不是专门的死亡识别 F1。线上只在当前明确读到 HP=0 时停止预测；缺测仍是未知。",
        "",
        "## 实体预测",
        "",
        "|时距|presence acc（含设施槽位）|机器人兵种槽位宏平均 acc|位置 MAE（归一化）|HP MAE|功能区 acc|精确区 acc|合法路线率|",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    metrics = report["entity_forecast_metrics"]
    for horizon in horizons:
        item = metrics[f"{horizon}s"]
        robot_slots = [value for key, value in item["by_slot"].items()
                       if not any(name in key for name in ("基地", "前哨站"))]
        robot_macro = sum(value["presence_accuracy"] for value in robot_slots) / max(1, len(robot_slots))
        lines.append(
            f"|{horizon}s|{item['presence_accuracy']:.3f}|{robot_macro:.3f}|"
            f"{item['position_mae_normalized']:.3f}|{item['hp_mae']:.3f}|"
            f"{item['functional_zone_accuracy']:.3f}|{item['canonical_zone_accuracy']:.3f}|"
            f"{item['transition_legal_rate']:.3f}|"
        )
    lines += [
        "",
        "机器人槽位宏平均将兵种槽位等权处理，会被大量存活样本主导，不能解释成阵亡召回率。下一次评估应补充死亡 precision/recall/F1 和逐兵种混淆矩阵。",
        "",
        "## 全局硬分类准确率与持久性基线",
        "",
        "|任务|10s 候选/基线|30s 候选/基线|60s 候选/基线|120s 候选/基线|",
        "|---|---:|---:|---:|---:|",
    ]
    for task in ("global_mode", "opponent_trend", "enemy_region"):
        cells = []
        for horizon in horizons:
            key = f"{task}@{horizon}s"
            candidate = report["accuracy_by_task_horizon"]["temporal_hybrid_v1"][key]["accuracy"]
            baseline = report["accuracy_by_task_horizon"]["persistence_v1"][key]["accuracy"]
            cells.append(f"{candidate:.3f} / {baseline:.3f}")
        lines.append(f"|{task}|" + "|".join(cells) + "|")
    scores = report["weighted_acceptance"]
    lines += [
        "",
        f"业务加权汇总：候选 global score `{scores['temporal_hybrid_v1']['global_score']:.3f}`，持久性 `{scores['persistence_v1']['global_score']:.3f}`；实体 score 两者均为 `{scores['temporal_hybrid_v1']['entity_score']:.3f}`。候选整体只小幅领先，且全局态势硬分类四个时距都落后持久性。",
        "",
        "## 时距用法",
        "",
        "暂时保留 10/30/60/120 秒作为评估采样点；10 秒显示位置/特殊状态候选，30 秒展示速度可达的局部路线，60/120 秒改成宽区域、侧翼压力和总体趋势。功能区、精确区、HP 数值均应显示概率/不确定性，不能用单类准确率替代决策效果。5/15/20/45/90 秒尚未评估，不能声称某个时距之后完全发散。",
        "",
        "模型文件、完整 JSON 指标和训练日志见同目录 `temporal_hybrid_v1.pt`、`temporal_hybrid_report.json`。",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path,
                        default=Path("out_m2/temporal_hybrid_motion_death_aware/temporal_hybrid_report.json"))
    parser.add_argument("--out", type=Path,
                        default=Path("out_m2/temporal_hybrid_motion_death_aware/death_aware_summary.md"))
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    args.out.write_text(build(report), encoding="utf-8")
    print(json.dumps({"report": str(args.report), "summary": str(args.out)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
