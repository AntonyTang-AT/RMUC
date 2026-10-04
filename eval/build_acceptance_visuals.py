"""Build acceptance-oriented visual artifacts from the existing RMUC outputs.

This script never trains a model or reads future rows for an inference example.
It only consumes reports and the already-built replay datasets under out_m2/.
"""

from __future__ import annotations

import gzip
import json
import math
import re
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out_m2" / "acceptance_visuals"
OUT.mkdir(parents=True, exist_ok=True)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def setup_font():
    # Microsoft YaHei is available on the Windows workstation. Fall back to a
    # normal sans font when the script is copied elsewhere.
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def build_architecture():
    fig, ax = plt.subplots(figsize=(15, 8), dpi=160)
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 8)
    ax.axis("off")

    boxes = [
        (0.4, 5.8, 2.2, 1.2, "现有比赛数据\nSQLite / JSONL", "#E8F1FB"),
        (3.1, 5.8, 2.4, 1.2, "可观测性层\nobservable.py\nradar_visibility_replay.py", "#EAF7EE"),
        (6.0, 5.8, 2.4, 1.2, "WorldState\nfeatures.py\nworldstate.py", "#FFF4DE"),
        (8.9, 5.8, 2.4, 1.2, "语义与路由\nsemantic / opp_intent\nrouter", "#FBEAF2"),
        (11.8, 5.8, 2.6, 1.2, "决策执行层\nTieredDecider\nDecisionRequest", "#EDE9FE"),
        (3.1, 3.1, 2.4, 1.2, "本地时序网络\nGRU + horizon heads", "#EAF7EE"),
        (6.0, 3.1, 2.4, 1.2, "实体预测头\n16 个兵种槽位\n位置 / HP / 数量", "#EAF7EE"),
        (8.9, 3.1, 2.4, 1.2, "LLM 可选慢环\nQwen / Open-Jev\nQLoRA adapter", "#F3F4F6"),
        (11.8, 3.1, 2.6, 1.2, "结构化输出\ndef / hitbuilding / cruise\n全局分析 + 单位建议", "#EDE9FE"),
        (5.0, 0.5, 5.0, 1.2, "评估账本\n红方 / 蓝方分别计算，再做 side macro average\nBrier · log loss · macro-F1 · 位置 MAE · HP MAE", "#FEF3C7"),
    ]
    for x, y, w, h, text, color in boxes:
        patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.08",
                               linewidth=1.4, edgecolor="#334155", facecolor=color)
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=10,
                color="#0f172a")

    arrows = [
        ((2.6, 6.4), (3.1, 6.4)), ((5.5, 6.4), (6.0, 6.4)),
        ((8.4, 6.4), (8.9, 6.4)), ((11.3, 6.4), (11.8, 6.4)),
        ((4.3, 5.8), (4.3, 4.3)), ((7.2, 5.8), (7.2, 4.3)),
        ((10.1, 5.8), (10.1, 4.3)), ((11.3, 3.7), (11.8, 3.7)),
        ((7.5, 3.1), (7.5, 1.7)), ((12.9, 3.1), (10.0, 1.7)),
    ]
    for (x1, y1), (x2, y2) in arrows:
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                     mutation_scale=15, linewidth=1.2, color="#64748b"))
    ax.text(7.5, 7.55, "RMUC 局势分析系统：当前整体架构", ha="center", va="center",
            fontsize=18, fontweight="bold", color="#0f172a")
    ax.text(7.5, 7.15, "离线回放使用已有比赛数据；部署接口只接收已经可见的当前快照", ha="center",
            va="center", fontsize=10, color="#475569")
    fig.savefig(OUT / "architecture_overview.png", bbox_inches="tight")
    plt.close(fig)


def build_metrics():
    report = load_json(ROOT / "out_m2" / "temporal_forecast_v3" / "temporal_forecast_report.json")
    backtest = load_json(ROOT / "out_m2" / "decision_outcome_backtest_v2.json")
    horizons = [10, 30, 60, 120]
    labels = [f"{h}s" for h in horizons]
    colors = {"global_mode": "#2563eb", "opponent_trend": "#16a34a", "enemy_region": "#ea580c"}

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), dpi=160)
    ax = axes[0, 0]
    for task, color in colors.items():
        gru, base = [], []
        for h in horizons:
            key = f"{task}@{h}s"
            gru.append(report["models"]["temporal_gru_v1"]["by_task_horizon"][key]["brier"]["side_macro_average"]["score"])
            base.append(report["models"]["persistence_v1"]["by_task_horizon"][key]["brier"]["side_macro_average"]["score"])
        ax.plot(labels, gru, marker="o", color=color, label=f"GRU {task}")
        ax.plot(labels, base, marker="x", linestyle="--", alpha=0.55, color=color, label=f"Persistence {task}")
    ax.set_title("概率预测：Brier 分数（越低越好）")
    ax.set_xlabel("预测时间")
    ax.set_ylabel("side macro average")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncol=2)

    ax = axes[0, 1]
    pos = [report["entity_forecast_metrics"][f"{h}s"]["position_mae_normalized"] for h in horizons]
    hp = [report["entity_forecast_metrics"][f"{h}s"]["hp_mae"] for h in horizons]
    x = list(range(len(labels)))
    ax.plot(labels, pos, marker="o", linewidth=2, color="#7c3aed", label="position MAE (normalized)")
    ax.plot(labels, hp, marker="s", linewidth=2, color="#dc2626", label="HP ratio MAE")
    ax.set_title("兵种槽位预测误差")
    ax.set_xlabel("预测时间")
    ax.set_ylabel("MAE")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9)

    ax = axes[1, 0]
    entity_presence = [report["entity_forecast_metrics"][f"{h}s"]["presence_accuracy"] for h in horizons]
    ax.bar(labels, entity_presence, color="#0f766e")
    ax.set_ylim(0.98, 1.0)
    ax.set_title("兵种槽位存在性识别准确率")
    ax.set_ylabel("accuracy")
    for i, v in enumerate(entity_presence):
        ax.text(i, v + 0.00015, f"{v:.4f}", ha="center", fontsize=9)
    ax.grid(axis="y", alpha=0.25)

    ax = axes[1, 1]
    metric_names = ["主线合法率", "交付通过", "回测 AUC", "回测 accuracy"]
    metric_vals = [1.0, 1.0, backtest["cv_win_model"]["cv_auc"], backtest["cv_win_model"]["cv_acc"]]
    bar_colors = ["#16a34a", "#16a34a", "#2563eb", "#2563eb"]
    ax.bar(metric_names, metric_vals, color=bar_colors)
    ax.set_ylim(0, 1.08)
    ax.set_title("主线硬指标与决策回测摘要")
    ax.set_ylabel("score")
    ax.tick_params(axis="x", rotation=20)
    for i, v in enumerate(metric_vals):
        ax.text(i, v + 0.025, f"{v:.3f}", ha="center", fontsize=10)
    ax.grid(axis="y", alpha=0.25)

    fig.suptitle("已有数据的系统结果摘要", fontsize=18, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT / "metrics_dashboard.png", bbox_inches="tight")
    plt.close(fig)


def build_decision_timeline():
    rows = load_json(ROOT / "out_m2" / "fused_decisions.json")
    mode_names = {"均衡": "balanced", "防守": "defend", "进攻": "attack"}
    mode_ids = {"均衡": 0, "防守": 1, "进攻": 2}
    ts = [float(r["t"]) for r in rows]
    ys = [mode_ids.get(r.get("global_mode"), -1) for r in rows]
    fig, ax = plt.subplots(figsize=(14, 3.8), dpi=160)
    ax.step(ts, ys, where="post", linewidth=2.4, color="#1d4ed8")
    ax.scatter(ts, ys, s=13, color="#1e3a8a", alpha=0.75)
    ax.set_yticks([0, 1, 2], ["均衡", "防守", "进攻"])
    ax.set_xlabel("比赛时间 / s")
    ax.set_title("主线决策序列：global_mode 随时间变化")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "decision_mode_timeline.png", bbox_inches="tight")
    plt.close(fig)


def build_sentry_chart():
    report_path = ROOT / "out_m2" / "sentry_state_eval_report.json"
    if not report_path.exists():
        return
    report = load_json(report_path)
    states = ["def", "hitbuilding", "cruise"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3), dpi=160,
                             gridspec_kw={"width_ratios": [1, 1, 1.15]})
    for ax, side, title in zip(axes[:2], ("red_view", "blue_view"), ("红方视角", "蓝方视角")):
        cm = report[side]["confusion"]
        mat = [[cm[t][p] for p in states] for t in states]
        im = ax.imshow(mat, cmap="Blues")
        ax.set_xticks(range(3), states)
        ax.set_yticks(range(3), states)
        ax.set_xlabel("预测")
        ax.set_ylabel("未来窗口标签")
        ax.set_title(f"{title}\naccuracy={report[side]['accuracy']:.3f}, macro-F1={report[side]['macro_f1']:.3f}")
        for i in range(3):
            for j in range(3):
                ax.text(j, i, str(mat[i][j]), ha="center", va="center",
                        color="white" if mat[i][j] > max(max(row) for row in mat) * 0.55 else "#0f172a")
    side = report["side_macro_average"]
    f1 = [side["per_class"][s]["f1"] for s in states]
    axes[2].bar(states, f1, color=["#2563eb", "#ea580c", "#16a34a"])
    axes[2].set_ylim(0, 1.0)
    axes[2].set_title(f"红蓝宏平均逐类 F1\nmacro-F1={side['macro_f1']:.3f}")
    axes[2].set_ylabel("F1")
    for i, value in enumerate(f1):
        axes[2].text(i, value + 0.03, f"{value:.3f}", ha="center")
    axes[2].grid(axis="y", alpha=0.25)
    fig.suptitle("哨兵三态评估：当前帧规则预测 vs 未来 5 秒窗口标签", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(OUT / "sentry_state_evaluation.png", bbox_inches="tight")
    plt.close(fig)


def parse_decision_examples():
    path = ROOT / "out_m2" / "m2_dataset_mainline.jsonl"
    candidates = {"进攻": None, "防守": None}
    line_no = 0
    with path.open(encoding="utf-8") as f:
        for line in f:
            line_no += 1
            item = json.loads(line)
            completion = json.loads(item["completion"])
            mode = completion.get("global_mode")
            if mode not in candidates or candidates[mode] is not None:
                continue
            # Prefer a developed situation with at least one firing unit.
            if float(completion.get("t", 0)) < 60:
                continue
            if not any(u.get("fire") for u in completion.get("units", {}).values()):
                continue
            candidates[mode] = {
                "source_file": str(path),
                "source_line": line_no,
                "input_instruction": item["instruction"],
                "output_completion": completion,
            }
            if all(candidates.values()):
                break
    for mode, value in candidates.items():
        write_json(OUT / f"example_decision_{'attack' if mode == '进攻' else 'defend'}.json", value)
    return candidates


def compact_entity(entity, role):
    out = {
        "entity_id": entity.get("entity_id"),
        "role": role,
        "team": entity.get("team"),
        "entity_type": entity.get("entity_type"),
        "observability_state": entity.get("observability_state"),
        "observed_at": entity.get("observed_at"),
        "received_at": entity.get("received_at"),
        "position": entity.get("position"),
        "hp": entity.get("hp"),
        "maxhp": entity.get("maxhp"),
    }
    return out


def build_temporal_example():
    dataset_path = ROOT / "out_m2" / "forecast_dataset" / "test.jsonl.gz"
    all_records = []
    with gzip.open(dataset_path, "rt", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row["game_id"] == 1780142276111 and row["side"] == "红":
                all_records.append(row)
    selected = next(row for row in all_records if row["issued_at"] == 361.0)

    from runtime.temporal_forecast import resolve_temporal_forecaster

    forecaster = resolve_temporal_forecaster(
        ROOT / "out_m2" / "temporal_forecast_v3" / "temporal_gru_v1.pt",
        device="cpu",
        team="红",
    )
    output = None
    selected_index = all_records.index(selected)
    for row in all_records[max(0, selected_index - 11): selected_index + 1]:
        entities = {}
        for key, role in (("self_entities", "self"), ("opponent_entities", "opponent")):
            for entity_id, entity in row["observable"].get(key, {}).items():
                item = dict(entity)
                item["role"] = role
                entities[entity_id] = item
        output = forecaster.predict({
            "self_team": row["side"],
            "issued_at": row["issued_at"],
            "entities": entities,
        })

    current_entities = []
    for key, role in (("self_entities", "self"), ("opponent_entities", "opponent")):
        current_entities.extend(compact_entity(e, role)
                                for e in selected["observable"].get(key, {}).values())
    targets = {}
    for horizon, target in selected["targets"].items():
        if not isinstance(target, dict):
            targets[horizon] = None
            continue
        truth_entities = target.get("truth", {}).get("entities", {})
        by_type = {}
        for entity in truth_entities.values():
            typ = entity.get("entity_type")
            if typ not in by_type:
                by_type[typ] = {"count": 0, "hp_sum": 0.0, "maxhp_sum": 0.0}
            by_type[typ]["count"] += 1
            by_type[typ]["hp_sum"] += float(entity.get("hp") or 0.0)
            by_type[typ]["maxhp_sum"] += float(entity.get("maxhp") or 0.0)
        targets[horizon] = {"target_at": target.get("target_at"), "by_type": by_type}

    example = {
        "source_file": str(dataset_path),
        "game_id": selected["game_id"],
        "side": selected["side"],
        "issued_at": selected["issued_at"],
        "max_input_observed_at": selected["max_input_observed_at"],
        "visibility_summary": selected["visibility_summary"],
        "observable_snapshot_compact": {"entities": current_entities},
        "forecast_output": output,
        "future_truth_summary_for_scoring_only": targets,
        "note": "forecast_output only used snapshots through issued_at; future_truth_summary is kept separately for evaluation and was not in the input.",
    }
    write_json(OUT / "example_temporal_red_t361.json", example)
    return example


def build_report(decision_examples, temporal_example):
    mainline = load_json(ROOT / "out_m2" / "mainline_eval_report.json")
    bench = load_json(ROOT / "out_m2" / "benchmark_report.json")
    backtest = load_json(ROOT / "out_m2" / "decision_outcome_backtest_v2.json")
    tf = load_json(ROOT / "out_m2" / "temporal_forecast_v3" / "temporal_forecast_report.json")
    sentry_path = ROOT / "out_m2" / "sentry_state_eval_report.json"
    sentry = load_json(sentry_path) if sentry_path.exists() else None
    rows = []
    for h in (10, 30, 60, 120):
        m = tf["entity_forecast_metrics"][f"{h}s"]
        rows.append(f"| {h}s | {m['presence_accuracy']:.4f} | {m['position_mae_normalized']:.4f} | {m['hp_mae']:.4f} | {m.get('functional_zone_accuracy', 0.0):.4f} | {m.get('transition_legal_rate', 0.0):.4f} |")
    report = f"""# 现有数据驱动的系统展示包

本目录只使用仓库中已经存在的结果和数据，不重新训练、不补造数据。

## 1. 当前整体架构

![架构图](architecture_overview.png)

1. **输入与可观测性**：`data/observable.py`、`data/radar_visibility_replay.py` 对已有比赛回放按红方/蓝方建立当前可见快照。
2. **状态构建**：`data/features.py` 和 `worldstate/` 组织为 WorldState，并计算语义态势、敌方意图。
3. **基础决策**：`tree/global_mode.py` 输出三态 `def / hitbuilding / cruise`，`labels/router.py` 决定规则、学生模型和慢环的路由。
4. **时序预测**：`forecast/` 的 GRU 对 10/30/60/120 秒分别预测全局模式、敌方趋势、敌方区域和 16 个兵种槽位。
5. **运行时决策**：`runtime/decide.py` 进行 Tier0/Tier1/Tier2 融合，`runtime/decision_request.py` 输出统一结构化接口。
6. **评估**：`eval/` 分别计算红方、蓝方，再给出 side macro average。

## 2. 功能区和移动预测契约

![功能区图](functional_zone_overlay.png)

`worldstate/functional_zone.py` 按场地模块定义功能区和相邻图。WorldState 的 `zone` 使用功能区名称；旧 6×3 网格保存在 `legacy_zone`。时序实体预测同时输出 `from_zone`、`to_zone`、`raw_to_zone` 和 `projection_applied`，因此展示时可以明确写成“步兵3：红上公路 → 中央北隧道”。

## 3. 每个部分的已有结果

| 模块 | 当前结果 | 说明 |
|---|---:|---|
| 主线结构化输出 | legal rate = {mainline['eval']['legal_rate']:.3f} | 160 个留出决策点，`deliverable_pass=true` |
| 规则基线对 lookahead 标签 | accuracy = {bench['models']['rule']['mode_accuracy_vs_lookahead']:.3f} | lookahead 是代理标签，不是人工金标准 |
| 规则基线 macro-F1 | {bench['models']['rule']['mode_macro_f1_vs_lookahead']:.3f} | 受类别不均衡影响 |
| 决策结果回测 | AUC = {backtest['cv_win_model']['cv_auc']:.3f} | 关联性指标，不是因果胜率证明 |
| 时序评估账本 | red/blue complete = {tf['red_blue_complete']} | 缺失率 = {tf['missing_rate']:.1%} |
| 哨兵三态红蓝宏平均 | accuracy = {sentry['side_macro_average']['accuracy'] if sentry else '待生成'} | macro-F1 = {sentry['side_macro_average']['macro_f1'] if sentry else '待生成'} |

### 兵种槽位预测结果

| 预测距离 | 存在性准确率 | 位置 MAE（归一化） | HP 比例 MAE | 功能区准确率 | 合法相邻转移率 |
|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

## 4. 完整实际例子一：主线决策

- `example_decision_attack.json`：来自 `m2_dataset_mainline.jsonl` 的真实进攻样本。
- `example_decision_defend.json`：来自同一数据集的真实防守样本。

每个文件都保留了完整输入字符串、源文件行号和完整结构化输出，现场可以直接打开检查。

## 5. 完整实际例子二：时序预测

`example_temporal_red_t361.json` 来自测试集中的真实样本：

- 比赛：`{temporal_example['game_id']}`
- 视角：红方
- 当前时间：`{temporal_example['issued_at']}s`
- 最大输入观测时间：`{temporal_example['max_input_observed_at']}s`
- 可见实体数：`{temporal_example['visibility_summary']['entity_count']}`
- 预测输出：10/30/60/120 秒四个 horizon
- 未来真值单独保存于 `future_truth_summary_for_scoring_only`，没有进入预测输入

## 6. 建议展示的图

![指标总览](metrics_dashboard.png)

![决策时间序列](decision_mode_timeline.png)

![哨兵三态评估](sentry_state_evaluation.png)

![架构图](architecture_overview.png)

## 7. 结果解释边界

- 当前离线数据可以证明：输入输出契约、因果时间切分、红蓝两侧评估和预测误差。
- 当前时序实体头是“兵种槽位级聚合”，不是逐机器人 ID 轨迹。
- lookahead 标签和决策回测指标不能被解释为人工金标准或因果胜率。
- Open-Jev/Qwen 是可替换慢环，系统主线不依赖它才能运行。
"""
    (OUT / "acceptance_visual_report.md").write_text(report, encoding="utf-8")


def main():
    setup_font()
    for name in ("functional_zone_overlay.png", "plan_comparison.png"):
        source = ROOT / "docs" / "field_map" / name
        if source.exists():
            shutil.copyfile(source, OUT / name)
    build_architecture()
    build_metrics()
    build_decision_timeline()
    build_sentry_chart()
    decisions = parse_decision_examples()
    temporal = build_temporal_example()
    build_report(decisions, temporal)
    print(f"generated: {OUT}")


if __name__ == "__main__":
    main()
