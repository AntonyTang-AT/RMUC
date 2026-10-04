# 全局态势精确时距基线修复（validation only）

日期：2026-09-27。修复 `eval/audit_validation_global_horizon.py` 中 current/future 标签合同混用。未训练模型、未读取 test archive、未修改生产默认；旧报告保留原样。

## 修复内容和数据时间

问题在于旧脚本把压缩档保存的 pre-death-aware `base_mode` 用作当前标签，却调用当前工作树的 death-aware `_mode_from_groups` 计算未来标签。这把旧当前类和新未来类直接比较，旧报告不再是统一合同的持久性基线。

修复后，每个 validation 行先在对齐的 radar-masked validation JSONL observable 上重建 issue-time 状态；再对 issue 时刻和精确未来时刻都分别应用同一规则。报告同时输出：

- `legacy`：旧 `relative_margin_v2` 语义，移动兵种不按零 HP 排除。
- `death_aware`：当前规则，移动兵种 HP≤0 或 destroyed 时排除。

原始录像数据每秒一条；预测起点每 3 秒一个。未来标签从 `issued_at + horizon` 的精确整数秒读取，无插值、最近邻、补值。只处理 validation.npz 中 122 场比赛，且 current JSONL 行键逐行对齐；旧合同 current 与压缩档 base_mode 100% 一致，death-aware current 与旧 base_mode 一致率为90.62%，对应标签口径差异。

## 同合同的持久性基线

指标按场均衡、场内红蓝观察方等权；区间对 122 场比赛做 2,000 次配对 bootstrap。Accuracy 就是当前类别保持到未来类别的比例；Macro-F1 固定三类，未出现类别按0分处理。

| 时距 | 旧合同 Accuracy [95% CI] | Macro-F1 | 状态变化率 | 可评分覆盖率 | death-aware Accuracy [95% CI] | Macro-F1 | 状态变化率 | 可评分覆盖率 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 10s | 75.18% [73.12, 77.19] | 54.67% | 24.82% | 97.84% | 72.25% [70.11, 74.38] | 53.16% | 27.75% | 97.84% |
| 30s | 63.71% [60.81, 66.47] | 39.79% | 36.29% | 92.85% | 60.39% [57.62, 63.26] | 38.59% | 39.61% | 92.85% |
| 60s | 57.40% [54.18, 60.51] | 33.14% | 42.60% | 85.70% | 55.89% [52.94, 58.69] | 33.73% | 44.11% | 85.70% |
| 120s | 51.95% [48.21, 55.66] | 28.51% | 48.05% | 71.39% | 50.23% [46.92, 53.66] | 28.72% | 49.77% | 71.39% |

同一合同下，Accuracy 与 Macro-F1 随时距平滑下降，状态变化率增加；没有单个时点之后突然发散。120 秒的 Accuracy 约为 50%、Macro-F1 约为 29%，并且只有 71.39% 的起点有有效目标，单一态势判断的可靠性有限，应该用于宏观趋势/风险摘要，不应强调确定类别。10/30/60/120 秒继续保留为标准报告点；若 UI 要展示连续变化，可增加 15/45/90 秒诊断点，不需要因此改模型输出时距。

## 验证记录

- 单元测试：`python -m pytest tests/test_validation_global_horizon.py -q`，3 passed。覆盖 legacy/death-aware 两种 issue/future 使用同一合同，并验证非整数秒拒绝最近邻取值及历史输出保护。
- 新 validation-only JSON：`out_m2/validation_global_horizon_consistent/report.json`。核心字段为 `metrics.legacy`、`metrics.death_aware`、`adjacent_horizon_paired_accuracy_deltas`；含逐侧分数、coverage、绝对 Accuracy CI、相邻时距整局配对 CI。
- 历史 `out_m2/temporal_forecast_canonical_side/global_horizon_validation.json` 未覆盖。旧脚本默认输出改为新目录；明确传入的输出路径仍可用于本地临时报告。
- 当前模型 auto Accuracy 另见 `out_m2/validation_mode_label_contract/report.json`；它是冻结 GRU+融合结果，不等同于本报告的持久性 baseline，也不应把两份指标混为一谈。
