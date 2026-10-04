# 30 秒全局态势候选：部署审计与结论修正

日期：2026-09-27。原记录将 30 秒 XGBoost 候选作为默认部署头，并把它与 `temporal_hybrid_motion_death_aware` 的融合结果比较。复核发现该对照并不对应当前 `auto` 的 `temporal_forecast_canonical_side` 检查点；两份数据虽然具有相同的比赛、观察方和时间行，但标签口径也不相同。因此原记录的“优于当前部署”结论撤回。候选保留为旁路结果，只有显式设置 `apply_experimental_mode_xgb=True` 才覆盖 30 秒态势头。

## 当前 `auto` 部署链

- 时序检查点：`out_m2/temporal_forecast_canonical_side/temporal_gru_v1.pt`。
- 该检查点内嵌 `persistence_blend`。运行时加载的 `global_mode` alpha 为 10/30/60/120 秒 `0 / 0 / 0.7 / 0.8`；配套 JSON 是同一融合配置的可读副本。
- 默认 `auto` 继续采用 canonical-side GRU 与该检查点融合；如果 XGBoost artifact 存在，它会作为实验候选出现在输出中，但不会默认替换生产 30 秒头。

## 当前 `auto` 在 canonical-side 测试集上的指标

在 `canonical_side/test.npz` 的已锁定测试划分上按实际运行时校准、平滑持久性融合重算。该测试集有 128 局；每个时距只统计可评分行。

| 时距 | 样本数 | 准确率 | Macro-F1 | 持久性准确率 | 进攻 F1 | 均衡 F1 | 防守 F1 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 秒 | 34,998 | 75.25% | 72.51% | 75.25% | 82.00% | 52.99% | 82.59% |
| 30 秒 | 33,208 | 64.17% | 60.79% | 64.17% | 72.36% | 37.36% | 72.66% |
| 60 秒 | 30,648 | 58.18% | 54.53% | 57.89% | 66.72% | 29.59% | 67.26% |
| 120 秒 | 25,528 | 56.11% | 52.81% | 54.32% | 63.98% | 30.54% | 63.91% |

“均衡”是明显弱项；整体准确率受进攻/防守占比影响，不能单独代表三类识别质量。逐类 precision、recall、support 和混淆矩阵见 `out_m2/temporal_forecast_canonical_side/runtime_default_mode_test_metrics.json`。

10 秒和 30 秒的融合权重为 0，因此这两档实际等于持久性输出，没有 GRU 增益；60 秒比持久性准确率高 0.29 个百分点，120 秒高 1.79 个百分点。这个测试划分过去已经用于分析，以下结果是诊断复核，不是全新盲测验收。

### 运行时校准修正

2026-09-27 复核发现，原运行时把模型 logits 直接送入概率校准；离线评估则先执行 softmax。运行时已修正为 `softmax(logits) -> horizon calibration -> embedded persistence blend`，并使用冻结检查点重新评估上表。验证集 256 行运行时/离线概率逐项最大差为 `5.21e-7`，类别预测 0 处不一致。此前本节 60 秒和 120 秒准确率、F1 数值作废；10 秒和 30 秒 alpha 为 0，结果未受该路径差异影响。

## 为什么不能报告 XGBoost 对当前 `auto` 的增益

XGBoost 候选的训练和评估数据来自 `temporal_hybrid_mode_stats_ablation`，其 feature manifest 指向 `temporal_hybrid_motion_death_aware` 与 `forecast_dataset`。当前 `auto` 时序模型来自 `canonical_side` 的 232 维 side-aware 数据。两份 test 文件虽然按 `game_id / side / issued_at` 能逐行对齐（35,768 行、128 局），但对齐后：

- 当前状态类别 `base_mode` 仅 88.23% 相同；
- 30 秒目标类别 `mode_y` 仅 88.15% 相同。

所以现有 XGBoost 的 60.20% 测试准确率（候选自己的标签口径）与当前 `auto` 的 64.17% 不能直接相减或据此判断优劣。此前 paired comparison 使用的 hybrid checkpoint也不是当前 canonical-side auto 检查点。测试集已经被候选报告查看过；这次审计只纠正模型/标签口径和默认部署路径，不进行测试集调参。

## XGBoost 候选的有效事实

- 已冻结候选：`out_m2/mode_change_xgb_probe/report.model.ubj`，validation 选择的 persistence alpha 为 0.95；不在本次审计重新调参。
- 原任务口径测试指标：红方 60.41%、蓝方 59.99%、等权 60.20%，但这些值只描述候选原始数据/标签合同。
- 在线输入特征与离线候选特征在既有 parity 检查中一致；这只验证输入复现，不证明候选胜过 canonical-side auto。
- 比较有效性状态：`not_comparable_to_canonical_auto_due_to_label_contract_mismatch`。

## 验证记录

检查点内嵌融合配置、resolver 的默认旁路行为以及 opt-in 覆盖行为由 `tests/test_mode_change_xgb_runtime.py` 覆盖。可复现的实际 auto 预测逐类结果保存在 `out_m2/temporal_forecast_canonical_side/runtime_default_mode_test_metrics.json`。
