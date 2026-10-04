# 全局态势准确率候选审计与下一步判断

日期：2026-09-27。只核对已有 train-OOF、validation、生产 resolver 与报告；本次未读 test archive、未训练新模型、未改运行时默认。

## 判断

目前不建议再启动一个新的全局态势分类候选训练。候选探索已经覆盖 GRU、TCN+Transformer、直接 LightGBM、mode-lag 特征、margin 回归、状态切换先验和切换门控。进一步训练最有可能先拟合到当前 `global_mode` 标签口径的差异，而不是带来可以可信确认的提升。

## 生产默认与同口径 validation

`runtime/temporal_forecast.py` 默认仍走 canonical-side GRU，并读取 checkpoint 内嵌持久性混合配置；XGBoost 30s 只是旁路候选，只有显式 opt-in 才能覆盖输出。当前 alpha@10/30/60/120 秒为 `0/0/0.7/0.8`。

在旧 `relative_margin_v2` 合同下，`out_m2/validation_mode_label_contract/report.json` 的 frozen auto 相对持久性基线：

| 时距 | Auto Accuracy | Auto−持久性 Accuracy（按122局配对95% CI） | Auto−持久性 Macro-F1（按122局配对95% CI） |
|---:|---:|---:|---:|
| 10s | 75.18% | 0.00pp [0.00, 0.00] | 0.00pp [0.00, 0.00] |
| 30s | 63.71% | 0.00pp [0.00, 0.00] | 0.00pp [0.00, 0.00] |
| 60s | 57.44% | +0.04pp [−0.54, +0.51] | −0.42pp [−0.78, −0.09] |
| 120s | 53.30% | +1.35pp [+0.36, +2.33] | +0.51pp [−0.02, +1.01] |

10/30秒没有模型增益；60秒准确率未显著优于持久性且 Macro-F1 略降；120秒准确率有约1.35个百分点的正差，但该 horizon alpha 在同一 validation 上选择，不能作为未见数据的泛化承诺。

## 已做实验为何暂不产生新的可信候选

- Death-aware 直接 LightGBM 有5折整局 GroupKFold 和冻结 validation 对照；60s validation Accuracy 57.83% 对持久性55.87%，但 Macro-F1 54.92% 对55.10%，配对 Macro-F1 差−0.50pp且CI跨0。它是可保留的实验候选，未达到类别均衡改善证明。
- Death-aware margin 连续回归候选 60s validation Accuracy/Macro-F1 约46.64%/46.37%，低于直接分类与持久性。
- 状态切换门控候选（本轮已审计）在60s验证集 Accuracy 57.36%、Balanced Accuracy 55.48%、Macro-F1 53.89%，均未胜过直接 LightGBM；相对持久性虽准确率+1.48pp，但 Macro-F1−0.93pp，且均衡类 recall 从39.7%降到24.3%。不保留。
- Causal mode-history 特征在某些时距有小幅 Accuracy 提升，但 Macro-F1 差均没有可靠正向证据；这条方向已试，不重复扩展 lag 特征。
- TCN+Transformer 没有比生产默认更高的 Accuracy 证据；10/30秒仍由持久性决定。
- 30秒 XGBoost 候选使用不同 hybrid/death-aware 标签合同，和当前 canonical-side auto 不可直接比较；运行时只保留旁路，不能据此宣称提升。

这些实验的完整口径、OOF、类别指标和限制分别见本目录下 `2026-09-27-death-aware-global-mode-candidates.md`、`2026-09-27-death-aware-mode-lag-history-corrected.md`、`2026-09-27-canonical-mode-transformer-experiment.md`、`2026-09-27-global-mode-30s-candidate-promotion.md`，以及本轮 `out_m2/death_aware_global_mode_experiments/transition_gated_lgbm_60s/report.json`。

## 根因与更高信息增益的下一步

1. **标签合同仍未统一。** 现有 canonical checkpoint/NPZ 是旧合同；当前工作树把 `hp <= 0` / destroyed 的移动单位从态势分数中排除。只在 validation 上重标时，目标标签改变约9.5–11.7%。旧模型、death-aware 持久性和新标签分类器的数字不可拼在一个排行榜里。准确率实验应先等死活单位语义冻结，再完整重建 train/validation，并把同一重建结果同时用于 OOF 和 validation。
2. **目前有用信号被当前类强烈支配。** 10/30秒 auto alpha 为0且准确率等于 persistence；现有分类候选的原始输出无法稳定超越它。单看 accuracy 容易奖励一直输出当前类的模型；必须同时报 Macro-F1、Balanced Accuracy 和三类召回，避免再出现“准确率涨、均衡类崩”的候选。
3. **validation 已反复参与筛选。** GRU混合权重、Transformer权重、LightGBM候选和多种诊断都看过同一 validation。再用它选一套新特征/阈值并报告漂亮数字的信息增益很有限；应避免继续 validation 搜索。

**不训练的原因：** 现在没有一个未尝试、又能在同一标签合同/分组方式下独立验证的假设。下一步高信息增益行动是修正精确时距审计中 issue/future 标签合同混用，统一目标重建，并先做按整局分组的错误分解（标签变换、状态边界距离、可见敌方信息量、阵亡/重生与剩余比赛时间）；之后只预注册一个假设，用 train 内 GroupKFold 做OOF，validation只做一次冻结比较。validation的多轮复用也意味着最终部署结论仍需尚未触碰的比赛数据验收。
