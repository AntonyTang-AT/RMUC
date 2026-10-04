# 时序网络是否学到可用信息：验证集诊断

日期：2026-09-27。范围：当前 canonical-side GRU 检查点，在既有 validation 上进行开发诊断。该 validation 已参与模型/策略开发，本报告不是独立盲测，也不触碰 test。

## 检查点和训练信号

检查点 `out_m2/temporal_forecast_canonical_side/temporal_gru_v1.pt` 使用 232 维、2 层、128 hidden 的 GRU，`business_weighted` 多任务损失，CUDA，训练 3 轮。训练损失为 0.5381 → 0.4684 → 0.4455；validation loss 为 0.5505 → 0.5069 → 0.5093。模型在拟合训练数据，validation 最优出现在第 2 轮，第 3 轮略回升，提示应按比赛级验证早停；不能靠继续增加 epoch 解决分类准确率问题。

## 输入依赖性消融

在同一检查点、同一 validation 标签上比较原始输入与三个诊断扰动：

- `repeat_current`：把最后一帧重复为全窗口，移除历史变化；
- `reverse_time`：反转时间顺序；
- `shuffle_other_prefix_same_side_timebin`：在相同观测方、相同 15 秒比赛时段内，交换其他样本的整段历史窗口，破坏样本—标签对应关系。

指标先在单场、单观测方分别计算，再红蓝等权，最后比赛等权。对每场的扰动减原始差值做 2,000 次成对 bootstrap。全局数字仅是同一 checkpoint 的输入敏感性诊断，不是独立泛化成绩。

### 30 秒敌方趋势

| 输入 | Accuracy | Macro-F1 | Brier |
|---|---:|---:|---:|
| 原始完整历史 | 45.13% | 39.39% | 0.6328 |
| 仅重复当前帧 | 42.84% | 36.86% | 0.6537 |
| 反转时间 | 38.40% | 32.80% | 0.7115 |
| 同侧同时间段错配历史 | 34.78% | 33.85% | 0.7323 |

相对原始输入，去掉历史的 Accuracy 差为 −2.29pp（95% CI [−2.77,−1.79]pp），Macro-F1 差为 −2.53pp（[−3.10,−2.00]pp），Brier 增加 0.0209（[0.0177,0.0245]）。反转时间后 Accuracy 下降 6.74pp（[−7.97,−5.53]pp）；错配整段输入后下降 10.35pp（[−11.58,−9.15]pp）。这支持模型使用了因果时序内容，而不只是依靠输出常数或类先验。

### 其他任务的诊断形态

- 错配历史输入后，global mode 四档时距 Accuracy 下降约 14.7–28.0pp，enemy region 下降约 9.5–29.1pp；各任务 Brier 都变差。网络输出明显依赖样本输入。
- 把历史换成当前帧重复，对 global mode 和 enemy region 的整体 Accuracy 改变很小，长时距甚至微升。这表示这些任务当前预测更多由当前局势/位置而非额外历史贡献；趋势任务更明确地需要时序信息。
- enemy region 虽然对错配输入敏感，但 Macro-F1 仅约 0.07–0.14。稀有区域仍难预测，不能用整体 Accuracy 或“网络确实依赖输入”掩盖类别质量不足。

完整任务 × 时距 × 输入消融和逐场置信区间见 `out_m2/temporal_input_dependence_audit/report.json`。

## 这不等于预测准确

`temporal_forecast_canonical_side/temporal_forecast_report.json` 中，原始 GRU 对 global mode 的 pooled Accuracy 为 63.20/56.87/54.88/55.62%，低于持久性 Accuracy 75.25/64.17/57.89/54.32%。硬类别特别是 10/30 秒仍被持久性占优。另一方面，原始 GRU 的概率 Brier 点估计低于尖锐的一热持久性输出：global mode 10/30/60/120 秒分别约 0.475/0.527/0.554/0.566，持久性约 0.481/0.696/0.818/0.887。概率分布含有有用信息，不表示 argmax 更准；Brier 差别也可能部分来自基线过度自信。

当前 global-mode 运行配置已把 10/30 秒混合权重设为 GRU=0、持久性=1，60/120 秒才使用部分 GRU。这是已有证据下的合理风险控制，但 validation 被反复开发使用，权重需在冻结方案后用新的比赛确认。

## 训练/方法问题判断

1. **不是“网络完全没学到”**：输入错配和时间反转造成显著退化，30 秒趋势的历史消融也有逐场区间支持。
2. **但全局态势硬分类没超过基线**：当前训练是多任务、多时距联合损失，部署关注的 global-mode Accuracy 与多任务加权 loss 不是同一个优化目标；训练 loss 降低不能当成准确率改善。
3. **轻微过拟合迹象**：validation loss 第 2 轮最低，第 3 轮回升；应采用组级验证早停/重训折估计 epoch，而非单靠加 epoch。
4. **训练窗口高度重叠**：3 秒步长、长序列和近未来标签让逐行随机分割不合格。必须完整比赛分组；报告应以比赛为统计单位。现有 OOF 已整场分组，但检查点结果只能视为 validation 开发结果。
5. **下一步应围绕预测用途设任务与损失**：趋势短期预测保留历史编码；态势先估计转换概率/幅度并按决策代价选用概率或类别；稀有区域采用层级区域/概率质量评估；通过整场 OOF 选早停、校准与阈值，避免再次用 test 选策略。

## 风格分类能否从这些数据验证

压缩训练样本确实不保留队伍身份，且 `data/rm.db` 为 0 字节；但原始比赛库 `数据与资料/rmuc_2026_region_dataset.sqlite` 的 `matches` 表保留红蓝学校名。只用这些名称构造折、不把名称输入模型后，363 场训练比赛可映射到 64 支队伍，已进行严格未见队伍评估。详见 `docs/analysis/2026-09-27-team-heldout-style-trend.md`。

## 可复核产物

- 指标：`out_m2/temporal_input_dependence_audit/report.json`
- 检查点训练史及 validation 原始 GRU/持久性比较：`out_m2/temporal_forecast_canonical_side/temporal_gru_v1.pt`、`out_m2/temporal_forecast_canonical_side/temporal_forecast_report.json`
- 模型选择出的 global-mode 持久性混合权重：`out_m2/temporal_forecast_canonical_side/temporal_gru_v1.pt` 内 `persistence_blend`
- 下一轮独立确认需要先冻结 checkpoint、训练 epoch 选择法、损失合同、校准和评分指标，再对未参与选择的比赛评分一次。
