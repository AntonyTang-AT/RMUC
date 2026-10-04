# Canonical-side TCN+Transformer 全局态势候选实验

日期：2026-09-27。目标是在不改变生产默认的前提下，测试复杂时序结构能否改善当前 `canonical_side` 全局态势预测。输入和标签沿用生产候选的同一契约（12 帧、232 维特征，红蓝两种观察视角）。

## 训练与选择

- 模型：5 epoch TCN+Transformer，GPU 训练；验证损失在 epoch 2 达到最低，后续 epoch 未继续改善。
- 仅用 train 训练、validation 选每个 horizon 的 persistence 融合权重；选择指标为红蓝等权的 `0.6 × accuracy + 0.4 × macro-F1`。
- 最终融合权重：10/30/60/120 秒 = `0 / 0 / 0.75 / 0.90`。10 秒和 30 秒因此完全退化为当前 persistence 融合，候选在这两个时距没有模型增益。
- 测试集在候选冻结后只做了一次配对评估。它是历史上已用过的锁定 test split，不是全新盲测；本实验没有用 test 调参。

## 配对测试结果

所有数值都来自相同的 test 行、标签、红蓝配对与时距。差值为候选减生产 `auto`。

| 时距 | Auto 准确率 / Macro-F1 | 候选准确率 / Macro-F1 | 准确率差及按比赛 bootstrap 95% CI | Macro-F1 差及 95% CI |
|---:|---:|---:|---:|---:|
| 10 秒 | 75.25% / 72.51% | 75.25% / 72.51% | 0.00 pp [0.00, 0.00] | 0.00 pp [0.00, 0.00] |
| 30 秒 | 64.17% / 60.79% | 64.17% / 60.79% | 0.00 pp [0.00, 0.00] | 0.00 pp [0.00, 0.00] |
| 60 秒 | 58.18% / 54.53% | 57.77% / 54.88% | −0.41 pp [−1.14, +0.28] | +0.35 pp [−0.40, +1.09] |
| 120 秒 | 56.11% / 52.81% | 55.86% / 54.48% | −0.25 pp [−1.85, +1.36] | +1.67 pp [+0.22, +3.01] |

候选在 120 秒 Macro-F1 有统计上可见的提升，但准确率没有提升证据，Brier 也没有可靠改善；60 秒 accuracy 和 Macro-F1 的配对区间都包含 0。若只以 accuracy 为主要验收指标，这个候选尚未满足晋升条件；若后续将类别均衡能力纳入验收，应继续用未触碰的新比赛确认 120 秒 Macro-F1 增益。

## 结论与后续

目前没有把该候选设为默认。复杂结构本身没有改善目标中的 accuracy：10/30 秒完全由 persistence 决定，60/120 秒候选 accuracy 略低于 Auto。下一步应优先改善标签和输入中的可预测信号、检查模式转换前兆及身份/运动特征，再以新比赛做预注册比较；不要因为模型更复杂而默认采用。

可复现文件：

- 训练与 validation-only 选择：`eval/train_canonical_mode_candidate.py`
- 冻结后的单次配对测试：`eval/compare_canonical_mode_candidate_test.py`
- GPU 候选权重：`out_m2/canonical_mode_candidate_hybrid/temporal_hybrid_v1.pt`
- validation：`out_m2/canonical_mode_candidate_hybrid/validation.json`
- paired test：`out_m2/canonical_mode_candidate_hybrid/locked_test_comparison.json`
