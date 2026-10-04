# 验证集准确率与预测时距审计

日期：2026-09-27。范围仅限已有 validation 工件和评估代码；没有重新训练、访问 test archive 或修改生产默认。

## 时间映射

- 原始比赛记录按 1 秒落盘；数据生成器每 3 秒产生一个预测起点。回放画面可以每秒更新，但模型可用输入只会每 3 秒刷新一次。
- 监督目标取 `issued_at + horizon` 的精确秒，不插值、不找最近邻。10/30/60/120 秒都是相对 issue 时刻的未来秒数，不是帧数。
- 有效目标受比赛结束截断。验证集有 122 场、34,114 个起点；死亡感知的原始时刻审计在 120 秒只有 71.39% 起点仍有目标，故长时距数字只代表仍可评分的目标，不代表全部回放帧。

## 当前 canonical 模型在 validation 的表现

应以 `out_m2/validation_mode_label_contract/report.json` 的机器结果为准。当前 checkpoint 和压缩档使用旧标签合同 `relative_margin_v2`；工作树中的 `_mode_from_groups` 已改为 death-aware，不能混用两个版本。以下 accuracy / Macro-F1 均按完整比赛等权、比赛内红蓝观察方等权；Macro-F1 是先按每个比赛方计算三类 F1 再汇总，不等于把所有行混在一起算的 pooled F1。

| 时距 | 旧合同持久性 Accuracy / Macro-F1 | frozen auto Accuracy / Macro-F1 | auto−持久性 Accuracy 差（整局配对 95% CI） | auto−持久性 Macro-F1 差（整局配对 95% CI） |
|---:|---:|---:|---:|---:|
| 10s | 75.18% / 54.67% | 75.18% / 54.67% | 0.00pp [0.00, 0.00] | 0.00pp [0.00, 0.00] |
| 30s | 63.71% / 39.79% | 63.71% / 39.79% | 0.00pp [0.00, 0.00] | 0.00pp [0.00, 0.00] |
| 60s | 57.40% / 33.14% | 57.44% / 32.72% | +0.04pp [−0.54, +0.51] | −0.42pp [−0.78, −0.09] |
| 120s | 51.95% / 28.51% | 53.30% / 29.02% | +1.35pp [+0.36, +2.33] | +0.51pp [−0.02, +1.01] |

当前 auto 的持久性混合系数是 10/30/60/120 秒 `0/0/0.7/0.8`。因此 10s 和 30s 的 runtime 输出准确率就是持久性基线，尚无模型增益；60s 的准确率增益不显著且 Macro-F1 下降；120s 的 Accuracy 有小幅正向差异，但 Macro-F1 区间跨 0。混合系数由同一 validation split 选择，表格只适合当开发集描述，不能当作盲测承诺。

## 标签合同与时距结论

按 validation JSONL 以同一合同重算的 death-aware 持久性结果为：10/30/60/120 秒 Accuracy `72.25/60.39/55.89/50.23%`，Macro-F1 `53.16/38.59/33.73/28.72%`；当前到目标标签改变率 `27.76/39.63/44.13/49.79%`。这是合同诊断，并非 death-aware 重训模型成绩。与旧合同相比，future 标签约有 9.5–11.7% 改变；生产 checkpoint 尚未按新合同重训。

全局态势随时距变难是连续过程，没有证据支持在某个具体秒数突然“失效”。10→30→60→120 秒的 death-aware 持久性准确率持续下降，Macro-F1 到 120 秒约 29%；建议保留四个既有输出时距作为兼容的标准评估点，但把 120 秒定位为宏观趋势/风险提示，少做精确类别断言。可视化可增加 15/45/90 秒诊断刻度以展示下降曲线，暂不替换现有模型输出时距；新时距若进入正式模型指标，应在冻结统一标签合同后另建训练与验证口径。

## 已发现的报告口径问题

`eval/audit_validation_global_horizon.py` 读取旧合同 validation `base_mode` 作当前预测，却用当前工作树的 death-aware `_mode_from_groups` 重算未来真值。其 JSON 标注 `relative_margin_v2` 与实际代码不一致，比较是旧当前标签对新未来标签。该报告中 70.24/60.41/55.45/50.35% 的持久性曲线及对应 CI 不应当作统一合同下的正式基线。它的 1 秒原始分辨率、3 秒 issue cadence、精确目标秒映射仍有参考价值。旧文档 `2026-09-27-exact-validation-global-horizon-audit.md` 的准确率与相邻时距 CI 需要在合同统一后重算。

另外，`2026-09-27-validation-global-mode-label-contract-audit.md` 的部分 auto 表格值与其引用的 `out_m2/validation_mode_label_contract/report.json` 当前机器字段不吻合；本报告直接引用机器报告的 `frozen_auto_legacy_prior` 和 `paired_delta_vs_persistence` 字段。评估汇报时应显示合同、alpha 来源和聚合方法，避免拿 death-aware 当前基线与旧 checkpoint 的标签成绩拼成一条指标。

## 证据位置

- 时间映射/生成逻辑：`data/forecast_dataset.py`（`time_resolution_seconds=1`, `issue_step_seconds=3`, `target_at=issued_at+horizon`）。
- 旧/新标签合同、122 场配对 CI 与 frozen auto：`out_m2/validation_mode_label_contract/report.json`；审计脚本 `eval/audit_mode_label_contract_validation.py`。
- 合同差异说明：`docs/analysis/2026-09-27-validation-global-mode-label-contract-audit.md`。
- 混合合同的精确原始时间曲线（仅用于定位口径问题）：`out_m2/temporal_forecast_canonical_side/global_horizon_validation.json`，复核 `eval/audit_validation_global_horizon.py`。
