# 全链路系统审计

## 结论

系统已经具备可运行的端到端链路，但目前更适合作为“可解释的离线分析与回放系统”，还不能把预测、建议和大模型效果宣称为已验证的实战收益。

## 1. 数据与评估完整性

- 固定契约为 1 秒状态分辨率、3 秒决策步长、10/30/60/120 秒预测时距。
- 现有混合模型报告覆盖 128 局完整比赛，但评估账本存在 18 条缺失/无效记录，`red_blue_complete=false`。
- 报告的 `scored_count=746174` 是预期记录数 `373096` 的约两倍，说明模型与基线记录被合并计数；展示时不能把它当成独立样本数。
- 所有后续成绩应同时报告：原始行数、去重后的 game×side×time 行数、红蓝配对数和有效标签数。

## 2. 全局态势与趋势

旧混合模型的全量结果如下，格式为“模型准确率 / 持久性准确率”：

| 任务 | 10s | 30s | 60s | 120s |
|---|---:|---:|---:|---:|
| 全局态势 | 0.582 / 0.705 | 0.517 / 0.592 | 0.506 / 0.542 | 0.482 / 0.508 |
| 敌方趋势 | 0.456 / 0.474 | 0.456 / 0.307 | 0.509 / 0.255 | 0.523 / 0.218 |
| 敌方区域 | 0.597 / 0.687 | 0.540 / 0.561 | 0.507 / 0.510 | 0.503 / 0.468 |

模型的 Brier 和 LogLoss 通常优于基线，说明概率分布比单纯持久性更有信息；但 top-1 分类并未稳定超过基线，尤其是全局态势 10/30 秒仍明显落后。

## 3. 单机器人状态与区域

旧模型实体指标：

| 时距 | 存在准确率 | 位置归一化 MAE | 功能区准确率 | 规范区准确率 |
|---|---:|---:|---:|---:|
| 10s | 0.875 | 0.097 | 0.370 | 0.228 |
| 30s | 0.869 | 0.121 | 0.275 | 0.214 |
| 60s | 0.861 | 0.130 | 0.241 | 0.204 |
| 120s | 0.847 | 0.135 | 0.209 | 0.200 |

需要特别注意：`transition_legal_rate=1.0` 是经过路线投影后的结果，不能代表原始位置预测正确。原始功能区和规范区识别仍然较弱，且标签覆盖率随时距降到约 0.52～0.71。

## 4. 决策建议

8 局红蓝对照审计显示：

- 首步路线合法率为 100%，说明规则约束有效。
- 但建议目标区在 t+3 的匹配率低于停留基线：红方约 71.9% 对 77.1%，蓝方约 61.3% 对 68.8%。
- t+10 的移动建议到达目标区约 3.5%～4.7%，合法路线进度约 4%～6%，部分结果比停留基线更差。
- 没有控制器指令/ACK 日志，所以执行率和因果收益目前都是 unknown，不能把后续移动当作“模型建议被执行”。

这说明建议生成器的路线合法性不错，但目标选择和行动价值还没有证明有效。

## 5. 哨兵、攻击和事件链

- 规则姿态覆盖率约 99.8%，说明绝大多数帧能生成一个细化姿态标签；这只是规则覆盖率，不是策略正确率。
- 8 局样本中英雄 42mm 弹药增量与发弹事件完全对应，共 389 次。
- 613 局全量数据中，弹药增量和事件总数为 23866 与 23869，存在 3 次不一致，事件链还需要逐条定位。
- 发弹事件不能证明目标匹配，也不能证明控制器执行了建议。

## 6. 在线自校准

在线路由已经真实运行在“模型/持久性”切换逻辑中，但收益很小：多数任务的准确率只提升零点几百分点，部分平滑版本的 Brier/LogLoss 反而变差。当前自校准可以作为安全回退机制，不能当作已经学会局内策略的证据。

## 7. Qwen/Jev 大模型对比

现有 Qwen/Jev scorecard 只有 8 条可见样本，未来行覆盖率虽为 100%，但：

- 没有控制器执行日志，execution coverage=0；
- 没有目标 ID 的可验证链路，目标 HP 指标大多不可评分；
- 缺少可移动目标和完整路线字段。

因此现在只能比较输出格式和可解释性，不能比较谁的策略效果更好。

## 8. 延迟和可视化

- 预热后单次预测 CPU 平均约 24.9ms，CUDA 约 27.5ms；GPU 模型前向约 2ms，主要时间在后处理和解码。
- 冷启动存在约 2.67 秒尖峰，部署前必须预加载模型并预热。
- 回放帧切换中位数约 1.7ms，浏览器启动约 391ms；这些测试不包含完整数据库提取和浏览器渲染链路。

## 当前总体等级

| 部分 | 状态 |
|---|---|
| 数据时间与可见性契约 | 基本完成，需修复 18 条账本异常 |
| 规则、地图和合法路线 | 可用，原始预测准确率仍需单独评分 |
| 单机器人存在/HP | 可用，存在准确率较好 |
| 单机器人位置/区域 | 未达标，低于持久性或覆盖不足 |
| 全局态势 | 概率质量有价值，分类准确率未达标 |
| 在线自校准 | 已接入，收益尚未证明稳定 |
| 决策建议 | 可解释但行动价值未证明，且低于停留基线 |
| 哨兵/攻击事件 | 规则链可运行，仍有 3 次全量计数不一致 |
| Qwen/Jev 对比 | 仅完成可见性 smoke test，不能下模型优劣结论 |
| 延迟/回放 | 预热后可用，冷启动需要处理 |

建议把下一阶段优先级定为：先修复账本和事件计数，再把“原始预测、合法投影、建议执行、结果响应”四层指标彻底拆开，最后才继续训练更复杂模型。

## 9. 本轮修正

本轮已落地并通过 `pytest -q tests`（525 passed）：

- `eval/benchmark.py` 的顶层 `scored_count` 改为唯一目标键数量，并新增 `model_slot_count`、`scored_count_by_model` 和 `unique_scored_key_count`，避免把模型与持久性基线相加后误认为独立样本；基线缺少 issue-time 标签的行记为 `unavailable`，不再伪装成错误预测。
- `forecast/entity.py` 保留 `unprojected_position` 与 `unprojected_canonical_zone`；`eval/run_temporal_forecast.py` 同时输出原始预测、投影后结果和路线合法率。
- 持久性实体基线独立计算，不再复用模型实体指标。历史报告可用 `eval/repair_temporal_report.py` 修复；修正后的账本见 `out_m2/temporal_hybrid_motion_death_aware/temporal_hybrid_report_repaired.json`。
- `eval/audit_hero_fire_event_coverage.py` 新增具体不一致分组。目前全库仅 2 个英雄-比赛-阵营组合不一致，分别为 -2 发和 -1 发。

历史报告的实体模型指标是在旧版投影流程中生成的，因此修正报告对它们明确标注为 projected；后续重新生成的报告才会同时提供 raw 与 projected 两套指标。

## 10. 全量修正评估结果

在限制实体详情写入账本后完成了全量重算，峰值进程内存约 6.1 GB：

- 唯一可评分目标 373,096；模型槽位 746,174；invalid=0、missing=0；持久性基线不可用 18 条。
- 加权全局分：混合模型 0.4891，持久性基线 0.4685。
- 加权实体分：混合模型 0.7961，持久性基线 0.7637。
- `opponent_trend` 在 30/60/120 秒模型准确率为 0.456/0.509/0.523，基线为 0.307/0.255/0.218；这是全局任务中模型稳定超过基线的部分。
- `global_mode` 和 `enemy_region` 的 10 秒准确率仍低于基线，分别为 0.582 对 0.705、0.597 对 0.687。
- 实体原始位置 MAE 为 0.122/0.127/0.131/0.135（10/30/60/120 秒）；投影后为 0.097/0.121/0.130/0.135。投影改善不能替代原始预测准确率。

完整文件：`out_m2/temporal_hybrid_motion_death_aware/temporal_hybrid_report_full_corrected.json`。

## 2026-09-28 runtime/LLM integration correction

- `runtime.temporal_forecast._classification` now preserves `model_probabilities`, `baseline_probabilities`, `blend_alpha`, and `selected_source` for every task/horizon.
- `AdaptiveStrategyRouter` scores the raw calibrated expert against the causal persistence baseline, even when the static validation blend has already been applied. This prevents the online loop from treating a persistence-heavy output as if it were the neural model.
- `run_m2.py` can be invoked with `--temporal-checkpoint auto` to align one-second radar/sensor snapshots to the three-second decision clock and attach the four-horizon forecast context to each TieredDecider output. The alignment is causal and uses `data.radar_visibility_replay`.
- `worldstate.forecast_context` now carries visible current facts, model-vs-baseline comparisons, selected source, and required analysis sections. Tier2 prompt capacity was raised from 576 to 896 tokens; the lower limit remains caller-configurable.
- CPU smoke evidence: `out_m2/temporal_llm_integration_smoke.json` (3/3 decision points had `forecast-context-v1`, current-state facts, and four horizons). Tier2 was intentionally Null in this smoke; no claim of GPU LoRA execution is made.
- Full regression: 527 passed, 2 warnings.
