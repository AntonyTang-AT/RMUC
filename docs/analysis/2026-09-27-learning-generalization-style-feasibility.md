# 训练是否学到可泛化信号：当前证据审计

日期：2026-09-27。范围：只整理现有训练、OOF、validation、在线反馈与早期行为画像证据；未训练新模型、未访问新的最终测试结果、未改变生产默认。

## 结论

当前网络的优化过程确实降低了训练损失，但现有证据**不能证明 GRU 已学到稳定且优于简单方法的可泛化规律**。在固定预算的比赛分组 OOF 中，GRU 明显弱于 LightGBM；继续训练或单独 warm-start 也没有扭转表现。更可靠的当前说法是：数据/标签存在可学信号，但现有 GRU 训练目标和结构没有有效提取它。

打法分型也不能先验假定有效。只用开局 30 秒敌方可见位置、HP、运动变化构成的初步画像，未显示稳定的队伍内聚集。该统计是可行性筛查，不是聚类模型的验收结果；风格专家是否值得保留，必须由严格留出比赛的预测增益决定。

## 1. 训练过程与预测泛化

### 当前 GRU 的优化曲线

`out_m2/temporal_forecast_canonical_side/temporal_forecast_report.json` 中三轮训练损失依次为 0.538、0.468、0.445；验证损失为 0.551、0.507、0.509。训练损失持续下降，验证损失在第二轮最好、第三轮略回升。这说明梯度优化有效，但也出现轻微的验证退化；**训练损失下降本身不等于预测泛化成功**。

### 按整场分组的 GRU 与 LightGBM 对照

现有固定 50:50 GRU+LightGBM 五折 OOF 中，两侧视角按同场分组。每折 GRU 训练两轮，比较未校准概率与持久性基线：

| 时距 | GRU Accuracy | LightGBM Accuracy | GRU Macro-F1 | LightGBM Macro-F1 |
|---:|---:|---:|---:|---:|
| 10 秒 | 58.24% | 69.08% | 39.24% | 52.14% |
| 30 秒 | 53.53% | 58.32% | 32.52% | 37.55% |
| 60 秒 | 50.56% | 54.73% | 28.78% | 33.08% |
| 120 秒 | 48.81% | 51.72% | 26.90% | 29.49% |

该 GRU 明显没有为现有候选提供有效增益；固定融合也未胜过 LightGBM。由于它每折只训练两轮，这不能证明所有神经时序结构都无效，但足以否定“当前 GRU 已验证优于基线”。参见 `docs/analysis/2026-09-27-death-aware-gru-lgbm-50-50-oof.md`。

### 其他验证证据

- 单独训练的 LightGBM 候选在冻结 validation 上，60 秒 Accuracy 相对持久性提高 1.95pp，95% CI 为 [+0.16, +3.71]pp；其余时距 Accuracy 区间跨零。60 秒 Macro-F1 增量区间仍跨零且点估计为负，因此只支持有限用途，不足以晋升全局默认。参见 `docs/analysis/2026-09-27-death-aware-global-mode-candidates.md`。
- 同一报告的 warm-start GRU 在 validation 的四个时距都弱于持久性；最佳 epoch 为 1。单纯增加 epoch 或从旧 checkpoint warm-start 不是当前问题的充分解决办法。
- death-aware OOF 候选的总体 Accuracy 在四个时距高于持久性，但 Macro-F1 在 10/30/60 秒更低；远期的总体命中率提升不能被解释成少数类也预测得更好。参见 `docs/analysis/2026-09-27-horizon-segment-review.md`。
- 最近的速度特征 OOF 对 10 秒 Accuracy 仅增加 0.23pp；30 秒及以后区间跨零，Macro-F1 没有改善。新增输入需证明独立增益，不能仅因“信息更多”就保留。

## 2. 训练与评估方法审查

现有数据构造按完整比赛做时间切分，红蓝视角同场同集合；训练目标取精确未来时间点，目标与预测时刻输入分开；历史观测检查不允许晚于 issue time 的数据进入特征。审计未发现明显的未来标签泄漏。

主要风险在于证据独立性和选模：

1. 三秒一个样本且输入窗口高度重叠；一帧不是独立实验单位，指标区间必须按整场计算。
2. OOF 候选已被多轮观察和改动，属于开发证据；不能当最终盲测成绩。
3. 现有训练/测试在队伍身份上隔离，但 validation 和 test 出现相同队伍。由于 validation 已被反复用于开发，测试结果不能再被描述为对这些队伍完全独立的泛化证明。
4. 部分 LightGBM 路由、温度或混合权重在 OOF 输出上拟合后又在同组 OOF 标签上报告，结果偏乐观；应以内层整场交叉验证选参、外层留出冻结确认。

### 比赛等权 Macro-F1 的口径修正

复核代码时发现两份报告把**全 validation 行 pooled Macro-F1 点估计**与**先逐场、场内红蓝等权再 bootstrap 的差值区间**并排展示，导致点估计与区间不是同一种统计量。已在 LightGBM 共用评估代码中新增明确的 `validation_game_equal_red_blue_equal` 汇总，并对原始特征模型和 mode-history 候选用同一训练折、同一 validation 重新生成报告。历史 pooled 指标保留作诊断，但不再作为与比赛级区间配套的主结论；原始输出没有覆盖。

- 基础 LightGBM 候选对持久性基线：只有 60 秒 Accuracy 的比赛等权增量区间完全高于 0（+1.95pp，95% CI [+0.16,+3.71]）；四档 Macro-F1 增量区间均不能证明改善，10 秒 Macro-F1 反而显著更低。
- 增加因果态势历史特征后，相对不带历史的 LightGBM：10/30/60 秒比赛等权 Accuracy 分别提高 +0.27/+1.11/+1.02pp，区间不跨 0；120 秒 +0.81pp 但区间跨 0。四档 Macro-F1 差值区间都跨 0；四档 Brier 均改善。
- 这支持将因果历史作为**概率和总体命中率候选**继续评估，但不支持宣称类别均衡能力改善。Validation 已多次被用于开发，这次重跑只是统计口径修复，不是新的盲测。

可复核的修订报告：`out_m2/death_aware_global_mode_experiments/lightgbm_game_equal/report.json`、`out_m2/death_aware_global_mode_experiments/mode_lag_history_game_equal/report.json`。文档已相应修正：`docs/analysis/2026-09-27-death-aware-global-mode-candidates.md`、`docs/analysis/2026-09-27-death-aware-mode-lag-history-corrected.md`。

## 3. 早期打法画像可分性筛查

从训练集 363 场中，每场红蓝各取 issue time 不超过 30 秒的最后一帧，形成 726 个因果画像，覆盖 64 支队伍。画像仅用敌方六个机动单位槽的可见位置、朝向己方的推进量、短窗位移、HP、高亮状态与可见覆盖率；不使用队伍 ID 作为预测输入，也不使用赛后统计。

在训练画像的标准化空间中，队伍自身画像到队伍中心的中位距离约为 0.746；不同队伍中心之间的中位距离约为 0.481，比例约 1.55。按队伍重采样的探索区间约为 [1.43, 1.66]。这套摘要下队伍内部差异大于队伍中心间差异，可能反映打法不稳定、对手/局势影响、可见性噪声或摘要特征不足。

这个比值仅用于判断是否值得做分型实验：它不是预测性能、不是正式聚类检验，也没有证明风格专家一定失败。后续同折专家预测对照及其结果见第 3.3 节。

### 训练拟合、validation 画像簇先验筛查

为检查分簇是否仅能区分训练图样、还是能解释未来标签，我又做了一个一次性筛查：只取训练/validation 中每场红蓝双方 issue time ≤30 秒的最后一帧（训练 726 个、validation 244 个画像）；标准化和 KMeans 原型只在训练画像拟合；K=2/3 簇内按训练数据的未来类别频率形成带 Laplace 平滑的先验。没有打开 test。这里没有训练预测网络，也没有把结果作为模型晋升证据。

| 目标及未来时距 | K=3 簇先验 Acc / Macro-F1 | 当前态势持久性 Acc / Macro-F1 | 全训练先验 Acc / Macro-F1 |
|---|---:|---:|---:|
| 全局态势 10 秒 | 56.1% / 47.1% | 64.8% / 63.9% | 26.6% / 14.0% |
| 全局态势 30 秒 | 49.2% / 41.4% | 62.7% / 62.4% | 29.5% / 15.2% |
| 全局态势 60 秒 | 54.9% / 44.3% | 56.1% / 54.7% | 39.3% / 18.8% |
| 全局态势 120 秒 | 38.5% / 31.3% | 48.0% / 47.6% | 28.7% / 14.9% |

按 Accuracy/Macro-F1 看，这个简单簇先验没有胜过持久性；K=2 也没有形成稳定增益。它在部分时距的 Brier/LogLoss 低于高度集中的 0.98 持久概率，但这可能只是概率更平滑，不能当作类别预测准确性提升。对手趋势簇先验对照训练总体先验也没有跨时距一致的改善，且该筛查没有适用的“沿用当前对手趋势”标签基线。

样本只有 122 场 validation，簇数与多个指标是在探索中同时查看，未计算可用于晋升的比赛级区间；validation 队伍也曾用于开发。因此本筛查仍只是弱证据：**单靠 KMeans 簇内标签先验没有足够收益。** 更完整的同折软专家对照见第 3.3 节。

### 聚类原型跨整场折稳定性

另用 train 的 363 场、726 个同样的早期因果画像做五折 `GroupKFold(game_id)` 稳定性检查。每折只用 4/5 场拟合标准化与 KMeans，再让该折模型为同一批 train 画像分簇；用 Adjusted Rand Index 比较任意两折的分配（不需要簇号对齐）。

- K=2 的折间 ARI 中位数 0.966（四分位距 0.957–0.972），但训练簇大小明显失衡：每折约 10–12% 的小簇和 88–90% 的大簇。最大原型差异来自敌方工程位置和推进量（特征索引 9、8，对应第二个敌方机动槽的 y 与推进），所以稳定的两分更可能是当前空间/布阵状态，而不能直接命名为两种队伍打法。
- K=3 的 ARI 中位数 0.889，但分布呈两组：部分折两两 ARI 约 0.90，另一些约 0.26；折内 silhouette 也在约 0.05 与 0.20 间变化，小簇训练样本约 32–65 个。三类原型的稳定性不足以支持现在给队伍贴固定打法标签。

这说明“聚类分组可复现”与“学到了稳定战术风格”不是一回事。要避免模型把工程当前站位、地图区域或可见性当作打法标签，后续专家候选应把静态位置与行为变化拆开，并通过整场留出预测结果验证。可复核脚本：`eval/audit_style_cluster_stability.py`；该分析只读取 train 的当前/历史画像，没有使用目标标签、validation 标签或 test。

### 同折软专家预测增益

随后对 `opponent_trend@30s` 完成五折整场 OOF 对照，合同与结果见 `docs/analysis/2026-09-27-style-conditioned-trend-oof.md`。相对通用模型，加因果行为摘要使 Accuracy 与 Macro-F1 都提高约 0.47pp，比赛级区间不跨零；K=2/3 软专家相对相同摘要模型的 Accuracy/Macro-F1 只提高约 0.03pp，区间跨零。当前证据更支持**保留因果运动摘要、暂不保留软聚类专家**。该试验是随机整场 OOF，不是 unseen-team 测试，也未使用 validation/test。

进一步把分型方式改为“每场开局 30 秒画像，固定用于之后预测”，只在 issue time ≥30 秒上做训练集五折 OOF。按每个 game-side 总权重相等后，行为摘要相对快照模型的 Accuracy 提高 +1.41pp [0.94,1.90]，Macro-F1 +1.23pp [0.78,1.71]；而 K=2/3 开局打法专家的 Accuracy 增益约 +0.007/−0.003pp，区间都跨零，平均路由专家强度只有约 1%。这确认了一个关键区别：模型能从近期敌方运动预测后续趋势，但现有画像仍不足以把不同比赛策略分成对准确率有帮助的稳定类别。详见 `out_m2/style_experts_pilot/early_match_style_experts_trend_30s.json`。没有可用学校/队伍 ID，因此这仍不能检验跨场稳定的战队风格。

## 4. 局内校准与对手逻辑反演状态

`runtime/temporal_forecast.py` 的顺序 `predict()` 路径已可登记预测、等待时距成熟，并从发出与当前 `observable` 传感器/雷达快照重算反馈；`runtime/observable_feedback.py` 不读取 `targets`，且每场/每观察方 `reset`。反馈闭环实现和因果门禁测试已通过。不过滚动 Brier、平滑混合、块路由以及行为条件频率方案都未在对应离线评估中胜过固定输出，因此 resolver 默认关闭自适应路由。硬件传感器采集器仍不在本仓库，尚未声称已在实车运行。

`labels/opp_intent.py` 的当前意图分数主要是启发式证据；其 tracker 会参考模型自己先前挑出的 top 意图，若把这当新证据用于训练，会造成循环强化。在线反演应从可见、独立的行为事件更新每局的风格概率；到时评分，缺测作删失，预测自己的历史输出不能作为真值证据。实现前还需解决稳定 forecast ID、精确成熟时间门禁和每局完整 reset。

## 5. 当前决策与下一步验收

- GRU 保持实验状态；不能以训练损失下降宣称它已泛化。
- 当前 30 秒对手趋势 OOF 支持因果行为摘要，不支持 K=2/3 专家带来额外类别增益；不先给队伍贴固定风格标签。
- 严格 held-out-team 比较已经完成：可见行为摘要改善概率质量但未确认硬分类增益；K=2/3 专家没有准确率增益。应冻结特征和任务合同，再决定是否用历史 chronological test 做一次确认；该 test 曾被项目多次查看，不能称为全新盲测。
- 按 `field × horizon` 单独评估 Accuracy、Macro-F1、Balanced Accuracy、逐类召回、Brier/LogLoss、覆盖率及连续位置误差。以交叉验证相对基线的增益与比赛级区间决定保留、降级或隐藏；相关系数不能单独决定权重。
- 只有预测候选通过整场留出验证后，再把成熟反馈接入实时链，并用 prequential 比较验证局内校准是否优于固定模型；不要在线更新网络梯度。

## 可复核来源

- `docs/analysis/2026-09-27-death-aware-global-mode-candidates.md`
- `docs/analysis/2026-09-27-death-aware-gru-lgbm-50-50-oof.md`
- `docs/analysis/2026-09-27-death-aware-motion-zone-score-oof.md`
- `docs/analysis/2026-09-27-horizon-segment-review.md`
- `docs/analysis/2026-09-27-inmatch-feedback-and-learning.md`
- `eval/experiment_death_aware_mode_lightgbm.py`
- `eval/experiment_death_aware_gru_lgbm_50_50_oof.py`
- `forecast/train.py`
- `data/forecast_splits.py`
- `eval/audit_early_style_cluster_prior.py`（可复现本节训练拟合、validation 簇先验筛查；未读取 test）
- `eval/audit_style_cluster_stability.py`（五折按比赛拟合原型并比较分簇稳定性；不使用标签）
- `eval/experiment_style_conditioned_trend.py`（30 秒对手趋势、因果行为摘要与软专家五折整场 OOF）
- `out_m2/style_experts_pilot/opponent_trend_30s/report.json`


## 6. 修正：原始库允许严格未见队伍评估

压缩 NPZ 不含队伍身份，但只读源库 `数据与资料/rmuc_2026_region_dataset.sqlite` 的 `matches` 表包含红蓝学校名。train 363/363 场可映射到 64 队；validation 122/122 场映射到 48 队，其中 16 队与 train 重合；test 128/128 场映射到 32 队，与 train 不重合。身份只用于 grouping 和评分，没有进入模型特征。

新增五折整队留出 OOF：64 支队伍均作为完全未见的敌方被评分，共 94,150 个 30 秒趋势预测。行为摘要相对通用模型的队伍等权 Accuracy +0.356pp [−0.005,+0.714]、Macro-F1 +0.226pp [−0.169,+0.628]，均未排除零；Brier −0.00344 [−0.00451,−0.00235]、LogLoss −0.00549 [−0.00715,−0.00383]。K=2/3 行为专家没有准确率增益，只有极小概率分数改善。详细口径见 `docs/analysis/2026-09-27-team-heldout-style-trend.md`。该结论取代本文中“无法按队伍分组测试”的旧说法。
