# 时序预测与 Open-Jev 2B 阶段结果

## 本地时序网络

`forecast/temporal.py` 从 `observable` 快照构造 167 维可见特征；输入窗口只包含当前及过去快照，目标由未来 `targets` 在离线训练阶段生成。`forecast/model.py` 使用两层、隐藏维度 128 的 GRU，共享时序编码后分别预测：

- 全局态势：进攻 / 均衡 / 防守；
- 敌方趋势：推进 / 稳定 / 回撤；
- 敌方位置区域：12 个固定网格；
- 预测窗口：10 / 30 / 60 / 120 秒。

训练和评估入口是 `python eval/run_temporal_forecast.py`。它按比赛和阵营分组构造因果窗口，按比赛时间切分，最后调用已有 `eval.benchmark.evaluate_forecast_ledger`，所以红蓝两侧都进入同一成熟预测账本并分别汇总。

正式 5 epoch 结果保存在 `out_m2/temporal_forecast/temporal_forecast_report.json`。在测试集上，GRU 相对持久性基线在 11 个任务×窗口的 log loss 全部下降，Brier 分数在 11 个桶中 10 个下降；唯一例外是 `global_mode@10s` 的 Brier 略差。当前 V1 checkpoint 是 `out_m2/temporal_forecast/temporal_gru_v1.pt`；V2 使用 `monotonic_horizon_v1` 跨时间置信度校准，部署前仍应在验证集复核校准误差。

## Open-Jev 2B 本地调试

已下载：

- Open-Jev 2B adapter/head/calibration 包：`models_hf/Open-Jev-2B`；
- 它要求的 Qwen3.5-2B 基座及固定 revision：`models_hf/Qwen3.5-2B`；
- 独立环境：`.venv_openjev_cuda`，使用 transformers 5.10.2、PEFT 0.19.1，并复用项目 CUDA torch。

Open-Jev 的本地 HTTP 服务可按以下命令启动（8GB 显存建议 4-bit 调试模式）：

```powershell
$env:JEV_LOAD_4BIT = "1"
$env:HF_HUB_OFFLINE = "1"
.venv_openjev_cuda\Scripts\python.exe -m jev.server `
  --checkpoint out_m2\openjev_local_checkpoint `
  --device cuda:0 --max-length 4096 --batch-size 1 --no-prefix-cache
```

`eval/probe_open_jev.py` 将相同的 observable-only 因果特征协议送入 Open-Jev，输出在 `out_m2/temporal_forecast/open_jev_probe.json`。当前 8 个红蓝平衡样本的诊断结果：

| 模型 | 10s accuracy / Brier | 30s accuracy / Brier | 60s accuracy / Brier | 120s accuracy / Brier |
|---|---:|---:|---:|---:|
| Open-Jev 2B（4-bit，本地） | 12.5% / 0.671 | 12.5% / 0.664 | 12.5% / 0.654 | 12.5% / 0.636 |
| Qwen2.5-3B + `finetuned_lookahead`（候选打分） | 0% / 1.796 | 0% / 1.835 | 0% / 1.852 | 0% / 1.823 |

这不是最终排名：样本很小，而且两个语言模型都没有用 RMUC 专门的结构化微调格式。它说明当前协议下直接把 167 维数值向量序列化成文本并交给语言模型效果很弱；本项目的 GRU 才是目前经过完整测试集验证的预测器。`eval/probe_qwen.py` 保留了 Qwen 候选概率对照方法，方便在改进状态文本协议或做 RMUC 领域微调后重跑。

## 使用边界

Open-Jev 是独立社区实现，不等同于 TypeSafe 官方 Jev。官方 Jev 以托管 API 形式提供，不能把其服务输出用于蒸馏、训练仿制模型或开发竞争服务；本项目没有用官方 API 输出训练任何权重。Open-Jev adapter 的本地训练或微调应只使用我们自己的 RMUC 标签和可观测快照，并保留其 Qwen 基座与 adapter 的许可证记录。


### V2 标签与时间粒度重标注

V2 使用 `relative_margin_v2` 标签：按当前阵营计算越线单位差，并加入双方基地血量差，红蓝视角对称。测试集四个未来窗口的全局态势分布约为：

| 窗口 | 进攻 | 均衡 | 防守 |
|---|---:|---:|---:|
| 10s | 38.0% | 24.1% | 38.0% |
| 30s | 38.3% | 23.3% | 38.4% |
| 60s | 38.3% | 23.3% | 38.4% |
| 120s | 38.4% | 23.1% | 38.5% |

模型输入对不同窗口采用不同时间聚合：10s 保留 12 个原始快照，30s/60s/120s 分别按 2/3/6 帧池化。因此近窗口保留细粒度变化，远窗口只保留趋势。V2 报告在 `out_m2/temporal_forecast_v2/temporal_forecast_report.json`。按红蓝两侧宏平均的概率指标，GRU 相对持久性基线如下（越低越好）：

| 任务 | 10s | 30s | 60s | 120s |
|---|---:|---:|---:|---:|
| 全局态势 Brier | 0.443 vs 0.481 | 0.512 vs 0.696 | 0.541 vs 0.818 | 0.545 vs 0.887 |
| 敌方趋势 Brier | 0.550 vs 0.865 | 0.614 vs 1.257 | 0.597 vs 1.405 | 0.591 vs 1.491 |
| 敌方区域 Brier | 0.372 vs 0.502 | 0.488 vs 0.751 | 0.535 vs 0.870 | 0.566 vs 0.967 |

全局态势的 10s 单一 top-1 为 65.6%，低于多数类占优的持久性 75.3%，但概率 Brier/log-loss 已改善；这说明 V2 预测分布更有信息，尚不能把 top-1 当作唯一上线门槛，下一步需要验证集校准和按类别 macro-F1 一起验收。

## 详细全局态势旁路

`runtime.TieredDecider` 现在额外输出 `global_analysis`（`global-analysis-v1`）。它包含当前模式、威胁等级、置信度、越线单位差、双方平均血量、基地/前哨站血量、敌方质心、敌方趋势、意图概率、证据列表，以及 10s/30--60s/120s 三个粒度的关注项。该字段是可审计旁路，不改变原有分层决策 schema；`DecisionRequest` 会透传它，`finetune.make_dataset.situation_digest` 也会把证据和时间粒度写入大模型输入。

现在 `global_analysis.narrative` 还提供 `description`、`recommendation` 和 `paragraph` 三个自然语言字段。默认由已计算的结构化证据生成，避免大模型补写不可观测事实；如果接入 Qwen/Open-Jev 慢环，可以让大模型改写这三个字段，但必须保留原始 `evidence`、`confidence` 和结构化决策作为审计依据。执行器仍只读取严格的 `global_mode + units` JSON。

哨兵三态单独使用 `eval.sentry_state_eval` 评估。当前帧规则预测与未来 5 秒窗口标签的红蓝宏平均结果保存在 `out_m2/sentry_state_eval_report.json`：accuracy=0.842、macro-F1=0.693；红方 macro-F1=0.550，蓝方 macro-F1=0.835。红方留出数据没有 `hitbuilding` 真值样本，因此该类的 F1=0 只说明覆盖不足，不能解释为模型已经学会或学不会该状态。验收时应同时展示混淆矩阵、逐类 F1 和样本支持数。

## 既有决策效果复检

已重新运行 `eval.outcome_backtest`，结果在 `out_m2/decision_outcome_backtest_v2.json`：基于 613 个按比赛分组的验证折，状态到比赛有利度的胜率模型 AUC=0.930、准确率=0.845。按哨兵三态的条件有利度均值为 `def=1.874`、`cruise=2.223`、`hitbuilding=2.525`，`def-cruise=-0.349`，说明三态排序与当前局面风险有稳定关联。

这项复检不是“模型动作导致胜负”的因果证明：回放环境的动作不会改变未来状态，而且数据没有完整的实际执行指令日志。因此现阶段只能把它作为决策区间对齐和风险排序证据；要检验“推荐是否真的被执行并带来收益”，还需要接入动作日志或可反事实的前向仿真。


V2 报告同时新增 `macro_f1_by_task_horizon` 和 `target_distribution_by_task_horizon`，避免只看总体准确率掩盖某一类失效。当前全局态势 macro-F1 为 10/30/60/120s 的 0.614/0.563/0.507/0.472；持久性基线为 0.725/0.608/0.546/0.517。概率质量已经超过基线，但离“每一类都能可靠识别”的上线标准还有差距，尤其均衡类 F1 随预测距离变长下降明显。

### 四个时间窗的输出契约

四个时间窗都保留同一组基础类别：全局态势（3 类）、敌方趋势（3 类）和敌方质心区域（12 个固定网格）。因此 30/60/120 秒不会突然删除类别。差异体现在输入聚合和解释粒度：10 秒看单位/建筑变化，30 秒看阵型与局部控制，60 秒看区域控制与推进深度，120 秒看总体攻防方向和基地风险。长期预测仍保留 12 格区域概率，但应把它解释为更分散的候选分布，并同时读取宏观区域汇总。

这套层级契约记录在 `forecast/temporal.py:HORIZON_PROFILES`，每条预测的 `visibility_summary` 也会写入对应粒度。后续校准会增加随 horizon 单调增大的不确定性约束：预测距离越远，允许的 top-1 置信度越低、概率熵越高，但不通过硬删除类别制造断层。

校准版本为 monotonic_horizon_v1：同一任务上，若远期分布比近期更尖锐，则只向均匀分布扩散到不超过前一窗口的 top-1 置信度。实测归一化熵从 10 秒到 120 秒逐步上升，全局态势为 0.681→0.796→0.848→0.868，敌方区域为 0.261→0.389→0.453→0.490。



## 实体级预测扩展

当前 V3 预测器在原有四个全局任务之外，增加了固定的 16 个实体槽位：`self/opponent × 英雄、工程、步兵3、步兵4、空中、哨兵、前哨站、基地`。每个时间窗都会输出每个槽位的：

- 存在概率 `presence_probability`；
- 预测位置 `position.x/y`（米）；
- 平均血量比例 `hp_ratio`；
- 数量估计 `count`；
- 状态 `active / critical / absent`。

实体槽位不会随时间窗删减。短期预测给出更具体的位置和血量，长期预测给出更宽的不确定性，仍然保留同一兵种集合。实体状态通过 `forecast/entity.py:decode_entity_forecast` 转为结构化 JSON，并挂在对应时间窗的全局预测记录 `entity_forecasts` 字段中。

V3 训练与评估报告：`out_m2/temporal_forecast_v3/temporal_forecast_report.json`。初步测试集结果如下，位置误差为归一化地图坐标的 MAE，血量误差为比例 MAE：

| 窗口 | 存在准确率 | 位置 MAE | 血量 MAE |
|---|---:|---:|---:|
| 10 秒 | 99.23% | 0.117 | 0.227 |
| 30 秒 | 99.21% | 0.124 | 0.239 |
| 60 秒 | 99.17% | 0.128 | 0.248 |
| 120 秒 | 99.18% | 0.130 | 0.259 |

存在准确率受“多数实体持续存在”影响，不能单独代表模型质量；位置和血量误差随时间变长，符合预测不确定性增加的规律。下一步应进一步按兵种分别报告位置/血量误差，并给位置输出验证集校准区间，而不是只给单点坐标。

V3 报告还包含 ntity_forecast_metrics.*.by_slot，可分别检查 self:英雄、opponent:哨兵 等槽位，避免平均指标掩盖某一兵种的位置或血量预测失败。

## 运行时接入方式

部署端应先用车载传感器和雷达构造 `build_observable_snapshot(...)`，再把快照交给 `TemporalForecastRuntime`。适配器会维护 12 个历史快照，只读取当前及过去可见字段：

```python
from runtime.temporal_forecast import resolve_temporal_forecaster
from runtime.decide import TieredDecider

forecaster = resolve_temporal_forecaster(
    "out_m2/temporal_forecast_weighted_v3/temporal_gru_v1.pt", device="cuda", team="红")
decider = TieredDecider(team="红", temporal_forecaster=forecaster)
result = decider.decide(dp, bounds, observable_snapshot=current_observable_snapshot)
```

结果中的 `temporal_forecast.horizons["10"/"30"/"60"/"120"]` 可被 UI、规则路由和大模型慢环共同读取。10 秒预测置信度达到阈值时，快路径会把它作为宏观模式提示；模型失败或没有合法可见快照时自动回退规则，不会用完整回放数据冒充部署传感器输入。

### 运行时自校准与策略切换

如果需要让部署端根据真实比赛持续校准，可显式挂载
`runtime.adaptive_strategy.AdaptiveStrategyRouter`：

```python
from runtime.adaptive_strategy import AdaptiveStrategyRouter
from runtime.temporal_forecast import resolve_temporal_forecaster

router = AdaptiveStrategyRouter(window=40, min_samples=20,
                                patience=3, recovery_patience=4,
                                margin=0.03)
forecaster = resolve_temporal_forecaster("auto", team="红",
                                         adaptive_router=router)
prediction = forecaster.predict({
    **current_observable_snapshot,
    # 当前帧可观测的全局标签，或由规则头得到的 persistence 分布。
    "current_labels": {"global_mode": "均衡", "opponent_trend": "稳定"},
})
# 仅在 +10/+30/+60/+120 秒的目标时刻真正到达后调用：
forecaster.observe_outcome(
    prediction["adaptive_strategy"]["forecast_id"],
    {"global_mode": {"10": matured_truth["global_mode"]},
     "opponent_trend": {"10": matured_truth["opponent_trend"]}},
)
```

路由器为每个任务和时间窗分别维护滚动 Brier 损失。模型在达到最小样本量后连续多次明显劣于持久性基线时，该任务暂时切换到持久性分布；模型恢复并连续优于基线后再切回模型。预测发布时不会读取未来帧，未提供可观测基线时保持模型输出并标记为 `model`，因此这条闭环不会把离线真值泄漏到部署热路径。

## 回放地图与步长

`eval/build_battle_replay_visual.py` 现在直接读取用户确认版 `out_m2/official/canonical_zone_regions.json` 和 `canonical_zone_labels.json`，以 59 个 canonical 区域为边界源；显示层会处理嵌套基地和线宽造成的细小重叠，并把剩余地面归入开放地面，形成完整无重叠分区。地图上的黄色箭头连接当前兵种位置和按“速度×时距”可达的合法路线；10 秒最多走一条边，30/60/120 秒允许沿多条合法边继续展开，不表示目标区域内的精确落点。

HTML 回放保留每一秒观测帧，页面中的“回放步长”下拉框可以在 1/3/6/12/30/60 秒之间切换，默认 1 秒。模型决策仍严格按 3 秒发布；两个决策点之间只更新当前观测位置，预测和建议沿用上一次决策，并在时间栏标明“沿用上一决策”。命令行的 `--display-step` 只决定首次打开页面时的默认步长，不会丢弃任何一秒观测帧。回放一致性评估只对真正的 3 秒决策帧计分，避免重复计算 held frame。

## 功能区位置与相邻移动预测

`docs/field_map/functional_zones.md` 和 `worldstate/functional_zone.py` 将完整平面图细分为基地、装配区、前哨区、公路、梯形高地、中央隧道和中央高地等功能区。`worldstate.robots[*].zone` 统一使用这些功能区名称，旧的 6×3 网格保存在 `legacy_zone`。

每个实体预测现在同时输出 `from_zone`、`to_zone`、`raw_to_zone` 和 `movement`。`movement.description` 采用“兵种：起点功能区 → 目标功能区”格式；10 秒预测只允许停留或走一条边，长时预测保留原始目标，并将可执行目标投影到满足通行图、路径长度和 `speed × horizon` 的合法路线。评估报告新增 `functional_zone_accuracy`、`transition_legal_rate` 和 `path_hops`，分别衡量区域判断、路线合法性和时距对应的路线长度。

## 兵种特化状态旁路

实体槽位还挂载 `special_state` 和 `special_state_forecast`。它们由 `worldstate/unit_state.py` 从当前可观测快照计算，并按 10/30/60/120 秒使用透明的持续性先验扩散概率：

- 英雄、步兵和哨兵：`attack_status`（正在攻击/可能攻击/未攻击/未知）、当前或可能目标、`attacking_energy_unit`；哨兵额外有守基地、守前哨和巡逻状态。
- 工程：`has_energy_unit`、`is_exchanging`、`resource_phase`（去资源区、回补给区、兑换、收取、未知）、`resource_direction` 和 `resource_route`。
- 空中：飞行状态和攻击状态同时输出，停机坪作为正式功能区。
- 基地、前哨站和能量机关：结构损伤状态及交互对象类型。

当前协议对敌方通常只提供类型、位置、血量、金币和雷达高亮，不能直接证明敌方正在开火、持有能量单元或锁定了谁。这些字段因此保持 `unknown`；由可见位置得到的目标只放在 `possible_target`，并标注 `visible_geometry_inference`。运行时顶层 `special_state_contract` 记录这一失败关闭约定。

## 全局分析与预测旁路扩展

全局预测仍保留 `global_mode`、`opponent_trend` 和 `enemy_region` 三个稳定任务，同时在每个时间窗增加 `global_outlook`：

- `control`：双方占用功能区、争夺区、控制关系；
- `threats`：敌方侵入、关键区域争夺和推进压力；
- `resource_posture`：双方工程资源阶段、目标补给区和可观测来源；
- `opponent_intent_hint`：趋势与敌方区域的组合解释；
- `recommended_actions`：主动作、触发条件和回退路线；
- `decision_summary`：当前姿态、控制趋势、承诺条件和回退条件；
- `force_balance`：按兵种汇总的存活、临界、攻击/资源/防守准备度；
- `engagement`：接触区、正在/可能攻击、目标兵种、能量机关关注和功能区转移；
- `objective_status`：基地、前哨站和能量机关的预测状态与交互状态；
- `mobility`：基于功能区邻接图的可行动转移和未知路线策略；
- `forecast_scope`：当前时间窗允许的空间、状态和决策粒度；10 秒保留到兵种功能区和特殊状态，30--60 秒收敛到兵种角色与目标，120 秒只保留区域和攻防方向；
- `uncertainty`：当前窗口的置信度、空间粒度和未知字段约束。

这些字段由实体预测聚合得到，标记为 `entity_forecast_aggregation_v1`，不伪装成额外训练出的分类头。结构化界面可以显示坐标、概率和血量；`global_analysis.narrative` 只解释控制关系、风险来源、敌方意图和条件建议，不再重复罗列功能区计数和原始数值。类别功能区头在 V4 实验中暂不替换 V3 的连续位置投影，只有通过准确率和合法转移门禁后才进入生产。

当前仍需继续补齐的学习数据包括：工程持矿/兑换事件、明确开火与目标 ID、空中机器人高度、建筑受击来源，以及 CAD 级功能区边界。它们进入标注后，可以训练特化状态头和功能区分类头；在此之前，系统会保留 `unknown` 和 `possible_*` 的失败关闭语义。

本轮复核结论：加权 V3 是新的生产候选。V4 的 27 类功能区辅助头已完成训练和离线评估，但 10/30/60/120 秒功能区准确率约为 0.322/0.272/0.235/0.200，尚未超过原 V3 的 0.565/0.516/0.480/0.450；因此运行时只保留 V4 类别头作为审计候选，行动路径仍由连续位置预测和合法邻接投影决定。真实回放样例见 `out_m2/global_forecast_example_1780143788827_t100.json`。

## 业务加权训练与验收

原始平均准确率偏低主要来自四个因素：三态全局模式的边界标签会随未来局部移动改变，远期位置本身存在不可预测性，功能区边界把连续坐标离散化会放大误差，敌方血量/开火/资源事件又受雷达字段限制。训练现在支持 `business_weighted`：10 秒比 120 秒权重高；全局模式和短期趋势比敌方粗区域权重高；存在性和位置比血量、数量及实验性功能区分类权重高；类别内部再使用截断的逆平方根频率权重处理长尾。

加权验收仍保留原始 accuracy、macro-F1、位置 MAE 和合法转移率，并额外输出 `weighted_acceptance`。在同一留出集上，无功能区辅助头的加权 V3 将全局加权分从 0.5193 提升到 0.5411，实体加权分从 0.8547 提升到 0.8563；10/30/60/120 秒全局模式 macro-F1 从 0.591/0.516/0.483/0.467 提升到 0.625/0.562/0.523/0.512。该候选模型位于 `out_m2/temporal_forecast_weighted_v3/`，运行时 `auto` 会优先加载它，文件不存在时回退到原 V3。

边界样本还会写入 `label_quality`：态势分数或敌方趋势位移靠近规则阈值时降低样本影响，避免把一帧的类别翻转当成确定规律；远期训练使用更高的 label smoothing，运行时使用 `monotonic_horizon_v2_temperature` 展开远期概率。对于雷达无法看到的开火、目标 ID、持矿和高度等字段，数据集不建立监督头，`unit_state` 明确输出 `unknown_fields`，自然语言和执行器不能把未知解释成“未发生”。

边界质量候选的质量加权准确率会单独写入 `quality_weighted_accuracy_by_task_horizon`。当前候选尚未超过生产加权 V3 的综合分，因此保留为实验目录 `out_m2/temporal_forecast_weighted_v2_nz/`，不会因为单个边界指标变好就替换整体模型。

## 本轮运行链路复核

`resolve_temporal_forecaster("auto")` 现在默认创建 `AdaptiveStrategyRouter`；只有显式传入 `adaptive_router=None` 才关闭。运行时从当前可见快照用同一套 `baseline_labels` 生成持久性基线，预测输出携带 `adaptive_strategy.forecast_id`，调用方必须在 `issued_at + horizon` 到达后调用 `observe_outcome`。回放构建器已按这一契约运行：每条预测在最长 120 秒目标成熟后才反馈，`adaptive_calibration` 保存红蓝两侧的成熟记录数、滚动 Brier 差和当前策略。它不是训练时偷看未来，而是成熟目标到达后的在线路由校准。

全局旁路新增 `motion_projection`、`event_probabilities`、`objective_projection` 和 `evidence`。前者汇总兵种速度、目标区距离、可达时间和速度约束，后者明确标记为实体旁路聚合概率；`evidence.future_data_used_at_issue_time` 固定为 `false`。这使全局输出可以同时回答“谁控制什么、哪类兵种构成压力、目标是否有风险、哪些路线在时限内可达、建议何时触发或回退”。

## 30 秒长预测与后续 10 秒短预测的一致性

`eval/temporal_consistency.py` 对每个时刻 `t` 配对：`t` 发布的 `+30s`、最接近 `t+20s` 发布的 `+10s`，以及 `t+30s` 的实际标签。原始决策粒度是 3 秒，因此当前数据无法精确落在 20 秒，评估器选择 `t+21s`，并将 1 秒量化误差写入报告。比较指标包括两者对实际的 Brier、准确率、短预测相对长预测的 Brier 改善、预测分布 JS 散度和兵种目标功能区一致率；不是只比较类别是否相同。回放 HTML 的全局面板会显示这组成熟配对结果。

## Qwen 与 Open-Jev 的比较口径

两者都已有微调产物，但旧探针只有 8 个相同红蓝样本，只能作为诊断，不能作为最终排名。`eval/compare_model_probes.py` 会检查 `(game_id, side, issued_at)` 完全重合、红蓝平衡，并在同一标签上计算准确率、Brier 和 log loss；少于 128 个配对样本自动标记 `diagnostic_only`。正式验收应扩展到同一 RM 留出集，至少 128 条配对、两侧等量、四个时间窗和相同 observable-only 输入契约后再比较。

## 用户确认版区域划分接入训练

用户确认的白底区域图 `F:/Desktop/download.png` 已固化为几何数据源：`out_m2/official/canonical_zone_regions.json` 保存 59 个无重叠区域多边形，`out_m2/official/canonical_zone_adjacency.json` 保存区域邻接、禁入基地和兵种通行规则。训练时不把图片像素直接喂给模型，而是对每个可观测实体追加四个因果空间特征：当前区域编号、区域邻接度、地面通行等级和基地标记；未来目标则按同一多边形规则生成 59 类区域标签。坐标统一转换为 RM 世界坐标，图片坐标的 y 轴翻转只发生在几何适配层。

新的紧凑数据版本为 `canonical-zone-v1`，输入维度从旧版 167 增加到 231；模型保留原 27 类功能区头，同时新增 59 类 `entity_canonical_zone_state` 头。训练损失中区域头权重为 0.45，存在性和连续位置仍保持更高权重，避免离散边界误差压过可执行预测。运行时会输出 `from_canonical_zone`、`to_canonical_zone` 以及前三个区域概率，但实际行动路线仍经过速度 × 时间和合法邻接投影，基地区域不会生成可进入箭头。

完整数据已用该版本重新缓存并训练了一轮，产物位于 `out_m2/temporal_forecast_canonical/`。这是一版空间标签已接入的实验 checkpoint；是否替换当前生产候选，仍需在相同训练轮数和验收门槛下与 `temporal_forecast_weighted_v3` 对比，不能仅依据一次 epoch 的结果直接切换。

### canonical-zone 续训结果

在首轮 checkpoint 基础上使用相同配置继续训练 8 个 epoch（总计 9 个 epoch，batch size 256，hidden=128，2 层 GRU，`business_weighted`）。验证集最低损失出现在第 2 个 epoch（0.6179），后续损失上升，因此 `temporal_forecast_canonical_e8/temporal_gru_v1.pt` 保留第 2 个 epoch 的最佳权重，并记录完整训练历史。测试集上 10 秒实体存在性为 0.9948、位置归一化 MAE 为 0.0865、canonical 区域准确率为 0.2507；全局加权分为 0.4934，仍低于持久性基线 0.5058。该结果说明空间区域头已正常学习，但当前配置继续训练后出现过拟合，模型暂不应直接替换生产候选，后续应以第 2 个 epoch 为起点做学习率衰减、早停和正则化对比。

### 低准确率修正：侧别特征与因果持久性融合

复核发现，原始特征没有显式编码红蓝观察侧，但态势和趋势标签按红蓝方向定义，导致模型必须从对称位置间接推断方向。现在 `canonical-zone-v2-side-aware` 在每个快照加入 side 特征，输入维度为 232；同时把 59 类区域辅助损失从 0.45 降为 0.15，避免高基数区域分类压制主任务。

这版 3 个 epoch 的最佳验证损失为 0.5069。测试集原始模型的全局加权准确率为 0.5529，略低于持久性基线的 0.5586；进一步在验证集拟合因果持久性融合系数后，测试集全局加权准确率达到 0.6074。融合只使用当前快照可计算的持久性标签，系数固定写入 checkpoint，部署时不会读取未来帧。配置和报告见 `out_m2/temporal_forecast_canonical_side/persistence_blend.json` 与 `persistence_blend_report.json`。

当前部署候选应使用 side-aware checkpoint 加持久性融合；未通过统一红蓝留出集和 Brier/log-loss 门禁前，不把单纯 raw model 概率作为最终输出。

### 最终候选：运动特征 + 因果 TCN/Transformer

本轮按最终候选完成了可复现实验。每个样本使用严格按比赛顺序截取的 40 帧上下文；每个实体除状态和 canonical 区域外追加位移、速度、加速度、方向等因果运动特征，输入维度为 329。网络由左填充膨胀 TCN 与上三角因果 Transformer 双流组成，192 隐层、3 层、4 个注意力头，输出保持现有的全局三任务、兵种实体状态、功能区和 59 类 canonical 区域头。

训练使用 `business_weighted` 损失和早停，在 CUDA 上运行 3 个 epoch；第 2 个 epoch 验证损失最低（0.509475），第 3 个 epoch 已回升。完整训练集、验证集和测试集缓存以及 checkpoint 位于 `out_m2/temporal_hybrid_motion/`，测试报告和集成清单分别为 `temporal_hybrid_motion_report.json`、`integration_manifest.json`。

测试集全局业务加权分为 0.523669，持久性基线为 0.505797，提升 0.017872。混合模型对趋势任务的提升最明显（10/30/60/120 秒准确率为 0.563/0.455/0.508/0.536），但 10 秒全局模式和敌方区域仍低于持久性基线；因此混合模型已接入 `resolve_temporal_forecaster("hybrid")` 作为可选候选，默认 `auto` 仍使用经过验证的侧别 GRU + 因果持久性融合。只有在同一红蓝留出集上同时通过加权分、Brier/log-loss、位置 MAE 和合法转移率门禁后，才应替换默认路径。

随后在验证集上对混合模型做了因果持久性融合调参，按任务和时间窗独立搜索 `alpha`，只使用当前快照可计算的基线标签。测试集原始分为 0.523669，调参后为 0.572880；配置保存在 `out_m2/temporal_hybrid_motion/persistence_blend_tuned.json`，并已写入混合模型 checkpoint。该结果说明主要收益来自“短期保留持久性、远期逐步增加模型趋势”的组合，不能把验证集调参结果误当作无偏泛化提升；后续仍需在新的红蓝留出比赛上复核。

### 预测结果作为大模型参考上下文

`worldstate/forecast_context.py` 将时序预测压缩为 `forecast-context-v1`：包括模型/校准版本、快照哈希、因果历史长度、10/30/60/120 秒的模式/趋势/敌方区域概率、兵种功能区转移、速度可达性、特殊状态和不确定性。上下文带有 `reference_only` 约束，明确预测不是事实，不能覆盖当前可观测状态、规则约束或结构化动作字段。

Tier2 慢环、QLoRA 训练数据和 `DecisionRequest` 都已接入该上下文。大模型能用预测来解释“短期风险是否连续、哪条功能区转移值得复核、远期是否只能保留攻防方向”，但生成结果仍必须通过 `schema.layered` 校验，执行器不会执行自然语言或预测字段。没有预测、模型失败或字段缺失时，接口回退到原有 WorldState 输入和规则路径。

### 回放可视化交互约束

最新回放 `rm-battle-replay.html` 使用 1 秒数据帧、3 秒决策帧；决策未更新的帧只沿用上一决策。黄色预测箭头的终点先经过 canonical 区域邻接、路径和速度×时间约束：10 秒保持局部，长时按可达路线展开，缺失起点时由当前观测坐标反算，不再跨越不可通行区域。页面可独立调整回放步长和每帧停留时间（10/20/33/50/80/180/350/600/1000 ms，默认 20 ms）。

地图边界采用细线，面板文字使用高对比色。全局说明按“当前看到什么—未来可能怎样—整体建议”三句展示；攻击中的可见单位显示攻击轨迹和目标，工程、空中、哨兵等特殊状态显示在单位上方气泡中。攻击轨迹只有在目标实体可见且结构化状态为 `attacking` 时绘制，雷达不可观测字段保持 `unknown`。

### 预测与大模型建议的有效性口径

短期预测不再用统一的“一跳限制”：`+10s` 只允许停留或走一条合法边，`+30/+60/+120s` 按单位速度、路径边长和最大跳数逐步展开。路线同时保留 `raw_to_zone`、`path_zones`、`path_hops`，因此可以区分“模型想去哪里”和“在这个时距内实际能走到哪里”。

大模型只接收当前可见事实和 `forecast-context-v1` 参考上下文。上下文新增每个时距的 `decision_guidance`：可能态势、目标区域、推荐目标、触发条件、回退动作和示例路线；预测不能覆盖当前事实，也不能把未知雷达字段说成已发生。自然语言建议按“现在看到什么—未来可能怎样—满足什么条件就做什么—否则退到哪里”输出，避免只说进攻/防守口号。

建议验收分两层：先验收数值预测（各时距的 Brier、log-loss、位置 MAE、路线可达率和 `path_hops` 误差），再验收建议（结构化动作合法率、触发条件命中率、回退覆盖率、与下一成熟时刻实际态势的一致率）。回放本身仍是相关性回测，不能把 replay 中的后续结果直接解释为“大模型导致了胜利”；要证明动作收益，需要接入可重复的前向动力学或仿真环境。
