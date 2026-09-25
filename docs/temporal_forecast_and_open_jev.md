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
    "out_m2/temporal_forecast_v3/temporal_gru_v1.pt", device="cuda", team="红")
decider = TieredDecider(team="红", temporal_forecaster=forecaster)
result = decider.decide(dp, bounds, observable_snapshot=current_observable_snapshot)
```

结果中的 `temporal_forecast.horizons["10"/"30"/"60"/"120"]` 可被 UI、规则路由和大模型慢环共同读取。10 秒预测置信度达到阈值时，快路径会把它作为宏观模式提示；模型失败或没有合法可见快照时自动回退规则，不会用完整回放数据冒充部署传感器输入。
