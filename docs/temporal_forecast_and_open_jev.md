# 时序预测与 Open-Jev 2B 阶段结果

## 本地时序网络

`forecast/temporal.py` 从 `observable` 快照构造 167 维可见特征；输入窗口只包含当前及过去快照，目标由未来 `targets` 在离线训练阶段生成。`forecast/model.py` 使用两层、隐藏维度 128 的 GRU，共享时序编码后分别预测：

- 全局态势：进攻 / 均衡 / 防守；
- 敌方趋势：推进 / 稳定 / 回撤；
- 敌方位置区域：12 个固定网格；
- 预测窗口：10 / 30 / 60 / 120 秒。

训练和评估入口是 `python eval/run_temporal_forecast.py`。它按比赛和阵营分组构造因果窗口，按比赛时间切分，最后调用已有 `eval.benchmark.evaluate_forecast_ledger`，所以红蓝两侧都进入同一成熟预测账本并分别汇总。

正式 5 epoch 结果保存在 `out_m2/temporal_forecast/temporal_forecast_report.json`。在测试集上，GRU 相对持久性基线在 11 个任务×窗口的 log loss 全部下降，Brier 分数在 11 个桶中 10 个下降；唯一例外是 `global_mode@10s` 的 Brier 略差。当前 checkpoint 是 `out_m2/temporal_forecast/temporal_gru_v1.pt`，校准版本仍标记为 `uncalibrated-v1`，部署前需要在验证集上单独拟合温度或向量校准。

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
| 全局态势 Brier | 0.443 vs 0.481 | 0.514 vs 0.696 | 0.540 vs 0.818 | 0.545 vs 0.887 |
| 敌方趋势 Brier | 0.550 vs 0.865 | 0.611 vs 1.257 | 0.590 vs 1.405 | 0.579 vs 1.491 |
| 敌方区域 Brier | 0.372 vs 0.502 | 0.488 vs 0.751 | 0.534 vs 0.870 | 0.566 vs 0.967 |

全局态势的 10s 单一 top-1 为 65.6%，低于多数类占优的持久性 75.3%，但概率 Brier/log-loss 已改善；这说明 V2 预测分布更有信息，尚不能把 top-1 当作唯一上线门槛，下一步需要验证集校准和按类别 macro-F1 一起验收。

## 详细全局态势旁路

`runtime.TieredDecider` 现在额外输出 `global_analysis`（`global-analysis-v1`）。它包含当前模式、威胁等级、置信度、越线单位差、双方平均血量、基地/前哨站血量、敌方质心、敌方趋势、意图概率、证据列表，以及 10s/30--60s/120s 三个粒度的关注项。该字段是可审计旁路，不改变原有分层决策 schema；`DecisionRequest` 会透传它，`finetune.make_dataset.situation_digest` 也会把证据和时间粒度写入大模型输入。

## 既有决策效果复检

已重新运行 `eval.outcome_backtest`，结果在 `out_m2/decision_outcome_backtest_v2.json`：基于 613 个按比赛分组的验证折，状态到比赛有利度的胜率模型 AUC=0.930、准确率=0.845。按哨兵三态的条件有利度均值为 `def=1.874`、`cruise=2.223`、`hitbuilding=2.525`，`def-cruise=-0.349`，说明三态排序与当前局面风险有稳定关联。

这项复检不是“模型动作导致胜负”的因果证明：回放环境的动作不会改变未来状态，而且数据没有完整的实际执行指令日志。因此现阶段只能把它作为决策区间对齐和风险排序证据；要检验“推荐是否真的被执行并带来收益”，还需要接入动作日志或可反事实的前向仿真。


V2 报告同时新增 `macro_f1_by_task_horizon` 和 `target_distribution_by_task_horizon`，避免只看总体准确率掩盖某一类失效。当前全局态势 macro-F1 为 10/30/60/120s 的 0.614/0.563/0.507/0.472；持久性基线为 0.725/0.608/0.546/0.517。概率质量已经超过基线，但离“每一类都能可靠识别”的上线标准还有差距，尤其均衡类 F1 随预测距离变长下降明显。
