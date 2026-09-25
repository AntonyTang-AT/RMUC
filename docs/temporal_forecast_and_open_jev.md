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

