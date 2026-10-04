# 全局态势模式标签合同审计（仅 validation）

日期：2026-09-27

## 发现

验证压缩档 `out_m2/temporal_forecast_canonical_side/validation.npz` 和对应 `out_m2/forecast_dataset/validation.jsonl.gz` 的行键（比赛、观察方、时刻）全部对齐。压缩档的 `base_mode` 与提交版本的旧 `_mobile` 规则（移动单位不因 `hp <= 0` 而排除）在 34,114/34,114 行一致。

当前工作树改为把 `hp <= 0` 或 `life_state=destroyed` 的单位从 `_mobile` 排除，但验证档与现有 checkpoint 没有按新口径重建/重训。于是新的 issue-time observable 重算值与旧 `base_mode` 在 3,201/34,114（9.38%）行不一致。按未来 exact-time truth 用旧规则复算，现有 `mode_y` 在 10/30/60/120 秒都 100% 一致；换成 death-aware 规则后，真值标签改变比例分别为 9.48%、9.96%、10.53%、11.72%。

这是标签版本漂移，不是 JSONL/NPZ 行偏移。已有基于旧合同的指标只描述旧定义。它们不能与新 death-aware 标签下的准确率混在一起；新定义也不能只更新实时基线而继续沿用旧 checkpoint 的目标标签。

## 两种标签合同下的 validation 结果

模型输出固定为现有 canonical-side checkpoint；“当前模型+旧基线”采用保存下来的 persistence blend；“当前模型+新基线”采用同一 checkpoint 概率和同一 alpha，仅把 prior 换成 death-aware 当前 observable 模式。所有分数按比赛等权、红蓝观察方各占一半汇总，配对 CI 对 122 场比赛做 2,000 次 bootstrap。

| 预测时距 | 旧合同：持久性准确率 | 旧合同：当前 auto 准确率 | 新合同：持久性准确率 | 新合同：auto 换新 prior 准确率 | 新合同 current→future 不同率 |
|---|---:|---:|---:|---:|---:|
| 10 秒 | 75.18% | 75.18% | 72.25% | 72.25% | 27.76% |
| 30 秒 | 63.71% | 63.71% | 60.39% | 60.39% | 39.63% |
| 60 秒 | 57.40% | 56.52% | 55.89% | 55.93% | 44.13% |
| 120 秒 | 51.95% | 52.50% | 50.23% | 51.44% | 49.79% |

旧合同 current→future 不同率依次为 24.82%、36.29%、42.61%、48.06%。由此可见，全局模式随时距增加而更难预测，约 120 秒接近抛硬币水平；不过这只是三类模式，不能代表位置、机器人状态或战术建议在该时间失效。

新合同与旧合同的 future 模式不同约一成。切换合同后，当前模型没有按新标签重训；上表新合同模型分数是诊断，不是经过训练的性能承诺。10/30 秒的 auto 因原配置 alpha 为 0，输出等于当前类；所以“预测模型”在短时距实际没有提供超越持久性的增益。

## 单独的轻量趋势实验

`eval/crossfit_global_mode_score_trend.py` 只用 validation observable 构造五个 issue-time 分数的因果滚动趋势，并按比赛分组五折选择每个 horizon 的缩放系数。为保证与冻结 checkpoint 同口径，它显式重建旧标签合同（旧 `_mobile` 不按零 HP 排除单位），与 compact `base_mode` 全量一致。每个 horizon 的五折系数均为 0，候选退化为持久性基线：没有观察到可用的趋势修正收益，未改生产默认。

## 时间段与后续

当前数据支持保留 10/30/60/120 秒四档用于不同用途，但不应把 120 秒模式预测描述为可靠。仅凭本次审计不建议移动档位或删掉 60 秒档：长时距模式变化率确实持续增加，且 60 秒仍优于 120 秒；更重要的是要先冻结统一的死亡状态语义。

若产品定义确认死亡单位不参与当前/未来模式分数，应在同一标签版本下重新生成训练、validation 数据并训练模型，再用 untouched test 做一次只读最终验收。若沿用目前已有 checkpoint/报告，则继续统一按旧规则计算和报告，不可把 death-aware 当前基线与旧 truth 混用。

## 可复现

```powershell
python eval\audit_mode_label_contract_validation.py --device cuda --bootstrap 2000
python eval\crossfit_global_mode_score_trend.py --device cuda --bootstrap 2000
```

完整数值和混淆矩阵见 `out_m2/validation_mode_label_contract/report.json`、`out_m2/validation_global_mode_score_trend/report.json`。两脚本只接收 validation 文件，未读取 train 或 test 样本；checkpoint 冻结，没有模型训练、生产默认修改或测试集调参。
