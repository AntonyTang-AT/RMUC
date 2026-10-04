# 可见战斗事件增强：train 整局 OOF

仅训练集 363 场五折整局 OOF；红蓝视角同折，比赛等权且比赛内红蓝等权。候选沿用已保存 motion+zone/score 模型的分折、LightGBM 参数与随机种子，仅追加因果事件特征。没有访问 validation/test，也没有选择阈值或调参。

| 时距 | 方法 | Accuracy | Macro-F1 | Balanced Accuracy | 相对持久性 Accuracy Δ [95% CI] |
|---:|---|---:|---:|---:|---:|
| 10s | 事件增强 | 69.49% | 51.66% | 52.17% | +0.85pp [+0.50, +1.21] |
| 10s | 原 motion | 69.56% | 51.66% | 52.17% | — |
| 10s | 持久性 | 68.64% | 52.38% | 53.14% | — |
事件增强减原 motion：10s Accuracy -0.07pp [-0.19, +0.05]；Macro-F1 -0.01pp [-0.16, +0.14]；Balanced Acc -0.00pp [-0.16, +0.14]。
| 30s | 事件增强 | 59.35% | 37.79% | 39.50% | +2.36pp [+1.64, +3.07] |
| 30s | 原 motion | 59.32% | 37.84% | 39.58% | — |
| 30s | 持久性 | 56.99% | 38.77% | 39.28% | — |
事件增强减原 motion：30s Accuracy +0.03pp [-0.14, +0.21]；Macro-F1 -0.05pp [-0.23, +0.13]；Balanced Acc -0.08pp [-0.26, +0.09]。

事件特征仅由当前或更早的可见快照生成；己方发弹来自同 ID 累计发弹差分；敌方只使用可见 HP，不能推断敌方发弹。HP 缺失和雷达消失不视为死亡。完整覆盖率和观测时间检查见 `visible_events_train_audit/report.json`。

该训练集此前经过多轮候选探索，OOF 为开发证据，不能当作新的独立盲测。

复现：先运行 `python eval/audit_death_aware_visible_events_train.py`，再运行 `python eval/experiment_death_aware_visible_events_oof.py`；motion 缓存由原 `extract_motion(train.npz, train.death_aware.jsonl.gz)` 生成。
