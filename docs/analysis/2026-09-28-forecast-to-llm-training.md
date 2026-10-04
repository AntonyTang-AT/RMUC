# 时序预测进入大模型训练样本（2026-09-28）

## 变更

`run_m2.py --stage dataset --temporal-checkpoint <checkpoint>` 现在会把每个决策点的
`temporal_forecast` sidecar 保留下来，并传给
`finetune.make_dataset.make_records(..., temporal_forecasts=...)`。训练指令因此同时包含：

- 当前可见的己方和敌方状态；
- 10/30/60/120 秒的预测参考；
- 模型输出与持久性基线的对照；
- 预测置信度、不确定性和触发条件；
- 只能把预测作为参考，不能把低置信度预测改写成事实的边界。

此前这些内容只在运行时 Tier2 注入，训练数据没有看到时序预测，导致大模型没有机会学习如何解释预测和基线差异。现在训练和运行时两条路径使用同一套 `forecast-context-v1` 格式。

另外，训练/运行时在决策点之间会按时间顺序预热全部 1 秒雷达快照，再取 3 秒决策点的预测结果；预测器不再只看到稀疏的决策点，仍然只使用当时及之前的可见快照。

## 验证

### 单元测试

```text
pytest -q tests/test_m2.py tests/test_forecast_context.py tests/test_run_m2_profile.py
26 passed
```

新增测试确认：提供 `temporal_forecasts` 后，样本 instruction 包含预测参考、时距和输入快照标识。

### 小规模真实数据检查

命令：

```text
python run_m2.py --stage dataset --profile mainline --dataset-games 1 --limit 3 \
  --all-fallback --temporal-checkpoint auto --temporal-device cpu \
  --out out_m2/temporal_dataset_smoke
```

结果生成 3 条样本，首条样本中可见：

- `预测参考信息（因果模型输出，仅供态势分析；不是事实，也不能直接改写动作）`；
- 10/30/60/120 秒预测；
- `schema_version=forecast-context-v1`；
- 模型/基线对照和触发条件。

证据文件：`out_m2/temporal_dataset_smoke/m2_dataset.jsonl`。

随后按相同配置生成了完整训练输入：

- 24 局、3317 条记录；
- 3317/3317 条 instruction 都包含 `预测参考信息`；
- 数据集：`out_m2/temporal_context_dataset/m2_dataset.jsonl`；
- 训练输出目录：`out_m2/finetuned_temporal_context`（Qwen2.5-3B-Instruct，1 epoch，GPU 安全配置）。

训练完成后使用 `eval/compare_lora_forecast_context.py` 做三路消融：旧适配器无预测、旧适配器带预测、新适配器带预测。验收只接受“新适配器带预测”在留出比赛上的事实一致性、合法率、冲突率和建议可执行性不劣于两类对照的结果。

## 后续评估要求

重新训练 QLoRA 后，需要额外报告：

1. 事实一致性：是否只引用当前可见输入；
2. 预测引用覆盖率：是否正确引用时距、概率和基线差异；
3. 建议可执行性：是否给出触发条件、动作和失败回退；
4. 预测帮助增益：加入 forecast context 后，相对不加入 context 的同配置模型提升多少。

只有第 4 项在留出比赛上为正，才可以认定大模型能力得到实际提升。
## Tier2 运行时事实与安全审计（2026-09-28）

Tier2 的结构化枚举校验只能证明输出格式正确，不能证明动作与当前可见状态
一致。现已接入 `runtime/decision_guard.py`：

- 明确阵亡的本方单位不再接受移动、开火或目标动作，自动回退到规则决策；
- `fire=true` 且 `target=无` 的输出被拒绝；
- 10/30 秒高置信预测与大模型选择不一致时只记录审计信息，不把预测覆盖成事实；
- guard 结果写入 `decision_meta.tier2_guard`，可在回放和离线评估中统计拒绝率、分歧率和原因。

这保证预测上下文进入大模型后具备可追溯的事实优先级，同时避免用低置信预测
强制改变战术动作。

## 运动/趋势专家进入上下文（2026-09-28）

运行时现在还会在 `forecast-context-v1.source` 中标出
`global_task_expert` 和 `entity_motion_expert`。单机器人区域 sidecar 的预测
以 `entity_motion_lgbm_zone`、`entity_motion_lgbm_canonical_zone` 进入每个单位，
趋势 sidecar 的模型/基线对照进入对应的全局分布。这样 Tier2 能区分当前可见
事实、神经头、LightGBM 专家和持久性基线；已有 Qwen 适配器可以直接读取这些
字段，但仍需在新的上下文数据上重跑留出集，确认冲突率是否继续下降后再替换
正式适配器。
