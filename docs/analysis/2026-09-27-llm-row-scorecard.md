# LLM 逐点决策响应评估

`eval/llm_row_scorecard.py` 对已有 issue-time JSONL 做逐点后评估，不加载模型，不生成决策。它只从 SQLite 读取 issue 时间加 3 秒、加 10 秒的精确遥测时间戳，并按 `game_id + entity_id + exact timestamp` 连接；缺行或重复键不做近似匹配，计入 unknown。

```powershell
python eval/llm_row_scorecard.py --rows out_m2/llm_backtest.rows.jsonl --out out_m2/llm_row_scorecard.json
```

报告按动作、单位类型、阵营分层，包括精确未来行覆盖率、移动/原地响应、日志中的开火事件、己方血量与存活、目标血量覆盖，以及路线和执行状态。现行分层决策 schema 只对单位类别给出 `move/target/fire/risk`，没有单位实例 ID、明确移动目的区或路径；因此同类别单位不唯一时不评分，未提供目的区时路线指标 unknown。目标 HP 只有 `target_unit_id` 加 `target_linkage=explicit_sensor_target` 才评分，symbolic target 类别不能冒充目标 ID。数据库也没有控制器命令/确认日志，执行覆盖为 0 / unknown。

以上是行为响应及观察性后果指标，不证明建议被执行，也不估计建议造成的因果收益。现存 `llm_backtest.py` 记录使用完整双方回放当前帧，报告将这种输入标记为不可用于部署质量结论；本 evaluator 会在 `visibility_audit` 中显示这一限制。尚未将 sensor/radar replay snapshot 安全转换为现有训练 prompt：现有 prompt 面向全场双方且模型输出没有明确决策视角。本工具不改变或默认调用该 legacy 路径。

Fixtures 覆盖精确时间/实体连接、近似时间不匹配、缺失值保留 unknown、单位归属不明、目标 ID 未确认不计目标 HP，以及 legacy omniscient 输入阻断部署声明。未运行 LLM、GPU 或全历史评估。
