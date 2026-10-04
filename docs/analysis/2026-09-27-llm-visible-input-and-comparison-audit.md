# Qwen / Open-JEV 决策输入与公平比较审计

## 已实现的输入合同

`eval/llm_visible_input.py` 从 `data.radar_visibility_replay` 的**指定阵营**快照构造 LLM 输入。`load_visible_llm_input` 的数据库查询只有 `game_id=? AND 时刻秒<=建议时刻`；没有未来行、未来事件或未来标签。要求建议时刻精确存在，按时间保留最近最多 4 帧。每个实体再次按白名单筛选：我方保留传感器的 HP、最大 HP、位置、朝向、热量等；敌方只保留雷达允许的类型、位置、HP、运动估计与宏观字段。敌方空中单位 HP 强制移除；己方基地和前哨站保留。历史帧只在 prompt 中展示位置，当前帧展示状态和运动等信息。完整的白名单快照留在 `visible_history` 供审计，后续标签单独进入评分器。

评分器 `eval/llm_row_scorecard.py` 现在要求每条记录的可见历史、当前 `issue_state` 和 prompt hash/内容相互匹配，单写 `visibility_policy` 字符串不能取得 `deployable_claim_allowed=true`。该验证证明记录符合本地输入合同；它不证明传感器本身真实、下发动作得到执行，或建议产生了因果收益。

在数据库的比赛 `1778668127515`、红方 11 秒做了真实快照烟测：按前缀查询构造 4 帧历史、12 个机器人和 2 个我方建筑；可见输入验证通过。精简后的 prompt 为 3,476 字符，用本地 Qwen2.5-3B tokenizer 得到 2,104 token。旧 `llm_backtest.py` 默认截断上限 576 token，因此不能将新输入直接塞进旧推理路径；正式推理需要提高上下文上限，并在 token 超限时显式拒绝而非静默截断。

## 现有本地模型与评估数据

- Qwen2.5-3B 基座及多套 LoRA adapter 已在 `models_hf` / `out_m2`，CUDA 设备可用。
- Open-JEV 2B 本地 checkpoint、Qwen3.5-2B 基座和本地源码在项目中；当前 8791 本地服务没有监听。Open-JEV 是候选决策头，返回候选概率，不生成分层 JSON。
- 已有 128 条、22 场、红蓝各 64 的 Qwen / Open-JEV 配对数据只对应**全局态势三分类**。它们用同一可观测特征窗口和样本键，但两个推理接口的 prompt/候选打分形式不同，不能称为逐字相同 prompt、相同解码的动作建议比较。两者四个时距的类别输出都坍缩到“进攻”，逐项准确率相同；详见 `2026-09-27-llm-probe-probability-audit.md`。
- 本次实验之前，没有 Qwen / Open-JEV 逐行、指定红蓝视角且经上述输入合同验证的**动作建议**及其后续行为记录。旧 Qwen 159 点文件只有聚合量，`llm_backtest` 旧路径使用完整回放当前帧；不能从那些旧数据计算两模型动作建议的可执行率或行为响应率。本次新生成的 8 行二选一 smoke 见下一节，其规模和候选覆盖也不足以给出决策优劣排名。

## 8 行本地配对动作 smoke

`eval/llm_visible_paired_smoke.py` 冻结 2 场 × 红蓝 × 60/180 秒的 8 个输入到 `out_m2/llm_visible_smoke/paired_inputs.jsonl`。每行附有原始可见输入、共同的哨兵二选一候选集（推进、坚守）、由 `jev.api.candidate_prompts` 编译出的**相同候选提示字节**及各自 SHA-256。输入查询逐行只读建议时刻之前的遥测。Qwen2.5-3B + `finetuned_lookahead` LoRA 和 Open-JEV 2B 本地 checkpoint 顺序跑完；无下载或训练。选择规则均为候选概率最大值，零采样。Qwen 根据同一提示后的 Yes/No 下一 token 对数几率给候选打分，Open-JEV 使用训练过的标量决策头。内部打分方式不同，因此不能称为“同一解码器”或完整策略公平排名。

| 指标 | Qwen | Open-JEV |
|---|---:|---:|
| 可见输入与决策输出行 | 8/8 | 8/8 |
| 红 / 蓝 | 4 / 4 | 4 / 4 |
| 哨兵动作分布 | 推进 8、坚守 0 | 推进 0、坚守 8 |
| 平均单行候选评分延迟 | 3.67 秒 | 2.07 秒 |
| 目的功能区、首步功能区可评分覆盖 | 0/8 | 0/8 |
| 3 秒精确未来遥测覆盖 | 8/8 | 8/8 |
| 3 秒相应区域行为代理 | 推进后换区 0/8 | 坚守后仍在区 8/8 |
| 10 秒精确未来遥测覆盖 | 8/8 | 8/8 |
| 10 秒相应区域行为代理 | 推进后换区 2/8 | 坚守后仍在区 6/8 |

逐行结果及完整评分卡：`out_m2/llm_visible_smoke/{qwen,jev}_rows.jsonl` 与 `{qwen,jev}_scorecard.json`。样本仅 2 场、固定 60/180 秒，且哨兵多停在本区；“坚守”在这种样本上自然更容易匹配回放。没有动作下发或确认日志，不能计算真正执行率。两种输出都没有目的区或目标 ID，路线合法率和目标 HP 效果无法评价。该 smoke 验证的是可见输入、相同候选提示、逐行记录和后验评分管线，并暴露了当前二选一建议不够具体的问题；不能据此判断哪种模型产生更好的战术收益。

硬件为 NVIDIA RTX 5060 Laptop GPU（8 GB），两模型均以 4-bit 顺序加载。Open-JEV 服务参数为 `--max-length 4096 --batch-size 1 --no-prefix-cache`；Qwen 使用 4096 token 上限且禁止截断，实际候选提示为 2,202–2,404 Qwen token。权重校验 SHA-256：Qwen adapter `59E715821F4D8FD49395A1E5B7E70D05DF0C7F64BA9793EAB5159475D555828E`；Open-JEV adapter `2D23935B1A7380DB444ABAC572C04646918BA794E59002D1588236182A3CA18F`；Open-JEV head `3532CD576C58D5AD5BF17C3E9F2DF4BE8C70E08C07FA6BB7FA673DCD7B401F2A`。服务已在实验后关闭。

复现命令（先运行 Open-JEV 本地服务；两模型顺序运行以免抢显存）：

```powershell
python -m eval.llm_visible_paired_smoke
python -m eval.llm_visible_paired_run --model jev
python -m eval.llm_visible_paired_run --model qwen
python -m eval.llm_row_scorecard --rows out_m2/llm_visible_smoke/jev_rows.jsonl --out out_m2/llm_visible_smoke/jev_scorecard.json
python -m eval.llm_row_scorecard --rows out_m2/llm_visible_smoke/qwen_rows.jsonl --out out_m2/llm_visible_smoke/qwen_scorecard.json
```

要评价完整动作建议，需冻结可达目的区和首步区候选，加入明确的风险/目标字段，并将候选覆盖、拒答和路线合法率一起评分。3/10 秒移动、守区和发弹只能称为**观察性行为响应**。目标 HP 需传感器显式目标 ID，控制器执行率需命令/确认日志，目前两项覆盖为未知。

验证：`python -m pytest tests/test_llm_visible_input.py tests/test_llm_row_scorecard.py tests/test_llm_backtest.py -q`，15 项通过。泄漏测试覆盖未来帧改写不影响 prompt、未来测量拒绝、非协议 provider 拒绝、敌方隐蔽字段筛除、停更轨迹不暴露测量字段、伪造可见策略字符串不通过评分门禁、数据库只查询时间前缀。Qwen 与 Open-JEV 8/8 行推理及两份评分卡已成功生成。
