# RMUC 2026 M2(RL + LLM 融合蒸馏)设计

> 上游:M0/M1/M3 已 COMPLETE。M3 已建立规则派生的分层决策(M1/M3 三方引擎:LLM / 博弈论 / 规则)。
> 本 M2 新增 **RL 引擎** + **三源融合蒸馏** + **QLoRA 微调**,把「世界状态 → 分层宏观决策」的映射教给本地 LLM(替代弱 prompt 基线)。

**目标(一句话):** 用真实赛局决策点驱动的多单位全队 MARL(IPPO)学出 RL 决策头,与规则头、LLM 哨兵头做三源置信融合,再以 QLoRA 蒸馏进 Qwen2.5-3B(1.5B 兜底),使 LLM 能直接从原始世界状态产出合法、局势感知的分层决策。

---

## 0. 场景与约束(从项目记忆 / 计划继承)

- 平台:Win11,GPU = **RTX 5060 Laptop 8GB**,torch 2.11 + cu128。硬件是硬约束 → 3B 需 4bit NF4 + 梯度检查点;OOM 则 1.5B。
- 数据:`数据与资料/rmuc_2026_region_dataset.sqlite`(gitignored,只读)。`matches` 表有 `game_id` 列(非 `id`)。
- 数据规则:所有机器人首帧时刻秒=1.0(无晚生成);建筑(前哨站/基地)坐标恒为 0 → 数敌方单位时**必须排除建筑**。
- 依赖:已装 `transformers 4.57 / peft 0.20 / datasets 5.0 / accelerate 1.12 / gymnasium 1.2`;**`trl`、`pettingzoo` 缺失 → M2 RL 用自研 IPPO(纯 PyTorch ActorCritic)**。
- **测试约束:** 单元测试**不得 import transformers/torch**;`data.db` 只读允许;RL 纯逻辑/分集/融合/黑板的测试均离线。torch 只在训练入口 `finetune/lora_train.py` 与 `rl/train.py` 的 `train()`/`rollout()` 内**惰性 import**(沿用 `model/predict.py` 模式)。
- `.gitignore` 只增不改;不提交任何 `.env`、`out_m2/`、`.sqlite`、`report.json`、`*_report.json`。

---

## 1. 分层决策 Schema(复用 `schema.layered.py`,标签唯一合法形态)

```
决策 = { 't': float, 'global_mode': 进攻|均衡|防守,
         'units': { 英雄|工程|步兵3|步兵4|空中|哨兵: {
            target, move, fire(bool), risk } } }
- TARGETS = 敌方哨兵|敌方英雄|敌方步兵|敌方工程|敌方前哨站|敌方基地|能量机关|飞镖|无
- MOVES   = 推进|回防|游击|坚守|绕行|无
- RISKS   = 高|中|低
```
每条融合决策必须通过 `schema.layered.validate(d)` → `(True, [])`。

## 2. 三源决策头(同一粒度:每单位类 4 通道)

| 头 | 来源 | 粒度 | 是否必选 |
|---|---|---|---|
| 规则头 `rule_head` | `run_m3.compute_decision_series(dps, b)` → `(decisions, prior, sigs)` | 全类分层 | 必选(保底/参照,已 100% 合规) |
| RL 头 `rl_head` | 自研 IPPO 在真实决策点上 rollout → `schema.layered` per-class | 全类(move/fire 学, target/risk 规则填充) | 可选(fallback 时无) |
| LLM 哨兵头 `llm_head` | DeepSeek 慢环(或离线跳过),`run_llm_m3.llm_state_to_intent` | 哨兵单类 | 可选(离线时无) |

## 3. RL 环境 `rl/env.py`(多单位全队 MARL,接地真实数据)

- **Episode = 一场真实比赛**的决策点序列(红 vs 蓝,step=3.0s)。不手搓战斗仿真——按真实位置/血量演变推进,reward 因此是「真实」RM 语义。
- **Agent = 每个红方单位类**(英雄/工程/步兵3/步兵4/空中/哨兵),共 6 个 agent,共享团队胜负条件 reward(合作式 IPPO)。
- **State**(每 step,扁平向量,全归一化): 每个红方单位类 + 每个蓝方单位类的 `(hp_ratio, norm_xy[2], dfire, vuln)`,加全局特征 `(红base_hp_ratio, 蓝base_hp_ratio, 红outpost_hp_ratio, 蓝outpost_hp_ratio, tick)`。缺失类以 0 填充,`dim` 固定。
- **Action**:每单位类 agent 的**离散** `(move ∈ MOVES, fire ∈ {0,1})` → 12 种组合(单策略输出 12 维 Categorical)。`target`/`risk` 由 `tree.global_mode.derive_unit_intent`/规则映射确定性填充(与 M3 一致),保证决策合法。
- **Reward(逐 step,来自 RM 胜负层级)**: 见 §4。回合末叠加 `win_bonus`。
- **转移**:对固定轨迹非交互式;每一步 env 直接取 `build_worldstate(dp, bounds)`,策略仅影响「选择哪个动作→对应决策」,reward 由该决策点的真实特征净变化给出。用 `truncate` 按真实时长。

**向量化**:`step()` 对整局预计算的决策点索引推进;支持 `reset(seed)` / `step(action_batch) -> (obs_batch, reward_batch, dones_batch, truncs)`。**纯 numpy,无 torch**。

## 4. 奖励塑形(已按 RMUC 规则定案)

RMUC 胜负判定层级(时间耗尽/基地被毁时依次比较):**①基地血 > ②前哨站血 > ③哨兵可战亡次数 / 哨兵血 > ④全队攻击伤害 > ⑤全队总剩余血量 > 平局**。参考血量:**基地 5000、前哨站 1500**。

逐 step 复合 reward(`composite_rew = Σ w_i * term_i`):

```python
R = 1.0 * 胜利                      # win_bonus, 仅回合结束 ±1
  + 0.5 * (红base_hp - 蓝base_hp)/5000
  + 0.3 * (红outpost_hp - 蓝outpost_hp)/1500
  + 0.2 * (红哨兵存活×hp - 蓝哨兵存活×hp)/1.0
  + 0.1 * (红队累计伤害 - 蓝队累计伤害)/TOTAL_DMG_REF
  + 0.2 * 存活单位比例               # 探索塑形, 防早期单位全灭的稀疏信号
```

`TOTAL_DMG_REF` 取该场双方累计发弹归一化上限(约 3000);每项按对战局整体作用 clip 到 `[-1,1]` 再加权,使量纲可比。**`win_bonus` 用等级判定判定胜负**(`_win_condition(w_end)` 按 §层级比较 base→outpost→哨兵死亡数→哨兵血→伤害→总血)。

## 5. RL 训练 `rl/train.py`(IPPO)

纯 PyTorch ActorCritic(小 MLP:hidden 64×2,+ action 12 维 Categorical + value head)。`train()` 惰性 import torch。超参:`lr=3e-4, gamma=0.99, gae_lambda=0.95, clip=0.2, epochs=4, batch=512, rollout=2048, ent_coef=0.01, vf_coef=0.5`。

产出:
- `train(env, n_steps, device)` → 保存策略 `rl/out/ppo.pth` + 训练统计。
- `rollout(policy, dps, bounds, device)` → 对给定决策点序列,每个 agent 选 `(move/fire)`,返回 `rl_head` 决策列表(经 `distill_labels`)。

## 6. 蒸馏标签与融合

### 6.1 `rl/distill_labels.py`
输入:策略 + 决策点;输出:`schema.layered` 决策列表(`rl_head`)。内部:跑 `rollout`,对每决策点填 `target/risk`(规则映射)、定 `global_mode`(博弈论 `choose_global_mode` 或派生)、`build_empty_decision` 起底 → `validate` 保证合法。

### 6.2 `labels/transfer.py`(跨单元迁移)
矩阵 `TRANSFER_SOURCE = {'哨兵':'工程', '工程':'哨兵', '空中':'工程', ...}`(后排/前排类在两队役中的站位相似)。当某类可达rollout样本稀疏(覆盖 < 阈值)时,复用源类策略的 `(move/fire)` 生成增广训练对,并给一个 `transfer_confidence`。产出:增广后的标签序列 + 每样本置信度。纯 numpy。

### 6.3 `labels/fusion.py`(三源置信融合 + 冲突标记)
每个决策点、每个单位类,对 (move, fire) 在 `rule_head / rl_head / llm_head(仅哨兵)` 之间做**加权置信融合**:
- `conf(rule)=0.5, conf(rl)=0.3, conf(llm)=0.2`(规则头为强先验;RL 头随收敛提升;LLM 头仅哨兵)。
- 三头一致 → 直接取;不一致 → 取**加权最高置信**值,并在该点标记 `conflict=True` 写入 `conflict_rate`。
- 融合后逐条 `validate`,不合法丢弃并记录。
产出:`fused_decisions`(每点含 `conflict` 标记)+ 报告(head 契合度 / conflict_rate)。

### 6.4 `blackboard.py`(异步黑板)
线程安全的共享存储,供三引擎管线(RL 产生线程 / LLM 慢环线程 / 融合消费线程)解耦。API:`put(key, value)`、`get(key, timeout=None)`、`get_or_default(key, default)`、`snapshot()`、`subscribe(key, callback)`、`close()`。用 `threading.Lock + Condition + dict`,无 torch,纯离线可测。

## 7. QLoRA 蒸馏 `finetune/`

### 7.1 `finetune/make_dataset.py`
把 `fused_decisions` + 对应 `worldstate` 渲染成 `instruction → completion` 对:
- **instruction**:来自 worldstate 的**紧凑中文局势摘要**(`_situation_digest(w)`):双方各单位 hp/位置/dfire、基地与前哨站血量比、全局模式诱发特征。可复用/仿 M3 的 digest 逻辑。
- **completion**:分层决策的 JSON(`json.dumps(ensure_ascii=False)`,已保证合法)。
- 写 jsonl + 返回 `datasets.Dataset`;纯函数、无 torch、离线可测。

### 7.2 `finetune/lora_train.py`(QLoRA)
**惰性 import torch/transformers/peft/datasets**。配置:
- `load_in_4bit=True, bnb_4bit_compute_dtype=float16, bnb_4bit_quant_type=nf4, bnb_4bit_use_double_quant=True`。
- `LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05, target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'], task_type='CAUSAL_LM')`。
- 训练:`batch_size=1, gradient_accumulation_steps=8, max_len=512, lr=2e-4, epochs=1-2, gradient_checkpointing=True`。
- 保存:LoRA adapter `out_m2/finetuned/adapter` + 合并后`out_m2/finetuned/model`。
- 返回 `(model_path, stats)`。base 名称 `Qwen/Qwen2.5-3B-Instruct`,OOM 自动回退 `Qwen/Qwen2.5-1.5B-Instruct`。

### 7.3 `finetune/eval.py`
加载合并模型,在留出决策点上 decode → `schema.layered` → 算 `legal_rate / agreement_with_rule_head / compliance_rate / conflict_rate`。这是「LLM 直接读原始世界状态」的诚实结果,对比 prompt 基线(原始 0.25)。torch 惰性 import。

### 7.4 `run_m2.py`(管线编排)
`--stage {fusion|dataset|train|eval|all}`。默认 `all`:
```
db → build_decision_points(dps) → rule_head
   → rl.train (IPPO) → rl_head
   → (可选 llm_head)
   → labels.transfer 增广 + labels.fusion 融合
   → make_dataset → Dataset
   → lora_train (QLoRA 3B/1.5B)
   → eval (留出) → out_m2/m2_report.json
```
关键:**rule_head 保底**,RL/LLM 头可按 `--skip-rl/--skip-llm` 关闭,使 `all` 在 RL 不收敛时仍能从规则头 + 融合出全流程(保证过夜有完整结果)。

## 8. 文档 / 产物

- `rl/out/ppo.pth`(gitignored)、`out_m2/finetuned/`、`out_m2/m2_report.json`、`out_m2/` 数据缓存。均 gitignored。
- 诚实报告:`m2_report.json` 记录 3B/1.5B、训练统计、留出 eval(P4 目标:legal≥0.95, agreement-with-rule 显著高于 prompt 基线 0.25 且 < 1.0——即「LLM 学到局势感知但不过拟合规则头」)。

## 9. 单元测试(离线,无 torch/transformers)

- `rl/env.py`(numpy) → 维度/动作/奖励边界/`build_empty_decision` 还原/胜负判定。
- `rl/distill_labels.py`(注入假策略) → 产出全合法、target/risk 填充一致。
- `labels/fusion.py`(注入三头假决策) → 三源一致/分歧/冲突标记。
- `labels/transfer.py`(构造稀疏类) → 增广样本 + 置信度。
- `blackboard.py` → put/get 阻塞超时/subscribe/release。
- `finetune/make_dataset.py` → digest 渲染 + 决策 JSON 往返 + validate。
- 全跑 `python -m pytest`(不加载任何 torch/transformers)。
