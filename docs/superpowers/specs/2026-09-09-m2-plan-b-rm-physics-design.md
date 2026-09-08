# M2 Plan B: RM 规则化对称物理 — 设计文档

**Goal:** 把符号物理从"回放标定/任意常量"重锚到 **RMUC 2026 规则权威量纲**,并让仿真环境**双方对称**——先修好"反事实 rollout 标注退化(恒守)"的物理层根因,再为 RL(轨2)提供真 MDP。做完后重跑标注,验证攻/守/均衡分布不再退化为单模。

**范围:** `rl/physics.py`(物理重锚定 + 意图化目标选择 + calibrate 修复)、`rl/env.py`(对称 PhysicsEnv)、`rl/train.py`(PPO ratio bug)。**不覆盖** Track1 LLM 重训、Track2 RL 长跑、融合重算——那些是下游,依赖本 plan 完成后的真实物理。

**Tech Stack:** 纯 numpy(无 torch),单测离线可跑。

**本文自身即 spec。** 承接 `docs/superpowers/specs/2026-09-09-rollout-labeling-design.md`(Plan A)与 `docs/superpowers/specs/2026-09-09-m2-retrain-eval-design.md`(Track1)。

---

## 1. 为什么必须做 Plan B(诊断结论)

Plan A 把标注从"行为锚定"升到"最优锚定",但反事实 rollout 依赖符号物理。诊断发现 **rollout 标注退化为恒守(97% 防守)的根因在物理保真度,不是评分器**:

1. **地平线内打不到胜负信号所在**——基地(BASE)在 x_norm≈0.92,前哨站在 ≈0.39/0.61。单位每帧只挪 ~0.04 norm(真实 ~0.3–0.55 m/s × dt=3s / 28m),T=5(15s)仅 0.19 norm。**进攻单位 T 帧内能摸到敌方前哨站,但摸不到基地;而胜负/fav 主要信号在基地血量** → 攻/守 counterfactual 无胜负差。
2. **目标选择不按意图**——`_apply_damage` 选"最近敌方任意物(含建筑)",`derive_unit_intent` 的 target 被忽略。防守单位也能"打掉敌方前哨站",攻/守轨迹的成败信号互相污染。
3. **常量非 RM 量纲**——`DMG=30/120`、`RANGE=12`、`FIRE_RATE=3/1.5`、`HEAT_MAX=100` 无规则依据;`SPEED` 靠 `calibrate` 从回放估,但抽样很碎。物理量纲与真实 RM 不对齐 → "进攻兑现战果"的条件不具备。

**结论: 先做 Plan B(按 RM 规则的真实物理 + 对称双方),物理真实后重跑标注,再写 Task 5。** 用户已裁决"先 Plan B 再重标(治本)"。

## 2. 权威常量(rm-battlescope + RMUC 2026 规则索引,带来源标记)

这些来自对 RMUC 2026 区域赛数据解析工具 `ezthor/rm-battlescope`(克隆到本地 `/tmp/rmb`)的实证,以及规则索引。**凡标 (P) 的为该项目实证值;标 (R) 的为规则索引/手册引用值;个别查不到精确值的标 (D) 为合理默认、需下游以真实规则核对。**

### 2.1 场地与坐标 (P, field.py)
| 量 | 值 |
|---|---|
| 场地 | 28 m × 15 m |
| x 轴 | 红方(左)→蓝方(右); y 轴 | 画布下→上 |
| 红基地 | (2.46, 7.44) | 蓝基地 | (25.73, 7.44) |
| 红前哨站 | (10.87, 3.58) | 蓝前哨站 | (17.12, 11.32) |
| 堡垒区 | 红(6.22,7.45)/蓝(21.78,7.45),半径 1.3 m |

> 实证: `features.field_bounds` 在真实局返回 ≈[0,27]×[0,15],即 raw 坐标**已经是 28×15 米帧**(99%+ 界内)。坐标归一化本无大碍。目标坐标固定,供"打基地/前哨站"判定用。

### 2.2 弹道 (P, combat.py)
| 量 | 17mm | 42mm |
|---|---|---|
| 弹速 m/s | 25.0 | 16.5 |
| 命中窗口 s | 2.0 | 3.0 |
| 热耗/发 | 10.0 | 100.0 |

### 2.3 伤害锚点 (P/R)
| 事件 | 伤害 |
|---|---|
| 飞镖打固定目标(基地) | 200 |
| 飞镖打随机固定 | 300 |
| 飞镖打随机移动 | 625 |
| 飞镖打末端移动 | 1000 |
| 英雄部署 42mm 打基地(regular/deployed) | 225 / 300(规则 V2.1 增基地 42mm +150%) |

### 2.4 胜负 (R)
- 比赛结束时**基地剩余血量高的一方获胜**(血量 > 前哨站 > 累计伤害 > 存活数的层级,见 `env.win_condition` —— 已正确)。
- 基地护甲删除"低于 2000 自动展开",改为其它机制触发。
- **英雄热耗上限**: 近战优先 1–10 级 200–300;远程优先 160–190。(R)

### 2.5 复活与增益 (R, revival.py/buffs.py)
- 复活进度 `round(10 + 阵亡时刻/10 + 20×此前立即复活次数)`;计时复活回 10% 血,立即复活回 100%。
- 地形增益(飞坡/中央高地/台阶)30 s;小能量机关 45 s;大能量机关 5–10 灯臂 → 30–60 s。

### 2.6 需默认/核对 (D)—— 权威源查不到精确值
| 量 | 建议默认 | 依据 |
|---|---|---|
| 基地最大血 | 3000 | 飞镖单发 200–1000、英雄 42mm 225/300 → 基地需千级才不秒爆;旧规则量纲参考,标 (D) 待核 |
| 前哨站最大血 | 750 | 同上,约为基地 1/4;标 (D) 待核 |
| 17mm 单发伤害(对单位) | 30 | RM 17mm 打装甲口径量纲,标 (D) |
| 17mm 单发伤害(对建筑) | 3 | 打建筑需大量发弹(见 2.4 层级),标 (D) |
| 42mm 单发伤害(对单位) | 75 | RM 42mm 重口径,标 (D) |
| 42mm 单发伤害(对建筑) | 90 | 英雄拆建筑主力,标 (D) |
| 17mm 有效射程 | 10 m | 弹速 25 m/s;中场单位够到前哨(5.3m)、够不到基地(11.7m) → 须推进才打基地(几何核算),标 (D) |
| 42mm 有效射程 | 12 m | 吊射,略长,标 (D) |
| 17mm 射速 | 10 发/s | RM 17mm 射速高(骚扰),标 (D) |
| 42mm 射速 | 2.5 发/s | RM 42mm 单发高伤+高热量,标 (D) |
| 地面/空中单位速度 | 哨兵0.1 / 工程0.35 / 英雄0.4 / 步兵3:0.7 / 步兵4:0.7 / 空中2.5 (m/s) | 有效机动速度(含停驻),实测 ~0.4–0.6 m/s 上浮,**刻意不快**——快则中场穿场直接打基地(反向退化);标 (D) |
| 基地最大血 | 3000 | 飞镖单发 200–1000、英雄 42mm 225/300 → 千级才不秒爆、但抗不住持续压;标 (D) |
| 前哨站最大血 | 750 | 约为基地 1/4;标 (D) |
| 17mm 单发(对单位/建筑) | 30 / 3 | 对建筑需大量发弹,标 (D) |
| 42mm 单发(对单位/建筑) | 70 / 90 | 英雄拆建筑主力,标 (D) |
| 冷却速率 | 每帧 `heat -= 25% * heat_max/dt` | 由 2.2 热耗 + 增益冷却倍率推出,标 (D) |

> **几何校验(为何默认值可接受):** 诊断证明"打不到基地"是**地平线 vs 速度 vs 目标坐标 vs RANGE** 的几何问题,而非某单发伤害的精确值。场地 28×15m,蓝前哨 x=17.12、蓝基地 x=25.73;中场(红单位 x_norm≈0.5,14 m)到蓝前哨 5.3 m、到蓝基地 11.7 m。**只要 RANGE≈10 m,单位在中场能打前哨、必须推进才能打基地**;配合意图化目标选择(攻击单位追前哨/基地,防守单位只打缠斗目标),攻/守 counterfactual 就会在"前哨攻伤 + dmg_deal + 击杀"上分化。精确伤害值不改变该定性结论。(D) 值尽量贴近规则量纲,并在 `RM_PARAMS` 字典里注释"需以真实规则 V1.x(区域赛)核对"。

## 3. 设计

### 3.1 常量驱动(zero 魔法数字)
`rl/physics.py` 顶部用一个 `RM_PARAMS` dict 集中所有规则量(§2.6 默认值 + §2.2/2.4 锚点),`step_robots`/`calibrate`/`_apply_damage`/`_enemy_fire` 全从它取。加 `RM_PARAMS_PROVENANCE` dict 记录每条来源标记(P/R/D),供文档与被审。

### 3.2 意图化目标选择(攻/守分化的关键)
`_apply_damage(shooter, ...)` 不再选"最近敌方任意物",而按**射手意图的目标类别**选目标:
- `derive_unit_intent` 给出 `target`(敌方哨兵/英雄/步兵/前哨站/基地/无)。`step_robots` 的 actions 里补一个 `target` 字段(rollout 的 `commit_actions` 已派生,直接透传;单步 PhysicsEnv/M2Env 的 `_phys_actions` 从 `derive_unit_intent` 补)。
- 单位在**该目标类别**内选最近者射击;若该类别为空(该敌类阵亡)或不在射程,则**不强求,无效不伤害**,但可回退到"最近敌方单位"(非建筑)以保留缠斗。
- 作用: 进攻单位追着敌方前哨站/基地打,防守单位只打缠斗目标,不再"顺手拆敌方前哨站" → 攻/守轨迹成败信号清晰。

### 3.3 对称 PhysicsEnv(RL 用)
`PhysicsEnv.step` 目前**单侧**调 `step_robots`(敌方静止)。改为:
- 每帧由 `derive_global_mode(w)` 读己方局势 → `enemy_best_response(ego_mode)` 得敌方应对模式 → `commit_actions(w, e_m, enemy_team)` 得敌方动作(M2Env 已复用)。在 `rl/env.py` 顶部 import `labels.rollout_label.{enemy_best_response, commit_actions}`。
- 调 `step_robots` **两次**: 我方 team 用 `_phys_actions(actions)`,敌方 team 用 `en_act`,两次都 `enemy_fire=False`(与 rollout 同构,避免 return-fire 双算)。
- obs/reward/win_condition/部分可观测/vis_radius 逻辑不改(只把 `_real_xy/_ego_sentry_xy` 保证取模拟 live —— 已实现)。

### 3.4 PPO ratio bug 修复
`rl/train.py:115-117` 的 `old_log_prob = log_prob.detach()` 使 `ratio = exp(log_prob - old_log_prob) = exp(0) = 1`,clip 恒不触发 → 退化为 A2C/REINFORCE+GAE。修法:
- 在 `_rollout_one_episode` 里,采样动作时**用 no_grad + 旧策略**计算并存下 `old_log_prob_decided`(即当次采样的 log prob);PPO 更新用**当前**策略在新 obs 上重算 `log_prob`,`ratio = exp(log_prob - old_log_prob_decided)`。
- 实现: `_rollout_one_episode` 返回 trace 里带 `old_log_prob`;`train` 组装 `old_log_prob_t`;`_ppo_step` 用它。为省显存/不引入第二个网络,可用"采样时在线旧 log prob"(同一网络前向时把 logits 先存下)——即 PPO 标准 off-policy: 存**行为策略**的 log-prob,更新时用行为策略 log-prob 作 old。

### 3.5 地图/边界
- `z`(高度)不建模(地面近似);`vuln`/`power`/`coins` 不参与物理,仅保留.
- 建筑为静态目标,坐标取 §2.1 固定值(替代 DB 的 (0,0) 占位);damage 打中后 `hp` 扣减,转 `hp_ratio`。

## 4. File map

| File | Action |
|---|---|
| `rl/physics.py` | Modify — `RM_PARAMS`/`RM_PARAMS_PROVENANCE` + 意图化目标选择 + 建筑坐标 + 速度/伤害/热量量纲；`calibrate` 改为给 RM_PARAMS 覆盖而非主源 |
| `rl/env.py` | Modify — `PhysicsEnv.step` 对称双 `step_robots`(敌方反应式对抗)；`_phys_actions` 补 target |
| `rl/train.py` | Modify — PPO ratio bug(存行为策略 log-prob) |
| `tests/test_physics.py` | Modify/Add — 意图化目标选择、射程、速度/伤害量纲、建筑坐标 |
| `tests/test_physics_env.py` | Modify/Add — 敌方也动、对称性断言 |
| `tests/test_train.py` | Modify/Add — PPO ratio≠1(clip 生效) |
| `RMUC2026_M2_优化分析与方案.md` | Modify(本次) — 记录 Plan B 与重标结果(原来 Task5 的活) |

> 测试约束: `tests/` 不 import torch/transformers;纯 numpy。`data.db` 只读。

## 5. 测试(TDD, 无 torch)

- `test_push_reaches_outpost_in_horizon`: 红步兵3 从 x_norm 0.55 起步,`推进` + `fire`,`T=8` 内**命中蓝前哨站**(其 `hp_ratio<1`)。证明"进攻兑现战果"。
- `test_defense_does_not_reach_enemy_outpost`: 红步兵3 从己方半场 `回防` + `fire`,`T=8` 内**不**命中蓝前哨站(单位未推进到射程),且打敌单位而非建筑。
- `test_target_intent_respected`: 单位 target=敌方基地 且基地在射程时,打基地;最近敌方**单位**在更近也不转移目标——除非敌方单位占据攻击优先。反例: target=敌方步兵 时不打建筑。
- `test_building_hp_not_instant_kill`: 英雄 42mm 打基地,T 帧内基地 hp_ratio 单调下降、但不为 0(量纲合理,不秒爆)。
- `test_physics_env_symmetric`: PhysicsEnv 下,敌方(蓝)英雄真实坐标在 step 后**改变**(敌方也动);红不动、蓝 `commit_actions` 推进 → 蓝 x 变化。
- `test_ppo_ratio_not_one`: PPO 更新后 `old_log_prob != log_prob`(首轮前向不同),ratio 由 clip 约束;给一个远离旧策略的动作,断言 `surr1 != surr2` 对某些 batch 元素(即 clip 真会触发)。
- 回归: 全量 `pytest` 通过;既有物理/PhysicsEnv 测试语义不变或按新常量更新后通过。

## 6. 成功判据(重跑标注)

Plan B 完成后,重跑 `python -m finetune.make_rollout_dataset --limit 24 ...`:
1. `label_dist_full` **不再单模**: 进攻/均衡 占比明显(≥10%,理想 20%+),而非 97% 防守。
2. `outcome_consistency` 的 `op_vs_def_gap > 0`: 标"进攻"的点在真实物理下其 Fav/胜机均值确实更高 —— 标签机器与物理现实对齐。
3. `win_model_cv.cv_auc` 仍在 [0.5,1] 且 >0.6(状态仍携带胜负信号)。
4. 全量既有测试无回归。

若仍单模: 说明伤害/目标/地平线还有一处不对,**停止**并回去查(按用户"每任务完成后检查"的指令),不硬闯下游。

## 7. 诚实上限

- 符号物理仍是**确定性的规则近似**: 不建模碰撞/地形/雷达反制/英雄部署模式/哨兵姿态/护甲展开/飞镖致盲全套。它只保证"攻/守/均衡在反事实内产生**与规则量纲一致的、可解释的**成败差异",不保证与真实对局逐帧一致。
- 敌方为**模式级反应式对抗**(best_response),非完整建模;最终权威性仍以"我方最优承诺"为上限,且受其影响(模型自我认知的全局最优,非绝对真值)。
- (D) 值需用户以真实规则 V1.x 核对;本 spec 已标注,改动只作用于 `RM_PARAMS` 一处。
- 区域赛数据早于全国赛 V2.1 规则;历史机制的规则版本差异已按索引说明标注。

## 8. Out of scope

- 不做 RL 长跑训练(轨2,需对称环境+PPO 修复后单独跑)、不做融合重算。
- 不做 LLM 重训(Track1)。
- 不改 `data.db`;不 commit 产物(`out_m*`/`_work`);`.gitignore` 只做增量添加。
- 不引入任何新训练模型/多超参扫描(8G 预算约束)。
