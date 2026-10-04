# 反事实最优承诺标注 (Counterfactual Optimal-Commitment Labeling) — 设计文档

**Goal:** 用"前向仿真 + 敌方博弈应对 + 单点胜负机"把训练数据的状态标注从**行为锚定**升格为**最优性锚定**,产出比回看(look-ahead)标签更权威的 `global_mode` + 六单位动作。

**Architecture:** 对每个决策点,枚举我方三个候选 `global_mode`,用现有符号物理 `step_robots` 逐帧推演 T 帧;敌方每帧按"对我方提交模式的最优应对(best_response)"行动;终点用一个**单点 win-prob 逻辑回归**(CV 校验)打"红方胜算"分;取得分最高的模式为该点标签——即"我方最优承诺(If I announce X, my opponent best-responds with Y, expected win is Z)"。复用 `derive_unit_intent`/`step_robots`/`best_response`,新增敌方推压 roll-in 与单点胜率机。

**Tech Stack:** 纯 numpy(无 torch/transformers,满足测试约束);`tree/global_mode.py`, `rl/physics.py`, `game/game_theory.py`, `rl/reward.py`, `rl/reward_learn.py`, `eval/outcome_backtest.py`, `data/features.py`, `worldstate/worldstate.py`。

**Spec:** 本文自身。承接 `docs/superpowers/specs/2026-09-08-rmuc-m2-design.md`。

---

## 1. 问题:为什么现有标注不够权威

M2 训练数据的状态(`global_mode` + 六单位动作)目前来自**瞬时逐帧规则**(`label_sentry`)或**回看窗口规则**(`relabel_decision_series`,Δ=5帧/margin=0.12)。两者都是**行为锚定**(描述队伍当时做了什么),不是**最优锚定**(此刻该做什么)。权威性短处:

1. **行为≠最优**:队伍当时可能就做错了,行为标签会继承错误。
2. **参数主观**:**Δ 与 margin 手设**,改变它们标签会变(标签不稳)。
3. **规则循环**:回看标签由规则派生,学生模型只能复刻老师,蒸馏上限被锁死(实测模型 vs 回看标签吻合率 0.566 ≈ 规则自身 0.562)。
4. **per-unit 复用规则**:六单位动作由 `derive_unit_intent` 从同一 mode 派生,与模式高度捆绑,不独立。
5. **无胜负锚**:瞬时/回看标签都不与"真实胜负"直接挂钩(仅 outcome_backtest 测过规则的相关性)。

**结论**:要更权威,须把标签锚到**我方在数学仿真 + 敌方最优应对下的预期胜率最大者**——这是"最优性"而不是"行为"。

## 2. 权威的定义与落点

把标注问题从"**什么是真实发生的**"改成"**什么是此刻的最优承诺**":

> **最优承诺 = argmax_m P(红胜 | 我方提交 m, 敌方按最优应对 e_m 行动, 推演 T 帧后的局面)**

其中 `e_m = enemy_best_response(m)` 是敌方对我方提交模式的真博弈应对。这是对策论"最优反应均衡"在标注层的落点。

**权威上限(诚实)**:此标签是"**在我们模型认知(符号物理 + 单点胜率机 + Payoff 矩阵)下的全局最优**",不是绝对真值。唯一能凌驾它的是**人类专家评审**。因此配套验证 rig(§9)用以证明标签与真实胜负对齐;专家抽样评审列入 optional。

## 3. 标注算法(逐决策点)

```
输入: w_t = build_worldstate(dp_t, bounds); 红方 team='红'
      枚举集合 M = GLOBAL_MODES = ('进攻','均衡','防守')
      推演帧数 T; 单点胜率机 win_proba(feats)

对 m in M:
  e_m    = enemy_best_response(m)       # 敌方对我方提交模式 m 的对抗应对(最小化我方 payoff;模式级, 窗口内固定)
  live   = copy(dp_t['robots'])         # 物理态(保留 maxhp/power/... )
  traj   = [w_t]
  for step in range(T):
      w_now = worldstate_from_live(live, bounds)                                   # 当前局面
      my_act = { cls: derive_unit_intent(my_unit_cls, w_now, m)     for cls in 红方六单位类 }  # 我方按模式 m 逐帧派生
      en_act = { cls: derive_unit_intent(enemy_unit_cls, w_now, e_m) for cls in 蓝方六单位类 }  # 敌方按 e_m 逐帧派生
      live   = step_robots(live, team,       phys(my_act), dt, params, enemy_fire=False)  # 我方行动(关返回敌火,R-E6)
      live   = step_robots(live, enemy_team, phys(en_act), dt, params, enemy_fire=False)  # 敌方行动(关返回敌火,R-E6)
      traj.append(worldstate_from_live(live, bounds))
  score_m = win_proba( window_features(traj, team) )                              # 单点胜率机评分

标签 = argmax_m score_m     # 平局时: 优先 '均衡' 作为不冒险默认, 注记 (scores, margin)
```

要点:
- **承诺的是"模式",不是"每帧微动作"**:`derive_unit_intent` 读取当前单位的 hp/vuln/位置/mode,单位血量随推演逐帧变,故敌我双方的单位级动作都**每帧从 live 重派生**——对称、真实。
- **敌方模式 e_m 在窗口内固定**(一次最优应对,不逐帧翻转),保证推演稳定、确定、可复现。
- `score_m` 是**校准概率**(§5),不是手加权和。

## 4. 组件与接口

### 4.1 复用(不改)
- `tree/global_mode.derive_unit_intent(unit, worldstate, global_mode) -> {'target','move','fire','risk'}`
  - 纯函数,由 unit 的 `hp_ratio/vuln/dfire/norm_xy` + mode 唯一决定,确定性无 seed。
- `rl/physics.step_robots(robots, team, actions, dt, P=None, enemy_fire=True) -> robots`
  - `actions={cls:{'move','fire'}}`;**只推演 `team` 侧单位**,敌方单位不移动、仅半径内开火。通过"我方一次、敌方一次"两次调用实现敌我双方活动。
  - **`enemy_fire` 参数(修正轮 R-E6)**:`step_robots` 末尾恒跑 `_enemy_fire`,故"我方+敌方"两次调用会在射程内让**双方各被开火两次**(自身 action-fire + 对方视角敌火 → 双算伤害),污染 rollout→胜率机→标签。为消除双算,`rollout_once` 的**两次调用均传 `enemy_fire=False`**:每侧每帧只经自身 action-fire 开火一次。默认 `enemy_fire=True` 使 PhysicsEnv 单次调用行为不变(既有物理测试原样通过)。
- `game/game_theory.best_response(matrix, opp_strategy) -> idx`, `PAYOFF_3`。
- `worldstate.worldstate.build_worldstate(dp, bounds)`(用 live 重建,依赖保留的 `maxhp` 等)。

### 4.2 新增
**`labels/rollout_label.py`** — 反事实最优承诺标注器(纯 numpy,无 torch):

- `enemy_best_response(our_mode, matrix=PAYOFF_3) -> mode`
  ```python
  i = GLOBAL_MODES.index(our_mode)
  row_i = matrix[i]                         # 我方提交 i 时,随敌取 j 而变的 payoff
  return GLOBAL_MODES[min(range(3), key=lambda j: row_i[j])]
  ```
  **语义(修正轮 R-E7-对抗)**:敌方视我提交的**纯策略**,取**使我方 payoff 最小**的应对模式——因为我方在 n 行,敌取 j 造成我方 payoff=PAYOFF_3[i][j],敌方欲最大化其自身收益即**最小化我方 PAYOFF_3[i][j]** ⇒ `argmin`。纯策略(非混合)的上限见 §10;此修正把标签从"乐观承诺(合作)"扳回"最优承诺(对抗)"。
- `commit_actions(worldstate, mode, team) -> {cls: {'move','fire'}}` — 遍历 team 六单位类,`derive_unit_intent` 后取 move/fire。
- `rollout_once(worldstate0, dp0, bounds, my_mode, enemy_team, T, dt, params) -> [w0..wT]` — 初始 `live=copy(dp0['robots'])`;`e_m = enemy_best_response(my_mode)`;每帧:由 `worldstate_from_live` 得 `w_now`,再 `derive_unit_intent` 派生我方(`my_mode`)/敌方(`e_m`)动作,分两次 `step_robots`,追加重建 worldstate。返回轨迹序列。
- `relabel_rollout(ws_list, dps, bounds, T, team='红', ...) -> (rollout_decisions, labels, scores)`
  - `labels[i] = argmax_m win_proba(window_features(traj_m))`;`rollout_decisions[i]={global_mode, units 由 derive_unit_intent 重派生}`(与回看标签同构,便于替换训练源);`scores[i]={m:prob}` 供注记/分析。

**`eval/win_proba.py`** — 单点胜率机(纯 numpy,可复用 `rl.reward_learn.fit_logistic`/`auc`/`outcome_backtest.cv_auc_win_model`):

- `def window_features(worldstates, team='红') -> np.array([f1..fk])` — 对 rollout 的 T+1 帧调用 `reward.reward_components` 并**累加**(每帧 k 项),得短窗窗长特征向量(非整局)。取 `outcome_backtest.WIN_FEATURES`(胜相关组件)为主,或全 `LR_FEATURES`。
- `def fit(features_matrix, win_labels, l2, seed)` / `def predict_proba(feats, w, b)` — 复用手写 GD 岭逻辑;`feats` 为逐决策点短窗特征,`win_labels` 为该局红方是否胜。
- `def cv(k, ...)` — K 折 held-out AUC/acc(镜像 `cv_auc_win_model`),并在主流程输出该 AUC 作为"单点胜率机可信度"的资格证。

## 5. 单点胜率机(终点打分)的关键细节

- **为什么不能直接用 reward_learn 的整局权重**:reward_learn 的 `P(win)` 是**整局累加特征**(幅值几十)学出,喂进 15-30s 的短 rollout(特征幅值 ~整局几分之一)会让 `z = w·X + b` 塌到先验(≈ sigmoid(b)),失去区分度。
- **正确做法**:另训一个**特征=逐决策点短窗累加反馈**、**标签=该局红方是否胜** 的单点 win-prob;CV 校验它确实判别胜负(held-out AUC≥某阈值,如 ≥0.75),才允许其作为标签打分器。这一步是不可跳过的"权威资格证"。
- **特征构成**:`window_features` 累加 `reward_components`(对 team 的归一化胜势势能),与 `outcome_backtest.WIN_FEATURES` 对齐;可拆分成"损失前 T/物理后 T"对比窗口。

## 6. 敌方 push 的 roll-in(本设计唯一的仿真行为新增)

`step_robots` 只推演传入 `team` 的单位,敌方不移动。为让反事实有区分度(尤其"防守 vs 均衡"),把**敌方活动**通过"以敌方为 team 再调一次 `step_robots`"实现——敌方单位同样走 `derive_unit_intent(敌方单位, w, e_m)` 得到 move/fire,再模拟其推进/开火。这样**不新增任何物理规则**,只是把现成 motion/fire 逻辑对称地施加到敌方。敌方以 `e_m`(对我方提交的最优应对)行动,即"真博弈 push"。

## 7. 数据流与假设

- 数据源:训练用决策点 `features.build_decision_points` + `features.field_bounds`(只读 `data.db`)。
- **范围**:对训练用 ~24 局(ca. 3317 决策点×3 模式×T 帧)重标注。标注集=训练集;留出评估集另行计算(避免把评估集卷进训练标签分布)。
- **maxhp/power 保留**:`step_robots` 用 `dict(r)` 全量复制,保留 `maxhp/power/small_heat/big_heat/vuln` 等原决策点字段 → 重建 worldstate 时 `hp_ratio/norm_xy/vuln/dfire` 齐备。
- **minor 偏移**:`step_robots` 改的是合并 `heat`,不改 `small_heat/big_heat`,故重建 worldstate 的 `heat` 字段略滞后(仅影响哨兵过热惩罚,量级小,可接受;后续如需精确可在 push 时分裂回 small/big)。

## 8. 结果别名与命名(必须遵守)

**数据集文件必须携带标注方法后缀**,避免同名覆盖(上轮事故教训)。产出:
- `out_m2/m2_dataset_rollout.jsonl`(反事实最优承诺标签)
- 对照保留 `m2_dataset_rebalanced_lookahead.jsonl`(回看标签,另行不复用旧名)
`labels` 字段及评估报告一律带 `method: 'rollout'` / 后缀。

## 9. 验证 rig(标注的可信度证明)

1. **结果一致性**:用单点胜率机对标注后各点打分;require 标"进攻"的点 mean(win-prob) > 标"防守"的点(即"进攻发生在高胜势区间")。若反向 → 标签机器与现实矛盾,标签不可信,停。
2. **horizon 消融**:T∈{3,5,8,12},报告跨 T 标签一致率(高=标签稳,填掉"参数依赖"短板)。
3. **vs 回看标签对照**:rollout 标签与 look-ahead 标签的同一点一致率;一致性适中(不同构)属预期,报告即可。
4. **重训回归**:用 rollout 标签重训(3B,同超参),报告与回看标签基线在 legal_rate / 与 rollout 标签吻合率上的变化;若 rollout 标签使一致率下降**不应**视为退化(因为标签更权威),应同时看 outcome-align gap 是否改善。

## 10. 诚实上限(权威边界)

- 单点胜率机是"单点→胜负"的**相关/短程预测**,非因果胜率(游戏尚未结束)。
- `PAYOFF_3` 是**手设先验**;敌方 best-response 只做了"模式级"应对,未建模敌方格斗/换策略。敌方策略变化仍不可预测(离线 RL 天花板)。
- 仿真是**区域符号物理**,非完整调度/计分规则。
- **人类专家评审仍是唯一能凌驾此标签的裁判**(可选;若可得,抽样 (worldstate→标签) 评审)。

## 11. Out of scope

- 不用 rollout 标签去训练"评分函数本文之外"的新模型;不改为全对抗性 rollout(Nash 混合策略,逐帧翻转应对)——留给后续。
- 不优化标注耗时(用户明示不讲效率;仅在真实训练/评估环节考虑速度)。
- 不改 `data.db`;不 commit `out_m*`/`_work` 产物。

## 12. 测试计划(纯函数,离线,无 torch)

`tests/test_rollout_label.py`:
- `enemy_best_response('进攻')` 应对在我方进攻下敌方取防守/均衡(依 PAYOFF_3 转置预期)——断言返回 `GLOBAL_MODES` 合法且符合转置 argmax。
- `commit_actions(w, mode, '红')` 对六个类各返回合法 move/fire(打 `sl.validate` 可过的 intent)。
- `rollout_once(...)` 在合成 worldstate 上推演 T 步后:单位 hp 单调不减/可降、位置随 move 改变(推进 x 增、回防 x 减),建筑掉血路径正确(用 `physics.py` 既有语义单测模式)。
- `window_features` 对 T 帧累加返回定长 k 向量,数值随"己方基地掉血/敌方前炮被毁"单调变化。
- `fit/predict_proba/cv` 在合成特征+标签上:预测与标签相关、CV 返回有限 AUC。
- `relabel_rollout` 对一小段真实 dps 返回与输入等长的合法 decisions + labels ∈ GLOBAL_MODES + scores 字典。

## 13. File map

| File | Action |
|---|---|
| `labels/rollout_label.py` | Create(标注器:enemy_best_response / commit_actions / rollout_once / relabel_rollout) |
| `eval/win_proba.py` | Create(单点胜率机:window_features / fit / predict_proba / cv) |
| `tests/test_rollout_label.py` | Create(纯函数单测) |
| `finetune/make_dataset.py` | Modify(接入 rollout 标签生成训练 jsonl;沿用 `_rollout` 后缀命名) |
| `RMUC2026_M2_优化分析与方案.md` | Modify(记录 rollout 标注 + 验证 rig 结果) |
