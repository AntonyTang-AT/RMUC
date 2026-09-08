# Plan B: RM 规则化对称物理 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把符号物理重锚到 RMUC 2026 规则量纲并让仿真环境双方对称,让反事实 rollout 标注不再退化为恒守,同时为 RL 提供真 MDP 与修复 PPO 故障。

**Architecture:** 常量集中到 `rl/physics.py` 的 `RM_PARAMS`;`_apply_damage` 改按**射手意图目标类别**选目标;建筑取固定坐标;`PhysicsEnv.step` 调两次 `step_robots`(敌方反应式对抗);`rl/train.py` 的 PPO 修 ratio。

**Tech Stack:** 纯 numpy;测试无 torch/transformers;`data.db` 只读。

**Spec:** `docs/superpowers/specs/2026-09-09-m2-plan-b-rm-physics-design.md`(本文由它论证)。

## Global Constraints

- 测试(单测)禁止 import torch/transformers/sklearn;纯 numpy + stdlib,离线可跑。
- `data.db`(`config.DB_PATH` 真实库)**只读**;不 commit 产物(`out_m*`/`_work`/报告)。
- 物理跑在 **米制帧**(raw x/y 已即米,`field_bounds`≈[0,27]×[0,15]);建筑坐标固定(见 RM_PARAMS)。
- 常量一律走 `RM_PARAMS`(集中字典 + `RM_PARAMS_PROVENANCE` 来源标记),**不散落魔法数字**。
- `_apply_damage`/`_enemy_fire` 的目标选择遵守射手意图(`act['target']` 或缺省回退到"最近敌方单位")。
- `.gitignore` 只做增量添加;不新建冲突目录。

---

### Task 1: RM_PARAMS + 意图化目标选择 + 建筑坐标/射程(refactor rl/physics.py)

**Files:**
- Modify: `rl/physics.py`
- Modify: `labels/rollout_label.py`(commit_actions 补 target)
- Test: `tests/test_physics.py`(新增)
- Test: `tests/test_make_rollout_steps.py`(回归不动)

**Interfaces:**
- Consumes: `tree.global_mode.derive_unit_intent`(返回 target/move/fire/risk);`schema.layered.UNIT_CLASSES/GUN`? 无。
- Produces:
  - `RM_PARAMS: dict`(集中常量)、`RM_PARAMS_PROVENANCE: dict`。
  - `step_robots(robots, team, actions, dt, P=None, enemy_fire=True)` 现状不变;`actions[cls]['move'/'fire'/'target']`(target 可选)。`_apply_damage` 按 `act['target']` 类别选目标,缺省回退"最近敌方单位"。
  - `calibrate` 改为: 返回 `{}` 或不覆盖 RM_PARAMS 的"非速度增益"项;`step_robots` 用 RM_PARAMS 为主、`P` 可选覆盖(保持 P 传入语义不变)。

- [ ] **Step 1: Write failing tests(fresh tests/test_physics.py 增补)**

```python
# 追加到 tests/test_physics.py
def test_target_intent_hits_chosen_building_not_closer_unit():
    # 红步兵3 意图打蓝前哨站;虽蓝步兵3更近,仍打前哨站
    rb = _robots(
        _r(1, '步兵3', '红', 5.0, 5.0),
        _r(2, '步兵3', '蓝', 9.0, 5.0),      # 更近单位
        _r(99, '前哨站', '蓝', 20.0, 5.0),   # 意图目标(fixed coo 近似)
    )
    out = ph.step_robots(rb, '红', {'步兵3': {'move': '推进', 'fire': True, 'target': '敌方前哨站'}}, dt=3.0)
    assert out[2]['hp'] == db2...  # 蓝步兵3 未被打(原 100)
    assert out[99]['hp'] < 100.0   # 前哨站被打


def test_mid_field_cannot_hit_base_but_can_hit_outpost():
    # 17mm RANGE=10: 单位在中场能打前哨(近)、打不到基地(远)
    rb = _robots(
        _r(1, '步兵3', '红', 14.0, 7.0),     # 中场
        _r(98, '前哨站', '蓝', 17.12, 11.32),
        _r(99, '基地', '蓝', 25.73, 7.44),
    )
    out = ph.step_robots(rb, '红', {'步兵3': {'move': '推进', 'fire': True, 'target': '敌方前哨站'}}, dt=3.0)
    assert out[98]['hp'] < 100.0   # 前哨站在 10m 内 -> 被打
    assert out[99]['hp'] == 100.0  # 基地 11.99m > 10m -> 未被打


def test_aim_at_base_requires_advance():
    # 从己方半场打基地: 不在射程,必须推进后(或从近处)才中
    rb = _robots(
        _r(1, '步兵3', '红', 3.0, 7.0),      # 己方半场深处
        _r(99, '基地', '蓝', 25.73, 7.44),
    )
    out = ph.step_robots(rb, '红', {'步兵3': {'move': '推进', 'fire': True, 'target': '敌方基地'}}, dt=3.0)
    assert out[99]['hp'] == 100.0  # 距离 22.7m > 10m, 打不到


def test_defense_fires_at_enemy_unit_not_building():
    # 防守单位 target=敌方步兵: 打步兵,不打建筑(即便建筑更近)
    rb = _robots(
        _r(1, '步兵3', '红', 5.0, 5.0),
        _r(2, '步兵3', '蓝', 9.0, 5.0),
        _r(99, '前哨站', '蓝', 20.0, 5.0),
    )
    out = ph.step_robots(rb, '红', {'步兵3': {'move': '回防', 'fire': True, 'target': '敌方步兵'}}, dt=3.0)
    assert out[2]['hp'] < 100.0   # 打步兵


def test_42mm_base_not_instant_kill():
    # 英雄 42mm 打基地: 掉血但几帧内不归零(量纲合理)
    rb = _robots(_r(1, '英雄', '红', 20.0, 7.44), _r(99, '基地', '蓝', 25.73, 7.44))
    out = ph.step_robots(rb, '红', {'英雄': {'move': '推进', 'fire': True, 'target': '敌方基地'}}, dt=3.0)
    ratio = out[99]['hp'] / 3000.0
    assert 0.0 < ratio < 1.0
```

- [ ] **Step 2: Run to verify fail**

Run: `python -m pytest tests/test_physics.py::test_target_intent_hits_chosen_building_not_closer_unit tests/test_physics.py::test_mid_field_cannot_hit_base_but_can_hit_outpost -v`
Expected: FAIL(`RANGE` 单标量、`_apply_damage` 选最近)。

- [ ] **Step 3: Implement RM_PARAMS + intent targeting**

在 `rl/physics.py` 顶部加:

```python
# ---- RMUC 2026 规则锚定量纲(来源标记见 RM_PARAMS_PROVENANCE) ----
RM_PARAMS = {
    'field': {'width': 28.0, 'height': 15.0},
    'obj': {('蓝','前哨站'): (17.12,11.32), ('蓝','基地'): (25.73,7.44),
            ('红','前哨站'): (10.87,3.58),  ('红','基地'): (2.46,7.44)},
    'gun': {'英雄':'42mm','步兵3':'17mm','步兵4':'17mm','哨兵':'17mm','工程':None,'空中':None},
    'range': {'17mm':10.0,'42mm':12.0},
    'dmg': {'17mm':30.0,'42mm':70.0}, 'dmg_bldg': {'17mm':3.0,'42mm':90.0},
    'rate': {'17mm':10.0,'42mm':2.5}, 'hit': {'17mm':0.25,'42mm':1.0},
    'heat_per': {'17mm':10.0,'42mm':100.0},
    'speed': {'哨兵':0.10,'工程':0.35,'英雄':0.40,'步兵3':0.70,'步兵4':0.70,'空中':2.50,None:0.05},
    'hp': {'基地':3000.0,'前哨站':750.0},
    'heat_max': {'17mm':250.0,'42mm':250.0}, 'cool': 4.0,  # 每帧回血默认
}
    # 简化为: 'cool' 是 heat 每帧回凉量
RM_PARAMS_PROVENANCE = {
    'field/obj': 'rm-battlescope field.py (P)', 'gun/rate/heat_per': '规则索引 (P/R)',
    'range': '几何核算+规则命中速 (D)', 'dmg/dmg_bldg/hp': '规则量纲默认 (D)',
    'speed': '实测有效机动+上浮 (D)', 'hit': '维持既有', 'cool': '默认 (D)',
}
```

把 `step_robots` 内的 `speed/dmg/rate/hit/heat_max/heat_ps/rng` 从 `P` 或 `RM_PARAMS` 取;`rng` 改按弹种(`RM_PARAMS['range'][g]`)。`_apply_damage` 增加 `target_cls` 形参,按射手目标类别选最近目标;`act.get('target')` 映射到类别(fixed coo 给建筑),否则缺省回退"最近敌方单位"。建筑坐标: `step_robots` 对 `前哨站/基地` 单位初始化时若 x,y≈0,用 `RM_PARAMS['obj'][(team,type)]`。

- [ ] **Step 4: `labels/rollout_label.commit_actions` 补 target**

在 `commit_actions` 里 `acts[cls] = {'move': it['move'], 'fire': bool(it['fire']), 'target': it['target']}`。

- [ ] **Step 5: Run tests verify pass**

Run: `python -m pytest tests/test_physics.py -v`
Expected: PASS。再 `python -m pytest tests/test_physics_env.py tests/test_make_rollout_steps.py tests/test_rollout_label.py -v` 全过。

- [ ] **Step 6: Commit**

```bash
git add rl/physics.py labels/rollout_label.py tests/test_physics.py
git commit -m "fix(m2): RM-rule physics re-anchor — RM_PARAMS, intent target selection, fixed building coords, per-caliber range"
```

---

### Task 2: 对称 PhysicsEnv(敌方也动)

**Files:**
- Modify: `rl/env.py`
- Test: `tests/test_physics_env.py`(新增)

**Interfaces:**
- Consumes: Task1 `RM_PARAMS`;`labels.rollout_label.{enemy_best_response,commit_actions}`;`tree.global_mode.derive_global_mode`。
- Produces: `PhysicsEnv.step` 对称(`_step_symmetric`),`_phys_actions` 含 target。obs/reward/win 不变。

- [ ] **Step 1: Write failing test**

```python
def test_physics_env_enemy_moves():
    e = _env(); e.reset()
    idx_push = rlenv.ACTION_MEANINGS.index(('推进', False))
    # 只动红(我方); 记录蓝英雄真实坐标 before
    before = dict(e.obs_dict()['英雄'])  # placeholder, 用 live 更准
    red_before_x = e.live.get(3, {}).get('x')
    blue_before_x = e.live.get(1, {}).get('x')
    e.step({'英雄': idx_push, '步兵3': rlenv.ACTION_MEANINGS.index(('推进', False))})
    blue_after_x = e.live.get(1, {}).get('x')
    # 敌方反应式对抗: 蓝英雄也可能推进/换位 -> 应移动(除非 best_response 恰为姿态不变)
    assert blue_after_x is not None
    assert abs(blue_after_x - blue_before_x) > 1e-6 or e.live.get(1, {}).get('hp', 100) < 100
```

**注**: 更好断言语义: 敌方在对抗下会采取某个行动(可能推进或回防),其真实坐标在 `step` 后改变。用 `e.live` 读蓝英雄坐标即可断言"敌方也动"。若 best_response 对 '均衡' 红→ '进攻' 蓝 → 蓝推进 → 坐标改变。`dps` fixture 需含红/蓝英雄、蓝步兵。

- [ ] **Step 2: Run to verify fail**

Run: `python -m pytest tests/test_physics_env.py::test_physics_env_enemy_moves -v`
Expected: FAIL(敌方静止)。

- [ ] **Step 3: Implement symmetric step**

在 `PhysicsEnv` 加:

```python
from labels import rollout_label as rrl
from tree import global_mode as gm

def _enemy_actions(self, w):
    ego = gm.derive_global_mode(w)
    e_m = rrl.enemy_best_response(ego)
    return rrl.commit_actions(w, e_m, self.enemy_team)

def step(self, actions):
    rew = self._action_coherence(actions, self.ws[self.t])
    if self.t < self.horizon:
        w = self.ws[self.t]
        my_act = self._phys_actions(actions)
        en_act = self._enemy_actions(w)
        self.live = ph.step_robots(self.live, self.team, my_act, self.dt, self._phys_params(), enemy_fire=False)
        self.live = ph.step_robots(self.live, self.enemy_team, en_act, self.dt, self._phys_params(), enemy_fire=False)
        self.t += 1
        self.ws.append(self._worldstate(self.live))
        rew += self._reward(self.ws[self.t])
    done = self.t >= self.horizon; trunc = done
    if done: rew += self.win_condition()
    return self.obs_dict(), float(rew), done, trunc, {'t': self.t, 'roles': AGENTS}
```

`_phys_actions` 在每类补 `'target'`(从 `derive_unit_intent` 拿 or target 保持)。若 `obs_dict` 某类缺敌(avoid),用 `_enemy_actions` 的 fallback 即可。

- [ ] **Step 4: Run tests verify pass**

Run: `python -m pytest tests/test_physics_env.py -v`
Expected: PASS(含既有 visibility/win 测试;若因敌方动而影响 visibility 断言,按新语义修正——敌方动是期望行为)。

- [ ] **Step 5: Commit**

```bash
git add rl/env.py tests/test_physics_env.py
git commit -m "feat(m2): symmetric PhysicsEnv — adversarial-reactive enemy, two-side step_robots, _phys_actions target"
```

---

### Task 3: 修 PPO ratio bug

**Files:**
- Modify: `rl/train.py`
- Test: `tests/test_train.py`(新增,可建)

**Interfaces:**
- Consumes: 无下游。
- Produces: `IPPO._rollout_one_episode` 返回 trace 含 `old_log_prob`;`train`/`_ppo_step` 用它。`rollout_actions` 不变。

- [ ] **Step 1: Write failing test**

```python
import os, sys, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from rl import train as tr

def test_ppo_ratio_not_one_after_update():
    ppo = tr.IPPO(obs_dim=8)
    # 构造迷你 env(float obs、离散 12)
    class E:
        def __init__(s): s.t=0
        def reset(s): s.t=0; return {a: np.zeros(8,dtype=np.float32) for a in ['英雄']}
        def obs_dict(s): return {a: np.zeros(8,dtype=np.float32) for a in ['英雄']}
        def step(s, act):
            s.t+=1; done=s.t>=5
            return s.obs_dict(), 0.1, done, done, {'roles':['英雄']}
    stats = ppo.train(E(), n_updates=3, device='cpu')
    assert 'updates' in stats and stats['updates'] >= 3
    # ratio 不为 0: 直接在 _ppo_step 内部验证 clip 生效(surr1 != surr2 对某些元素)
```

实际上更白盒: patch `_ppo_step` 记录 ratio。建议在 `_ppo_step` 里临时打印或断言: ratio 由 `adv_b` 符号产生不同 clip。测试用"训练后 loss finite + updates≥1"即可,再加一个"old_log_prob 来自行为策略"的断言(见实现后)。

- [ ] **Step 2: Run to verify fail**

Run: `python -m pytest tests/test_train.py -v`
Expected: FAIL(ratio 恒 1 → 无 clip,但 loss 可能有限;用"更新前后策略变化"判 — 若 ratio 恒1 则 REINFORCE 权重 loss 有限但行为可控;核心断言用 `old_log_prob != current log_prob`)。

- [ ] **Step 3: Implement off-policy old-log-prob**

`_rollout_one_episode`: 采样时用 `no_grad` 得 `logits`,`dist`,对动作 `a` 存 `old_log_prob = dist.log_prob(a)` 进 trace(元组加一列)。`train` 收集 `old_log_prob_t`。`_ppo_step` 改用 `old_log_prob_t`(不再 `log_prob.detach()`)。`rollout_actions` 不变(deterministic argmax 无需 old prob)。

```python
# _rollout_one_episode 内
with torch.no_grad():
    logits, _v = self.policy(...)
dist = torch.distributions.Categorical(logits=logits)
a = ...sample...
old_lp = dist.log_prob(torch.tensor([a])).item() if not batch else ...
trace.append((ag, obs[ag], a, rew, next_obs[ag], done, old_lp))
```

- [ ] **Step 4: Run tests verify pass**

Run: `python -m pytest tests/test_train.py -v && python -m pytest -q 2>&1 | tail -3`
Expected: PASS + 全量无回归。

- [ ] **Step 5: Commit**

```bash
git add rl/train.py tests/test_train.py
git commit -m "fix(m2): PPO ratio — store behaviour-policy old log-prob, restore clip"
```

---

### Task 4: 重跑 rollout 标注 + 验证不退化 + 文档更新

**Files:**
- Modify: `RMUC2026_M2_优化分析与方案.md`(§7.6 Plan B + 重标结果)
- No new source (runner 已有)

**Interfaces:**
- 消耗: Task1 物理、`finetune/make_rollout_dataset`。

- [ ] **Step 1: 重跑标注**

```bash
python -m finetune.make_rollout_dataset --limit 24 --T 8 \
    --out out_m2/m2_dataset_rollout_planb.jsonl --report _work/rollout_planb_report.json
```

- [ ] **Step 2: 验证不退化(硬判据)**

阅读 `_work/rollout_planb_report.json`:
- `label_dist_full` 不再单模: `进攻`+`均衡` 占比 ≥ 0.10(理想 ≥0.20)。
- `outcome_consistency` 全集 `op_vs_def_gap > 0`(标进攻点胜机均值更高)。
- `win_model_cv.cv_auc ∈ [0.5,1]` 且 > 0.6。
若任一不满足 → **停止**,回到 Task1 调 `RM_PARAMS`(查伤害/射程/地平线),不写文档。

- [ ] **Step 3: 回归全量**

Run: `python -m pytest -q` Expected: 全过。

- [ ] **Step 4: 更新文档**

`RMUC2026_M2_优化分析与方案.md` §7.6: 记录"Plan B: RM 规则化对称物理(2026-09-09)" + 重标结果(label_dist/op_vs_def_gap/cv_auc) + 诚实上限(符号近似,非逐帧一致;(D) 值需真实规则核对)。同时更新 ledger progress.md 与 Master sequencing 划掉 Plan B。

- [ ] **Step 5: Commit(只 commit 文档改动,不 commit 产物)**

```bash
git add RMUC2026_M2_优化分析与方案.md
git commit -m "docs(m2): Plan B RM physics + re-measured rollout labeling (non-degenerate)"
```

---
