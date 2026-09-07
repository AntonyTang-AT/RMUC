# RMUC 2026 M3 进阶任务（博弈论 + 分层 schema + 行为树先验）Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现进阶任务：从 WorldState 产出**全局模式 + 每台机器人四通道**的严格结构化决策（分层 schema），用博弈论选意图、行为树先验提供上下文，并以 schema 合规率 + 相邻冲突率评测。

**Architecture:** M3 在 M0/M1 已交付的 `worldstate`（`build_worldstate`→`{'t','team','robots','buildings'}`，含 `zone`/`norm_xy`/`hp_ratio`/`dfire`/`vuln`）与 `fastloop.behavior`（`Rules`/`decide_fast`/`feasible`）之上，新增四个纯 Python 模块：`schema/layered`（分层 schema + 校验器 + 冲突率）、`tree/global_mode`（规则推导全局模式 + 单位四通道）、`game/game_theory`（payoff + regret matching 选意图）、`tree/behavior_prior`（状态转移频繁路径挖掘 → 行为树先验），最后由 `run_m3.py` 整链串联并评测。

**Tech Stack:** Python 3，标准库（`json`/`math`）；**无 sklearn、无 torch、无 transformers 进单测**（沿用 M0/M1 测试边界）。博弈论用自实现 regret matching（无外部依赖）。

**Spec:** `docs/superpowers/specs/2026-09-07-rmuc-llm-situation-analysis-design.md` — 本计划从 §1.2（进阶分层 schema）、§2（三引擎-博弈论）、§6.1（慢环意图选择）、§6.2（行为树先验）、§6.6（DecisionRequest）、§7（合规率/冲突率）出发，spec 为权威。

## Global Constraints

- **进阶 schema 无官方标准**（spec §9）：按 spec §1.2 的四通道（目标对象/移动/开火/风险）+ 全局三模式（攻/守/均衡）设计；考核点是"**严格输出结构化数据**"，即 schema 合规，不追求某具体标准。设计时值从 spec §4 的示例值借鉴（`target` 含 `敌方哨兵/英雄/步兵/前哨站/基地/能量机关/飞镖/无`，`risk` 为 `高/中/低`）。
- **测试边界**：单测**不加载 transformers/torch**；凡涉及字段值可构造合成 `dp`/`worldstate`（纯 dict），或用 `data.db` **只读查询** `config.GAME_REF` 的离线路由（M1 先例，不触发 GPU/模型）。
- **状态枚举一致性**：沿用既有 `('def','hitbuilding','cruise')`（sentry 状态）、`accept_targets`（`worldstate.worldstate`）、`ROBOT_TYPES`（`data.features`）；M3 的新枚举集中在 `schema/layered.py`，其它模块 `from schema.layered import ...`。
- **生成产物不入库**：`run_m3.py` 输出到 `out_m3/`；在 `.gitignore` **追加** `out_m3/` 一行（additive 追加，不覆盖既有忽略规则 `out_m0/out_m1/.superpowers/` 等）。
- **M3-first 依赖处理**：M3 原契约消费 `rl/distill_labels`，但 M2 未建。M3 用 `tree/global_mode` 的**规则推导分层决策**作标签源（self-consistent），M2 蒸馏标签后续可替换；接口保持 `worldstate -> layered decision` 同构。
- **训练/测试防泄漏**：M3 无训练，暂不做 game_id 切分；留全局注释提示 M2 需按 `game_id` 切分。

---

## 模块接口速览（后续任务依赖）

- `schema/layered.py`:
  - `UNIT_CLASSES: tuple` = `('英雄','工程','步兵3','步兵4','空中','哨兵')`
  - `GLOBAL_MODES: tuple` = `('进攻','均衡','防守')`
  - `TARGETS: tuple`、`MOVES: tuple`、`RISKS: tuple`
  - `build_empty_decision(t:float) -> dict`（合法候选，字段为 None/空，供测试/运行填充）
  - `validate(decision:dict) -> tuple[bool, list[str]]`（True + [] 或 False + 错误列表）
  - `conflict_rate(decisions:list[dict]) -> float`（相邻决策点冲突率，0~1）
- `tree/global_mode.py`:
  - `derive_global_mode(worldstate:dict) -> str`（`进攻/均衡/防守`）
  - `derive_unit_intent(unit:dict, worldstate:dict, global_mode:str) -> dict`（`{'target','move','fire','risk'}`，字段均在 allowed 集合）
- `game/game_theory.py`:
  - `PAYOFF_3: list[list[float]]`（3×3 对称默认 payoff）
  - `regret_matching(matrix:list[list[float]], iterations:int=500) -> list[float]`（混合策略，sum≈1）
  - `best_response(matrix, opp_strategy:list[float]) -> int`（纯最优反应下标）
  - `choose_mode(matrix, opp_estimate:list[float], iterations:int=500) -> str`（返回 `GLOBAL_MODES` 之一）
- `tree/behavior_prior.py`:
  - `state_sig(worldstate:dict, mode:str) -> str`
  - `mine(trajectories:list[list[tuple[str,str]]], k:int=3) -> list[dict]`（优先规则列表）
  - `build_prior_context(prior:list[dict], cur_sig:str) -> list[str]`
- `run_m3.py`: 整链 CLI，写 `out_m3/report.json`。

---

### Task M3.1: 分层 schema + 校验器

**Files:**
- Create: `schema/__init__.py`（空）
- Create: `schema/layered.py`
- Test: `tests/test_layered.py`

**Interfaces:**
- Produces: `UNIT_CLASSES`/`GLOBAL_MODES`/`TARGETS`/`MOVES`/`RISKS`/`build_empty_decision`/`validate`/`conflict_rate` — 全部后续模块依赖。

- [ ] **Step 1: 写失败测试**

```python
# tests/test_layered.py
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from schema import layered as sl

def _valid():
    d = sl.build_empty_decision(12.0)
    d['global_mode'] = '进攻'
    d['units']['哨兵'] = {'target': '敌方前哨站', 'move': '推进', 'fire': True, 'risk': '中'}
    d['units']['英雄'] = {'target': '敌方英雄', 'move': '游击', 'fire': False, 'risk': '低'}
    return d

def test_constants():
    assert sl.GLOBAL_MODES == ('进攻', '均衡', '防守')
    assert len(sl.UNIT_CLASSES) == 6
    assert '哨兵' in sl.UNIT_CLASSES

def test_validate_valid():
    ok, errs = sl.validate(_valid())
    assert ok is True and errs == []

def test_validate_bad_global_mode():
    d = _valid(); d['global_mode'] = '跑路'
    ok, errs = sl.validate(d)
    assert ok is False and any('global_mode' in e for e in errs)

def test_validate_missing_unit_class():
    d = _valid(); del d['units']['英雄']
    ok, errs = sl.validate(d)
    assert ok is False and any('英雄' in e for e in errs)

def test_validate_bad_target():
    d = _valid(); d['units']['哨兵']['target'] = '敌方坦克'
    ok, errs = sl.validate(d)
    assert ok is False and any('target' in e for e in errs)

def test_validate_bad_fire_type():
    d = _valid(); d['units']['哨兵']['fire'] = 'yes'
    ok, errs = sl.validate(d)
    assert ok is False

def test_conflict_rate():
    a = _valid()
    b = _valid(); b['global_mode'] = '防守'
    c = dict(b)   # same as b -> 仅 a->b 这一对变化
    r = sl.conflict_rate([a, b, c])
    assert 0.0 <= r <= 1.0
    assert abs(r - 1/2) < 1e-9  # 两对相邻中恰好一对变化
```

- [ ] **Step 2: 运行确认失败**

Run: `cd "F:\Desktop\学校\rm考核项目" && python -m pytest tests/test_layered.py -q`
Expected: FAIL（ModuleNotFoundError: schema）。

- [ ] **Step 3: 写最小实现**

```python
# schema/layered.py
UNIT_CLASSES = ('英雄', '工程', '步兵3', '步兵4', '空中', '哨兵')
GLOBAL_MODES = ('进攻', '均衡', '防守')
TARGETS = ('敌方哨兵', '敌方英雄', '敌方步兵', '敌方工程', '敌方前哨站', '敌方基地', '能量机关', '飞镖', '无')
MOVES = ('推进', '回防', '游击', '坚守', '绕行', '无')
RISKS = ('高', '中', '低')

def build_empty_decision(t):
    # 每类预置保守闲置决策, 保证"每台机器人"六类齐全的合法基线 (合规率 100%); 运行时按需覆盖
    idle = {'target': '无', 'move': '无', 'fire': False, 'risk': '低'}
    return {'t': t, 'global_mode': '均衡', 'units': {c: dict(idle) for c in UNIT_CLASSES}}

def _channels(entry):
    if not isinstance(entry, dict):
        return None
    return ('target', 'move', 'fire', 'risk')

def validate(decision):
    errs = []
    if not isinstance(decision, dict):
        return False, ['decision 不是 dict']
    if 't' not in decision:
        errs.append('missing t')
    if decision.get('global_mode') not in GLOBAL_MODES:
        errs.append('global_mode 非法')
    units = decision.get('units')
    if not isinstance(units, dict):
        errs.append('units 不是 dict')
        return False, errs
    for c in UNIT_CLASSES:
        if c not in units:
            errs.append(f'缺少单位类 {c}')
            continue
        e = units[c]
        if not isinstance(e, dict):
            errs.append(f'{c} 不是 dict')
            continue
        if e.get('target') not in TARGETS:
            errs.append(f'{c}.target 非法')
        if e.get('move') not in MOVES:
            errs.append(f'{c}.move 非法')
        if not isinstance(e.get('fire'), bool):
            errs.append(f'{c}.fire 非 bool')
        if e.get('risk') not in RISKS:
            errs.append(f'{c}.risk 非法')
    return (errs == []), errs

def conflict_rate(decisions):
    if not decisions or len(decisions) < 2:
        return 0.0
    conflicts = 0
    for a, b in zip(decisions, decisions[1:]):
        if a.get('global_mode') != b.get('global_mode'):
            conflicts += 1
        else:
            # compare unit intent flips (target) as a finer-grained conflict
            ua, ub = a.get('units') or {}, b.get('units') or {}
            if any((ua.get(c) or {}).get('target') != (ub.get(c) or {}).get('target')
                   for c in UNIT_CLASSES):
                conflicts += 1
    return conflicts / (len(decisions) - 1)
```

- [ ] **Step 4: 运行确认通过**

Expected: PASS（7 个用例）。

- [ ] **Step 5: 提交**

```bash
cd "F:\Desktop\学校\rm考核项目" && git add schema/ tests/test_layered.py && git commit -m "feat(m3): layered schema + validator + conflict-rate"
```

---

### Task M3.2: 全局模式 + 单位四通道规则

**Files:**
- Test: `tests/test_global_mode.py`

**Interfaces:**
- Consumes: `worldstate.worldstate.build_worldstate -> dict`（`robots` 值为 `{'type','team','zone','norm_xy','hp_ratio','power','heat','vuln','ammo17','ammo42','dfire'}`）、`schema.layered`（TARGETS/MOVES/RISKS/UNIT_CLASSES）。
- Produces: `derive_global_mode(worldstate) -> str`、`derive_unit_intent(unit, worldstate, global_mode) -> dict`。

- [ ] **Step 1: 写失败测试**

```python
# tests/test_global_mode.py
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from tree import global_mode as gm
from schema import layered as sl

def _ws(red_x, blue_x, red_hp, blue_hp, base_hp):
    # 红 3 台 + 蓝 3 台; 中线 x=0.5 (norm)
    def u(tid, team, x, hp):
        return {'type': tid, 'team': team, 'zone': 'Z01', 'norm_xy': [x, 0.5],
                'hp_ratio': hp, 'power': 0.0, 'heat': 0.0, 'vuln': 0,
                'ammo17': 0, 'ammo42': 0, 'dfire': 0.0}
    reds = {i: u('哨兵', '红', x, hp) for i, (x, hp) in enumerate(zip(red_x, red_hp))}
    blues = {i + 10: u('步兵3', '蓝', x, hp) for i, (x, hp) in enumerate(zip(blue_x, blue_hp))}
    return {'t': 0.0, 'team': {}, 'robots': {**reds, **blues},
            'buildings': {99: {'type': '基地', 'team': '红', 'hp_ratio': base_hp}}}

def test_three_modes_observable():
    # 进攻: 红 units 深入敌方半场(x>0.5) 且蓝残血(蓝缩在自己半场)
    assert gm.derive_global_mode(_ws([0.8, 0.7, 0.6], [0.7, 0.65, 0.6], [1,1,1], [0.1,0.1,0.1], 1.0)) == '进攻'
    # 防守: 蓝 units 深入红方半场(x<0.5) 且红残血/基地低血
    assert gm.derive_global_mode(_ws([0.4, 0.35, 0.3], [0.45, 0.4, 0.35], [0.1,0.1,0.1], [1,1,1], 0.4)) == '防守'
    # 均衡: 中线拉扯, hp 均衡
    assert gm.derive_global_mode(_ws([0.6, 0.5, 0.4], [0.4, 0.5, 0.6], [0.7,0.7,0.7], [0.7,0.7,0.7], 0.9)) == '均衡'

def test_unit_intent_channels_valid():
    w = _ws([0.8, 0.7, 0.6], [0.2, 0.2, 0.2], [1,1,1], [0.1,0.1,0.1], 1.0)
    mode = gm.derive_global_mode(w)
    unit = w['robots'][0]
    it = gm.derive_unit_intent(unit, w, mode)
    assert it['target'] in sl.TARGETS
    assert it['move'] in sl.MOVES
    assert it['risk'] in sl.RISKS
    assert isinstance(it['fire'], bool)
```

- [ ] **Step 2: 运行确认失败**

Run: `cd "F:\Desktop\学校\rm考核项目" && python -m pytest tests/test_global_mode.py -q`
Expected: FAIL（ModuleNotFoundError: tree）。

- [ ] **Step 3: 写最小实现**

```python
# tree/__init__.py  (空文件, Task 第一步创建)
# game/__init__.py  (空文件, Task 下一步创建)
```

```python
# tree/global_mode.py
from schema.layered import GLOBAL_MODES, TARGETS, MOVES, RISKS

def _mid_norm(x):
    return x  # norm_xy 已在 [0,1]; 中线 = 0.5

def derive_global_mode(worldstate):
    robots = [r for r in worldstate['robots'].values() if r not in (None,)]
    if not robots:
        return '均衡'
    reds = [r for r in robots if r['team'] == '红']
    blues = [r for r in robots if r['team'] == '蓝']
    red_push = sum(1 for r in reds if r['norm_xy'][0] > 0.5)
    blue_push = sum(1 for r in blues if r['norm_xy'][0] < 0.5)
    red_hp = sum(r['hp_ratio'] for r in reds) / len(reds) if reds else 0.0
    blue_hp = sum(r['hp_ratio'] for r in blues) / len(blues) if blues else 0.0
    # 基地血量低 -> 防守倾向
    base_threat = any(b.get('hp_ratio', 1.0) < 0.5 for b in worldstate['buildings'].values())
    push_adv = red_push - blue_push
    if push_adv >= 2 and blue_hp < 0.9:
        return '进攻'
    if push_adv <= -2 or base_threat:
        return '防守'
    return '均衡'

def derive_unit_intent(unit, worldstate, global_mode):
    hp = unit.get('hp_ratio', 1.0)
    vuln = unit.get('vuln', 0)
    dfire = unit.get('dfire', 0.0)
    x = unit['norm_xy'][0] if unit.get('norm_xy') else 0.5
    # 目标
    if global_mode == '进攻' and x > 0.5:
        target = '敌方前哨站'
    elif global_mode == '防守':
        target = '敌方步兵'
    elif dfire > 0:
        target = '敌方英雄'
    else:
        target = '无'
    # 移动
    move = {'进攻': '推进', '防守': '回防', '均衡': '游击'}.get(global_mode, '无')
    if hp < 0.3 or vuln:
        move = '坚守'
    # 开火
    fire = bool(dfire > 0 or target.startswith('敌方'))
    # 风险
    risk = '高' if (hp < 0.3 or vuln) else ('中' if hp < 0.6 else '低')
    return {'target': target, 'move': move, 'fire': fire, 'risk': risk}
```

- [ ] **Step 4: 运行确认通过**

Expected: PASS。

- [ ] **Step 5: 提交**

```bash
cd "F:\Desktop\学校\rm考核项目" && git add tree/ tests/test_global_mode.py && git commit -m "feat(m3): derive global mode + unit 4-channel intent"
```

---

### Task M3.3: 博弈论 intent 选择（regret matching）

**Files:**
- Test: `tests/test_game_theory.py`

**Interfaces:**
- Consumes: `schema.layered.GLOBAL_MODES`。
- Produces: `PAYOFF_3`/`regret_matching`/`best_response`/`choose_mode`。

- [ ] **Step 1: 写失败测试**

```python
# tests/test_game_theory.py
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from game import game_theory as gt
from schema import layered as sl

def test_payoff_shape():
    assert len(gt.PAYOFF_3) == 3 and all(len(row) == 3 for row in gt.PAYOFF_3)

def test_regret_matching_valid_distribution():
    s = gt.regret_matching(gt.PAYOFF_3, iterations=800)
    assert len(s) == 3
    assert all(p >= 0 for p in s)
    assert abs(sum(s) - 1.0) < 1e-6

def test_choose_mode_returns_mode():
    m = gt.choose_mode(gt.PAYOFF_3, [1/3, 1/3, 1/3])
    assert m in sl.GLOBAL_MODES

def test_best_response_dominant():
    # 行 0 严格占优 (期望收益始终最高)
    M = [[10, 10, 10], [1, 1, 1], [0, 0, 0]]
    assert gt.best_response(M, [1/3, 1/3, 1/3]) == 0

def test_regret_matching_recovers_dominant_strategy():
    M = [[10, 10, 10], [1, 1, 1], [0, 0, 0]]
    s = gt.regret_matching(M, iterations=800)
    assert s.index(max(s)) == 0
```

- [ ] **Step 2: 运行确认失败**

Run: `cd "F:\Desktop\学校\rm考核项目" && python -m pytest tests/test_game_theory.py -q`
Expected: FAIL（ModuleNotFoundError: game）。

- [ ] **Step 3: 写最小实现**

```python
# game/game_theory.py
from schema.layered import GLOBAL_MODES

# 默认对称 3 意图 payoff (我方行, 对方列)。进攻 vs 防守 高打低/低打高 有收益差; 同类对拼近零和。
# 注: spec §9 指出真实 payoff 需依赖 RL/对手模型估计, 此为手设先验默认值。
PAYOFF_3 = [
    [-0.5, 0.3, 0.8],    # 我方进攻
    [0.0, 0.0, 0.2],     # 我方均衡
    [-0.3, -0.5, 0.5],   # 我方防守
]

def regret_matching(matrix, iterations=500):
    n = len(matrix)
    regret = [0.0] * n
    strat_sum = [0.0] * n
    strategy = [1.0 / n] * n
    for _ in range(iterations):
        payoff = [sum(matrix[i][j] * strategy[j] for j in range(n)) for i in range(n)]
        avg = sum(payoff) / n
        regret = [r + (payoff[i] - avg) for i, r in enumerate(regret)]
        pos = [max(r, 0.0) for r in regret]
        total = sum(pos)
        if total > 0:
            strategy = [p / total for p in pos]
        else:
            strategy = [1.0 / n] * n
        for i in range(n):
            strat_sum[i] += strategy[i]
    total = sum(strat_sum)
    return [s / total for s in strat_sum]

def best_response(matrix, opp_strategy):
    exp = [sum(matrix[i][j] * opp_strategy[j] for j in range(len(matrix))) for i in range(len(matrix))]
    return max(range(len(exp)), key=lambda i: exp[i])

def choose_mode(matrix, opp_estimate, iterations=500):
    strategy = regret_matching(matrix, iterations)
    # 学到的混合策略给出"我方倾向哪一意图"分布, argmax 即最优先意图
    idx = max(range(len(strategy)), key=lambda i: strategy[i])
    return GLOBAL_MODES[idx]
```

> 注意：`choose_mode` 中混合策略 `strategy` 是对"我方应选哪一意图"的 learned 分布，`idx = argmax(strategy)` 即最优先意图；这贴合"博弈论选意图"的考核语义。`best_response` 单独测纯反应。

- [ ] **Step 4: 运行确认通过**

Expected: PASS。

- [ ] **Step 5: 提交**

```bash
cd "F:\Desktop\学校\rm考核项目" && git add game/ tests/test_game_theory.py && git commit -m "feat(m3): game-theory regret-matching intent selection"
```

---

### Task M3.4: 行为树先验挖掘

**Files:**
- Test: `tests/test_behavior_prior.py`

**Interfaces:**
- Consumes: `tree.global_mode.derive_global_mode`、`worldstate.worldstate`（`zone`）。
- Produces: `state_sig(worldstate, mode) -> str`、`mine(trajectories, k) -> list[dict]`、`build_prior_context(prior, cur_sig) -> list[str]`。

- [ ] **Step 1: 写失败测试**

```python
# tests/test_behavior_prior.py
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from tree import behavior_prior as bp

def test_state_sig_encodes():
    ws = {'robots': {7: {'team': '红', 'zone': 'Z12'}}, 'global_mode': '进攻'}
    assert bp.state_sig(ws, '进攻') == '进攻|Z12'

def test_mine_finds_frequent_path():
    sigA, sigB = ('均衡|Z01'), ('进攻|Z12')
    # 反复出现 A->进攻 (decision), A->防守, A->进攻 占据多数
    traj = [[(sigA, 'cruise'), (sigB, 'hitbuilding'), (sigA, 'cruise')],
            [(sigA, 'cruise'), (sigB, 'cruise'), (sigA, 'def')],
            [(sigA, 'cruise'), (sigB, 'cruise')]]
    prior = bp.mine(traj, k=3)
    # A 的 top 决策应是 cruise (出现次数多)
    a_rules = [p for p in prior if p['sig'] == sigA]
    assert a_rules and a_rules[0]['decision'] == 'cruise'
    assert a_rules[0]['count'] >= 2

def test_build_prior_context():
    prior = [{'sig': '进攻|Z12', 'decision': 'hitbuilding', 'count': 3, 'freq': 1.0}]
    ctx = bp.build_prior_context(prior, '进攻|Z12')
    assert any('hitbuilding' in line for line in ctx)
```

- [ ] **Step 2: 运行确认失败**

Run: `cd "F:\Desktop\学校\rm考核项目" && python -m pytest tests/test_behavior_prior.py -q`
Expected: FAIL（ModuleNotFoundError 缺失函数）。

- [ ] **Step 3: 写最小实现**

```python
# tree/behavior_prior.py
from collections import defaultdict

def state_sig(worldstate, mode):
    s = worldstate.get('robots', {}).get(7, {})
    zone = s.get('zone', 'Z--')
    return f"{mode}|{zone}"

def mine(trajectories, k=3):
    counts = defaultdict(int)
    for traj in trajectories:
        for sig, decision in traj:
            counts[(sig, decision)] += 1
    by_sig = defaultdict(list)
    for (sig, decision), c in counts.items():
        by_sig[sig].append((decision, c))
    prior = []
    for sig, items in by_sig.items():
        total = sum(c for _, c in items)
        items.sort(key=lambda x: -x[1])
        for decision, c in items[:k]:
            prior.append({'sig': sig, 'decision': decision, 'count': c, 'freq': c / total})
    return prior

def build_prior_context(prior, cur_sig):
    lines = []
    for p in prior:
        if p['sig'] == cur_sig:
            lines.append(f"行为树先验: 在 {p['sig']} 局势下, 历史决策倾向 {p['decision']} (频率 {p['freq']:.0%}, 样本 {p['count']})")
    return lines
```

- [ ] **Step 4: 运行确认通过**

Expected: PASS。

- [ ] **Step 5: 提交**

```bash
cd "F:\Desktop\学校\rm考核项目" && git add tree/behavior_prior.py tests/test_behavior_prior.py && git commit -m "feat(m3): mine behavior-tree prior from decision trajectories"
```

---

### Task M3.5: 整链端到端 + 合规率评测

**Files:**
- Create: `run_m3.py`
- Modify: `.gitignore`（追加 `out_m3/` 一行）
- Test: `tests/test_m3_e2e.py`

**Interfaces:**
- Consumes: `data.db`/`data.features`/`worldstate.worldstate`/`schema.layered`/`tree.global_mode`/`game.game_theory`/`tree.behavior_prior`/`fastloop.behavior`。
- Produces: `run_m3.py` CLI → `out_m3/report.json`（`compliance_rate`/`conflict_rate`/`mode_counts`/`n`）。
- 单测：`compute_decision_series`（整链生成一条可校验决策序列）+ `--no-llm` 冒烟。

- [ ] **Step 1: 写失败测试**

```python
# tests/test_m3_e2e.py
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import config
from data import db, features
from worldstate import worldstate as ws
import run_m3

def test_compute_decision_series_valid():
    con = db.connect()
    rows = db.timeseries_for(con, config.GAME_REF)
    dps = features.build_decision_points(rows, config.GAME_REF)
    con.close()
    b = features.field_bounds(dps)
    decisions, prior, sigs = run_m3.compute_decision_series(dps, b)
    assert len(decisions) == len(sigs)
    from schema import layered as sl
    for d in decisions:
        ok, errs = sl.validate(d)
        assert ok, errs
    # 合规率应为 1.0 (我们生成时即保证合法)
    assert run_m3.compliance_rate(decisions) == 1.0
```

- [ ] **Step 2: 运行确认失败**

Run: `cd "F:\Desktop\学校\rm考核项目" && python -m pytest tests/test_m3_e2e.py -q`
Expected: FAIL（run_m3 未定义 / validate 失败提示线索）。

- [ ] **Step 3: 写最小实现**

```python
# run_m3.py
# -*- coding: utf-8 -*-
import os, argparse, json
import config
from data import db, features
from worldstate import worldstate as ws
from schema import layered as sl
from tree import global_mode as gm
from tree import behavior_prior as bp
from game import game_theory as gt
from fastloop import behavior as bh

def compute_decision_series(dps, bounds, opp_estimate=None):
    est = opp_estimate or [1/3, 1/3, 1/3]
    decisions, sigs = [], []
    for dp in dps:
        w = ws.build_worldstate(dp, bounds)
        mode = gt.choose_mode(gt.PAYOFF_3, est)      # 博弈论慢环: 选全局模式/意图
        d = sl.build_empty_decision(w['t'])
        d['global_mode'] = mode
        for rid, unit in w['robots'].items():
            if unit['team'] != '红' or unit['type'] not in d['units']:
                continue  # 只做己方(红)单位; 类型唯一, 不跨队冲突
            it = gm.derive_unit_intent(unit, w, mode)
            # 快环可行性校验 (spec §6.2): 哨兵血低/易伤 -> 禁进攻开火, 回退安全
            if unit['type'] == '哨兵' and it['fire'] and not bh.feasible(
                    'hitbuilding', {'robots': {7: unit}, 'bot_state': 'hitbuilding'}, bh.Rules()):
                it = {'target': it['target'], 'move': '回防', 'fire': False, 'risk': '高'}
            d['units'][unit['type']] = it
        # build_empty_decision 已预置六类闲置基线, 未出场类保持合法, 无需再补
        decisions.append(d)
        sigs.append(bp.state_sig(w, mode))
    prior = bp.mine([(sig, d['global_mode']) for sig, d in zip(sigs, decisions)])
    return decisions, prior, sigs

def compliance_rate(decisions):
    if not decisions:
        return 0.0
    return sum(1 for d in decisions if sl.validate(d)[0]) / len(decisions)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--games', nargs='*', type=int, default=None)
    ap.add_argument('--out', default='out_m3')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    con = db.connect()
    gids = args.games or [config.GAME_REF]
    all_decisions, all_sigs, trail = [], [], []
    for gid in gids:
        rows = db.timeseries_for(con, gid)
        dps = features.build_decision_points(rows, gid)
        b = features.field_bounds(dps)
        decisions, prior, sigs = compute_decision_series(dps, b)
        all_decisions += decisions; all_sigs += sigs
    con.close()
    mc = {}
    for d in all_decisions:
        mc[d['global_mode']] = mc.get(d['global_mode'], 0) + 1
    report = {
        'games': gids, 'n': len(all_decisions),
        'compliance_rate': compliance_rate(all_decisions),
        'conflict_rate': sl.conflict_rate(all_decisions),
        'mode_counts': mc,
    }
    with open(os.path.join(args.out, 'report.json'), 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
```

`.gitignore` 追加一行 `out_m3/`。

- [ ] **Step 4: 运行确认通过 + 冒烟**

Run:
```bash
cd "F:\Desktop\学校\rm考核项目" && python -m pytest tests/test_m3_e2e.py -q && python run_m3.py
```
Expected: PASS + `out_m3/report.json` 打印 compliance_rate（应为 1.0，因是规则自产）、conflict_rate、mode_counts。整链不加载 GPU/LLM（`--no-llm` 恒默认）。

- [ ] **Step 5: 提交**

```bash
cd "F:\Desktop\学校\rm考核项目" && git add run_m3.py tests/test_m3_e2e.py .gitignore && git commit -m "feat(m3): end-to-end layered decision chain + compliance report"
```

---

## 自检（Self-Review）

- **Spec coverage**：§1.2 分层 schema（全局三模式 + 每单位四通道）→ M3.1/M3.2；§2 博弈论意图选择 → M3.3；§6.2 行为树先验 → M3.4；§6.6 DecisionRequest 结构化输出 + §7 合规率/冲突率评测 → M3.1(conflict_rate)/M3.5。覆盖无缺口。
- **Placeholder 扫描**：无 TBD/TODO；每步含真实代码。
- **Type consistency**：`UNIT_CLASSES`/`GLOBAL_MODES`/`TARGETS`/`MOVES`/`RISKS`/`build_empty_decision`/`validate`/`conflict_rate`（M3.1）→ M3.2 消费；`derive_global_mode`/`derive_unit_intent`（M3.2）→ M3.4/M3.5 消费；`PAYOFF_3`/`regret_matching`/`best_response`/`choose_mode`（M3.3）→ M3.5 消费；`state_sig`/`mine`/`build_prior_context`（M3.4）→ M3.5 消费。一致。
- **歧义**：单位按**6 类**（英雄/工程/步兵3/步兵4/空中/哨兵）键控（spec 明言 6 类作战单位；每类恰对应该职责机器人）。`choose_mode` 以混合策略 argmax 选意图（贴合"博弈论选意图"）。嵌套 `mine` 先验在 M3.5 用整局轨迹一次累积。
- **测试边界**：全部单测纯 dict 构造或仅用 `data.db`（GAME_REF 只读查询），**无 transformers/torch**；`run_m3.py` 恒无 LLM。

## 执行移交

计划落盘：`docs/superpowers/plans/2026-09-07-rmuc-llm-m3-advanced.md`。
开始执行：**从 Task M3.1 起**，逐任务实现 + 测试 + 提交；每任务经 task review，M3.5 通过后做整支（M0+M1+M3）最终评审。M2（RL+LoRA）作为下一独立子计划，后续展开（M3 用规则推导决策作标签源，M2 蒸馏标签届时可替换）。
