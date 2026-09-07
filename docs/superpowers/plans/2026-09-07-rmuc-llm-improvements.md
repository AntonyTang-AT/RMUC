# RMUC LLM 系列 M1/M3 改进 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 修复 M1/M3 的已知退化行为(M1 无诚实规则基线、评测指标单一、缺逐类口径;M3 全局模式恒「进攻」、行为先验频率退化、compliance 无信息、若干死代码/语义矛盾),使评估与断言可靠,并为三引擎融合铺路。全部 GPU-free。

**Architecture:** 每个任务改一个独立归属模块;`run_m3.py` 被 P2/P4 顺序复用(不同函数),其余文件不跨任务共享。P1 在 `eval/metrics.py` 增 `per_class`、在 `tree/global_mode.py` 增 `derive_sentry_state`(诚实规则基线)、改 `run_m1.py` 增 `--rule-baseline`。P2 在 `game/game_theory.py` 增 `choose_global_mode`(有对手模型走博弈 best_response,否则回退局势感知)、改 `run_m3.py` 使用它。P3 新建 `run_llm_m3.py`:把 LLM 慢环哨兵状态映射成 4 通道分层决策,与规则派生序列对齐度量(合规/LLM-vs-规则 agreement/逐类/conflict)。P4 清理 `run_m3.py`(未用 `sigs`、哨兵回退 target 语义矛盾)。

**Tech Stack:** Python 3, `argparse`, `json`, `urllib`(DeepSeek API), pytest, 无 scikit-learn/transformers(手写指标)。仅 read-only `data.db`。

**Spec:** 基于对话中分析结论(M1/M3 现状 + 文献)提出的改进清单;本 plan 实现其中四个最高价值项。

## Global Constraints

- No transformers/torch in unit tests; read-only `data.db` allowed (M1 precedent).
- `.gitignore` 仅可追加,绝不覆盖/收窄既有忽略行(含 `.env` 行,API key 永不提交)。
- M1 哨兵三态 `('def','hitbuilding','cruise')`;M3 分层 schema 六类 × 四通道(target/move/fire/risk),全局模式 `('进攻','均衡','防守')`。
- 主队 = 红;敌方进己方半场以中线为界。
- 每个 Task 只改「Files」里列出的文件。
- 不提交生成的输出(`out_m1/`, `out_m3/` 等已 gitignore)。
- 数值、字符串、签名逐字采用本 plan 给定值;注释/报告用中文。
- TDD:先写失败测试确认 RED,再实现确认 GREEN,再提交。

---

### Task P1: M1 — 逐类指标 + 诚实规则基线(derive_sentry_state)

**Files:**
- Modify: `eval/metrics.py`(加 `per_class`)
- Modify: `tree/global_mode.py`(加 `derive_sentry_state`)
- Modify: `run_m1.py`(加 `--rule-baseline` + 报告逐类)
- Test: `tests/test_metrics.py`, `tests/test_global_mode.py`(加 `test_derive_sentry_state`)

**Interfaces:**
- Consumes: `model.prompt.VALID_STATES`(既有);`tree.global_mode.derive_global_mode`(既有)。
- Produces: `metrics.per_class(y_true, y_pred, labels=None) -> {label: {'precision','recall','f1','support'}}`;`global_mode.derive_sentry_state(worldstate) -> str(三态之一)`。

- [ ] **Step 1: 写 `per_class` 失败测试**(`tests/test_metrics.py`)

```python
# tests/test_metrics.py
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from eval import metrics

def test_per_class_counts_and_scores():
    y_true = ['cruise', 'def', 'cruise', 'cruise']
    y_pred = ['cruise', 'def', 'cruise', 'def']
    pc = metrics.per_class(y_true, y_pred)
    # cruise: tp=2 fp=0 fn=1 support=3 -> precision 1.0 recall 2/3
    assert pc['cruise']['precision'] == 1.0
    assert abs(pc['cruise']['recall'] - 2/3) < 1e-9
    assert pc['cruise']['support'] == 3
    # def: tp=1 fp=1 fn=0 support=1 -> precision 0.5 recall 1.0
    assert abs(pc['def']['precision'] - 0.5) < 1e-9
    assert pc['def']['recall'] == 1.0
    assert pc['def']['support'] == 1

def test_per_class_unknown_label_present_in_pred():
    y_true = ['cruise']
    y_pred = ['hitbuilding']
    pc = metrics.per_class(y_true, y_pred)
    assert 'hitbuilding' in pc  # 出现在 y_pred 的标签也应被计入
```

- [ ] **Step 2: 运行确认失败**

Run: `python -m pytest tests/test_metrics.py -q`
Expected: `AttributeError: module 'eval.metrics' has no attribute 'per_class'`

- [ ] **Step 3: 写 `per_class` 实现**(`eval/metrics.py`,追加到 `f1_macro` 后)

```python
def per_class(y_true, y_pred, labels=None):
    # 无 scikit-learn 依赖: 逐类 precision/recall/f1 宏口径, 并带支持数(support)
    if labels is None:
        labels = [l for l in VALID_STATES if l in set(y_true) or l in set(y_pred)]
    out = {}
    for l in labels:
        tp = sum(1 for a, b in zip(y_true, y_pred) if a == l and b == l)
        fp = sum(1 for a, b in zip(y_true, y_pred) if a != l and b == l)
        fn = sum(1 for a, b in zip(y_true, y_pred) if a == l and b != l)
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f = 2 * p * r / (p + r) if (p + r) else 0.0
        out[l] = {'precision': round(p, 4), 'recall': round(r, 4), 'f1': round(f, 4), 'support': tp + fn}
    return out
```

- [ ] **Step 4: 运行确认通过**

Run: `python -m pytest tests/test_metrics.py -q`
Expected: 2 passed

- [ ] **Step 5: 写 `derive_sentry_state` 失败测试**(`tests/test_global_mode.py` 追加)

```python
def test_derive_sentry_state_maps_mode():
    import tree.global_mode as gm
    # 防守 -> def: 红基地血低触发 base_threat(需有一台红方单位, 空则 derive_global_mode 直接返回 均衡)
    red = {'robots': {'r': {'type': '步兵3', 'team': '红', 'norm_xy': [0.6, 0.5],
                            'hp_ratio': 1.0, 'vuln': 0, 'dfire': 0}},
           'buildings': {'b': {'team': '红', 'hp_ratio': 0.4}}}
    assert gm.derive_sentry_state(red) == 'def'
    # 无机器人 -> 均衡 -> cruise
    assert gm.derive_sentry_state({'robots': {}, 'buildings': {}}) == 'cruise'
```

- [ ] **Step 6: 写 `derive_sentry_state` 实现**(`tree/global_mode.py`,追加到文件末尾)

```python
_MODE_TO_SENTRY = {'防守': 'def', '均衡': 'cruise', '进攻': 'cruise'}

def derive_sentry_state(worldstate):
    """从世界状态派生一个**诚实**的哨兵三态(规则基线): map 全局模式, 不接触标注器真值。

    供 M1 --rule-baseline 作为不泄漏真值的下限; 也让 LLM-raw 得分有参照系。
    """
    return _MODE_TO_SENTRY.get(derive_global_mode(worldstate), 'cruise')
```

- [ ] **Step 7: 运行确认通过与无回归**

Run: `python -m pytest tests/test_global_mode.py -q`
Expected: 全部通过(含新增)。

- [ ] **Step 8: 改 `run_m1.py` — `--rule-baseline` + 报告逐类**

顶部加导入:

```python
from tree import global_mode as gm
```

在 `main()` 参数区加:

```python
    ap.add_argument('--rule-baseline', action='store_true',
                    help='诚实规则基线: 用 derive_sentry_state(非 LLM/非真值)提议, 作下限参照')
```

循环内 `w = ws.build_worldstate(dp, b)` 之后的分支改成:

```python
            if args.rule_baseline:
                pred = bh.decide_fast({'robots': w['robots'], 'bot_state': gm.derive_sentry_state(w)}, rules)
                p = pred['state']
            elif args.no_llm:
                pred = bh.decide_fast({'robots': w['robots'], 'bot_state': state}, rules)
                p = pred['state']
            else:
                ...(原 LLM 分支不变)...
```

报告字典改:

```python
    report = {
        'mode': ('rule_baseline' if args.rule_baseline
                 else ('rule_baseline_smoke' if args.no_llm else 'llm_slowloop')),
        'games': gids, 'n': len(samples),
        'accuracy': metrics.accuracy(y_true, y_pred),
        'f1_macro': metrics.f1_macro(y_true, y_pred),
        'legal_rate': legal,
        'per_class': metrics.per_class(y_true, y_pred),
        'mean_latency_s': (sum(latencies)/len(latencies) if latencies else None),
        'confusion': metrics.confusion(y_true, y_pred, list(sl.SENTRY_STATES)),
    }
```

- [ ] **Step 9: 手动冒烟(诚实基线应 ≈ 多数类, 低于恒等 1.0)**

Run: `python run_m1.py --rule-baseline`
Expected: `mode: rule_baseline`,accuracy 约为多数类水平(≈0.82),明显低于 `--no-llm` 的恒等 1.0。

- [ ] **Step 10: 提交**

```bash
git add eval/metrics.py tree/global_mode.py run_m1.py tests/test_metrics.py tests/test_global_mode.py
git commit -m "feat(m1): per-class metrics + honest rule-baseline (derive_sentry_state)"
```

---

### Task P2: M3 — 局势感知全局模式(choose_global_mode)+ 行为先验多样化

**Files:**
- Modify: `game/game_theory.py`(加 `choose_global_mode`)
- Modify: `run_m3.py`(`compute_decision_series` 用 `choose_global_mode`)
- Test: `tests/test_situation_mode.py`(新)

**Interfaces:**
- Consumes: `game.game_theory.choose_mode/PAYOFF_3`(既有);`tree.global_mode.derive_global_mode`(P1 已加入);`schema.layered.build_empty_decision`(既有)。
- Produces: `game_theory.choose_global_mode(worldstate, opp_estimate=None) -> str(三模式之一)`。

- [ ] **Step 1: 写失败测试**(`tests/test_situation_mode.py`)

```python
# tests/test_situation_mode.py
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import config
from data import db, features
from game import game_theory as gt
import run_m3

def _worldstate(red_xs=None, blue_xs=None, base_hp=1.0, blue_hp=1.0):
    def unit(x, hp):
        return {'type': '步兵3', 'team': '蓝', 'norm_xy': [x, 0.5], 'hp_ratio': hp,
                'vuln': 0, 'dfire': 0.0}
    robots = {}
    for i, x in enumerate(red_xs or [0.9, 0.8, 0.95]):
        robots[i] = dict(unit(x, 1.0), team='红')
    for i, x in enumerate(blue_xs or [0.2, 0.3]):
        robots[100 + i] = unit(x, blue_hp)
    return {'t': 0.0, 'robots': robots,
            'buildings': {'b': {'team': '红', 'hp_ratio': base_hp}}}

def test_choose_global_mode_situation_vs_game_theory():
    # 无对手模型 -> 局势感知: red 深推(push差>=2)+ 蓝低血 + 无基地威胁 -> 进攻
    w_attack = _worldstate(red_xs=[0.9, 0.8, 0.95], blue_xs=[0.4], blue_hp=0.5)
    assert gt.choose_global_mode(w_attack, None) == '进攻'
    # 有对手均匀模型 -> 博弈 best_response(落在 row0), 返回合法模式
    assert gt.choose_global_mode(w_attack, [1/3, 1/3, 1/3]) in ('进攻', '均衡', '防守')
    # 无机器人 -> 均衡
    w_empty = {'t': 0.0, 'robots': {}, 'buildings': {}}
    assert gt.choose_global_mode(w_empty, None) == '均衡'

def test_compute_decision_series_mode_counts_diversify():
    con = db.connect()
    rows = db.timeseries_for(con, config.GAME_REF)
    dps = features.build_decision_points(rows, config.GAME_REF)
    con.close()
    b = features.field_bounds(dps)
    decisions, prior, sigs = run_m3.compute_decision_series(dps, b)
    modes = {d['global_mode'] for d in decisions}
    assert len(modes) > 1  # 修复此前恒 {'进攻':140} 的病态
```

- [ ] **Step 2: 运行确认失败**

Run: `python -m pytest tests/test_situation_mode.py -q`
Expected: `AttributeError: module 'game.game_theory' has no attribute 'choose_global_mode'`;且 GAME_REF 的 `mode_counts` 会因**现有**恒 `{'进攻':140}` 而使 `len(modes)>1` 断言 False(即先 RED)。

- [ ] **Step 3: 写 `choose_global_mode` 实现**(`game/game_theory.py`,追加到文件末尾)

```python
def choose_global_mode(worldstate, opp_estimate=None):
    """M3 慢环全局模式: 有对手模型 -> 博弈 best_response(强对抗推演); 否则**回退局势感知规则**。

    PAYOFF_3 在均匀对手下恒落 row0(进攻), 导致 mode_counts 恒一类、行为先验频率退化。
    无对手模型时用 derive_global_mode(读世界状态, 会变)作上下文驱动, 修复病态; 对手模型就绪时仍走博弈。
    """
    if opp_estimate is not None:
        return choose_mode(PAYOFF_3, opp_estimate)
    from tree import global_mode as gm
    return gm.derive_global_mode(worldstate)
```

- [ ] **Step 4: 改 `run_m3.compute_decision_series`**

删掉函数体开头的 `est = opp_estimate or [1/3, 1/3, 1/3]`,并把循环内的模式选择改为:

```python
        mode = gt.choose_global_mode(w, opp_estimate)   # None -> derive_global_mode(局势感知)
```

(不再预置均匀占位;`opp_estimate` 原样透传。)

- [ ] **Step 5: 运行确认通过与无回归**

Run: `python -m pytest tests/test_situation_mode.py tests/test_m3_e2e.py -q`
Expected: 全部通过;`test_m3_e2e` 的 `compliance_rate==1.0` 仍成立(决策仍由 build_empty_decision + derive_unit_intent 生成,合法)。

- [ ] **Step 6: 手动冒烟(mode_counts 应多样)**

Run: `python run_m3.py`
Expected: `mode_counts` 不再是 `{"进攻":140}`,出现 均衡/进攻(可能含 防守);`prior` 频率因 move 分布变化而不再是全 1.0。

- [ ] **Step 7: 提交**

```bash
git add game/game_theory.py run_m3.py tests/test_situation_mode.py
git commit -m "feat(m3): situation-aware global mode via choose_global_mode; prior diversifies"
```

---

### Task P3: M3 — LLM 决策 → 分层对齐评估(合规/LLM-vs-规则/逐类/conflict)

**Files:**
- Create: `run_llm_m3.py`
- Test: `tests/test_llm_m3.py`(新, 用注入的假 `llm_state`, 不走网络; 直接测纯函数 `_as_decision`/`assess`)

**Interfaces:**
- Consumes: `run_llm_api.llm_state`(既有,DeepSeek 慢环,返回 `(state_str, raw_text)`);`worldstate.worldstate.build_worldstate`、`schema.layered.*`、`tree.global_mode.derive_global_mode`、`fastloop.behavior.feasible`、`eval.metrics`。
- Produces: `run_llm_m3.llm_state_to_intent(state, sent=None) -> dict`;`run_llm_m3._build_rule_decision(w) -> dict`;`run_llm_m3._as_decision(w, llm_state) -> dict`;`run_llm_m3.build_rule_decisions(dps, bounds) -> list`;`run_llm_m3.build_llm_decisions(dps, bounds, llm_state_fn, *, api_key=None, base_url=None, model=None) -> list`;`run_llm_m3.assess(decisions_llm, decisions_rule) -> dict`。

- [ ] **Step 1: 写失败测试**(`tests/test_llm_m3.py`)

```python
# tests/test_llm_m3.py
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import run_llm_m3
from schema import layered as sl

def _world():
    # 一张红方哨兵(中线处)的世界状态
    return {'t': 0.0,
            'robots': {'7': {'type': '哨兵', 'team': '红', 'norm_xy': [0.5, 0.5],
                             'hp_ratio': 1.0, 'vuln': 0, 'dfire': 0.0}},
            'buildings': {'b': {'team': '红', 'hp_ratio': 1.0}}}

def test_llm_state_to_intent_valid():
    assert run_llm_m3.llm_state_to_intent('def')['move'] == '回防'
    assert run_llm_m3.llm_state_to_intent('hitbuilding')['fire'] is True
    assert run_llm_m3.llm_state_to_intent('cruise')['target'] == '无'
    assert run_llm_m3.llm_state_to_intent('unknown')['move'] == '游击'  # 未知回退 cruise

def test_as_decision_uses_llm_state_and_is_valid():
    d = run_llm_m3._as_decision(_world(), 'def')
    assert d['units']['哨兵']['move'] == '回防'
    ok, errs = sl.validate(d)
    assert ok, errs
    d2 = run_llm_m3._as_decision(_world(), 'cruise')
    assert d2['units']['哨兵']['move'] == '游击'
    assert sl.validate(d2)[0] is True

def test_assess_reports_agreement_and_sentry_valid():
    rule = [sl.build_empty_decision(0.0)] * 3
    llm = [sl.build_empty_decision(0.0)] * 3
    for d in (rule + llm):
        d['global_mode'] = '均衡'
    for d in llm:
        d['units']['哨兵']['move'] = '游击'
    for d in rule:
        d['units']['哨兵']['move'] = '推进'   # 与 llm 不一致 -> agreement < 1
    rep = run_llm_m3.assess(llm, rule)
    assert rep['llm_sentry_valid'] == 1.0
    assert rep['compliance_rate'] == 1.0
    assert rep['agreement'] == 0.0           # 恰各不相同(匀速) -> 0.0
```

- [ ] **Step 2: 运行确认失败**

Run: `python -m pytest tests/test_llm_m3.py -q`
Expected: `ModuleNotFoundError: No module named 'run_llm_m3'`

- [ ] **Step 3: 写 `run_llm_m3.py`**(完整文件)

```python
# -*- coding: utf-8 -*-
"""M3 分层决策的 LLM 对齐评估: 把 DeepSeek 慢环的哨兵三态映射成分层 4 通道决策,
与规则派生(局势感知)序列对齐度量。GPU-free; 仅 --llm 分支真正调 API。

用法:
  python run_llm_m3.py                    # 规则派生(不调 API)
  python run_llm_m3.py --llm --limit 20   # LLM 慢环 (DeepSeek, 读 .env)
"""
import os, json, argparse
import config
from data import db, features
from worldstate import worldstate as ws
from schema import layered as sl
from tree import global_mode as gm
from fastloop import behavior as bh
from eval import metrics

# 哨兵三态 -> 4 通道(合法值域见 schema.layered)
_INTENT = {
    'def':         {'target': '敌方步兵', 'move': '回防', 'fire': False, 'risk': '高'},
    'hitbuilding': {'target': '敌方前哨站', 'move': '推进', 'fire': True,  'risk': '中'},
    'cruise':      {'target': '无',        'move': '游击', 'fire': False, 'risk': '低'},
}


def llm_state_to_intent(state, sent=None):
    """把 LLM 哨兵状态映射为合法 4 通道; 未知状态回退 cruise。"""
    return dict(_INTENT.get(state, _INTENT['cruise']))


def _build_rule_decision(w):
    """规则派生: 用局势感知全局模式 + 单位意图生成一个合法分层决策(核心, 供兜底与参照)。"""
    mode = gm.derive_global_mode(w)
    d = sl.build_empty_decision(w['t'])
    d['global_mode'] = mode
    for rid, unit in w['robots'].items():
        if unit['team'] != '红' or unit['type'] not in d['units']:
            continue
        if unit['type'] == '哨兵' and gm.derive_unit_intent(unit, w, mode).get('fire') \
                and not bh.feasible('hitbuilding', {'robots': {7: unit}, 'bot_state': 'hitbuilding'}, bh.Rules()):
            d['units'][unit['type']] = {'target': '敌方步兵', 'move': '回防', 'fire': False, 'risk': '高'}
        else:
            d['units'][unit['type']] = gm.derive_unit_intent(unit, w, mode)
    return d


def _as_decision(w, llm_state):
    """LLM 哨兵状态 -> 合法分层决策(其余类用规则派生兜底以保 schema 完整)。"""
    d = _build_rule_decision(w)
    sent = next((u for u in w['robots'].values() if u['team'] == '红' and u['type'] == '哨兵'), None)
    if sent is not None:
        d['units']['哨兵'] = llm_state_to_intent(llm_state, sent)
    return d


def build_rule_decisions(dps, bounds):
    return [_build_rule_decision(ws.build_worldstate(dp, bounds)) for dp in dps]


def build_llm_decisions(dps, bounds, llm_state_fn, *, api_key=None, base_url=None, model=None):
    """对每个决策点: 先出规则决策, 再由 llm_state_fn 的哨兵状态覆盖哨兵通道。"""
    out = []
    for dp in dps:
        w = ws.build_worldstate(dp, bounds)
        state, _raw = llm_state_fn(w, api_key, base_url, model)
        out.append(_as_decision(w, state))
    return out


def assess(decisions_llm, decisions_rule):
    """LLM vs 规则 对齐评估: 哨兵通道合法性、与规则一致性、逐类、冲突率。"""
    n = len(decisions_llm)
    if n == 0:
        return {'n': 0}
    llm_sentry = [d['units']['哨兵']['move'] for d in decisions_llm]
    rule_sentry = [d['units']['哨兵']['move'] for d in decisions_rule]
    valid = sum(1 for d in decisions_llm if sl.validate(d)[0])
    return {
        'n': n,
        'llm_sentry_valid': valid / n,
        'compliance_rate': valid / n,
        'agreement': sum(1 for a, b in zip(llm_sentry, rule_sentry) if a == b) / n,
        'per_class': metrics.per_class(rule_sentry, llm_sentry),
        'conflict_rate': sl.conflict_rate(decisions_llm),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--games', nargs='*', type=int, default=None)
    ap.add_argument('--llm', action='store_true', help='用 DeepSeek 慢环(默认仅规则派生)')
    ap.add_argument('--limit', type=int, default=20, help='LLM 分支跑多少个决策点(0=全部)')
    ap.add_argument('--out', default='out_m3')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    con = db.connect()
    gids = args.games or [config.GAME_REF]

    all_llm, all_rule = [], []
    for gid in gids:
        rows = db.timeseries_for(con, gid)
        dps = features.build_decision_points(rows, gid)
        b = features.field_bounds(dps)
        if args.limit > 0 and args.llm:
            dps = dps[:args.limit]
        all_rule += build_rule_decisions(dps, b)
        if args.llm:
            from run_llm_api import _load_env, llm_state
            cfg = _load_env()
            api_key = cfg['DEEPSEEK_API_KEY']
            base_url, model = cfg['DEEPSEEK_BASE_URL'], cfg['DEEPSEEK_MODEL']
            all_llm += build_llm_decisions(dps, b, llm_state, api_key=api_key, base_url=base_url, model=model)
        else:
            all_llm += build_rule_decisions(dps, b)  # 参照: 自洽, agreement 应=1.0
    con.close()

    report = assess(all_llm, all_rule)
    report['mode'] = 'llm_slowloop' if args.llm else 'rule_reference'
    with open(os.path.join(args.out, 'llm_m3_report.json'), 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
```

- [ ] **Step 4: 运行确认通过**

Run: `python -m pytest tests/test_llm_m3.py -q`
Expected: 3 passed(纯函数路径, 无网络)。

- [ ] **Step 5: 手动冒烟(规则分支自洽; LLM 分支相对便宜)**

Run: `python run_llm_m3.py`   → Expected: `mode: rule_reference`,`agreement: 1.0`,`per_class` 全满。
Run: `python run_llm_m3.py --llm --limit 20` → Expected: `mode: llm_slowloop`,`agreement` 显著低于 1.0(LLM 与规则差异化),并给出逐类 P/R/F1。

- [ ] **Step 6: 提交**

```bash
git add run_llm_m3.py tests/test_llm_m3.py
git commit -m "feat(m3): LLM layered-decision alignment assessment (run_llm_m3)"
```

---

### Task P4: M3 — 清理死代码 + 哨兵回退语义矛盾

**Files:**
- Modify: `run_m3.py`(加 `_sentry_decision_after_feasibility` 辅助并替换内联回退;`main` 里 `sigs` 改 `_sigs`)
- Test: `tests/test_m3_e2e.py`(加哨兵回退一致性断言)

**Interfaces:**
- Consumes: `schema.layered.TARGETS`(既有);`fastloop.behavior.feasible`、`tree.global_mode.derive_unit_intent`(既有)。
- Produces: `run_m3._sentry_decision_after_feasibility(unit, w, mode) -> dict(4通道)`。

- [ ] **Step 1: 写哨兵回退一致性失败测试**(`tests/test_m3_e2e.py` 追加)

```python
def test_sentry_fallback_move_target_coherent():
    import run_m3
    from schema import layered as sl
    # hp 0.1 < Rules.min_hp_to_attack 0.2, 故 进攻 姿态下 hitbuilding 不可行 -> 应回退
    w = {'t': 0.0,
         'robots': {7: {'type': '哨兵', 'team': '红', 'norm_xy': [0.9, 0.5],
                        'hp_ratio': 0.1, 'vuln': 0, 'dfire': 0.0}},
         'buildings': {'b': {'team': '红', 'hp_ratio': 1.0}}}
    it = run_m3._sentry_decision_after_feasibility(w['robots'][7], w, '进攻')
    assert it['move'] == '回防'
    assert it['target'] != '敌方前哨站'   # target 不再与 move=回防 矛盾
    assert it['target'] in sl.TARGETS
```

- [ ] **Step 2: 运行确认失败**

Run: `python -m pytest tests/test_m3_e2e.py::test_sentry_fallback_move_target_coherent -q`
Expected: `AttributeError: module 'run_m3' has no attribute '_sentry_decision_after_feasibility'`

- [ ] **Step 3: 加辅助函数并替换内联回退(`run_m3.py`)**

在 `compute_decision_series` 之前新增:

```python
def _sentry_decision_after_feasibility(unit, w, mode):
    """哨兵单位意图经快环可行性校验: 进攻开火不可行 -> 安全回退, 并修正 target 与 move 语义一致。"""
    it = gm.derive_unit_intent(unit, w, mode)
    if it['fire'] and not bh.feasible(
            'hitbuilding', {'robots': {7: unit}, 'bot_state': 'hitbuilding'}, bh.Rules()):
        it = {'target': '敌方步兵', 'move': '回防', 'fire': False, 'risk': '高'}
    return it
```

把 `compute_decision_series` 循环里的哨兵回退段替换为:

```python
            if unit['type'] == '哨兵':
                d['units'][unit['type']] = _sentry_decision_after_feasibility(unit, w, mode)
            else:
                d['units'][unit['type']] = gm.derive_unit_intent(unit, w, mode)
```

(删除原 `if unit['type']=='哨兵' and it['fire'] and not bh.feasible(...): it = {...}` 整段与 `it = gm.derive_unit_intent(...)`。)

- [ ] **Step 4: `main()` 未用 `sigs` 改名**

`main()` 里:

```python
        decisions, prior, _sigs = compute_decision_series(dps, b)
```

(保留函数返回值三元组不变以兼容 `test_m3_e2e` 的三元解包;只把 `main` 接收者改名并继续不用。)

- [ ] **Step 5: 运行确认通过与无回归**

Run: `python -m pytest tests/test_m3_e2e.py tests/test_situation_mode.py -q`
Expected: 全部通过;`compliance_rate` 仍 1.0;哨兵回退 target='敌方步兵' 与 move='回防' 一致。

- [ ] **Step 6: 提交**

```bash
git add run_m3.py tests/test_m3_e2e.py
git commit -m "chore(m3): coherent sentry fallback target; drop unused sigs in main"
```

---

## Execution Handoff
四个 Task 通过后,做一次**整支 final review**(opus),然后复跑 `python run_m1.py --rule-baseline`、`python run_m3.py`、`python run_m3.py --llm --limit 20`、`python -m pytest -q`,并跑一个 **613 局抽样 sweep**(P2/P3 泛化确认:mode_counts 多样、compliance 稳定、agreement 有差异)。无遗留 load-bearing 则本 plan 关闭。
