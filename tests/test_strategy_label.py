# -*- coding: utf-8 -*-
"""strategy_label 纯逻辑单测(无 GPU): 回看标签 + 滞后一致率。"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from labels import strategy_label as slb


def _mw(robots, buildings=None):
    """最小世界状态: robots=[(team,nx)], buildings=[(team,hp)]。"""
    return {'robots': {i: {'team': t, 'norm_xy': [nx, 0.5]} for i, (t, nx) in enumerate(robots)},
            'buildings': {i: {'team': t, 'hp_ratio': hp} for i, (t, hp) in enumerate(buildings or [])}}


def _ws(teams_nx, bldg=None, n=6):
    return [_mw(teams_nx, bldg) for _ in range(n)]


def test_forward_defense_when_enemy_in_own_half():
    # 敌(蓝)持续在红方半场(nx<0.5) -> 防守
    ws = _ws([('蓝', 0.2)])
    assert slb.forward_strategy(ws, 0, 5, team='红') == '防守'


def test_forward_offense_when_pushing_enemy_half():
    ws = _ws([('红', 0.8)])   # 己方单位深入敌半场
    assert slb.forward_strategy(ws, 0, 5, team='红') == '进攻'


def test_forward_neutral_when_no_crossing():
    ws = _ws([('红', 0.3)])  # 己方在己方半场, 无敌越线
    assert slb.forward_strategy(ws, 0, 5, team='红') == '均衡'


def test_forward_building_damage_drives_offense():
    # 无推进但敌建筑掉血(1.0->0.3) -> 进攻
    ws = []
    for k in range(6):
        hp = 1.0 if k == 0 else 0.3
        ws.append(_mw([('红', 0.3)], [('蓝', hp)]))
    assert slb.forward_strategy(ws, 0, 5, team='红') == '进攻'


def test_forward_building_leak_drives_defense():
    ws = []
    for k in range(6):
        hp = 1.0 if k == 0 else 0.4
        ws.append(_mw([('红', 0.3)], [('红', hp)]))   # 己方建筑掉血
    assert slb.forward_strategy(ws, 0, 5, team='红') == '防守'


def test_strategy_series_length():
    ws = _ws([('红', 0.5)], n=8)
    assert len(slb.strategy_series(ws, delta=3)) == 8


def test_lag_consistency_horizon0():
    preds = ['进攻', '防守', '进攻', '防守']
    labels = ['进攻', '进攻', '防守', '防守']
    r0 = slb.lag_consistency(preds, labels, 0)
    assert r0['rate'] == 0.5 and r0['n'] == 4
    # horizon1: 用"一帧前预测"对照"当前真实" -> 一致率更高(滞后反而更准)
    r1 = slb.lag_consistency(preds, labels, 1)
    assert abs(r1['rate'] - 2 / 3) < 1e-4 and r1['n'] == 3


def test_lag_curve_shape_and_monotonic_examples():
    preds = ['进攻', '进攻', '进攻', '进攻']
    labels = ['进攻', '进攻', '进攻', '进攻']
    curve = slb.lag_curve(preds, labels, horizons=(0, 1, 3, 8))
    assert len(curve) == 4
    assert curve[0]['rate'] == 1.0  # 全一致


# ---- 回看哨兵三态(忠于 def/cruise/hitbuilding) ----
def _wr(cls, team, nx, hp=1.0, dfire=0.0, vuln=0):
    """单个单位(结构与 build_worldstate 的输出一致)。"""
    return {'type': cls, 'team': team, 'norm_xy': [nx, 0.5], 'hp_ratio': hp,
            'dfire': dfire, 'vuln': vuln}


def _ws2(robots, n=6):
    return [{'robots': dict(enumerate(robots)), 'buildings': {}, 't': 0.0} for _ in range(n)]


def test_sentry_def_when_enemy_in_own_half_window():
    # 敌(蓝)步兵在红方半场(nx<0.5) -> 窗口内 def(最警觉)
    ws = _ws2([_wr('步兵3', '蓝', 0.2)])
    assert slb.sentry_state_series(ws, delta=5) == ['def'] * 6


def test_sentry_hitbuilding_when_sentry_pushes_and_fires():
    # 红哨兵越线(nx>0.5)且开火(dfire>0) -> hitbuilding
    ws = _ws2([_wr('哨兵', '红', 0.7, dfire=1.0)])
    assert slb.sentry_state_series(ws, delta=5) == ['hitbuilding'] * 6


def test_sentry_cruise_default():
    # 哨兵在己方半场未开火, 无敌进犯 -> cruise
    ws = _ws2([_wr('哨兵', '红', 0.3)])
    assert slb.sentry_state_series(ws, delta=5) == ['cruise'] * 6


# ---- 决策回看重标注(整体自洽) ----
def test_relabel_makes_decision_coherent_with_lookahead_mode():
    from schema import layered as sl
    # 红英雄在敌半场(nx>0.5) -> 前向窗口判 进攻 -> 动作整体按进攻派生(推进)
    ws = _ws2([_wr('英雄', '红', 0.75)])
    dps = [{'t': 0.0, 'robots': {}, 'team': {}} for _ in ws]  # 仅占位长度
    decisions = [sl.build_empty_decision(0.0) for _ in ws]
    out, labels = slb.relabel_decision_series(ws, decisions, delta=5, team='红')
    assert labels[0] == '进攻'
    d0 = out[0]
    assert d0['global_mode'] == '进攻'
    # 英雄动作与 mode 一致: 进攻 -> 推进 / 打前哨
    assert d0['units']['英雄']['move'] == '推进'
    assert d0['units']['英雄']['target'] == '敌方前哨站'


def test_relabel_defense_mode_derives_fallback():
    from schema import layered as sl
    # 敌(蓝)在红方半场(nx<0.2) -> 防守 -> 各单位回防
    ws = _ws2([_wr('英雄', '红', 0.3), _wr('步兵3', '蓝', 0.2)])
    decisions = [sl.build_empty_decision(0.0) for _ in ws]
    out, labels = slb.relabel_decision_series(ws, decisions, delta=5, team='红')
    assert labels[0] == '防守'
    assert out[0]['global_mode'] == '防守'
    assert out[0]['units']['英雄']['move'] == '回防'
    assert out[0]['units']['英雄']['target'] == '敌方步兵'


def test_relabel_preserves_length_and_missing_units():
    from schema import layered as sl
    ws = _ws2([_wr('英雄', '红', 0.75)], n=8)
    decisions = [sl.build_empty_decision(0.0) for _ in ws]
    out, labels = slb.relabel_decision_series(ws, decisions, delta=3, team='红')
    assert len(out) == len(ws) == len(labels)
    assert all(u['move'] in sl.MOVES for d in out for u in d['units'].values())
