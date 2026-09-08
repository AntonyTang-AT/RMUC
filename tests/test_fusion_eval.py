# -*- coding: utf-8 -*-
"""eval.fusion_eval 纯函数单测(无 torch/transformers/sklearn, 离线可跑)。

覆盖: global_mode 分布/退化判据、合法性、RL 源 fav-align(攻击倾向随胜势调整)、
compare_rl_sources 的 physics-vs-bandit delta、fusion_overview。
用合成 worldstate(schema 合法单位)控制红方有利度 fav; 用 sl.build_empty_decision 造合法决策。
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from schema import layered as sl
from eval import fusion_eval as fe


# ---- 合成 worldstate: 复用 test_decision_gap 的构造, 控制红方 fav ----
def _ws(bh, red_alive, blue_alive, red_coins, blue_coins, red_ammo=0.0, blue_ammo=0.0):
    robots = {}
    for i in range(red_alive):
        robots[100 + i] = {'type': '步兵3', 'team': '红', 'hp_ratio': 1.0,
                           'ammo17': red_ammo, 'ammo42': 0.0, 'heat': 0.0, 'norm_xy': (0.2, 0.5)}
    for i in range(blue_alive):
        robots[200 + i] = {'type': '步兵3', 'team': '蓝', 'hp_ratio': 1.0,
                           'ammo17': blue_ammo, 'ammo42': 0.0, 'heat': 0.0, 'norm_xy': (0.8, 0.5)}
    buildings = {}
    for rid, ((tag, typ), hp) in enumerate(bh.items(), start=1):
        buildings[rid] = {'type': typ, 'team': tag, 'hp_ratio': hp}
    return {'robots': robots, 'buildings': buildings,
            'team': {'红': {'coins': red_coins}, '蓝': {'coins': blue_coins}}}


def _high_fav():
    return _ws({('红', '基地'): 1.0, ('红', '前哨站'): 1.0, ('蓝', '基地'): 0.2, ('蓝', '前哨站'): 0.1},
               3, 1, 800, 200, red_ammo=3000.0)


def _low_fav():
    return _ws({('红', '基地'): 0.2, ('红', '前哨站'): 0.3, ('蓝', '基地'): 1.0, ('蓝', '前哨站'): 1.0},
               1, 3, 100, 900, blue_ammo=3000.0)


def _decision(t, mode, move_map):
    """构造合法分层决策: 全局 mode + 各单位类 move(其余字段 build_empty 兜底)。"""
    d = sl.build_empty_decision(t)
    d['global_mode'] = mode
    for cls, mv in move_map.items():
        d['units'][cls]['move'] = mv
    return d


CLS = list(sl.UNIT_CLASSES)


def test_mode_distribution_normalized_and_empty():
    ds = [_decision(0, '进攻', {}), _decision(1, '均衡', {}), _decision(2, '进攻', {})]
    md = fe.mode_distribution(ds)
    assert md == {'进攻': round(2 / 3, 4), '均衡': round(1 / 3, 4)}, md
    assert fe.mode_distribution([]) == {}


def test_legal_rate():
    assert fe.legal_rate([_decision(0, '进攻', {})]) == 1.0
    bad = _decision(0, '进攻', {})
    bad['global_mode'] = '邪魔外道'  # 非法 mode
    assert fe.legal_rate([bad]) == 0.0
    assert fe.legal_rate([]) == 0.0


def test_attack_balance_and_degenerate():
    assert fe.is_degenerate({'防守': 1.0}) is True          # 单模守
    assert fe.is_degenerate({'进攻': 0.05, '防守': 0.95}) is True   # 攻+均<0.10
    assert fe.is_degenerate({'进攻': 0.5, '防守': 0.5}) is False
    assert fe.is_degenerate({'进攻': 0.05, '均衡': 0.05, '防守': 0.9}) is False  # 攻+均=0.10 恰不退化
    assert abs(fe.attack_balance_share({'进攻': 0.3, '均衡': 0.2, '防守': 0.5}) - 0.5) < 1e-9


def test_rl_source_fav_align_positive_when_aggressive_on_high_fav():
    """规则恒坚守; RL 在高 fav 点推进、低 fav 点回防/坚守 -> fav_align>0 且 override>0。"""
    worldstates = [_high_fav(), _low_fav()]
    # 规则: 两点全类 '坚守'(防御)
    rule = [_decision(0, '均衡', {c: '坚守' for c in CLS}),
            _decision(1, '防守', {c: '坚守' for c in CLS})]
    # RL: 高 fav 点全类 '推进'(进攻), 低 fav 点全类 '坚守'(与规则同)
    rl = [_decision(0, '进攻', {c: '推进' for c in CLS}),
          _decision(1, '防守', {c: '坚守' for c in CLS})]
    m = fe.rl_source_metrics(rl, rule, worldstates, '红')
    assert m['override_rate'] > 0.0
    assert m['fav_align'] is not None and m['fav_align'] > 0.0
    assert len(m['move_diversity']) == len(CLS)
    assert m['move_diversity']['步兵3'] >= 2  # RL 在步兵3 上至少出过 2 种 move(推进+坚守)
    assert m['legal_rate'] == 1.0


def test_rl_source_fav_align_zero_when_no_override():
    """RL 与规则完全一致 -> override=0, fav_align 定义上为 0(无 Δ)。"""
    worldstates = [_high_fav(), _low_fav()]
    rule = [_decision(0, '进攻', {c: '坚守' for c in CLS}),
            _decision(1, '防守', {c: '坚守' for c in CLS})]
    m = fe.rl_source_metrics(list(rule), rule, worldstates, '红')  # 深拷贝同值
    assert m['override_rate'] == 0.0
    gain = m['fav_align']
    assert gain is None or gain == 0.0  # 无 Δ -> hi.mean()-lo.mean()=0


def test_compare_rl_sources():
    phys = {'fav_align': 0.12, 'legal_rate': 1.0, 'override_rate': 0.3}
    band = {'fav_align': -0.05, 'legal_rate': 0.95, 'override_rate': 0.2}
    c = fe.compare_rl_sources(phys, band)
    assert c['phys_ge_bandit_fav_align'] is True
    assert abs(c['fav_align_delta_phys_minus_bandit'] - 0.17) < 1e-9
    assert c['both_legal'] is True and c['both_override'] is True
    # 空对齐(单局无 fav 区分) -> 不可比
    c2 = fe.compare_rl_sources({'fav_align': None, 'legal_rate': 1.0, 'override_rate': 0.1},
                               {'fav_align': None, 'legal_rate': 1.0, 'override_rate': 0.1})
    assert c2['phys_ge_bandit_fav_align'] is None


def test_fusion_overview():
    worldstates = [_high_fav(), _low_fav(), _high_fav()]
    ds = [_decision(0, '进攻', {}), _decision(1, '防守', {}), _decision(2, '均衡', {})]
    res = {'decisions': ds, 'conflict_rate': 0.5}
    o = fe.fusion_overview(res, worldstates, '红')
    assert o['n'] == 3
    assert o['legal_rate'] == 1.0
    assert o['conflict_rate'] == 0.5
    assert o['degenerate'] is False  # 攻+均=0.67 ≥0.10
    assert o['align_gap']['gap'] is not None
