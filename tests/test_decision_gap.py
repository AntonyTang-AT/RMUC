# -*- coding: utf-8 -*-
"""eval.decision_gap.global_mode_align_gap 单测(纯 numpy, 无 torch/transformers/sklearn)。

用合成 worldstate(手工构造 reward_components 读到的字段)控制红方有利度 fav,
验证: 把'进攻'放到高有利度点、'防守'放到低有利度点 -> gap>0; 缺一方 -> gap=None。
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from eval import decision_gap as dg


def _ws(bh, red_alive, blue_alive, red_coins, blue_coins, red_ammo=0.0, blue_ammo=0.0):
    """构造 worldstate dict(buildings/robots/team), 供 reward_components 算红方有利度。
    bh: {(team, type): hp_ratio} 建筑血量表。alive 用 hp_ratio>0 的单位个数近似。"""
    robots = {}
    for i in range(red_alive):
        robots[100 + i] = {'type': '步兵3', 'team': '红', 'hp_ratio': 1.0,
                           'ammo17': red_ammo, 'ammo42': 0.0, 'heat': 0.0,
                           'norm_xy': (0.2, 0.5)}
    for i in range(blue_alive):
        robots[200 + i] = {'type': '步兵3', 'team': '蓝', 'hp_ratio': 1.0,
                           'ammo17': blue_ammo, 'ammo42': 0.0, 'heat': 0.0,
                           'norm_xy': (0.8, 0.5)}
    buildings = {}
    for rid, ((tag, typ), hp) in enumerate(bh.items(), start=1):
        buildings[rid] = {'type': typ, 'team': tag, 'hp_ratio': hp}
    return {'robots': robots, 'buildings': buildings,
            'team': {'红': {'coins': red_coins}, '蓝': {'coins': blue_coins}}}


def _high_fav():
    # 红方大优: 敌基地/前哨残血 + 红满血 + 红多活 + 红伤高 + 红金币领先
    return _ws({('红', '基地'): 1.0, ('红', '前哨站'): 1.0, ('蓝', '基地'): 0.2, ('蓝', '前哨站'): 0.1},
               3, 1, 800, 200, red_ammo=3000.0, blue_ammo=0.0)


def _low_fav():
    # 红方大劣: 红基地/前哨残血 + 红少活 + 红伤低 + 红金币落后
    return _ws({('红', '基地'): 0.2, ('红', '前哨站'): 0.3, ('蓝', '基地'): 1.0, ('蓝', '前哨站'): 1.0},
               1, 3, 100, 900, red_ammo=0.0, blue_ammo=3000.0)


def test_gap_positive_when_attack_on_high_fav():
    """进攻放高有利度、防守放低有利度 -> gap > 0(标签/模型与胜势对齐)。"""
    meta = dg.global_mode_align_gap(['进攻', '防守'], [_high_fav(), _low_fav()])
    assert meta['gap'] is not None and meta['gap'] > 0
    assert meta['counts']['进攻'] == 1 and meta['counts']['防守'] == 1
    assert meta['mean_fav_by_mode']['进攻'] > meta['mean_fav_by_mode']['防守']


def test_gap_reversed_when_attack_on_low_fav():
    """反例: 进攻放低有利度、防守放高有利度 -> gap < 0(自洽但方向反 -> 判为未对齐)。"""
    meta = dg.global_mode_align_gap(['进攻', '防守'], [_low_fav(), _high_fav()])
    assert meta['gap'] is not None and meta['gap'] < 0


def test_gap_none_when_no_defense():
    """只有进攻点(无防守点) -> gap None, 不可比, 但 count/mean 仍给出。"""
    meta = dg.global_mode_align_gap(['进攻', '进攻'], [_high_fav(), _high_fav()])
    assert meta['gap'] is None
    assert meta['counts']['进攻'] == 2 and '防守' not in meta['counts']
    assert meta['mean_fav_by_mode']['进攻'] > 0

def test_gap_lte_comparator():
    """gap_lte: rollout(0.05) >= lookahead(-0.1) -> True; None 侧 -> None(不可比)。"""
    assert dg.gap_lte({'gap': 0.05}, -0.1) is True
    assert dg.gap_lte({'gap': -0.2}, -0.1) is False
    assert dg.gap_lte({'gap': None}, -0.1) is None
    assert dg.gap_lte({'gap': 0.05}, None) is None
