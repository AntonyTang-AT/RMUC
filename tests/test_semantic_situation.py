# -*- coding: utf-8 -*-
"""semantic_situation 无 torch 单测。"""
from worldstate import semantic as sem


def _w(enemy_x=0.7, our_x=0.3, base_hp=1.0, enemy_types=('步兵3',)):
    robots = {
        's': {'type': '哨兵', 'team': '红', 'norm_xy': [our_x, 0.5], 'hp_ratio': 1.0},
    }
    for i, t in enumerate(enemy_types):
        robots[f'e{i}'] = {'type': t, 'team': '蓝', 'norm_xy': [enemy_x, 0.5], 'hp_ratio': 1.0}
    return {
        't': 0.0,
        'robots': robots,
        'buildings': {
            'rb': {'type': '基地', 'team': '红', 'hp_ratio': base_hp},
            'ro': {'type': '前哨站', 'team': '红', 'hp_ratio': 1.0},
            'bo': {'type': '前哨站', 'team': '蓝', 'hp_ratio': 1.0},
        },
    }


def test_base_threat_on_low_base():
    s = sem.derive_semantic_situation(_w(base_hp=0.3, enemy_x=0.8))
    assert s['base_threat'] is True
    assert s['source'] == 'replay_approx'


def test_enemy_trend_press():
    prev = _w(enemy_x=0.7)
    cur = _w(enemy_x=0.55)
    s = sem.derive_semantic_situation(cur, prev_w=prev)
    assert s['enemy_trend'] == 'press'


def test_lost_target():
    prev = _w(enemy_types=('步兵3', '英雄'))
    cur = _w(enemy_types=('步兵3',))
    s = sem.derive_semantic_situation(cur, prev_w=prev)
    assert s['lost_target'] is True
    assert '英雄' in s['lost_types']


def test_format_line():
    s = sem.derive_semantic_situation(_w(base_hp=0.2))
    line = sem.format_semantic_line(s)
    assert 'base_threat=1' in line
    assert '语义:' in line


def test_global_analysis_has_evidence_and_horizon_specificity():
    analysis = sem.derive_global_analysis(_w(enemy_x=0.2, our_x=0.8), team='红')
    assert analysis['schema_version'] == 'global-analysis-v1'
    assert analysis['state']['opponent_push_count'] >= 1
    assert analysis['horizon_views']['10s']['specificity'] == 'high'
    assert analysis['horizon_views']['120s']['specificity'] == 'coarse'
