# -*- coding: utf-8 -*-
"""mode_features 单测（无 torch）。"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from schema import layered as sl
from rl import mode_features as mf
from worldstate import worldstate as ws


def _dp(t=0.0):
    robots = {
        '7': {'type': '哨兵', 'team': '红', 'x': 900.0, 'y': 0.0, 'hp': 1000.0, 'maxhp': 1000.0,
              'small_heat': 0.0, 'big_heat': 0.0, 'power': 0.0, 'ammo17': 0.0, 'ammo42': 0.0, 'vuln': 0,
              'team_coins_total': 0.0, 'team_coins_left': 0.0},
        '1': {'type': '英雄', 'team': '红', 'x': 850.0, 'y': -50.0, 'hp': 2000.0, 'maxhp': 2000.0,
              'small_heat': 0.0, 'big_heat': 0.0, 'power': 0.0, 'ammo17': 0.0, 'ammo42': 0.0, 'vuln': 0,
              'team_coins_total': 0.0, 'team_coins_left': 0.0},
        '100': {'type': '英雄', 'team': '蓝', 'x': 100.0, 'y': 0.0, 'hp': 2000.0, 'maxhp': 2000.0,
                'small_heat': 0.0, 'big_heat': 0.0, 'power': 0.0, 'ammo17': 0.0, 'ammo42': 0.0, 'vuln': 0,
                'team_coins_total': 0.0, 'team_coins_left': 0.0},
        'b1': {'type': '基地', 'team': '红', 'x': 0.0, 'y': 0.0, 'hp': 5000.0, 'maxhp': 5000.0,
               'small_heat': 0.0, 'big_heat': 0.0, 'power': 0.0, 'ammo17': 0.0, 'ammo42': 0.0, 'vuln': 0,
               'team_coins_total': 0.0, 'team_coins_left': 0.0},
        'b2': {'type': '基地', 'team': '蓝', 'x': 1200.0, 'y': 0.0, 'hp': 5000.0, 'maxhp': 5000.0,
               'small_heat': 0.0, 'big_heat': 0.0, 'power': 0.0, 'ammo17': 0.0, 'ammo42': 0.0, 'vuln': 0,
               'team_coins_total': 0.0, 'team_coins_left': 0.0},
    }
    return {'t': float(t), 'robots': robots,
            'team': {'红': {'coins_total': 0.0, 'coins_left': 0.0},
                     '蓝': {'coins_total': 0.0, 'coins_left': 0.0}}}


def _bounds():
    return {'xmin': -500.0, 'xmax': 1400.0, 'ymin': -500.0, 'ymax': 500.0}


def test_obs_dim_is_40():
    obs = mf.pack_obs(_dp(), _bounds())
    assert obs.shape == (mf.OBS_DIM,)
    assert obs.dtype == np.float32


def test_mode_id_roundtrip():
    for m in sl.GLOBAL_MODES:
        assert mf.id_to_mode(mf.mode_to_id(m)) == m
    assert mf.mode_to_id('???') == mf.mode_to_id('均衡')


def test_sample_row_fields():
    row = mf.sample_row(_dp(), _bounds(), teacher_mode='进攻', game_id=1)
    assert len(row['obs']) == 40
    assert row['mode'] == '进攻' and row['y'] == 0
    assert 'need_llm' in row and 'rule_mode' in row


def test_refill_units_valid():
    w = ws.build_worldstate(_dp(), _bounds())
    d = mf.refill_units_for_mode(sl.build_empty_decision(0.0), w, '进攻')
    assert d['global_mode'] == '进攻'
    assert sl.validate(d)[0]


def test_rows_to_xy_tier1_filter():
    rows = [
        {'obs': [0.0] * 40, 'mode': '进攻', 'need_llm': False},
        {'obs': [1.0] * 40, 'mode': '防守', 'need_llm': True},
    ]
    X, y, m = mf.rows_to_xy(rows, tier1_only=True)
    assert len(y) == 1 and y[0] == mf.mode_to_id('进攻')
    assert m.shape[0] == 1
