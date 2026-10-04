# -*- coding: utf-8 -*-
"""unit_features / unit_student 形状测。"""
import numpy as np
import pytest
from schema import layered as sl
from rl import unit_features as uf


def _dp():
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
    return {'t': 0.0, 'robots': robots,
            'team': {'红': {'coins_total': 0.0, 'coins_left': 0.0},
                     '蓝': {'coins_total': 0.0, 'coins_left': 0.0}}}


def _bounds():
    return {'xmin': -500.0, 'xmax': 1400.0, 'ymin': -500.0, 'ymax': 500.0}


def test_feat_dim():
    feat = uf.pack_unit_feat(_dp(), _bounds(), mode='进攻',
                             opp_intent={'top': '盯哨兵', 'probs': {'盯哨兵': 0.7}})
    assert feat.shape == (uf.FEAT_DIM,)
    assert feat.dtype == np.float32


def test_sample_sentry_row():
    d = sl.build_empty_decision(0.0)
    d['global_mode'] = '进攻'
    d['units']['哨兵'] = {'target': '敌方哨兵', 'move': '游击', 'fire': True, 'risk': '低'}
    row = uf.sample_sentry_row(_dp(), _bounds(), d, game_id=1)
    assert len(row['feat']) == uf.FEAT_DIM
    assert row['y_move'] == uf.move_to_id('游击')
    assert row['fire'] == 1


def test_unit_student_shapes_optional_torch():
    torch = pytest.importorskip('torch')
    from rl.unit_student import UnitStudent, train_unit_student
    X = np.random.randn(16, uf.FEAT_DIM).astype(np.float32)
    y_m = np.random.randint(0, uf.N_MOVE, size=16)
    y_f = np.random.randint(0, 2, size=16).astype(np.float32)
    stu, stats = train_unit_student(X, y_m, y_f, epochs=1, batch=8, device='cpu')
    move, fire = stu.predict(X[0])
    assert move in sl.MOVES
    assert isinstance(fire, bool)
    assert stats['n'] == 16
