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
    w_attack = _worldstate(red_xs=[0.9, 0.8, 0.95], blue_xs=[0.4], blue_hp=0.5)
    assert gt.choose_global_mode(w_attack, None) == '进攻'
    assert gt.choose_global_mode(w_attack, [1/3, 1/3, 1/3]) in ('进攻', '均衡', '防守')
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
