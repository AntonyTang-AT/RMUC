# -*- coding: utf-8 -*-
"""M2 模块离线测试(无 torch/transformers)。"""
import os, sys, json, threading, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from schema import layered as sl
from rl import env as renv
from rl import distill_labels as dl
from labels import fusion as fu, transfer as tr
from blackboard import Blackboard
from finetune import make_dataset as md


def _dp(t):
    robots = {
        '7': {'type': '哨兵', 'team': '红', 'x': 900.0, 'y': 0.0, 'hp': 1000.0, 'maxhp': 1000.0,
              'small_heat': 0.0, 'big_heat': 0.0, 'power': 0.0, 'ammo17': 0.0, 'ammo42': 0.0, 'vuln': 0,
              'team_coins_total': 0.0, 'team_coins_left': 0.0},
        '1': {'type': '工程', 'team': '红', 'x': 850.0, 'y': -50.0, 'hp': 2000.0, 'maxhp': 2000.0,
              'small_heat': 0.0, 'big_heat': 0.0, 'power': 0.0, 'ammo17': 0.0, 'ammo42': 0.0, 'vuln': 0,
              'team_coins_total': 0.0, 'team_coins_left': 0.0},
        '100': {'type': '工程', 'team': '蓝', 'x': 100.0, 'y': 0.0, 'hp': 2000.0, 'maxhp': 2000.0,
                'small_heat': 0.0, 'big_heat': 0.0, 'power': 0.0, 'ammo17': 0.0, 'ammo42': 0.0, 'vuln': 0,
                'team_coins_total': 0.0, 'team_coins_left': 0.0},
        'b1': {'type': '前哨站', 'team': '红', 'x': 0.0, 'y': 0.0, 'hp': 1500.0, 'maxhp': 1500.0,
               'small_heat': 0.0, 'big_heat': 0.0, 'power': 0.0, 'ammo17': 0.0, 'ammo42': 0.0, 'vuln': 0,
               'team_coins_total': 0.0, 'team_coins_left': 0.0},
    }
    return {'t': float(t), 'robots': robots,
            'team': {'红': {'coins_total': 0.0, 'coins_left': 0.0}, '蓝': {'coins_total': 0.0, 'coins_left': 0.0}}}


def _bounds():
    return {'xmin': -500.0, 'xmax': 1400.0, 'ymin': -500.0, 'ymax': 500.0}


# -------- RL env --------
def test_m2env_obs_and_action_meanings():
    e = renv.M2Env([_dp(i) for i in range(4)], _bounds())
    e.reset()
    assert e.obs_dim > 0
    obs = e.obs('哨兵')
    assert isinstance(obs, np.ndarray) and obs.shape == (e.obs_dim,)
    assert len(e.action_meanings) == 12
    for move, fire in e.action_meanings:
        assert move in sl.MOVES and isinstance(fire, bool)


def test_m2env_building_excluded_from_enemy_count():
    e = renv.M2Env([_dp(0)], _bounds())
    # 蓝方单位仅 100(工程); 建筑 b1 结构被排除
    assert e._count_team(e.ws[0], '蓝') == 1


def test_m2env_step_and_win():
    e = renv.M2Env([_dp(i) for i in range(3)], _bounds())
    obs = e.reset()
    acts = {c: i % 12 for i, c in enumerate(renv.AGENTS)}
    nxt, rew, done, trunc, info = e.step(acts)
    assert set(nxt.keys()) == set(renv.AGENTS)
    assert e.win_condition() in (1.0, -1.0, 0.0)


# -------- distill --------
def test_rollout_to_decisions_all_valid():
    acts = [{'英雄': ('推进', True), '工程': ('推进', True), '步兵3': ('推进', True),
             '步兵4': ('推进', True), '空中': ('推进', True), '哨兵': ('推进', True)} for _ in range(3)]
    dps = [_dp(0.0), _dp(3.0), _dp(6.0)]
    bounds = {'xmin': 0, 'xmax': 1, 'ymin': 0, 'ymax': 1}
    ds = dl.rollout_to_decisions(acts, dps, bounds)
    assert len(ds) == 3
    for d in ds:
        assert sl.validate(d)[0]
        assert d['units']['哨兵']['move'] == '推进'


# -------- fusion --------
def _d(move, t):
    d = sl.build_empty_decision(t)
    d['global_mode'] = '均衡'
    for c in sl.UNIT_CLASSES:
        d['units'][c] = {'target': '敌方步兵', 'move': move, 'fire': False, 'risk': '低'}
    return d


def test_fuse_agree_no_conflict():
    out = fu.fuse([_d('推进', 0.0)], [_d('推进', 0.0)], [_d('推进', 0.0)])
    assert out['decisions'][0]['units']['哨兵']['move'] == '推进'
    assert out['conflict_rate'] == 0.0


def test_fuse_disagree_picks_top_weight():
    out = fu.fuse([_d('推进', 0.0)], [_d('回防', 0.0)], None)
    assert out['conflict_rate'] == 1.0
    assert out['decisions'][0]['units']['哨兵']['move'] == '推进'
    assert sl.validate(out['decisions'][0])[0]


# -------- transfer --------
def test_transfer_augment_sparse():
    decisions = [_d('推进', 0.0)]
    coverage = {'哨兵': 0.1, '工程': 0.9}
    aug, n = tr.augment_transfer(decisions, coverage, None, threshold=0.5)
    assert n >= 1
    assert tr.TRANSFER_SOURCE.get('哨兵') == '工程'


# -------- blackboard --------
def test_blackboard_put_get_timeout_snapshot():
    bb = Blackboard()
    bb.put('a', 1)
    assert bb.get('a') == 1
    t0 = time.time()
    try:
        bb.get('missing', timeout=0.1)
        raised = False
    except KeyError:
        raised = True
    assert raised and (time.time() - t0) >= 0.09
    bb.put('b', 2)
    assert bb.snapshot() == {'a': 1, 'b': 2}
    bb.close()


def test_blackboard_subscribe():
    bb = Blackboard()
    got = []
    ev = threading.Event()
    bb.subscribe('x', lambda k, v: (got.append((k, v)), ev.set()))
    bb.put('x', 5)
    ev.wait(1.0)
    assert ('x', 5) in got
    bb.close()


# -------- make_dataset --------
def test_dataset_records_valid():
    d = _d('推进', 0.0)
    dps = [_dp(0.0)]
    bounds = {'xmin': -500.0, 'xmax': 1400.0, 'ymin': -500.0, 'ymax': 500.0}
    recs = md.make_records([d], dps, bounds)
    assert len(recs) == 1
    assert isinstance(md.situation_digest({'t': 0.0, 'robots': {}, 'buildings': {},
                                           'team': {}}), str)
    assert sl.validate(json.loads(recs[0]['completion']))[0]
