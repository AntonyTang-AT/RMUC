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


def test_rollout_to_decisions_from_units_mode_is_attack_when_all_push():
    """默认 mode_source=from_units: 全推进 -> global_mode=进攻。"""
    acts = [{'英雄': ('推进', True), '工程': ('推进', True), '步兵3': ('推进', True),
             '步兵4': ('推进', True), '空中': ('推进', True), '哨兵': ('推进', True)}]
    dps = [_dp(0.0)]
    bounds = {'xmin': 0, 'xmax': 1, 'ymin': 0, 'ymax': 1}
    ds = dl.rollout_to_decisions(acts, dps, bounds, mode_source='from_units')
    assert ds[0]['global_mode'] == '进攻'


def test_rollout_from_units_enables_fuse_mode_diverge_from_rule():
    """RL 全推进(进攻) vs rule 防守 -> fuse_mode 取 RL。"""
    acts = [{'英雄': ('推进', False), '工程': ('推进', False), '步兵3': ('推进', False),
             '步兵4': ('推进', False), '空中': ('推进', False), '哨兵': ('推进', False)}]
    dps = [_dp(0.0)]
    bounds = {'xmin': 0, 'xmax': 1, 'ymin': 0, 'ymax': 1}
    rl = dl.rollout_to_decisions(acts, dps, bounds, mode_source='from_units')
    rule = [_d('回防', 0.0)]
    rule[0]['global_mode'] = '防守'
    out = fu.fuse(rule, rl, None, mode_policy='fuse_mode')
    assert rl[0]['global_mode'] == '进攻'
    assert out['decisions'][0]['global_mode'] == '进攻'


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
    # 哨兵通道: CHANNEL_CONFS {'rule':0.4,'rl':0.6,'llm':0.0} -> RL 权重最高, 冲突取 RL
    out = fu.fuse([_d('推进', 0.0)], [_d('回防', 0.0)], None)
    assert out['conflict_rate'] == 1.0
    assert out['decisions'][0]['units']['哨兵']['move'] == '回防'
    # 非哨兵通道: 走 DEFAULT_CONFS {'rule':0.5,'rl':0.3,...} -> 规则权重最高, 冲突仍取规则(推进)
    assert out['decisions'][0]['units']['英雄']['move'] == '推进'
    assert sl.validate(out['decisions'][0])[0]
    # 默认锚定: global_mode 来自 rule
    assert out['decisions'][0]['global_mode'] == '均衡'
    assert out['mode_policy'] == 'anchor_rule'


def test_fuse_mode_policy_fuse_mode_prefers_rl():
    """fuse_mode: MODE_CONFS rl=0.6 > rule=0.4 -> 取 RL 的 global_mode。"""
    rule = [_d('推进', 0.0)]
    rule[0]['global_mode'] = '防守'
    rl = [_d('回防', 0.0)]
    rl[0]['global_mode'] = '进攻'
    out = fu.fuse(rule, rl, None, mode_policy='fuse_mode')
    assert out['mode_policy'] == 'fuse_mode'
    assert out['decisions'][0]['global_mode'] == '进攻'


def test_fuse_mode_from_units_when_missing():
    """RL 无 global_mode 时由 move 聚合导出后再融合。"""
    rule = [_d('推进', 0.0)]
    rule[0]['global_mode'] = '均衡'
    rl = [_d('推进', 0.0)]
    del rl[0]['global_mode']
    # 全类推进 -> mode_from_units='进攻'; rl 权重更高 -> 进攻
    out = fu.fuse(rule, rl, None, mode_policy='fuse_mode')
    assert out['decisions'][0]['global_mode'] == '进攻'
    assert fu.mode_from_units(rl[0]) == '进攻'


def test_mode_from_units_attack_bias():
    """进攻偏向: 单推进即可进攻; 防守需 ≥3 回防。"""
    one_push = {
        'units': {
            '英雄': {'move': '推进'},
            '工程': {'move': '坚守'},
            '步兵3': {'move': '回防'},
        }
    }
    assert fu.mode_from_units(one_push) == '进攻'
    two_back = {
        'units': {
            '英雄': {'move': '回防'},
            '工程': {'move': '回防'},
            '步兵3': {'move': '坚守'},
        }
    }
    assert fu.mode_from_units(two_back) == '均衡'
    three_back = {
        'units': {
            '英雄': {'move': '回防'},
            '工程': {'move': '回防'},
            '步兵3': {'move': '回防'},
            '步兵4': {'move': '坚守'},
        }
    }
    assert fu.mode_from_units(three_back) == '防守'


def test_fuse_anchor_ignores_rl_mode():
    rule = [_d('推进', 0.0)]
    rule[0]['global_mode'] = '防守'
    rl = [_d('推进', 0.0)]
    rl[0]['global_mode'] = '进攻'
    out = fu.fuse(rule, rl, None, mode_policy='anchor_rule')
    assert out['decisions'][0]['global_mode'] == '防守'


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
    back = json.loads(recs[0]['completion'])
    # 默认 completion 剥离旁路；意图只在 instruction
    assert 'opponent_intent' not in back and 'decision_meta' not in back
    assert '敌方意图' in recs[0]['instruction']
    assert sl.validate(back)[0]
    recs2 = md.make_records([d], dps, bounds, keep_sidecars_in_completion=True)
    back2 = json.loads(recs2[0]['completion'])
    assert 'opponent_intent' in back2 and 'decision_meta' in back2
    from labels import router as rt
    assert sl.validate(rt.strip_sidecar_for_validate(back2))[0]


def test_dataset_records_include_temporal_forecast_context_when_supplied():
    """训练样本必须携带时序预测参考，而不是只在运行时注入。"""
    d = _d('推进', 0.0)
    dps = [_dp(0.0)]
    bounds = {'xmin': -500.0, 'xmax': 1400.0, 'ymin': -500.0, 'ymax': 500.0}
    forecast = {
        'schema_version': 'temporal-entity-forecast-v1',
        'model_version': 'test-model',
        'calibration_version': 'test-cal',
        'input_snapshot_hash': 'test-hash',
        'team': '红',
        'issued_at': 0.0,
        'horizons': {
            '10': {
                'global_mode': {'top': '进攻', 'confidence': .8,
                                'probabilities': {'进攻': .8, '均衡': .2}},
                'opponent_trend': {'top': '推进', 'confidence': .7,
                                   'probabilities': {'推进': .7, '稳定': .3}},
                'enemy_region': {'top': '蓝中路', 'confidence': .6,
                                 'probabilities': {'蓝中路': .6}},
                'entities': {},
            },
        },
    }
    recs = md.make_records([d], dps, bounds, temporal_forecasts=[forecast])
    assert len(recs) == 1
    assert '预测参考信息' in recs[0]['instruction']
    assert '10s' in recs[0]['instruction']
    assert 'test-hash' in recs[0]['instruction']
