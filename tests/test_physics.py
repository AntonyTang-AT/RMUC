# -*- coding: utf-8 -*-
"""rl.physics 符号物理单测(无 torch, 离线可跑)。"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from rl import physics as ph


def _r(rid, cls, team, x, y, hp=100.0, a17=0.0, a42=0.0, heat=0.0):
    return rid, {'type': cls, 'team': team, 'x': x, 'y': y, 'hp': hp, 'maxhp': 100.0,
                 'ammo17': a17, 'ammo42': a42, 'heat': heat, 'dfire': 0.0, 'yaw': 0.0,
                 'alive': 1}


def _robots(*items):
    return {r[0]: r[1] for r in items}


def _act(move='无', fire=False):
    return {'move': move, 'fire': bool(fire)}


def _act_t(move, fire, target):
    return {'move': move, 'fire': bool(fire), 'target': target}


def test_push_moves_red_toward_enemy_x():
    # 红队: 推进 -> +x
    rb = _robots(_r(1, '英雄', '红', 5.0, 5.0), _r(2, '英雄', '蓝', 25.0, 5.0))
    out = ph.step_robots(rb, '红', {'英雄': _act('推进')}, dt=3.0)
    assert out[1]['x'] > 5.0  # 红英雄推进 -> 前移
    assert out[2]['x'] == 25.0  # 蓝方(非我方动作)运动不动

def test_fallback_moves_red_toward_own_x():
    rb = _robots(_r(1, '英雄', '红', 20.0, 5.0), _r(2, '英雄', '蓝', 25.0, 5.0))
    out = ph.step_robots(rb, '红', {'英雄': _act('回防')}, dt=3.0)
    assert out[1]['x'] < 20.0  # 回防 -> 向己方(低x)

def test_hold_stays_put():
    rb = _robots(_r(1, '英雄', '红', 10.0, 5.0))
    out = ph.step_robots(rb, '红', {'英雄': _act('坚守')}, dt=3.0)
    assert abs(out[1]['x'] - 10.0) < 1e-9 and abs(out[1]['y'] - 5.0) < 1e-9

def test_fire_damages_nearest_enemy():
    rb = _robots(_r(1, '步兵3', '红', 5.0, 5.0), _r(2, '步兵3', '蓝', 9.0, 5.0))
    before = rb[2]['hp']
    out = ph.step_robots(rb, '红', {'步兵3': _act('推进', True)}, dt=3.0)
    assert out[2]['hp'] < before  # 红开火打蓝 -> 蓝掉血

def test_enemy_fire_hits_friendly():
    rb = _robots(_r(1, '英雄', '红', 5.0, 5.0), _r(2, '步兵3', '蓝', 9.0, 5.0))
    before = rb[1]['hp']
    # 我方不动, 敌(蓝)打最近我方(红) -> 红掉血
    out = ph.step_robots(rb, '红', {'英雄': _act('坚守', False)}, dt=3.0)
    assert out[1]['hp'] < before
    # 蓝方 ammo 涨
    assert out[2]['ammo17'] > 0.0

def test_building_hp_touchable():
    rb = _robots(_r(1, '英雄', '红', 5.0, 5.0),
                 _r(99, '基地', '蓝', 14.0, 5.0))   # 距离 9 < RANGE
    out = ph.step_robots(rb, '红', {'英雄': _act('推进', True)}, dt=3.0)
    assert out[99]['hp'] < 100.0  # 红打蓝基地 -> 基地掉血

def test_calibrate_runs_and_returns_speed_map():
    # 构造 minimal dps: 英雄从 x 递增 -> 速度>0
    def dps(dt=3.0):
        return [{'t': dt * i, 'robots': {
            1: {'type': '英雄', 'team': '红', 'x': 5.0 + 2.0 * i, 'y': 5.0,
                'ammo42': 0.0, 'ammo17': 0.0, 'hp': 100.0},
        }} for i in range(4)]
    P = ph.calibrate([dps()])
    assert P['speed']['英雄'] > 0
    assert P['speed']['英雄'] > ph.SPEED.get('英雄', 0)  # 明显比默认速度大

def test_calibrate_detects_42mm():
    def dps():
        return [{'t': 3.0 * i, 'robots': {
            1: {'type': '英雄', 'team': '红', 'x': 5.0, 'y': 5.0, 'ammo42': 50.0, 'ammo17': 0.0},
        }} for i in range(3)]
    P = ph.calibrate([dps()])
    assert P['gun']['英雄'] == '42mm'


def test_unit_velocity_direction_red():
    # 红: 推进=+x, 回防=-x
    assert ph._unit_velocity('推进', (5, 5), '英雄', '红', None) == (1.0, 0.0)
    assert ph._unit_velocity('回防', (5, 5), '英雄', '红', None) == (-1.0, 0.0)
    # 蓝镜像
    assert ph._unit_velocity('推进', (5, 5), '英雄', '蓝', None) == (-1.0, 0.0)


# ---- Plan B: 意图化目标选择 + RM 量纲几何 ----

def test_target_intent_hits_chosen_building_not_closer_unit():
    # 红步兵3 意图打蓝前哨站; 虽蓝步兵3(15,6)更近, 仍打前哨站(弹道意图-建筑优先)
    rb = _robots(_r(1, '步兵3', '红', 14.0, 7.0),
                 _r(2, '步兵3', '蓝', 15.0, 6.0),
                 _r(99, '前哨站', '蓝', 17.12, 11.32))
    out = ph.step_robots(rb, '红', {'步兵3': _act_t('坚守', True, '敌方前哨站')}, dt=3.0)
    assert out[2]['hp'] == 100.0         # 更近单位未被转移目标(不'顺手'打单位)
    assert out[99]['hp'] < 100.0         # 意图建筑被命中

def test_mid_field_cannot_hit_base_but_can_hit_outpost():
    # 17mm RANGE=10: 中场单位能打前哨(近)、打不到基地(远) -> 攻/守分化于基地血量
    rb = _robots(_r(1, '步兵3', '红', 14.0, 7.0),
                 _r(98, '前哨站', '蓝', 17.12, 11.32),
                 _r(99, '基地', '蓝', 25.73, 7.44))
    out = ph.step_robots(rb, '红', {'步兵3': _act_t('坚守', True, '敌方前哨站')}, dt=3.0)
    assert out[98]['hp'] < 100.0          # 前哨站在 10m 内 -> 被打
    assert out[99]['hp'] == 100.0         # 基地 11.74m > 10m -> 未被打

def test_aim_at_base_requires_advance():
    # 己方半场深处打基地: 不在射程; 单位必须推进才能击打基地(证明'进攻需前进'的量纲)
    rb = _robots(_r(1, '步兵3', '红', 3.0, 7.0), _r(99, '基地', '蓝', 25.73, 7.44))
    out = ph.step_robots(rb, '红', {'步兵3': _act_t('推进', True, '敌方基地')}, dt=3.0)
    assert out[99]['hp'] == 100.0  # 单步(推进2.1m)不足, 仍在射程外

def test_defense_fires_at_enemy_unit_not_building():
    # 防守(意图=敌方步兵): 只打缠斗单位, 不打建筑(即便建筑更近)
    rb = _robots(_r(1, '步兵3', '红', 5.0, 5.0),
                 _r(2, '步兵3', '蓝', 9.0, 5.0),
                 _r(99, '前哨站', '蓝', 20.0, 5.0))
    out = ph.step_robots(rb, '红', {'步兵3': _act_t('坚守', True, '敌方步兵')}, dt=3.0)
    assert out[2]['hp'] < 100.0    # 打步兵
    assert out[99]['hp'] == 100.0  # 不打建筑

def test_42mm_base_not_instant_kill():
    # 英雄 42mm 打基地: 量纲合理(掉血但不秒爆)
    rb = _robots(_r(1, '英雄', '红', 20.0, 7.44),
                 _r(99, '基地', '蓝', 25.73, 7.44, hp=3000.0))
    out = ph.step_robots(rb, '红', {'英雄': _act_t('推进', True, '敌方基地')}, dt=3.0)
    ratio = out[99]['hp'] / 3000.0
    assert 0.0 < ratio < 1.0
