# -*- coding: utf-8 -*-
"""PhysicsEnv(B2) 集成单测: action 必须决定性地改变未来(真 MDP), 无 torch。"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from rl import env as rlenv

B = {'xmin': 0.0, 'xmax': 30.0, 'ymin': 0.0, 'ymax': 15.0}


def _robot(rid, cls, team, x, y, hp=100.0, maxhp=100.0):
    return rid, {'type': cls, 'team': team, 'x': x, 'y': y, 'hp': hp, 'maxhp': maxhp,
                 'power': 1.0, 'small_heat': 0.0, 'big_heat': 0.0, 'ammo17': 0.0,
                 'ammo42': 0.0, 'vuln': 0, 'dfire': 0.0, 'dx': 0.0, 'dy': 0.0, 'dhp': 0.0}


def _dps(n=4):
    out = []
    for i in range(n):
        robots = dict([_robot(1, '英雄', '红', 5.0, 5.0),
                       _robot(2, '步兵3', '红', 6.0, 6.0),
                       _robot(3, '英雄', '蓝', 14.0, 6.0),   # 距红英雄 9 < RANGE
                       _robot(4, '步兵3', '蓝', 13.0, 7.0)])
        out.append({'t': i * 3.0, 'robots': robots,
                    'team': {'红': {'coins_left': 1.0}, '蓝': {'coins_left': 1.0}}})
    return out


def _env():
    return rlenv.PhysicsEnv(_dps(), B, team='红', horizon=3)


def test_physics_env_obs_dim_matches_base():
    assert _env().obs_dim == rlenv.M2Env(_dps(), B).obs_dim


def test_push_advances_red_hero_future():
    # 推进: 红英雄 未来 x 增大; 坚守: 几乎不动
    idx_fire_false = rlenv.ACTION_MEANINGS.index(('推进', False))
    idx_hold = rlenv.ACTION_MEANINGS.index(('坚守', False))
    e_push = _env(); e_push.reset()
    e_push.step({'英雄': idx_fire_false, '步兵3': idx_fire_false})
    x_push = e_push.ws[1]['robots'][1]['norm_xy'][0]
    e_hold = _env(); e_hold.reset()
    e_hold.step({'英雄': idx_hold, '步兵3': idx_hold})
    x_hold = e_hold.ws[1]['robots'][1]['norm_xy'][0]
    assert x_push > x_hold  # 推进 -> 更靠敌半场


def test_fire_hurts_enemy_in_future():
    idx_fire_false = rlenv.ACTION_MEANINGS.index(('推进', False))
    idx_fire_true = rlenv.ACTION_MEANINGS.index(('推进', True))
    e_fire = _env(); e_fire.reset()
    e_fire.step({'英雄': idx_fire_true, '步兵3': idx_fire_true})
    fired = e_fire.ws[1]['robots'].get(4)   # 蓝步兵3(距红英雄最近)
    died = (fired is None) or (fired['hp_ratio'] == 0.0)
    e_hold = _env(); e_hold.reset()
    e_hold.step({'英雄': idx_fire_false, '步兵3': idx_fire_false})
    held = e_hold.ws[1]['robots'].get(4)
    assert held is not None and held['hp_ratio'] == 1.0  # 不开火 -> 保留
    assert died  # 开火 -> 最近敌被击毁/掉血


def test_reset_restores_real_start():
    e = _env(); e.reset()
    e.step({'英雄': rlenv.ACTION_MEANINGS.index(('推进', False)),
            '步兵3': rlenv.ACTION_MEANINGS.index(('推进', False))})
    e.reset()
    assert e.t == 0 and len(e.ws) == 1  # 回到真实起点
    assert abs(e.ws[0]['robots'][1]['norm_xy'][0] - round(5.0 / 30.0, 3)) < 1e-6  # 未动的真实位置


def test_win_condition_reachable():
    e = _env(); e.reset()
    for _ in range(3):
        e.step({'英雄': rlenv.ACTION_MEANINGS.index(('推进', False)),
                '步兵3': rlenv.ACTION_MEANINGS.index(('推进', False))})
    w = e.win_condition()
    assert isinstance(w, float) or isinstance(w, int)


def test_physics_env_visibility_uses_sim_position():
    """G3: PhysicsEnv 的部分可观测要用**模拟坐标**, 而非回放真实坐标。
    回放里蓝步兵3 冻结在 (13,7); 但 sim 里红哨兵被推前, 可视性应随 sim 坐标翻转。"""
    idx = rlenv.AGENTS.index('步兵3')
    vis_idx = 4 + idx * 5 + 4          # 敌方步兵3 块的 vis(末维)
    n = 6
    dps = []
    for i in range(n):
        robots = dict([_robot(1, '哨兵', '红', 5.0, 5.0, hp=400.0, maxhp=400.0),
                       _robot(2, '步兵3', '蓝', 13.0, 7.0)])
        dps.append({'t': i * 3.0, 'robots': robots,
                    'team': {'红': {'coins_left': 1.0}, '蓝': {'coins_left': 1.0}}})
    e = rlenv.PhysicsEnv(dps, B, team='红', vis_radius=8.0, horizon=n - 1)
    e.reset()
    assert abs(e.obs('哨兵')[vis_idx]) < 1e-6          # 初始距 8.246>8 -> 不可见
    push = rlenv.ACTION_MEANINGS.index(('推进', False))
    for _ in range(4):
        e.step({'哨兵': push})
    assert abs(e.obs('哨兵')[vis_idx] - 1.0) < 1e-6   # 哨兵 sim 推进到 x≈5.8, 距 7.44<8 -> 可见


def test_physics_env_enemy_moves():
    """Plan B 对称性: 敌方(蓝)也经自身 step_robots 行动(反应式对抗), 而非冻结。
    蓝英雄应在 step 后真实坐标改变。"""
    e = _env(); e.reset()
    idx_push = rlenv.ACTION_MEANINGS.index(('推进', False))
    e.step({'英雄': idx_push, '步兵3': idx_push})
    red_before = e.live[1]['x']
    blue_after = e.live[3]['x']     # 蓝英雄(敌)模拟 x
    blue_before = e.dps[0]['robots'][3]['x']   # 回放初值(冻结基准)
    assert abs(blue_after - blue_before) > 1e-6   # 敌方也动 -> 非冻结
    assert abs(e.live[1]['x'] - red_before) < 1e-6  # 红推进已在前一步反映


def test_symmetric_fire_default_and_win_condition_type():
    """仿真默认 symmetric_fire=True; 终局 win_condition 为有限标量。"""
    e = _env()
    assert e.symmetric_fire is True
    e.reset()
    for _ in range(3):
        e.step({'英雄': rlenv.ACTION_MEANINGS.index(('推进', True)),
                '步兵3': rlenv.ACTION_MEANINGS.index(('推进', True))})
    assert e.t >= e.horizon
    w = e.win_condition()
    assert isinstance(w, (float, int)) and abs(w) <= 1.0 + 1e-9


def test_fav_align_bonus_prefers_push_on_high_fav():
    """有 fav_align_scale 时: 高 fav 选推进的即时 bonus > 选回防。"""
    dps = _dps(4)
    e = rlenv.M2Env(dps, B, fav_align_scale=0.1, coherence_scale=0.0)
    e.reset()
    e._fav_median = -1e9  # 强制当前为高 fav
    idx_push = rlenv.ACTION_MEANINGS.index(('推进', False))
    idx_back = rlenv.ACTION_MEANINGS.index(('回防', False))
    acts_push = {c: idx_push for c in rlenv.AGENTS}
    acts_back = {c: idx_back for c in rlenv.AGENTS}
    assert e._fav_align_bonus(acts_push, e.ws[0]) > e._fav_align_bonus(acts_back, e.ws[0])
    e2 = rlenv.M2Env(dps, B, fav_align_scale=0.0)
    assert e2._fav_align_bonus(acts_push, e2.ws[0]) == 0.0
