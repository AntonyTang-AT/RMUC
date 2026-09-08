# -*- coding: utf-8 -*-
"""B2 符号物理模拟: 把 replay-bandit 变成真 MDP 的确定性前向步。

为什么: env.step() 之前只是 `t+=1`(跳到下一真实决策点), 动作不改未来。这里用**规则化符号物理**
在 (state_t, 我方动作) 上推进 Δt, 使动作改变位置/冷却/掉血 -> 未来后果进入价值。

哪些是确定的(可学动作效应): 运动(每类速度×语义方向)、射击过热/冷却、对目标掉血、建筑被毁;
哪些是不可约随机(风格化): 敌方"开火打谁"按最近可见+半径, 只造伤害不学策略 —— 这正是离线 RL
天花板(只能对"我方动作"反事实, 不能预测"敌方换策略")。

参数从回放标定(见 calibrate): 速度=中位位移/Δt, 弹种=哪个类 ammo42 递增,
伤害吞吐=队伍 hp 掉率 / 该队综合火力。纯 numpy, 单测离线可跑(无 torch)。
"""
import numpy as np
from schema import layered as sl

# ---- 弹种与典型物理量(可被 calibrate 覆盖) ----
GUN_BY_CLASS = {'英雄': '42mm', '步兵3': '17mm', '步兵4': '17mm', '哨兵': '17mm',
                '工程': None, '空中': None}
# 单发伤害(区域 sim 自洽量纲)
DMG = {'17mm': 30.0, '42mm': 120.0}
# 射速(发/s)
FIRE_RATE = {'17mm': 3.0, '42mm': 1.5}
# 命中率(每发)
HIT_PROB = {'17mm': 0.25, '42mm': 1.0}
# 有效射程(数据单位/归一化 x 的距离)
RANGE = 12.0
# 过热阈值(超过则节流) 与 每发升温
HEAT_MAX = {'17mm': 100.0, '42mm': 100.0}
HEAT_PER_SHOT = {'17mm': 4.0, '42mm': 10.0}
# 敌火接战半径
ENEMY_RANGE = 12.0
# 移动速度(数据单位/s; 回放标定后的中位值, calibrate 可覆盖)
SPEED = {'英雄': 0.054, '工程': 0.069, '步兵3': 0.184, '步兵4': 0.159, '空中': 0.069, '哨兵': 0.069}


def _enemy_dir(team):
    return 1.0 if team == '红' else -1.0   # 红主场低x: 推敌=+x; 蓝镜像


def _unit_velocity(move, pos, clss, team, enemy_disp):
    """move -> 归一化速度方向(与速度幅值无关)。返回 (ux, uy)。"""
    ex = _enemy_dir(team)
    if move == '推进':
        return (ex, 0.0)
    if move == '回防':
        return (-ex, 0.0)
    if move == '坚守':
        return (0.0, 0.0)
    if move == '无':
        return (0.0, 0.0)
    if move == '游击':
        # 向最近敌人方向
        if enemy_disp is not None:
            dx, dy = enemy_disp
            n = (dx * dx + dy * dy) ** 0.5 + 1e-9
            return (dx / n, dy / n)
        return (ex, 0.0)
    if move == '绕行':
        # 侧翼: 垂直(±y)稍向前
        return (ex * 0.4, 0.9)
    return (0.0, 0.0)


def _caliber(cls):
    return GUN_BY_CLASS.get(cls)


def step_robots(robots, team, actions, dt, P=None, enemy_fire=True):
    """在 (robots_t, 我方动作) 上推进 dt。robots: {rid: 物理状态dict}。
    actions: {unit_class: {'move': move, 'fire': bool}} —— 我方每类动作。
    enemy_fire: 是否施加风格化敌火(敌方打最新可见我方)。多侧联合推演(rollout)时传 False,
    避免双方各被对方 step 重复开火(return-fire double-count); 单侧调用(PhysicsEnv)保持 True。
    返回新的 robots dict(单位 x/y/hp/ammo/heat 改变, 建筑 hp 改变)。"""
    P = P or {}
    speed = lambda c: (P.get('speed') or SPEED).get(c, 0.05)
    dmg = lambda g: (P.get('dmg') or DMG).get(g, 10.0) if g else 0.0
    rate = lambda g: (P.get('rate') or FIRE_RATE).get(g, 1.0) if g else 0.0
    hit = lambda g: (P.get('hit') or HIT_PROB).get(g, 0.2) if g else 0.0
    heat_max = lambda g: (P.get('heat_max') or HEAT_MAX).get(g, 100.0) if g else 0.0
    heat_ps = lambda g: (P.get('heat_ps') or HEAT_PER_SHOT).get(g, 5.0) if g else 0.0
    rng = P.get('range', RANGE)
    erange = P.get('enemy_range', ENEMY_RANGE)

    # 全量复制, 以便我方打敌、敌火都作用在统一副本上
    new = {rid: dict(r) for rid, r in robots.items()}

    # 1) 我方运动 + 开火
    for rid, s in new.items():
        if s.get('type') in ('基地', '前哨站'):
            continue
        if s.get('hp', 0.0) <= 0.0:
            s['alive'] = 0
            continue
        if s.get('team') != team:
            continue
        act = actions.get(s['type']) or {}
        mv = act.get('move', '无') if isinstance(act, dict) else '无'
        fire = bool(act.get('fire', False)) if isinstance(act, dict) else False
        enemy_disp = _nearest_enemy_disp(s, new, team)
        ux, uy = _unit_velocity(mv, (s['x'], s['y']), s['type'], team, enemy_disp)
        d = speed(s['type']) * dt
        s['x'] = s['x'] + ux * d
        s['y'] = s['y'] + uy * d
        s['yaw'] = s.get('yaw', 0.0)
        g = _caliber(s['type'])
        if g and fire and s.get('heat', 0.0) < heat_max(g):
            shots = rate(g) * dt
            if g == '17mm':
                s['ammo17'] = s.get('ammo17', 0) + shots
            else:
                s['ammo42'] = s.get('ammo42', 0) + shots
            s['heat'] = s.get('heat', 0) + heat_ps(g) * shots
            s['dfire'] = s.get('dfire', 0) + shots
            _apply_damage(rid, s, new, dmg(g), hit(g), rng, team, shots)
        s['heat'] = max(0.0, s.get('heat', 0.0) - 4.0 * dt)

    # 2) 风格化敌火(敌方打最新可见我方)
    if enemy_fire:
        _enemy_fire(new, team, dmg, hit, rate, erange, dt)
    return new


def _nearest_enemy_disp(r, robots, team):
    """从单位 r 到其最近敌方(基地除外)的 (dx, dy)。"""
    best, bd = None, None
    for rr in robots.values():
        if rr.get('team') == team or rr.get('type') in ('基地', '前哨站'):
            continue
        if rr.get('hp', 1.0) <= 0.0:
            continue
        dx, dy = rr['x'] - r['x'], rr['y'] - r['y']
        d = (dx * dx + dy * dy) ** 0.5
        if bd is None or d < bd:
            bd, best = d, (dx, dy)
    return best


def _apply_damage(shooter_id, shooter, new, dmg, hit, rng, team, shots):
    """射击者对最近敌方(单位/建筑)造成 dmg*shots, 若命中且射程内。写回 new。"""
    target, td = None, None
    for rid, rr in new.items():
        if rid == shooter_id or rr.get('team') == team:
            continue
        dx, dy = rr['x'] - shooter['x'], rr['y'] - shooter['y']
        d = (dx * dx + dy * dy) ** 0.5
        if td is None or d < td:
            td, target = d, rid
    if target is None or td > rng:
        return
    e = new[target]
    e['hp'] = max(0.0, e.get('hp', 0.0) - dmg * shots * hit)
    if e['type'] in ('基地', '前哨站') or e.get('hp', 1.0) <= 0.0:
        e['alive'] = 0


def _enemy_fire(new, team, dmg, hit, rate, erange, dt):
    """风格化敌火: 敌方单位向半径内最近我方开火, 造成伤害不学策略。"""
    for r in new.values():
        if r.get('team') == team or r.get('hp', 1.0) <= 0.0:
            continue
        if r.get('type') in ('基地', '前哨站'):
            continue
        g = _caliber(r['type'])
        if not g:
            continue
        tgt, td = None, None
        for tr in new.values():
            if tr.get('team') != team or tr.get('hp', 1.0) <= 0.0:
                continue
            if tr.get('type') in ('基地', '前哨站'):
                continue
            dx, dy = tr['x'] - r['x'], tr['y'] - r['y']
            d = (dx * dx + dy * dy) ** 0.5
            if td is None or d < td:
                td, tgt = d, tr
        if tgt is None or td > erange:
            continue
        shots = rate(g) * dt
        if g == '17mm':
            r['ammo17'] = r.get('ammo17', 0) + shots
        else:
            r['ammo42'] = r.get('ammo42', 0) + shots
        r['heat'] = r.get('heat', 0) + 6.0 * shots
        tgt['hp'] = max(0.0, tgt.get('hp', 0.0) - dmg(g) * shots * hit(g))


def calibrate(decision_points_list):
    """从多局决策点标定: 每类速度(中位) + 每类弹种(看 ammo42 是否出现)。
    返回 {'speed': {...}, ...} 字典, 供 P 传入。纯启发/标定, 不依赖外部库。"""
    from collections import defaultdict
    disp = defaultdict(list)
    ammo42 = defaultdict(set)
    for dps in decision_points_list:
        prev = {}
        for dp in dps:
            for rid, r in dp['robots'].items():
                typ = r.get('type')
                if typ in ('基地', '前哨站') or typ not in SLOT_SPEED:
                    continue
                a42 = r.get('ammo42', 0) or 0
                if a42 > 5:
                    ammo42[typ].add('42mm')
                if rid in prev:
                    prt, pr = prev[rid]
                    dt = dp['t'] - prt
                    dxx = r['x'] - pr['x']
                    dyy = r['y'] - pr['y']
                    v = (dxx * dxx + dyy * dyy) ** 0.5 / max(dt, 1e-3)
                    if 0.0 < v < 3.0:
                        disp[typ].append(v)
                prev[rid] = (dp['t'], r)
    sp = {}
    for typ, vs in disp.items():
        vs = sorted(vs)
        sp[typ] = vs[len(vs) // 2] if vs else SPEED.get(typ, 0.05)
    gun = dict(GUN_BY_CLASS)
    for typ, s in ammo42.items():
        if '42mm' in s:
            gun[typ] = '42mm'
    return {'speed': sp, 'gun': gun}


SLOT_SPEED = set(SPEED)  # 处理"基地/前哨站"外常见类型
