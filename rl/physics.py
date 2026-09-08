# -*- coding: utf-8 -*-
"""B2 符号物理模拟: 把 replay-bandit 变成真 MDP 的确定性前向步。

为什么: env.step() 之前只是 `t+=1`(跳到下一真实决策点), 动作不改未来。这里用**规则化符号物理**
在 (state_t, 我方动作) 上推进 Δt, 使动作改变位置/冷却/掉血 -> 未来后果进入价值。

哪些是确定的(可学动作效应): 运动(每类速度×语义方向)、射击过热/冷却、对目标掉血、建筑被毁;
哪些是不可约随机(风格化): 敌方"开火打谁"按最近可见+半径, 只造伤害不学策略 —— 这正是离线 RL
天花板(只能对"我方动作"反事实, 不能预测"敌方换策略")。

量纲: Plan B 将常量集中到 RM_PARAMS(RMUC 2026 规则锚定,来源标记见 RM_PARAMS_PROVENANCE)。
RANGE/伤害/速度/热量全部按弹种/单位类取自 RM_PARAMS; `calibrate` 只作为**覆盖源**(P 传入),
不再做主源。目标选择按**射手意图目标类别**(act['target'], 见 derive_unit_intent): 攻击单位追敌方
建筑(前哨/基地), 防守单位只打缠斗目标, 不再"顺手拆敌方前哨站"。
纯 numpy, 单测离线可跑(无 torch)。
"""
import numpy as np
from schema import layered as sl

# ---- 弹种与典型物理量(可被 calibrate 覆盖, 主源是 RM_PARAMS) ----
GUN_BY_CLASS = {'英雄': '42mm', '步兵3': '17mm', '步兵4': '17mm', '哨兵': '17mm',
                '工程': None, '空中': None}

# ---- RMUC 2026 规则锚定量纲(集中常量; 来源标记见 RM_PARAMS_PROVENANCE) ----
RM_PARAMS = {
    'field': {'width': 28.0, 'height': 15.0},
    # 目标固定坐标(米; 红主场低x). 用于"打基地/前哨站"判定; 覆盖回放里的 (0,0) 占位.
    'obj': {('蓝', '前哨站'): (17.12, 11.32), ('蓝', '基地'): (25.73, 7.44),
            ('红', '前哨站'): (10.87, 3.58), ('红', '基地'): (2.46, 7.44)},
    'gun': dict(GUN_BY_CLASS),
    'range': {'17mm': 10.0, '42mm': 12.0},      # 有效射程米: 中场够到前哨(≈5m)、须推进才够到基地(≈11.7m)
    'dmg': {'17mm': 30.0, '42mm': 70.0},        # 对单位单发伤害
    'dmg_bldg': {'17mm': 3.0, '42mm': 90.0},    # 对建筑单发伤害(英雄拆建筑主力)
    'rate': {'17mm': 10.0, '42mm': 2.5},        # 射速发/s
    'hit': {'17mm': 0.25, '42mm': 1.0},         # 每发命中率
    'heat_per': {'17mm': 10.0, '42mm': 100.0},  # 每发升温
    'heat_max': {'17mm': 250.0, '42mm': 250.0}, # 过热阈值(超过节流)
    'speed': {'哨兵': 0.10, '工程': 0.35, '英雄': 0.40, '步兵3': 0.70,
              '步兵4': 0.70, '空中': 2.50, None: 0.05},   # 有效机动(刻意不快, 见 spec §2.6)
    'hp': {'基地': 3000.0, '前哨站': 750.0},
    'cool': 4.0,   # 每秒天然回凉(heat 每秒下调)
    'enemy_range': 12.0,  # 风格化敌火接战半径
}

RM_PARAMS_PROVENANCE = {
    'field/obj': 'rm-battlescope field.py (P)',
    'gun/rate/heat_per': '规则索引+弹道 (P/R)',
    'range': '几何核算+命中窗口 (D: 需真实规则核对)',
    'dmg/dmg_bldg/hp': '规则伤害量纲默认 (D)',
    'speed': '实测有效机动+上浮 (D)',
    'hit': '维持既有 (P)',
    'cool': '由热耗+增益冷却倍率推出 (D)',
    'enemy_range': '接战风格化 (D)',
}

# 兼容下游引用(测试/回放脚本读取这些名字; 值从 RM_PARAMS 派生)
SPEED = dict(RM_PARAMS['speed'])
DMG = dict(RM_PARAMS['dmg'])
FIRE_RATE = dict(RM_PARAMS['rate'])
HIT_PROB = dict(RM_PARAMS['hit'])
HEAT_MAX = dict(RM_PARAMS['heat_max'])
HEAT_PER_SHOT = dict(RM_PARAMS['heat_per'])
RANGE = RM_PARAMS['range']['17mm']
ENEMY_RANGE = RM_PARAMS['enemy_range']

_BUILDING_TYPES = ('基地', '前哨站')


def _target_building_type(target):
    """意图 target 字符串 -> 若指建筑则返建筑类型('基地'/'前哨站'), 否则 None。"""
    if not target:
        return None
    if '基地' in target:
        return '基地'
    if '前哨站' in target:
        return '前哨站'
    return None


def _target_unit_types(target):
    """意图 target 字符串 -> 匹配的单位类型集合(空=任何敌方单位)。"""
    if not target:
        return None
    m = {'敌方英雄': {'英雄'}, '敌方哨兵': {'哨兵'}, '敌方工程': {'工程'},
         '敌方步兵3': {'步兵3'}, '敌方步兵4': {'步兵4'},
         '敌方步兵': {'步兵3', '步兵4'}, '敌方空中': {'空中'}}
    return m.get(target)


def _select_target(shooter, new, team, target_cls):
    """按射手意图选目标 rid。返回 rid 或 None。
    建筑意图 -> 最近存活敌方建筑(前哨/基地; 前哨先毁则自然转向基地);
    单位意图 -> 匹配类型>任意敌方单位; '无'/缺省 -> 最近敌方任意(单位或建筑)。"""
    def dist(rid):
        rr = new[rid]
        dx, dy = rr['x'] - shooter['x'], rr['y'] - shooter['y']
        return (dx * dx + dy * dy) ** 0.5

    def nearest(id_candidates):
        best, bd = None, None
        for rid in id_candidates:
            d = dist(rid)
            if bd is None or d < bd:
                bd, best = d, rid
        return best

    if target_cls is None:  # 缺省(旧测试/无意图): 最近敌方任意(单位或建筑)
        any_cand = [rid for rid, rr in new.items()
                    if rid != shooter.get('id') and rr.get('team') != team
                    and rr.get('hp', 1.0) > 0.0]
        return nearest(any_cand)
    if _target_building_type(target_cls):  # 建筑意图(攻击): 最近存活敌方建筑(前哨先毁则自然转向基地)
        b_cand = [rid for rid, rr in new.items()
                  if rid != shooter.get('id') and rr.get('team') != team
                  and rr.get('type') in _BUILDING_TYPES and rr.get('hp', 1.0) > 0.0]
        if b_cand:
            return nearest(b_cand)
        u_cand = [rid for rid, rr in new.items()
                  if rid != shooter.get('id') and rr.get('team') != team
                  and rr.get('type') not in _BUILDING_TYPES and rr.get('hp', 1.0) > 0.0]
        return nearest(u_cand) if u_cand else None
    # 单位/无意图: 只打敌方单位(绝不打建筑), 保留缠斗
    u_cand = [rid for rid, rr in new.items()
              if rid != shooter.get('id') and rr.get('team') != team
              and rr.get('type') not in _BUILDING_TYPES and rr.get('hp', 1.0) > 0.0]
    utypes = _target_unit_types(target_cls)
    if utypes:
        matched = [rid for rid in u_cand if new[rid]['type'] in utypes]
        if matched:
            return nearest(matched)
    return nearest(u_cand) if u_cand else None


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


def _shots(g, heat0, rate_v, dt, heat_max_v, heat_ps_v, cool_dt):
    """本帧实际开火发数(受射速与过热预算约束) + 更新后热度。
    过热按"射击把 heat 推向 heat_max 以下"逐帧计算, 使持续开火但被热节流(而非一次性爆发)。"""
    if g is None or heat_max_v <= 0.0 or heat_ps_v <= 0.0:
        return 0.0, max(0.0, heat0)
    budget = max(0.0, heat_max_v - heat0)
    by_heat = budget / heat_ps_v
    by_rate = rate_v * dt
    shots = max(0.0, min(by_rate, by_heat))
    heat_new = max(0.0, heat0 + shots * heat_ps_v - cool_dt)
    return shots, heat_new


def step_robots(robots, team, actions, dt, P=None, enemy_fire=True):
    """在 (robots_t, 我方动作) 上推进 dt。robots: {rid: 物理状态dict}。
    actions: {unit_class: {'move': move, 'fire': bool, 'target': str}} —— 我方每类动作(target 可选)。
    enemy_fire: 是否施加风格化敌火(敌方打最新可见我方)。多侧联合推演(rollout)时传 False,
    避免双方各被对方 step 重复开火(return-fire double-count); 单侧调用(PhysicsEnv)保持 True。
    返回新的 robots dict(单位 x/y/hp/ammo/heat 改变, 建筑 hp 改变)。"""
    P = P or {}
    PM = RM_PARAMS
    def spg(cls):
        d = P.get('speed', PM['speed'])
        return d.get(cls, PM['speed'].get(cls, 0.05))
    def dmg_u(g):
        d = P.get('dmg', PM['dmg'])
        return d.get(g, 10.0) if g else 0.0
    def dmg_b(g):
        d = P.get('dmg_bldg', PM['dmg_bldg'])
        return d.get(g, 10.0) if g else 0.0
    def rate(g):
        d = P.get('rate', PM['rate'])
        return d.get(g, 1.0) if g else 0.0
    def hit(g):
        d = P.get('hit', PM['hit'])
        return d.get(g, 0.2) if g else 0.0
    def heat_max(g):
        d = P.get('heat_max', PM['heat_max'])
        return d.get(g, 100.0) if g else 0.0
    def heat_ps(g):
        d = P.get('heat_ps', PM['heat_per'])
        return d.get(g, 5.0) if g else 0.0
    def rng(g):
        d = P.get('range', PM['range'])
        return d.get(g, 10.0) if isinstance(d, dict) else (d if g else 0.0)
    erange = P.get('enemy_range', PM['enemy_range'])
    cool = P.get('cool', PM['cool']) * dt

    # 全量复制, 以便我方打敌、敌火都作用在统一副本上
    new = {}
    for rid, r in robots.items():
        s = dict(r)
        if s.get('type') in _BUILDING_TYPES:
            obj = PM['obj'].get((s.get('team'), s.get('type')))
            if obj is not None and abs(s.get('x', 0.0)) < 1e-9 and abs(s.get('y', 0.0)) < 1e-9:
                # 回放里建筑常落在 (0,0) 占位 -> 用 RM 固定坐标替代(保证打基地/前哨站几何一致)
                s['x'], s['y'] = obj
            s['maxhp'] = s.get('maxhp', PM['hp'].get(s['type'], 100.0))
        new[rid] = s
    # 给每个单位补 id(供 _select_target 排除自己)
    for rid, s in new.items():
        s['id'] = rid

    # 1) 我方运动 + 开火
    for rid, s in new.items():
        if s.get('type') in _BUILDING_TYPES:
            continue
        if s.get('hp', 0.0) <= 0.0:
            s['alive'] = 0
            continue
        if s.get('team') != team:
            continue
        act = actions.get(s['type']) or {}
        mv = act.get('move', '无') if isinstance(act, dict) else '无'
        fire = bool(act.get('fire', False)) if isinstance(act, dict) else False
        target = act.get('target') if isinstance(act, dict) else None
        enemy_disp = _nearest_enemy_disp(s, new, team)
        ux, uy = _unit_velocity(mv, (s['x'], s['y']), s['type'], team, enemy_disp)
        d = spg(s['type']) * dt
        s['x'] = s['x'] + ux * d
        s['y'] = s['y'] + uy * d
        s['yaw'] = s.get('yaw', 0.0)
        g = _caliber(s['type'])
        if g and fire:
            shots, heat_new = _shots(g, s.get('heat', 0.0), rate(g), dt, heat_max(g), heat_ps(g), cool)
            if shots > 0:
                if g == '17mm':
                    s['ammo17'] = s.get('ammo17', 0) + shots
                else:
                    s['ammo42'] = s.get('ammo42', 0) + shots
                s['dfire'] = s.get('dfire', 0) + shots
                _apply_damage(s, new, g, shots, team, target, dmg_u, dmg_b, hit, rng)
            s['heat'] = heat_new
        else:
            s['heat'] = max(0.0, s.get('heat', 0.0) - cool)

    # 2) 风格化敌火(敌方打最新可见我方)
    if enemy_fire:
        _enemy_fire(new, team, dmg_u, hit, rate, erange, dt)
    return new


def _nearest_enemy_disp(r, robots, team):
    """从单位 r 到其最近敌方(基地除外)的 (dx, dy)。"""
    best, bd = None, None
    for rr in robots.values():
        if rr.get('team') == team or rr.get('type') in _BUILDING_TYPES:
            continue
        if rr.get('hp', 1.0) <= 0.0:
            continue
        dx, dy = rr['x'] - r['x'], rr['y'] - r['y']
        d = (dx * dx + dy * dy) ** 0.5
        if bd is None or d < bd:
            bd, best = d, (dx, dy)
    return best


def _apply_damage(shooter, new, g, shots, team, target_cls, dmg_u, dmg_b, hit, rng):
    """按射手意图目标类别对目标造成 dmg*shots, 若命中且射程内。写回 new(含建筑)。"""
    target = _select_target(shooter, new, team, target_cls)
    if target is None:
        return
    e = new[target]
    dx, dy = e['x'] - shooter['x'], e['y'] - shooter['y']
    d = (dx * dx + dy * dy) ** 0.5
    if d > rng(g):
        return
    dmg = dmg_b(g) if e['type'] in _BUILDING_TYPES else dmg_u(g)
    e['hp'] = max(0.0, e.get('hp', 0.0) - dmg * shots * hit(g))
    if e['type'] in _BUILDING_TYPES or e.get('hp', 1.0) <= 0.0:
        e['alive'] = 0


def _enemy_fire(new, team, dmg_u, hit, rate, erange, dt):
    """风格化敌火: 敌方单位向半径内最近我方开火, 造成伤害不学策略。"""
    for r in new.values():
        if r.get('team') == team or r.get('hp', 1.0) <= 0.0:
            continue
        if r.get('type') in _BUILDING_TYPES:
            continue
        g = _caliber(r['type'])
        if not g:
            continue
        tgt, td = None, None
        for tr in new.values():
            if tr.get('team') != team or tr.get('hp', 1.0) <= 0.0:
                continue
            if tr.get('type') in _BUILDING_TYPES:
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
        tgt['hp'] = max(0.0, tgt.get('hp', 0.0) - dmg_u(g) * shots * hit(g))


def calibrate(decision_points_list):
    """从多局决策点标定: 每类速度(中位) + 每类弹种(看 ammo42 是否出现)。
    返回 {'speed': {...}, ...} 字典, 供 P 传入。**覆盖源**(主源 RM_PARAMS), 纯启发/标定, 不依赖外部库。"""
    from collections import defaultdict
    disp = defaultdict(list)
    ammo42 = defaultdict(set)
    for dps in decision_points_list:
        prev = {}
        for dp in dps:
            for rid, r in dp['robots'].items():
                typ = r.get('type')
                if typ in _BUILDING_TYPES or typ not in SLOT_SPEED:
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
