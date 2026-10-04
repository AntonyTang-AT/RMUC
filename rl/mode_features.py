# -*- coding: utf-8 -*-
"""Tier1 global_mode 蒸馏特征：40 维 M2Env.obs('哨兵') + 三分类标签。
模块级无 torch。"""
from __future__ import annotations

import numpy as np
from schema import layered as sl
from rl.env import M2Env
from worldstate import worldstate as ws
from tree import global_mode as gm
from labels import router as rt

MODES = sl.GLOBAL_MODES  # ('进攻', '均衡', '防守')
OBS_DIM = 40


def mode_to_id(mode):
    m = mode if mode in MODES else '均衡'
    return MODES.index(m)


def id_to_mode(i):
    i = int(i)
    if i < 0 or i >= len(MODES):
        return '均衡'
    return MODES[i]


def pack_obs(dp, bounds, team='红'):
    """单决策点 -> float32[40]（与 IPPO 哨兵观测同构）。"""
    env = M2Env([dp], bounds, team=team)
    env.reset()
    vec = env.obs('哨兵')
    out = np.asarray(vec, dtype=np.float32).reshape(-1)
    if out.shape[0] != OBS_DIM:
        raise ValueError(f'obs dim {out.shape[0]} != {OBS_DIM}')
    return out


def refill_units_for_mode(decision, w, mode, team='红', opp_intent=None):
    """用给定 global_mode 规则重填六类单位（Tier0 填洞）；可选敌方意图偏置。"""
    from labels import intent_policy as ip
    d = dict(decision) if decision else sl.build_empty_decision((w or {}).get('t', 0.0))
    d['global_mode'] = mode if mode in MODES else '均衡'
    units = dict(d.get('units') or sl.build_empty_decision(d.get('t', 0.0))['units'])
    for cls in sl.UNIT_CLASSES:
        unit = next((r for r in (w.get('robots') or {}).values()
                     if r.get('team') == team and r.get('type') == cls), None)
        if unit is None:
            continue
        raw = gm.derive_unit_intent(unit, w, d['global_mode'])
        units[cls] = ip.apply_intent_bias(
            raw, opp_intent=opp_intent, unit_class=cls, global_mode=d['global_mode'])
    d['units'] = units
    return d


def sample_row(dp, bounds, teacher_mode, mode_history=None, conflict_rate=0.0,
               team='红', game_id=None):
    """一条蒸馏样本 dict（json 可序列化）。"""
    w = ws.build_worldstate(dp, bounds)
    route = rt.route_tier(w, mode_history=mode_history, our_team=team,
                          conflict_rate=conflict_rate)
    obs = pack_obs(dp, bounds, team=team)
    mode = teacher_mode if teacher_mode in MODES else '均衡'
    return {
        'obs': [float(x) for x in obs.tolist()],
        'mode': mode,
        'y': mode_to_id(mode),
        'need_llm': bool(route.get('need_llm')),
        'tier': route.get('tier'),
        'game_id': game_id,
        't': float(dp.get('t') or 0.0),
        'rule_mode': gm.derive_global_mode(w),
    }


def load_jsonl(path):
    import json
    rows = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def rows_to_xy(rows, tier1_only=False):
    """rows -> (X[n,40], y[n], mask_tier1[n])。"""
    if tier1_only:
        rows = [r for r in rows if not r.get('need_llm')]
    if not rows:
        return (np.zeros((0, OBS_DIM), dtype=np.float32),
                np.zeros((0,), dtype=np.int64),
                np.zeros((0,), dtype=np.bool_))
    X = np.asarray([r['obs'] for r in rows], dtype=np.float32)
    y = np.asarray([mode_to_id(r.get('mode', '均衡')) for r in rows], dtype=np.int64)
    m = np.asarray([not bool(r.get('need_llm')) for r in rows], dtype=np.bool_)
    return X, y, m
