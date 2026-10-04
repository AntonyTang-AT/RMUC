# -*- coding: utf-8 -*-
"""单位层蒸馏特征（脚手架）：40 维 obs + mode one-hot + 意图 top one-hot。

模块级无 torch。本阶段默认只蒸馏哨兵 move/fire。
"""
from __future__ import annotations

import numpy as np
from schema import layered as sl
from labels.opp_intent import INTENTS
from labels import intent_policy as ip
from rl import mode_features as mf

UNIT_FOCUS = '哨兵'
MOVES = sl.MOVES
N_MOVE = len(MOVES)
MODE_DIM = len(sl.GLOBAL_MODES)
INTENT_DIM = len(INTENTS)
# 40 + 3 + 6
FEAT_DIM = mf.OBS_DIM + MODE_DIM + INTENT_DIM


def mode_onehot(mode):
    v = np.zeros(MODE_DIM, dtype=np.float32)
    m = mode if mode in sl.GLOBAL_MODES else '均衡'
    v[sl.GLOBAL_MODES.index(m)] = 1.0
    return v


def intent_onehot(opp_intent):
    v = np.zeros(INTENT_DIM, dtype=np.float32)
    top = ip.resolve_top_intent(opp_intent)
    if top in INTENTS:
        v[INTENTS.index(top)] = 1.0
    return v


def pack_unit_feat(dp, bounds, mode='均衡', opp_intent=None, team='红'):
    """float32[FEAT_DIM]。"""
    obs = mf.pack_obs(dp, bounds, team=team)
    return np.concatenate([obs, mode_onehot(mode), intent_onehot(opp_intent)]).astype(np.float32)


def move_to_id(move):
    m = move if move in MOVES else '无'
    return MOVES.index(m)


def id_to_move(i):
    i = int(i)
    if i < 0 or i >= N_MOVE:
        return '无'
    return MOVES[i]


def sample_sentry_row(dp, bounds, decision, opp_intent=None, team='红', game_id=None):
    """一条哨兵蒸馏样本。"""
    mode = (decision or {}).get('global_mode', '均衡')
    unit = ((decision or {}).get('units') or {}).get(UNIT_FOCUS) or {}
    move = unit.get('move', '无')
    fire = 1 if unit.get('fire') else 0
    feat = pack_unit_feat(dp, bounds, mode=mode, opp_intent=opp_intent, team=team)
    return {
        'feat': [float(x) for x in feat.tolist()],
        'move': move if move in MOVES else '无',
        'y_move': move_to_id(move),
        'fire': int(fire),
        'mode': mode if mode in sl.GLOBAL_MODES else '均衡',
        'unit_class': UNIT_FOCUS,
        'game_id': game_id,
        't': float(dp.get('t') or 0.0),
    }


def rows_to_xy(rows):
    X = np.asarray([r['feat'] for r in rows], dtype=np.float32)
    y_move = np.asarray([r['y_move'] for r in rows], dtype=np.int64)
    y_fire = np.asarray([r['fire'] for r in rows], dtype=np.int64)
    return X, y_move, y_fire
