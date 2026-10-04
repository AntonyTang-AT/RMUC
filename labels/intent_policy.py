# -*- coding: utf-8 -*-
"""敌方意图 → 我方单位 move/target 轻量偏置（纯规则，无学习/无 torch）。

在 derive_unit_intent 之后调用；只改合法 schema 枚举，不引入新字段。
"""
from __future__ import annotations

from schema import layered as sl
from labels.opp_intent import INTENTS

# 意图 → 偏置动作（按 unit_class 可选覆盖）
_BIAS = {
    '拆基地': {
        'default': {'move': '回防', 'target': '敌方基地'},
        '哨兵': {'move': '回防', 'target': '敌方步兵'},
        '工程': {'move': '回防', 'target': '无'},
    },
    '压前哨': {
        'default': {'move': '回防', 'target': '敌方前哨站'},
        '哨兵': {'move': '回防', 'target': '敌方前哨站'},
    },
    '盯哨兵': {
        'default': {'move': '游击', 'target': '敌方哨兵'},
        '哨兵': {'move': '绕行', 'target': '敌方哨兵'},
        '步兵3': {'move': '游击', 'target': '敌方哨兵'},
        '步兵4': {'move': '游击', 'target': '敌方哨兵'},
    },
    '换血前排': {
        'default': {'move': '推进', 'target': '敌方英雄'},
        '哨兵': {'move': '游击', 'target': '敌方步兵'},
        '英雄': {'move': '推进', 'target': '敌方英雄'},
    },
    '回防收缩': {
        'default': {'move': '推进', 'target': '敌方步兵'},
        '哨兵': {'move': '推进', 'target': '敌方前哨站'},
    },
    '摸资源/中立': {
        'default': {'move': '游击', 'target': '能量机关'},
        '工程': {'move': '游击', 'target': '能量机关'},
        '哨兵': {'move': '游击', 'target': '能量机关'},
    },
}


def _clamp_intent(unit_intent):
    """保证输出仍在 schema 合法集合内。"""
    out = dict(unit_intent or {})
    if out.get('target') not in sl.TARGETS:
        out['target'] = '无'
    if out.get('move') not in sl.MOVES:
        out['move'] = '无'
    if out.get('risk') not in sl.RISKS:
        out['risk'] = '中'
    out['fire'] = bool(out.get('fire'))
    return out


def resolve_top_intent(opp_intent):
    """从旁路结构取 top；无则 None。"""
    if not opp_intent:
        return None
    top = opp_intent.get('top')
    if top in INTENTS:
        return top
    probs = opp_intent.get('probs') or {}
    if not probs:
        return None
    best = max(probs.items(), key=lambda kv: float(kv[1]))
    return best[0] if best[0] in INTENTS else None


def apply_intent_bias(unit_intent, opp_intent=None, unit_class=None, global_mode=None):
    """对单单位意图做偏置。无意图或未知意图时原样返回（经 clamp）。

    global_mode 仅作弱约束：防守模式下不把 move 强推成「推进」。
    """
    base = _clamp_intent(unit_intent)
    top = resolve_top_intent(opp_intent)
    if top is None or top not in _BIAS:
        return base
    table = _BIAS[top]
    patch = dict(table.get('default') or {})
    if unit_class and unit_class in table:
        patch.update(table[unit_class])
    out = dict(base)
    if 'move' in patch:
        move = patch['move']
        if global_mode == '防守' and move == '推进':
            move = '回防'
        out['move'] = move
    if 'target' in patch:
        out['target'] = patch['target']
    # 有明确敌方目标时默认开火；能量机关/无则保持原 fire
    tgt = out.get('target') or '无'
    if isinstance(tgt, str) and tgt.startswith('敌方'):
        out['fire'] = True
    return _clamp_intent(out)


def bias_units_dict(units, opp_intent=None, global_mode=None):
    """对 decision['units'] 整表偏置。"""
    units = units or {}
    out = {}
    for cls in sl.UNIT_CLASSES:
        u = units.get(cls)
        if u is None:
            continue
        out[cls] = apply_intent_bias(u, opp_intent=opp_intent,
                                     unit_class=cls, global_mode=global_mode)
    return out
