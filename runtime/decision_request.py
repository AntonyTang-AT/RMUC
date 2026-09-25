# -*- coding: utf-8 -*-
"""薄 DecisionRequest 适配（考核原图统一接口名）。

不改变热路径默认返回类型；供演示/对接实车主执行器。
"""
from __future__ import annotations

from tree import global_mode as gm
from labels import router as rt


def to_decision_request(decision):
    """layered(+sidecars) → DecisionRequest 形态 dict。"""
    d = dict(decision) or {}
    meta = dict(d.get('decision_meta') or {})
    sentry = meta.get('sentry_state')
    if sentry not in gm.SENTRY_STATES:
        # 无 meta 时尽量从空壳补；调用方应已写入
        sentry = 'cruise'
    strategic = meta.get('strategic_intent') or gm.sentry_to_strategic(sentry)
    return {
        'interface': 'DecisionRequest',
        'sentry_state': sentry,
        'strategic_intent': strategic,
        'global_mode': d.get('global_mode'),
        'units': dict(d.get('units') or {}),
        't': d.get('t'),
        'decision_meta': meta or None,
        'opponent_intent': d.get('opponent_intent'),
        'semantic_situation': d.get('semantic_situation'),
        'global_analysis': d.get('global_analysis'),
        'temporal_forecast': d.get('temporal_forecast'),
    }


def from_decision_request(req):
    """DecisionRequest → 可 validate 的 layered（剥旁路）。"""
    req = req or {}
    d = {
        't': req.get('t', 0.0),
        'global_mode': req.get('global_mode', '均衡'),
        'units': dict(req.get('units') or {}),
    }
    return rt.strip_sidecar_for_validate(d)
