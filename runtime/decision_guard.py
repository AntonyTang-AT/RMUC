# -*- coding: utf-8 -*-
"""Tier2 输出的事实与安全审计。

语言模型输出的结构化枚举合法，并不代表它和当前可见状态一致。本模块只
拦截可以由当前事实直接证明不安全的情况；对预测分歧只做记录，避免把低
置信预测错误地提升为事实。
"""
from __future__ import annotations

from typing import Any, Mapping

from schema import layered as sl
from labels import router as rt


_TARGET_TO_TYPE = {
    '敌方哨兵': '哨兵',
    '敌方英雄': '英雄',
    '敌方步兵': '步兵',
    '敌方工程': '工程',
}


def _explicitly_destroyed(unit: Mapping[str, Any]) -> bool:
    if unit.get('life_state') == 'destroyed':
        return True
    hp = unit.get('hp_ratio')
    return (isinstance(hp, (int, float)) and not isinstance(hp, bool)
            and float(hp) <= 0.0)


def _is_inert_action(action: Any) -> bool:
    return (isinstance(action, Mapping) and action.get('target') == '无'
            and action.get('move') == '无' and action.get('fire') is False)


def _active_team_units(worldstate: Mapping[str, Any], team: str) -> list[Mapping[str, Any]]:
    out = []
    for unit in (worldstate.get('robots') or {}).values():
        if not isinstance(unit, Mapping) or unit.get('team') != team:
            continue
        # 缺失 HP 不证明阵亡；只有明确的 destroyed/<=0 才视为已毁。
        if _explicitly_destroyed(unit):
            continue
        out.append(unit)
    return out


def audit_tier2_decision(decision: Mapping[str, Any], worldstate: Mapping[str, Any],
                         forecast_context: Mapping[str, Any] | None = None,
                         *, our_team: str = '红',
                         forecast_confidence_threshold: float = 0.80) -> dict[str, Any]:
    """返回 Tier2 输出是否可接受，以及可审计的原因。

    ``forecast_context`` 的作用是记录高置信预测和决策是否一致；它不能单独
    让一个动作变成非法。当前事实和结构化规则的优先级更高。
    """
    base = rt.strip_sidecar_for_validate(decision)
    ok, errors = sl.validate(base)
    if not ok:
        return {'accepted': False, 'reasons': ['schema_invalid'], 'errors': list(errors),
                'forecast_disagreements': []}

    robots = list((worldstate.get('robots') or {}).values())
    active_own = _active_team_units(worldstate, our_team)
    active_types = {str(r.get('type')) for r in active_own}
    reasons: list[str] = []
    # 已毁单位必须保留结构化槽位，但只能是保守空动作；不得继续调度它们。
    for unit_class, action in (base.get('units') or {}).items():
        if unit_class not in active_types:
            matching = [r for r in robots if r.get('team') == our_team
                        and r.get('type') == unit_class]
            if (matching and all(_explicitly_destroyed(r) for r in matching)
                    and not _is_inert_action(action)):
                reasons.append(f'destroyed_unit_action:{unit_class}')
        if isinstance(action, Mapping) and bool(action.get('fire')) and action.get('target') == '无':
            reasons.append(f'fire_without_target:{unit_class}')

    # 仅当本方没有该兵种存活单位时，才认为该类动作确实与当前事实冲突。
    # 其余“目标当前不可见”的情况允许保留，因为目标可能来自雷达/预测上下文。
    accepted = not reasons
    forecast_disagreements: list[dict[str, Any]] = []
    if isinstance(forecast_context, Mapping):
        for item in list(forecast_context.get('predictions') or []):
            if not isinstance(item, Mapping) or int(item.get('horizon_seconds') or 0) not in (10, 30):
                continue
            head = item.get('global_mode') or {}
            top = head.get('top')
            conf = float(head.get('confidence') or 0.0)
            if top in sl.GLOBAL_MODES and conf >= float(forecast_confidence_threshold):
                selected = base.get('global_mode')
                if selected != top:
                    forecast_disagreements.append({
                        'horizon_seconds': int(item.get('horizon_seconds')),
                        'forecast_mode': top,
                        'forecast_confidence': round(conf, 4),
                        'decision_mode': selected,
                        'reason': 'high_confidence_forecast_disagrees_with_action',
                    })
    return {
        'accepted': bool(accepted),
        'reasons': reasons,
        'errors': [],
        'forecast_disagreements': forecast_disagreements,
        'active_own_unit_types': sorted(active_types),
    }
