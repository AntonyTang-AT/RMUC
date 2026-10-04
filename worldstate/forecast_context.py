# -*- coding: utf-8 -*-
"""将因果时序预测转换为大模型可读的参考上下文。

这个模块只做信息整理，不把预测结果提升为执行指令。Tier2 可以阅读它来
补充态势分析，但最终动作仍必须通过 ``schema.layered`` 的结构化校验。
"""
from __future__ import annotations

import json
from typing import Any, Mapping


def _finite_probability(value: Any, default: float = 0.0) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return max(0.0, min(1.0, value))


def _distribution(head: Mapping[str, Any] | None) -> dict[str, Any]:
    head = head or {}
    probs = head.get('probabilities') or {}
    if not isinstance(probs, Mapping):
        probs = {}
    clean = {str(k): round(_finite_probability(v), 6) for k, v in probs.items()}
    top = head.get('top')
    if top is None and clean:
        top = max(clean, key=clean.get)
    confidence = head.get('confidence')
    if confidence is None and top in clean:
        confidence = clean[top]
    comparison = head.get('model_vs_baseline') or {}
    baseline_top = comparison.get('baseline_top', head.get('baseline_top'))
    model_top = comparison.get('model_top', head.get('model_top', top))
    source = head.get('source', head.get('selected_source'))
    return {
        'top': top,
        'confidence': round(_finite_probability(confidence), 6),
        'probabilities': clean,
        'model_top': model_top,
        'baseline_top': baseline_top,
        'model_confidence': head.get('model_confidence', confidence),
        'baseline_confidence': head.get('baseline_confidence',
                                        comparison.get('baseline_confidence')),
        'blend_alpha': head.get('blend_alpha', comparison.get('alpha')),
        'selected_source': source,
    }


def _compact_entity(item: Mapping[str, Any]) -> dict[str, Any]:
    """保留对态势推理有用的字段，避免把完整预测表塞进 prompt。"""
    state = item.get('special_state_forecast') or item.get('special_state') or {}
    movement = item.get('movement') or {}
    result = {
        'entity_id': item.get('entity_id'),
        'side': item.get('side'),
        'entity_type': item.get('entity_type'),
        'state': item.get('state'),
        'presence_probability': round(_finite_probability(item.get('presence_probability')), 6),
        'death_probability': round(_finite_probability(item.get('death_probability',
                                                                  1.0 - _finite_probability(item.get('presence_probability')))), 6),
        'from_zone': movement.get('from_zone'),
        'to_zone': movement.get('to_zone') or item.get('to_zone'),
        'legal_adjacent': movement.get('legal_adjacent', movement.get('legal')),
        'legal_path': movement.get('legal_path', movement.get('legal_adjacent', movement.get('legal'))),
        'path_zones': movement.get('path_zones') or [],
        'path_hops': movement.get('path_hops'),
        'reachable_within_horizon': movement.get('reachable_within_horizon'),
        'estimated_travel_time_s': movement.get('estimated_travel_time_s'),
        'distance_constraint': movement.get('distance_constraint'),
        'position_probability': item.get('position_probability'),
        'entity_motion_lgbm_zone': item.get('entity_motion_lgbm_zone'),
        'entity_motion_lgbm_canonical_zone': item.get('entity_motion_lgbm_canonical_zone'),
        'entity_motion_lgbm_position_applied': item.get('entity_motion_lgbm_position_applied'),
        'special_state': {
            key: state.get(key) for key in (
                'attack_status', 'target', 'possible_target', 'attack_probability',
                'attacking_energy_unit', 'possible_energy_unit_target',
                'resource_phase', 'resource_phase_probabilities', 'has_energy_unit',
                'defense_status', 'interaction',
            ) if state.get(key) is not None
        },
    }
    # Do not emit a large empty object for absent/unknown units.
    return {k: v for k, v in result.items() if v is not None and v != {}}


def _compact_current_state(current_state: Mapping[str, Any] | None,
                           *, max_units: int = 32) -> dict[str, Any] | None:
    """Keep only visible current facts for the decision model.

    Forecasts are reference evidence.  A separate current-facts block makes
    that ordering explicit and prevents a language model from treating a
    future top class as if it were an observed fact.
    """
    if not isinstance(current_state, Mapping):
        return None
    units = current_state.get('units') or []
    if isinstance(units, Mapping):
        units = list(units.values())
    compact = []
    for item in list(units)[:max(1, int(max_units))]:
        if not isinstance(item, Mapping):
            continue
        keep = ('entity_id', 'side', 'entity_type', 'position', 'yaw',
                'speed_mps', 'hp', 'hp_ratio', 'life_state',
                'functional_zone', 'canonical_zone', 'attack_status',
                'target', 'resource_phase', 'has_energy_unit')
        row = {key: item.get(key) for key in keep
               if item.get(key) is not None}
        if row:
            compact.append(row)
    return {
        'issued_at': current_state.get('issued_at'),
        'visibility': current_state.get('visibility', 'sensor_radar_masked_snapshot_only'),
        'future_rows_used_at_issue_time': bool(
            current_state.get('future_rows_used_at_issue_time', False)),
        'units': compact,
    }


def _comparison(head: Mapping[str, Any] | None) -> dict[str, Any]:
    """Expose model/baseline disagreement without duplicating all heads."""
    head = head or {}
    nested = head.get('model_vs_baseline') or {}
    selected = head.get('top')
    model = head.get('model_top', nested.get('model_top', selected))
    baseline = head.get('baseline_top', nested.get('baseline_top'))
    return {
        'selected': selected,
        'model': model,
        'baseline': baseline,
        'selected_confidence': head.get('confidence'),
        'model_confidence': head.get('model_confidence', nested.get('model_confidence')),
        'baseline_confidence': head.get('baseline_confidence', nested.get('baseline_confidence')),
        'blend_alpha': head.get('blend_alpha', nested.get('alpha')),
        'source': head.get('source', head.get('selected_source')),
        'model_agrees_with_baseline': (
            None if model is None or baseline is None else model == baseline),
        'selected_agrees_with_baseline': (
            None if selected is None or baseline is None else selected == baseline),
    }


def build_forecast_context(forecast: Mapping[str, Any] | None,
                           *, max_entities_per_horizon: int = 32) -> dict[str, Any] | None:
    """Build a versioned, auditable LLM reference context.

    ``forecast`` is expected to be the output of ``TemporalForecastRuntime``.
    The function is deliberately tolerant of old/fake forecast objects so it can
    be used by replay tools and tests.  It never invents missing probabilities.
    """
    if not isinstance(forecast, Mapping):
        return None
    horizons = forecast.get('horizons') or {}
    if not isinstance(horizons, Mapping) or not horizons:
        return None
    records = []
    decision_guidance = []
    for raw_horizon, raw in sorted(horizons.items(), key=lambda pair: int(pair[0])):
        if not isinstance(raw, Mapping):
            continue
        try:
            horizon = int(raw_horizon)
        except (TypeError, ValueError):
            continue
        entities = raw.get('entities') or {}
        if isinstance(entities, Mapping):
            entity_values = list(entities.values())
        elif isinstance(entities, list):
            entity_values = entities
        else:
            entity_values = []
        compact = [_compact_entity(item) for item in entity_values
                   if isinstance(item, Mapping)][:max(1, int(max_entities_per_horizon))]
        outlook = raw.get('global_outlook') or {}
        uncertainty = dict(outlook.get('uncertainty') or {})
        action = (outlook.get('recommended_actions') or [{}])[0] or {}
        scope = outlook.get('forecast_scope') or {}
        guidance = {
            'horizon_seconds': horizon,
            'likely_mode': _distribution(raw.get('global_mode')).get('top'),
            'likely_trend': _distribution(raw.get('opponent_trend')).get('top'),
            'likely_region': _distribution(raw.get('enemy_region')).get('top'),
            'detail_level': scope.get('spatial_detail'),
            'action': action.get('objective') or action.get('action'),
            'trigger': action.get('trigger'),
            'fallback': action.get('fallback'),
            'route_examples': [
                {'unit': item.get('entity_type'),
                 'path': (item.get('movement') or {}).get('path_zones') or []}
                for item in entity_values
                if isinstance(item, Mapping)
                and (item.get('movement') or {}).get('path_zones')
            ][:4],
        }
        records.append({
            'horizon_seconds': horizon,
            'global_mode': _distribution(raw.get('global_mode')),
            'opponent_trend': _distribution(raw.get('opponent_trend')),
            'enemy_region': _distribution(raw.get('enemy_region')),
            'model_comparison': {
                'global_mode': _comparison(raw.get('global_mode')),
                'opponent_trend': _comparison(raw.get('opponent_trend')),
                'enemy_region': _comparison(raw.get('enemy_region')),
            },
            'outlook': outlook,
            'entities': compact,
            'uncertainty': uncertainty,
            'decision_guidance': guidance,
        })
        decision_guidance.append(guidance)
    if not records:
        return None
    source = {
        'kind': 'causal_temporal_forecast',
        'model_version': forecast.get('model_version'),
        'calibration_version': forecast.get('calibration_version'),
        'input_snapshot_hash': forecast.get('input_snapshot_hash'),
        'team': forecast.get('team'),
        'issued_at': forecast.get('issued_at'),
        'causal_history_length': forecast.get('causal_history_length'),
    }
    if forecast.get('entity_motion_expert') is not None:
        source['entity_motion_expert'] = forecast.get('entity_motion_expert')
    if forecast.get('global_task_expert') is not None:
        source['global_task_expert'] = forecast.get('global_task_expert')
    return {
        'schema_version': 'forecast-context-v1',
        'source': source,
        'horizon_semantics': {
            '10': '短期：单位、功能区和特殊状态，支持立即行动判断',
            '30': '中短期：单位/区域变化和局部控制转移',
            '60': '中期：区域控制、目标和攻防趋势',
            '120': '远期：宏观方向和目标压力，不应据此确定精确坐标',
        },
        'decision_guidance': decision_guidance,
        'current_state': _compact_current_state(forecast.get('current_state')),
        'predictions': records,
        'usage_policy': {
            'role': 'reference_only',
            'may_inform': ['态势解释', '候选行动排序', '需要复核的风险提示'],
            'must_not_override': ['当前可观测事实', '规则约束', '结构化单位动作字段'],
            'unknown_policy': '缺失或低置信度字段保持未知，不补写为事实',
            'priority_order': [
                'current_state.visible_facts',
                'prediction.model_comparison',
                'prediction.outlook',
                'language_model_reasoning',
            ],
            'required_analysis_sections': [
                'current_situation_enemy_and_self',
                'forecast_interpretation_with_horizon',
                'actionable_recommendation_with_trigger_and_fallback',
            ],
        },
    }


def format_forecast_context(context: Mapping[str, Any] | None,
                            *, max_chars: int = 12000) -> str:
    """为 prompt 生成稳定 JSON 文本，明确其仅是参考信息。"""
    if not context:
        return ''
    payload = json.dumps(context, ensure_ascii=False, separators=(',', ':'))
    if len(payload) > max_chars:
        compact_guidance = []
        for raw_guidance in list(context.get('decision_guidance') or [])[:4]:
            item = dict(raw_guidance or {})
            item['route_examples'] = [
                {'unit': route.get('unit'), 'path': list(route.get('path') or [])[:4]}
                for route in list(item.get('route_examples') or [])[:2]
            ]
            compact_guidance.append(item)
        # Keep the source and task distributions first.  Entity-level details
        # are useful to the UI but can make a slow-loop prompt needlessly large.
        trimmed = {
            'schema_version': context.get('schema_version'),
            'source': context.get('source'),
            'horizon_semantics': context.get('horizon_semantics'),
            'decision_guidance': compact_guidance,
            'current_state': _compact_current_state(context.get('current_state'), max_units=8),
            'predictions': [],
            'usage_policy': context.get('usage_policy'),
            'truncation': {'applied': True, 'max_chars': int(max_chars)},
        }
        for item in list(context.get('predictions') or [])[:4]:
            item = dict(item)
            item['outlook'] = {}
            item['entities'] = list(item.get('entities') or [])[:4]
            item['model_comparison'] = item.get('model_comparison') or {}
            trimmed['predictions'].append(item)
        payload = json.dumps(trimmed, ensure_ascii=False, separators=(',', ':'))
        if len(payload) > max_chars:
            # A final compact form retains exactly what the LLM needs for
            # trend comparison while keeping the prompt bounded and valid JSON.
            compact = {
                'schema_version': context.get('schema_version'),
                'decision_guidance': compact_guidance,
                'source': context.get('source'),
                'predictions': [],
                'usage_policy': context.get('usage_policy'),
                'truncation': {'applied': True, 'max_chars': int(max_chars),
                               'level': 'probability_heads_only'},
            }
            for item in list(context.get('predictions') or [])[:4]:
                compact['predictions'].append({
                    'horizon_seconds': item.get('horizon_seconds'),
                    'global_mode': _distribution(item.get('global_mode')),
                    'opponent_trend': _distribution(item.get('opponent_trend')),
                    'enemy_region': _distribution(item.get('enemy_region')),
                    'model_comparison': item.get('model_comparison') or {},
                    'uncertainty': item.get('uncertainty') or {},
                    'decision_guidance': item.get('decision_guidance') or {},
                })
            payload = json.dumps(compact, ensure_ascii=False, separators=(',', ':'))
        if len(payload) > max_chars:
            # This should only occur with an unusually long snapshot hash or
            # custom metadata.  Keep the contract valid rather than slicing
            # JSON in the middle.
            minimal = {
                'schema_version': context.get('schema_version'),
                'decision_guidance': compact_guidance,
                'predictions': [
                    {'horizon_seconds': item.get('horizon_seconds'),
                     'global_mode': _distribution(item.get('global_mode')),
                     'opponent_trend': _distribution(item.get('opponent_trend')),
                     'enemy_region': _distribution(item.get('enemy_region'))}
                    for item in list(context.get('predictions') or [])[:4]
                ],
                'usage_policy': {'role': 'reference_only'},
                'truncation': {'applied': True, 'level': 'minimal'},
            }
            payload = json.dumps(minimal, ensure_ascii=False, separators=(',', ':'))
            if len(payload) > max_chars:
                # A very small caller budget should still receive valid JSON.
                # Keep one compact head per horizon and the reference policy.
                minimal['predictions'] = [
                    {
                        'horizon_seconds': item.get('horizon_seconds'),
                        'selected': {
                            'mode': (item.get('global_mode') or {}).get('top'),
                            'trend': (item.get('opponent_trend') or {}).get('top'),
                            'region': (item.get('enemy_region') or {}).get('top'),
                        },
                    }
                    for item in list(context.get('predictions') or [])[:4]
                ]
                minimal['decision_guidance'] = [
                    {
                        'horizon_seconds': item.get('horizon_seconds'),
                        'action': item.get('action'),
                        'trigger': item.get('trigger'),
                    }
                    for item in compact_guidance[:4]
                ]
                payload = json.dumps(minimal, ensure_ascii=False, separators=(',', ':'))
            if len(payload) > max_chars:
                minimal['predictions'] = minimal['predictions'][:2]
                minimal['decision_guidance'] = minimal['decision_guidance'][:2]
                payload = json.dumps(minimal, ensure_ascii=False, separators=(',', ':'))
    return ('\n\n预测参考信息（因果模型输出，仅供态势分析；不是事实，也不能直接改写动作）：\n'
            + payload)


__all__ = ['build_forecast_context', 'format_forecast_context']
