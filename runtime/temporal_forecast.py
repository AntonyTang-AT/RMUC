"""Causal runtime adapter for the V3 per-entity temporal forecaster.

The adapter accepts an already visibility-masked observable snapshot.  It does
not build a snapshot from the replay database, and therefore cannot leak
future or hidden opponent fields into deployment inference.
"""

from __future__ import annotations

from collections import deque
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from forecast.entity import ENTITY_SLOTS


DEFAULT_CHECKPOINT = "out_m2/temporal_forecast_canonical_side/temporal_gru_v1.pt"
FALLBACK_CHECKPOINT = "out_m2/temporal_forecast_weighted_v3/temporal_gru_v1.pt"
HYBRID_CHECKPOINT = "out_m2/temporal_hybrid_motion/temporal_hybrid_motion.pt"
MODE_XGB_CHECKPOINT = "out_m2/mode_change_xgb_probe/report.model.ubj"
CANONICAL_MODE_LGBM_MANIFEST = "out_m2/canonical_mode_lightgbm/production_manifest.json"
ENTITY_MOTION_LGBM_MANIFEST = "out_m2/entity_motion_lightgbm/production_manifest.json"
GLOBAL_TASK_LGBM_MANIFEST = "out_m2/global_task_lightgbm/production_manifest.json"
ENEMY_REGION_TRANSITION_MANIFEST = "out_m2/enemy_region_transition/production_manifest.json"
ENEMY_REGION_TRANSITION_MANIFEST_10 = "out_m2/enemy_region_transition/production_manifest_10.json"
VALIDATION_POSITION_FUSION_WEIGHTS = {10: 0.25, 30: 0.75, 60: 0.75, 120: 1.0}
ADAPTIVE_PERSISTENCE_CONFIDENCE = 0.98


def _persistence_blend_from_checkpoint(checkpoint):
    """Return only an explicit tuned blend config; legacy scalar metadata is not a tune."""
    config = checkpoint.get("persistence_blend") if isinstance(checkpoint, dict) else None
    if not isinstance(config, dict) or config.get("enabled", True) is False:
        return None
    if not isinstance(config.get("alphas"), dict):
        return None
    return config


def _forecast_zone_counts(entities, side):
    counts = {}
    for item in (entities or {}).values():
        if item.get('side') != side or item.get('state') in ('absent', 'destroyed'):
            continue
        zone = item.get('to_zone') or item.get('current_functional_zone')
        if not zone:
            continue
        counts[zone] = counts.get(zone, 0) + 1
    return counts


def _forecast_role_summary(entities, side):
    """Summarise forecast readiness without exposing a coordinate table."""
    items = [item for item in (entities or {}).values()
             if item.get('side') == side and item.get('state') not in ('absent', 'destroyed')]
    by_type = {}
    critical = 0
    active = 0
    for item in items:
        kind = item.get('entity_type') or 'unknown'
        entry = by_type.setdefault(kind, {
            'present': 0, 'critical': 0, 'active': 0,
            'attacking': 0, 'possible_attack': 0,
            'resource_phases': {}, 'defense_states': {},
        })
        entry['present'] += 1
        state = item.get('special_state_forecast') or item.get('special_state') or {}
        if item.get('state') == 'critical':
            entry['critical'] += 1
            critical += 1
        else:
            entry['active'] += 1
            active += 1
        attack = state.get('attack_status')
        if attack in ('attacking', 'possible_attack'):
            entry[attack] += 1
        phase = state.get('resource_phase')
        if phase:
            phases = entry['resource_phases']
            phases[phase] = phases.get(phase, 0) + 1
        defence = state.get('defense_status')
        if defence:
            defenses = entry['defense_states']
            defenses[defence] = defenses.get(defence, 0) + 1
    return {
        'present': len(items), 'active': active, 'critical': critical,
        'by_type': by_type,
    }


def _forecast_objectives(entities):
    """Return coarse objective health and interaction posture."""
    result = {}
    for item in (entities or {}).values():
        kind = item.get('entity_type')
        if kind not in ('基地', '前哨站', '能量机关'):
            continue
        side = item.get('side') or 'unknown'
        state = item.get('state')
        result[f'{side}:{kind}'] = {
            'state': state,
            'hp_ratio': item.get('hp_ratio'),
            'zone': item.get('to_zone') or item.get('current_functional_zone'),
            'interaction': (item.get('special_state_forecast') or
                            item.get('special_state') or {}).get('interaction'),
        }
    return result


def _forecast_engagement(entities, contested):
    """Aggregate contact and target semantics from unit sidecars."""
    contacts = []
    attacking = {'self': 0, 'opponent': 0}
    possible = {'self': 0, 'opponent': 0}
    target_types = {}
    energy_targets = {'self': 0, 'opponent': 0}
    transitions = []
    for item in (entities or {}).values():
        side = item.get('side')
        if side not in ('self', 'opponent') or item.get('state') in ('absent', 'destroyed'):
            continue
        state = item.get('special_state_forecast') or item.get('special_state') or {}
        status = state.get('attack_status')
        if status == 'attacking':
            attacking[side] += 1
        elif status == 'possible_attack':
            possible[side] += 1
        target = state.get('target') or state.get('possible_target')
        if target:
            if isinstance(target, dict):
                target_label = target.get('entity_type') or target.get('objective_type') or 'unknown'
            else:
                target_label = str(target)
            target_types[target_label] = target_types.get(target_label, 0) + 1
        if state.get('attacking_energy_unit') or state.get('possible_energy_unit_target'):
            energy_targets[side] += 1
        move = item.get('movement') or {}
        if move.get('from_zone') and move.get('to_zone'):
            transitions.append({
                'side': side, 'entity_type': item.get('entity_type'),
                'from_zone': move.get('from_zone'), 'to_zone': move.get('to_zone'),
                'legal': move.get('legal_path', move.get('legal_adjacent', move.get('legal'))),
                'legal_adjacent': move.get('legal_adjacent'),
                'path_zones': move.get('path_zones') or [],
                'path_hops': move.get('path_hops'),
            })
    return {
        'contested_zones': list(contested),
        'active_attackers': attacking,
        'possible_attackers': possible,
        'target_types': target_types,
        'energy_unit_focus': energy_targets,
        'planned_transitions': transitions,
    }


def _forecast_motion(entities):
    """Summarise speed/distance constrained route forecasts."""
    transitions = []
    reachable = 0
    constrained = 0
    travel_times = []
    for item in (entities or {}).values():
        if item.get('state') in ('absent', 'destroyed'):
            continue
        move = item.get('movement') or {}
        if not move.get('from_zone') or not move.get('to_zone'):
            continue
        row = {
            'side': item.get('side'),
            'entity_type': item.get('entity_type'),
            'from_zone': move.get('from_zone'),
            'to_zone': move.get('to_zone'),
            'speed_mps': move.get('speed_mps'),
            'max_travel_m': move.get('max_travel_m'),
            'distance_to_target_m': move.get('distance_to_target_m'),
            'estimated_travel_time_s': move.get('estimated_travel_time_s'),
            'reachable_within_horizon': move.get('reachable_within_horizon'),
            'distance_constraint': move.get('distance_constraint'),
            'route_constraint': move.get('route_constraint'),
            'path_zones': move.get('path_zones') or [],
            'path_hops': move.get('path_hops'),
            'legal_path': move.get('legal_path', move.get('legal_adjacent', move.get('legal'))),
        }
        transitions.append(row)
        if move.get('reachable_within_horizon') is True:
            reachable += 1
        if move.get('projection_applied'):
            constrained += 1
        if isinstance(move.get('estimated_travel_time_s'), (int, float)):
            travel_times.append(float(move['estimated_travel_time_s']))
    return {
        'transition_count': len(transitions),
        'reachable_count': reachable,
        'distance_constrained_count': constrained,
        'mean_estimated_travel_time_s': (round(sum(travel_times) / len(travel_times), 3)
                                         if travel_times else None),
        'transitions': transitions,
        'policy': 'legal_functional_route_and_speed_x_horizon',
    }


def _forecast_event_probabilities(entities, contested, own_prefix):
    """Make auditable event probabilities from per-entity sidecars.

    These are aggregation probabilities, not a separately trained event
    head.  Keeping the derivation explicit prevents the global layer from
    presenting heuristic counts as model certainty.
    """
    total = 0
    intrusion_survival = 1.0
    attack_survival = 1.0
    resource_survival = 1.0
    for item in (entities or {}).values():
        if item.get('side') != 'opponent' or item.get('state') in ('absent', 'destroyed'):
            continue
        total += 1
        p = max(0.0, min(1.0, float(item.get('presence_probability') or 0.0)))
        if str(item.get('to_zone') or '').startswith(own_prefix):
            intrusion_survival *= (1.0 - p)
        state = item.get('special_state_forecast') or {}
        attack_p = state.get('attack_probability')
        if attack_p is None:
            attack_p = 1.0 if state.get('attack_status') in ('attacking', 'possible_attack') else 0.0
        attack_survival *= (1.0 - max(0.0, min(1.0, float(attack_p))))
        if state.get('resource_phase') in ('collecting', 'returning', 'delivering'):
            resource_survival *= (1.0 - p)
    contested_p = min(1.0, len(contested) / max(1.0, len(set(
        (item.get('to_zone') or '') for item in (entities or {}).values()
        if item.get('to_zone')))))
    return {
        'enemy_intrusion': round(1.0 - intrusion_survival, 6),
        'enemy_attack_intent': round(1.0 - attack_survival, 6),
        'resource_interruption': round(1.0 - resource_survival, 6),
        'contested_control': round(contested_p, 6),
        'sample_opponent_units': total,
        'source': 'entity_sidecar_aggregation',
    }


def _tactical_situation(contested, enemy_intruders, engagement, resource,
                        objectives, control_trend):
    """Name an evidence-supported tactical candidate, with no invented score.

    The three-way learned mode is intentionally excluded. A mode alone cannot
    establish which objective, resource route, or defensive line is involved.
    """
    label = '细分态势待确认'
    evidence = []
    if enemy_intruders:
        label = '纵深拦截压力'
        evidence = [f"敌方预测进入{item['to_zone']}" for item in enemy_intruders[:2]
                    if item.get('to_zone')]
    elif contested:
        label = '关键区域争夺'
        evidence = list(contested[:2])
    elif any(key.startswith('self:') and (value.get('state') in ('critical', 'damaged')
                                                or isinstance(value.get('hp_ratio'), (int, float))
                                                and value['hp_ratio'] < .35)
             for key, value in objectives.items()):
        label = '关键设施防护'
        evidence = [key for key, value in objectives.items()
                    if key.startswith('self:') and (value.get('state') in ('critical', 'damaged')
                                                      or isinstance(value.get('hp_ratio'), (int, float))
                                                      and value['hp_ratio'] < .35)][:2]
    elif engagement.get('active_attackers', {}).get('self', 0) > 0:
        label = '交战压制'
        evidence = [f"我方预计交战单位{engagement['active_attackers']['self']}类"]
    elif (resource.get('own') or {}).get('phase') in (
            'collecting_energy_unit', 'going_to_mine', 'returning_to_exchange', 'exchanging'):
        label = '资源运营'
        evidence = [f"工程阶段：{resource['own']['phase']}"]
    elif control_trend == 'own_pressure':
        label = '控制性推进'
        evidence = ['我方预测推进槽位多于敌方']
    elif control_trend == 'opponent_pressure':
        label = '前沿防守压力'
        evidence = ['敌方预测推进槽位多于我方']
    return {
        'label': label, 'evidence': evidence,
        'probability': None,
        'source': 'forecast_evidence_rule',
    }


def suppress_destroyed_entity_forecasts(forecasts, current_entities):
    """Stop type-level forecasts only when every current track is explicitly dead.

    Missing/stale observations and unchanged positions do not establish death.
    Since the model aggregates units by side and type, a slot is suppressed
    only if every current track for that role/type has HP <= 0 (or an explicit
    destroyed lifecycle flag). A positive HP reading, including after respawn,
    leaves the slot forecastable.
    """
    def explicitly_dead(item):
        hp = item.get('hp')
        hp_dead = (isinstance(hp, (int, float)) and not isinstance(hp, bool)
                   and math.isfinite(float(hp)) and hp <= 0)
        return (hp_dead
                or item.get('life_state') == 'destroyed')

    for forecast in (forecasts or {}).values():
        role = forecast.get('side')
        entity_type = forecast.get('entity_type')
        matching = [item for item in (current_entities or [])
                    if item.get('role') == role and item.get('entity_type') == entity_type]
        if matching and all(explicitly_dead(item) for item in matching):
            forecast.update({
                'life_state': 'destroyed',
                'state': 'destroyed',
                'presence_probability': 0.0,
                'position': None,
                'raw_predicted_zone': None,
                'zone_head_predicted_zone': None,
                'zone_head_probabilities': None,
                'canonical_zone_predicted_zone': None,
                'canonical_zone_probabilities': None,
                'to_zone': None,
                'to_canonical_zone': None,
                'movement': None,
                'special_state_forecast': None,
                'forecast_suppressed': True,
                'suppression_reason': 'explicit_current_death_evidence',
            })
    return forecasts


def _derive_global_outlook(entities, global_mode, opponent_trend, enemy_region,
                           team, horizon_seconds=None):
    """Derive a richer, auditable global outlook from the entity forecast.

    These fields are a structured aggregation layer, not a new learned head.
    They let the UI and a language model reason about control, threats and
    actions without repeating every coordinate in prose.
    """
    opponent = '蓝' if team == '红' else '红'
    own_counts = _forecast_zone_counts(entities, 'self')
    opp_counts = _forecast_zone_counts(entities, 'opponent')
    zones = sorted(set(own_counts) | set(opp_counts))
    zone_control = {}
    contested = []
    for zone in zones:
        own_n, opp_n = own_counts.get(zone, 0), opp_counts.get(zone, 0)
        posture = 'contested' if own_n and opp_n else (
            'own_control' if own_n else 'opponent_control')
        zone_control[zone] = {'posture': posture, 'own_units': own_n,
                              'opponent_units': opp_n}
        if posture == 'contested':
            contested.append(zone)
    own_prefix = '红' if team == '红' else '蓝'
    enemy_intruders = [
        {'entity_type': item.get('entity_type'),
         'to_zone': item.get('to_zone'),
         'possible_target': (item.get('special_state') or {}).get('possible_target')}
        for item in (entities or {}).values()
        if item.get('side') == 'opponent' and item.get('state') not in ('absent', 'destroyed')
        and str(item.get('to_zone') or '').startswith(own_prefix)
    ]
    resource = {'own': {}, 'opponent': {}}
    for side_key, label in (('self', 'own'), ('opponent', 'opponent')):
        for item in (entities or {}).values():
            if (item.get('side') != side_key or item.get('entity_type') != '工程'
                    or item.get('state') in ('absent', 'destroyed')):
                continue
            state = item.get('special_state') or {}
            forecast = item.get('special_state_forecast') or {}
            probs = forecast.get('resource_phase_probabilities') or {}
            top = max(probs, key=probs.get) if probs else state.get('resource_phase', 'unknown')
            resource[label] = {
                'phase': top,
                'has_energy_unit': state.get('has_energy_unit'),
                'target_zone': state.get('resource_target_zone'),
                'source': state.get('observability', 'unavailable'),
            }
    threats = []
    if enemy_intruders:
        threats.append({'type': 'enemy_intrusion', 'level': 'high',
                        'units': enemy_intruders})
    if contested:
        threats.append({'type': 'contested_control', 'level': 'medium',
                        'zones': contested})
    mode_top = (global_mode or {}).get('top')
    trend_top = (opponent_trend or {}).get('top')
    if trend_top in ('推进', 'advance'):
        threats.append({'type': 'opponent_pressure', 'level': 'medium',
                        'description': '敌方推进趋势需要连续帧确认'})
    if mode_top == '防守':
        primary = 'protect_base_and_outpost'
    elif mode_top == '进攻' and not enemy_intruders:
        primary = 'pressure_objective_with_retreat_route'
    elif contested:
        primary = 'stabilize_contested_zone'
    else:
        primary = 'maintain_control_and_observe'
    actions = [{
        'priority': 'P0' if enemy_intruders else 'P1',
        'action': primary,
        'objective': ('先拦截入侵单位并保护己方建筑' if enemy_intruders else
                      '保持当前区域控制并等待可验证的意图变化'),
        'trigger': '出现连续帧推进、建筑受击或明确目标字段',
        'fallback': '回到最近己方控制区并保留相邻撤退路线',
    }]
    own_roles = _forecast_role_summary(entities, 'self')
    opp_roles = _forecast_role_summary(entities, 'opponent')
    engagement = _forecast_engagement(entities, contested)
    objectives = _forecast_objectives(entities)
    motion = _forecast_motion(entities)
    event_probabilities = _forecast_event_probabilities(entities, contested, own_prefix)
    opponent_prefix = '蓝' if team == '红' else '红'
    own_push = sum(1 for item in (entities or {}).values()
                   if item.get('side') == 'self' and item.get('state') not in ('absent', 'destroyed') and
                   str(item.get('to_zone') or '').startswith((opponent_prefix, 'central')))
    opp_push = sum(1 for item in (entities or {}).values()
                   if item.get('side') == 'opponent' and item.get('state') not in ('absent', 'destroyed') and
                   str(item.get('to_zone') or '').startswith((own_prefix, 'central')))
    if contested:
        control_trend = 'contested_and_unresolved'
    elif own_push > opp_push:
        control_trend = 'own_pressure'
    elif opp_push > own_push:
        control_trend = 'opponent_pressure'
    else:
        control_trend = 'distributed'
    tactical_situation = _tactical_situation(
        contested, enemy_intruders, engagement, resource, objectives, control_trend)
    mode_conf = (global_mode or {}).get('confidence')
    trend_conf = (opponent_trend or {}).get('confidence')
    horizon = int(horizon_seconds or 10)
    if horizon <= 10:
        detail_level = 'unit-zone-and-state'
        state_detail = 'specialized-unit-sidecars'
        action_detail = 'commitment_with_trigger_and_fallback'
    elif horizon <= 60:
        detail_level = 'unit-zone-with-role-summary'
        state_detail = 'role-and-objective-sidecars'
        action_detail = 'local-control-and-objective-pressure'
    else:
        detail_level = 'region-and-objective'
        state_detail = 'objective-and-trend-sidecars'
        action_detail = 'coarse-direction-only'
    return {
        'control': {
            'posture': 'contested' if contested else 'distributed',
            'own_zones': own_counts,
            'opponent_zones': opp_counts,
            'by_zone': zone_control,
            'contested_zones': contested,
        },
        'threats': threats,
        'resource_posture': resource,
        'opponent_intent_hint': {
            'trend': trend_top,
            'region': (enemy_region or {}).get('top'),
        },
        'recommended_actions': actions,
        'decision_summary': {
            'posture': (global_mode or {}).get('top'),
            'control_trend': control_trend,
            'priority': actions[0]['action'],
            'commitment_rule': actions[0]['trigger'],
            'fallback_rule': actions[0]['fallback'],
        },
        'force_balance': {
            'self': own_roles,
            'opponent': opp_roles,
            'push_margin_estimate': own_push - opp_push,
        },
        'engagement': engagement,
        'motion_projection': motion,
        'event_probabilities': event_probabilities,
        'tactical_situation': tactical_situation,
        'objective_projection': {
            'objects': objectives,
            'at_risk': [key for key, value in objectives.items()
                        if value.get('state') in ('critical', 'damaged')],
            'source': 'visible_entity_objective_sidecars',
        },
        'evidence': {
            'classification_heads': ['global_mode', 'opponent_trend', 'enemy_region'],
            'entity_heads': ['presence', 'position', 'hp_ratio', 'count',
                             'functional_zone', 'canonical_zone'],
            'derived_layers': ['control', 'threats', 'engagement', 'motion_projection',
                               'event_probabilities', 'tactical_situation', 'recommended_actions'],
            'future_data_used_at_issue_time': False,
        },
        'objective_status': objectives,
        'mobility': {
            'legal_transition_count': sum(1 for item in engagement['planned_transitions']
                                          if item.get('legal') is not False),
            'transition_count': len(engagement['planned_transitions']),
            'adjacency_constrained': True,
            'unknown_route_policy': 'hold_current_zone',
        },
        'forecast_scope': {
            'horizon_seconds': horizon,
            'spatial_detail': detail_level,
            'state_detail': state_detail,
            'action_detail': action_detail,
            'decision_horizon': ('short_commitment_with_conditional_fallback'
                                 if horizon <= 10 else
                                 ('medium_control_shift' if horizon <= 60
                                  else 'long_directional_trend')),
        },
        'uncertainty': {
            'global_mode_confidence': mode_conf,
            'opponent_trend_confidence': trend_conf,
            'joint_confidence': round(float(mode_conf or 0.0) *
                                     float(trend_conf or 0.0), 6),
            'spatial_detail': detail_level,
            'missing_action_fields_remain_unknown': True,
            'calibration': 'monotonic_horizon_v2_temperature',
            'detail_decay_policy': 'nearer_horizon_more_specific',
        },
        'derivation': 'entity_forecast_aggregation_v1',
    }


def apply_mode_xgb_candidate(horizons, candidate, team="红"):
    """Promote a valid frozen candidate into only the 30-second mode head.

    The neural output is retained verbatim for side-by-side audit, and the
    global outlook is rebuilt so its tactical summary uses the selected head.
    """
    if (not isinstance(candidate, dict) or candidate.get("status") != "ok"
            or int(candidate.get("horizon_seconds") or 0) != 30):
        return False
    row = (horizons or {}).get("30")
    if not isinstance(row, dict) or not isinstance(row.get("global_mode"), dict):
        return False
    probabilities = candidate.get("probabilities")
    if not isinstance(probabilities, dict) or set(probabilities) != set(("进攻", "均衡", "防守")):
        return False
    baseline = dict(row["global_mode"])
    row["global_mode_neural_baseline"] = baseline
    row["global_mode"] = {
        "top": candidate.get("top"),
        "confidence": candidate.get("confidence"),
        "probabilities": dict(probabilities),
    }
    row["global_mode_source"] = "mode_change_xgb_v1"
    row["global_outlook"] = _derive_global_outlook(
        row.get("entities") or {}, row["global_mode"],
        row.get("opponent_trend") or {}, row.get("enemy_region") or {},
        team, horizon_seconds=30)
    return True


class TemporalForecastRuntime:
    """Maintain a causal feature window and emit structured forecasts."""

    def __init__(self, checkpoint_path, *, device="auto", context=12,
                 team="红", fail_closed=True, adaptive_router=None,
                 auto_mature_feedback=None,
                 opponent_residual_calibrator=None,
                 experimental_mode_xgb_path=None, experimental_mode_xgb_alpha=0.95,
                 apply_experimental_mode_xgb=False,
                 apply_validation_position_fusion=False,
                 canonical_mode_lgbm_path=None,
                 apply_canonical_mode_lgbm=False,
                 entity_motion_lgbm_path=None,
                 apply_entity_motion_lgbm=False,
                 global_task_lgbm_path=None,
                 apply_global_task_lgbm=False,
                 enemy_region_transition_path=None,
                 apply_enemy_region_transition=False):
        self.checkpoint_path = str(checkpoint_path)
        self.device_name = device
        self.context = int(context)
        self.team = team
        self.fail_closed = bool(fail_closed)
        # Optional feedback loop.  It is disabled by default so existing
        # deployments remain deterministic; callers can opt in once matured
        # outcomes are available.
        self.adaptive_router = adaptive_router
        self.opponent_residual_calibrator = opponent_residual_calibrator
        self.auto_mature_feedback = (adaptive_router is not None
                                     or opponent_residual_calibrator is not None
                                     if auto_mature_feedback is None
                                     else bool(auto_mature_feedback))
        if (self.auto_mature_feedback and self.adaptive_router is None
                and self.opponent_residual_calibrator is None):
            raise ValueError("auto_mature_feedback requires a router or residual calibrator")
        self._observable_feedback_pending = {}
        self._observable_feedback_order = deque()
        self._observable_feedback_counts = {
            "completed_forecast_records": 0,
            "matured_task_horizon_outcomes": 0,
        }
        self._last_observable_feedback = {
            "matured_forecast_records": 0,
            "matured_task_horizon_outcomes": 0,
        }
        # The XGBoost sidecar can be exposed for auditing or promoted for the
        # 30 s mode head; other horizons always remain on the neural model.
        self.experimental_mode_xgb_path = experimental_mode_xgb_path
        self.apply_experimental_mode_xgb = bool(apply_experimental_mode_xgb)
        self.experimental_mode_xgb_alpha = float(experimental_mode_xgb_alpha)
        if not 0.0 <= self.experimental_mode_xgb_alpha <= 1.0:
            raise ValueError("experimental_mode_xgb_alpha must be in [0, 1]")
        # Validation-derived motion-position candidate; disabled by default
        # and reversible with a single constructor flag.
        self.apply_validation_position_fusion = bool(apply_validation_position_fusion)
        self._experimental_mode_xgb = None
        self._experimental_mode_xgb_error = None
        self._experimental_mode_xgb_history = deque()
        self.canonical_mode_lgbm_path = canonical_mode_lgbm_path
        self.apply_canonical_mode_lgbm = bool(apply_canonical_mode_lgbm)
        self._canonical_mode_lgbm = None
        self._canonical_mode_lgbm_error = None
        self._canonical_mode_lgbm_history = deque(maxlen=12)
        self.entity_motion_lgbm_path = entity_motion_lgbm_path
        self.apply_entity_motion_lgbm = bool(apply_entity_motion_lgbm)
        self._entity_motion_lgbm = None
        self._entity_motion_lgbm_error = None
        self.global_task_lgbm_path = global_task_lgbm_path
        self.apply_global_task_lgbm = bool(apply_global_task_lgbm)
        self._global_task_lgbm = None
        self._global_task_lgbm_error = None
        self.enemy_region_transition_path = enemy_region_transition_path
        self.apply_enemy_region_transition = bool(apply_enemy_region_transition)
        self._enemy_region_transition = None
        self._enemy_region_transitions = {}
        self._enemy_region_transition_error = None
        self.history = deque(maxlen=self.context)
        # Raw causal snapshots are retained only for the motion-aware
        # candidate.  They are converted to features immediately and never
        # expose future replay rows to the model.
        self._raw_history = deque(maxlen=self.context)
        self._position_history = deque(maxlen=2)
        self._behavior_observable_history = deque(maxlen=4)
        self._loaded = False
        self._load_error = None
        self._torch = None
        self._model = None
        self._device = None
        self._feature_vector = None
        self._horizons = None
        self._labels = None
        self._calibrate = None
        self._decode_entities = None
        self._persistence_blend = None

    def reset(self):
        self.history.clear()
        self._raw_history.clear()
        self._position_history.clear()
        self._behavior_observable_history.clear()
        self._experimental_mode_xgb_history.clear()
        self._canonical_mode_lgbm_history.clear()
        self._observable_feedback_pending.clear()
        self._observable_feedback_order.clear()
        self._observable_feedback_counts = {
            "completed_forecast_records": 0,
            "matured_task_horizon_outcomes": 0,
        }
        self._last_observable_feedback = {
            "matured_forecast_records": 0,
            "matured_task_horizon_outcomes": 0,
        }
        if self.adaptive_router is not None and hasattr(self.adaptive_router, "reset"):
            self.adaptive_router.reset()
        if (self.opponent_residual_calibrator is not None
                and hasattr(self.opponent_residual_calibrator, "reset")):
            self.opponent_residual_calibrator.reset()

    def available(self):
        self._ensure_loaded()
        return self._loaded

    def _ensure_loaded(self):
        if self._loaded or self._load_error is not None:
            return self._loaded
        if not Path(self.checkpoint_path).exists():
            self._load_error = f"checkpoint not found: {self.checkpoint_path}"
            return False
        try:
            import numpy as np
            import torch
            from forecast.entity import (decode_entity_forecast,
                                         current_positions_from_feature_row,
                                         causal_group_centroid_motion)
            from forecast.temporal import (HORIZONS, MODES, REGION_LABELS, TRENDS,
                                           MOTION_FEATURE_DIM, baseline_labels,
                                           feature_vector, motion_feature_vector)
            from forecast.train import load_model
            from forecast.calibration import (CALIBRATION_VERSION,
                                               calibrate_horizon_probabilities)

            if self.device_name in (None, "auto", ""):
                device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                device = self.device_name
            self._torch = torch
            self._np = np
            self._device = torch.device(device)
            self._model, self._checkpoint = load_model(
                self.checkpoint_path, device=self._device)
            self._persistence_blend = _persistence_blend_from_checkpoint(self._checkpoint)
            if not getattr(self._model, "entity_slots", 0):
                raise ValueError("checkpoint has no per-entity head; use temporal_forecast_v3")
            self._feature_vector = feature_vector
            self._motion_feature_vector = motion_feature_vector
            self._feature_schema = "motion" if int(getattr(self._model, "input_dim", 0)) == int(MOTION_FEATURE_DIM) else "legacy"
            if self._feature_schema == "motion" and self.context < 40:
                self.context = 40
                self.history = deque(self.history, maxlen=self.context)
                self._raw_history = deque(self._raw_history, maxlen=self.context)
            self._baseline_labels = baseline_labels
            self._horizons = HORIZONS
            self._labels = {
                "global_mode": MODES,
                "opponent_trend": TRENDS,
                "enemy_region": REGION_LABELS,
            }
            self._calibrate = calibrate_horizon_probabilities
            self._calibration_version = CALIBRATION_VERSION
            self._decode_entities = decode_entity_forecast
            self._current_positions = current_positions_from_feature_row
            self._causal_group_centroid_motion = causal_group_centroid_motion
            if self.apply_canonical_mode_lgbm and self.canonical_mode_lgbm_path:
                try:
                    from forecast.canonical_mode_lgbm import CanonicalModeLGBMEnsemble
                    self._canonical_mode_lgbm = CanonicalModeLGBMEnsemble(
                        self.canonical_mode_lgbm_path)
                except Exception as exc:
                    # The neural model remains a safe fallback if the optional
                    # CPU expert is unavailable.
                    self._canonical_mode_lgbm_error = str(exc)
            if self.apply_entity_motion_lgbm and self.entity_motion_lgbm_path:
                try:
                    from forecast.entity_motion_lgbm import EntityMotionLGBMEnsemble
                    self._entity_motion_lgbm = EntityMotionLGBMEnsemble(
                        self.entity_motion_lgbm_path)
                except Exception as exc:
                    # The neural entity head remains the safe fallback.
                    self._entity_motion_lgbm_error = str(exc)
            if self.apply_global_task_lgbm and self.global_task_lgbm_path:
                try:
                    from forecast.global_task_lgbm import GlobalTaskLGBMEnsemble
                    self._global_task_lgbm = GlobalTaskLGBMEnsemble(
                        self.global_task_lgbm_path)
                except Exception as exc:
                    self._global_task_lgbm_error = str(exc)
            if self.apply_enemy_region_transition and self.enemy_region_transition_path:
                try:
                    from forecast.enemy_region_transition import EnemyRegionTransitionEnsemble
                    paths = [self.enemy_region_transition_path]
                    if self.enemy_region_transition_path == ENEMY_REGION_TRANSITION_MANIFEST:
                        paths.append(ENEMY_REGION_TRANSITION_MANIFEST_10)
                    for manifest_path in paths:
                        if not manifest_path or not Path(manifest_path).exists():
                            continue
                        manifest = json.loads(Path(manifest_path).read_text(
                            encoding="utf-8"))
                        if not manifest.get("enabled"):
                            raise ValueError(f"enemy region transition manifest is not promoted: {manifest_path}")
                        candidate = EnemyRegionTransitionEnsemble(manifest_path)
                        horizon = int(manifest.get("horizon_seconds", 30))
                        if horizon not in (10, 30):
                            raise ValueError(f"unsupported enemy region transition horizon: {horizon}")
                        self._enemy_region_transitions[horizon] = candidate
                    self._enemy_region_transition = self._enemy_region_transitions.get(30)
                    if not self._enemy_region_transitions:
                        raise FileNotFoundError("no enabled enemy region transition manifest")
                except Exception as exc:
                    self._enemy_region_transition_error = str(exc)
            self._loaded = True
        except Exception as exc:  # deployment must fail closed, not guess
            self._load_error = f"temporal forecast unavailable: {exc}"
            if not self.fail_closed:
                raise
        return self._loaded

    @staticmethod
    def _classification(logits_by_horizon, labels, calibrate, *,
                        baseline_labels=None, task=None, blend_config=None):
        raw = {
            horizon: logits_by_horizon[horizon].detach().cpu().numpy()
            for horizon in logits_by_horizon
        }
        # Model heads return logits. Normalize them before any probability
        # calibration or persistence blending; clipping raw logits in the
        # calibrator does not produce a valid softmax distribution and made
        # runtime predictions disagree with offline evaluation.
        probabilities = {}
        for horizon, values in raw.items():
            values = np.asarray(values, dtype=np.float64)
            values = values - float(np.max(values))
            exp_values = np.exp(values)
            probabilities[horizon] = exp_values / max(float(exp_values.sum()), 1e-12)
        calibrated = calibrate(probabilities)
        output = {}
        for horizon, values in calibrated.items():
            # Retain the calibrated neural distribution before the causal
            # persistence fusion.  This is useful to the LLM and the audit
            # layer: a selected prediction must be distinguishable from the
            # raw expert output and from the current-state baseline.
            model_values = np.asarray(values, dtype=np.float64).copy()
            model_probs = {label: round(float(value), 6)
                           for label, value in zip(labels, model_values)}
            alpha = 1.0
            if blend_config and task:
                alpha = float((blend_config.get("alphas") or {}).get(
                    f"{task}@{int(horizon)}s", 1.0))
                alpha = max(0.0, min(1.0, alpha))
            baseline_label = None
            baseline_probs = None
            if baseline_labels is not None and task:
                current = baseline_labels.get(task)
                if isinstance(current, dict):
                    current = current.get(str(horizon), current.get(int(horizon)))
                if current in labels:
                    baseline_label = current
                    baseline_prior = np.full(
                        len(labels), 0.02 / max(1, len(labels) - 1),
                        dtype=np.float64)
                    baseline_prior[labels.index(current)] = 0.98
                    baseline_probs = {label: round(float(value), 6)
                                      for label, value in zip(labels, baseline_prior)}
                    # The baseline is intentionally smoothed so the model can
                    # still contribute probability mass to a changed state.
                    values = alpha * model_values + (1.0 - alpha) * baseline_prior
                    values = values / max(float(np.sum(values)), 1e-12)
            probs = {label: round(float(value), 6)
                     for label, value in zip(labels, values)}
            top = max(probs, key=probs.get)
            model_top = max(model_probs, key=model_probs.get)
            output[str(horizon)] = {
                "top": top,
                "confidence": probs[top],
                "probabilities": probs,
                "model_top": model_top,
                "model_confidence": model_probs[model_top],
                "model_probabilities": model_probs,
                "baseline_top": baseline_label,
                "baseline_confidence": (baseline_probs or {}).get(baseline_label),
                "baseline_probabilities": baseline_probs,
                "blend_alpha": round(float(alpha), 6),
                "selected_source": (
                    "model" if baseline_label is None or alpha >= 1.0
                    else ("persistence" if alpha <= 0.0 else "causal_blend")
                ),
            }
        return output

    def _opponent_behavior_context(self, example):
        """Infer a coarse current motion regime from causal visible enemy tracks."""
        now = float(example.get("issued_at", 0.0))
        if not self._behavior_observable_history:
            return "unknown"
        previous = self._behavior_observable_history[-1]
        dt = now - float(previous.get("issued_at", now))
        if not 0.0 < dt <= 6.0:
            return "unknown"

        def selected(snapshot, entity_type):
            values = ((snapshot.get("observable") or {}).get("opponent_entities") or {})
            values = list(values.values()) if isinstance(values, dict) else list(values)
            candidates = []
            for entity in values:
                if not isinstance(entity, dict) or entity.get("entity_type") != entity_type:
                    continue
                if entity.get("life_state") == "destroyed":
                    continue
                hp = entity.get("hp")
                if isinstance(hp, (int, float)) and hp <= 0:
                    continue
                if entity.get("observability_state", "current") != "current":
                    continue
                if entity.get("source") not in ("sensor", "radar", "camera", "vision"):
                    continue
                try:
                    observed = float(entity.get("observed_at", snapshot.get("issued_at", 0.0)))
                    received = float(entity.get("received_at", observed))
                except (TypeError, ValueError):
                    continue
                if observed > float(snapshot.get("issued_at", 0.0)) + 1e-6 or received > float(snapshot.get("issued_at", 0.0)) + 1e-6:
                    continue
                pos = entity.get("position")
                if not isinstance(pos, (list, tuple)) or len(pos) < 2:
                    continue
                try:
                    x = float(pos[0])
                except (TypeError, ValueError):
                    continue
                if not math.isfinite(x):
                    continue
                candidates.append(entity)
            if not candidates:
                return None
            return sorted(candidates, key=lambda item: str(item.get("entity_id", "")))[0]

        from forecast.temporal import MAP_X, MOBILE_TYPES
        toward = -1.0 if self.team == "红" else 1.0
        speeds = []
        for entity_type in MOBILE_TYPES:
            current = selected(example, entity_type)
            prior = selected(previous, entity_type)
            if current is None or prior is None:
                continue
            cur_pos, prev_pos = current.get("position"), prior.get("position")
            try:
                # Replay/runtime features store field coordinates normalised
                # to [0, 1]; the behavior gate is defined in m/s.
                speed = ((float(cur_pos[0]) - float(prev_pos[0]))
                         * MAP_X * toward / dt)
            except (TypeError, ValueError, IndexError):
                continue
            if math.isfinite(speed):
                speeds.append(speed)
        if not speeds:
            return "unknown"
        advancing = sum(speed >= 0.10 for speed in speeds) / len(speeds)
        retreating = sum(speed <= -0.10 for speed in speeds) / len(speeds)
        mean_toward_speed = sum(speeds) / len(speeds)
        if mean_toward_speed >= 0.10 and advancing >= retreating + 0.15:
            return "advance"
        if mean_toward_speed <= -0.10 and retreating >= advancing + 0.15:
            return "retreat"
        return "hold_or_mixed"

    def predict(self, observable_snapshot):
        """Predict from one current observable snapshot and past snapshots."""
        if not isinstance(observable_snapshot, dict):
            raise ValueError("observable_snapshot must be a mapping")
        if observable_snapshot.get("self_team", self.team) != self.team:
            raise ValueError("observable snapshot side does not match forecaster team")
        if not self._ensure_loaded():
            return None
        matured_feedback = self._mature_observable_feedback(observable_snapshot)
        issued = float(observable_snapshot.get("issued_at", 0.0))
        grouped = {"self_entities": {}, "opponent_entities": {}}
        for entity_id, entity in (observable_snapshot.get("entities") or {}).items():
            role = entity.get("role")
            if role not in ("self", "opponent"):
                role = "self" if entity.get("team") == self.team else "opponent"
            item = dict(entity)
            item.setdefault("entity_id", str(entity_id))
            item["role"] = role
            item.setdefault("team", self.team if role == "self" else
                            ("蓝" if self.team == "红" else "红"))
            grouped[f"{role}_entities"][str(entity_id)] = item
        example = {
            "side": self.team,
            "issued_at": issued,
            "observable": grouped,
        }
        opponent_behavior_context = self._opponent_behavior_context(example)
        self._behavior_observable_history.append(deepcopy(example))
        if getattr(self, "_feature_schema", "legacy") == "motion":
            raw_values = list(self._raw_history)
            previous = raw_values[-1] if raw_values else None
            previous_previous = raw_values[-2] if len(raw_values) >= 2 else None
            vector = self._motion_feature_vector(example, previous, previous_previous)
            self._raw_history.append(example)
        else:
            vector = self._feature_vector(example)
        canonical_mode_vector = np.asarray(vector, dtype=np.float32).copy()
        # New checkpoints use the side-aware 232-field schema.  Keep old
        # checkpoints runnable as rollback artifacts by projecting the causal
        # vector to their recorded input width instead of failing on shape.
        model_dim = int(getattr(self._model, "input_dim", vector.shape[0]))
        if vector.shape[0] > model_dim:
            vector = vector[:model_dim]
        elif vector.shape[0] < model_dim:
            padded = self._np.zeros(model_dim, dtype=self._np.float32)
            padded[:vector.shape[0]] = vector
            vector = padded
        self.history.append(vector)
        # Keep the full causal 232-field row for the independently trained
        # LightGBM global-mode expert.  The main checkpoint may truncate or pad
        # its own input for rollback compatibility, so this history is kept
        # before that projection.
        if canonical_mode_vector.shape[0] == 232:
            self._canonical_mode_lgbm_history.append(canonical_mode_vector)
        window = self._np.zeros((self.context, vector.shape[0]), dtype=self._np.float32)
        values = list(self.history)
        window[-len(values):] = self._np.asarray(values, dtype=self._np.float32)
        tensor = self._torch.from_numpy(window[None]).to(self._device)
        with self._torch.no_grad():
            outputs = self._model(tensor)
        baseline_now = self._baseline_labels(example)
        global_mode = self._classification(
            {h: outputs[h]["global_mode"][0] for h in self._horizons},
            self._labels["global_mode"], self._calibrate,
            baseline_labels=baseline_now, task="global_mode",
            blend_config=self._persistence_blend)
        canonical_mode_meta = {
            "enabled": bool(self.apply_canonical_mode_lgbm),
            "applied": False,
            "source": "neural",
        }
        if self.apply_canonical_mode_lgbm and self._canonical_mode_lgbm is not None:
            try:
                mode_idx = self._labels["global_mode"].index(baseline_now["global_mode"])
                refined = self._canonical_mode_lgbm.predict(
                    list(self._canonical_mode_lgbm_history), issued, mode_idx)
                for horizon in self._horizons:
                    row = dict(global_mode[str(horizon)])
                    row["neural_baseline"] = dict(row)
                    row.update(refined[str(horizon)])
                    row["source"] = "canonical_mode_lightgbm_locked_test_pass"
                    comparison = row.get("model_vs_baseline") or {}
                    row["model_top"] = row.get("top")
                    row["baseline_top"] = comparison.get("baseline_top")
                    row["model_confidence"] = row.get("confidence")
                    row["baseline_confidence"] = comparison.get("baseline_confidence")
                    row["blend_alpha"] = comparison.get("alpha")
                    row["selected_source"] = row["source"]
                    global_mode[str(horizon)] = row
                canonical_mode_meta.update({
                    "applied": True,
                    "source": "canonical_mode_lightgbm_locked_test_pass",
                    "manifest": self.canonical_mode_lgbm_path,
                })
            except Exception as exc:
                canonical_mode_meta["error"] = str(exc)
        elif self._canonical_mode_lgbm_error:
            canonical_mode_meta["error"] = self._canonical_mode_lgbm_error
        opponent_trend = self._classification(
            {h: outputs[h]["opponent_trend"][0] for h in self._horizons},
            self._labels["opponent_trend"], self._calibrate,
            baseline_labels=baseline_now, task="opponent_trend",
            blend_config=self._persistence_blend)
        residual_calibration_meta = None
        if self.opponent_residual_calibrator is not None:
            trend_head = opponent_trend.get("10", {})
            corrected, residual_calibration_meta = self.opponent_residual_calibrator.apply(
                trend_head.get("probabilities"), opponent_behavior_context)
            if corrected is not None:
                trend_head["probabilities"] = {
                    label: round(float(probability), 6)
                    for label, probability in corrected.items()
                }
                trend_head["top"] = max(corrected, key=corrected.get)
                trend_head["confidence"] = round(float(corrected[trend_head["top"]]), 6)
                trend_head["online_calibration"] = residual_calibration_meta
        enemy_region = self._classification(
            {h: outputs[h]["enemy_region"][0] for h in self._horizons},
            self._labels["enemy_region"], self._calibrate,
            baseline_labels=baseline_now, task="enemy_region",
            blend_config=self._persistence_blend)
        global_task_meta = {
            "enabled": bool(self.apply_global_task_lgbm),
            "applied": False,
            "source": "neural",
        }
        if self.apply_global_task_lgbm and self._global_task_lgbm is not None:
            try:
                from forecast.temporal import REGION_LABELS, TRENDS
                mode_index = self._labels["global_mode"].index(baseline_now["global_mode"])
                region_index = (REGION_LABELS.index(baseline_now["enemy_region"])
                                if baseline_now.get("enemy_region") in REGION_LABELS else -1)
                refined_tasks = self._global_task_lgbm.predict(
                    list(self._canonical_mode_lgbm_history), issued,
                    mode_index, region_index)
                for horizon in self._horizons:
                    refined = refined_tasks.get(str(horizon), {})
                    for task, head in (("opponent_trend", opponent_trend),
                                       ("enemy_region", enemy_region)):
                        if task not in refined:
                            continue
                        row = dict(head[str(horizon)])
                        row["neural_baseline"] = dict(row)
                        row.update(refined[task])
                        row["model_top"] = row.get("top")
                        row["baseline_top"] = (row.get("model_vs_baseline") or {}).get("baseline_top")
                        row["model_confidence"] = row.get("confidence")
                        row["baseline_confidence"] = (row.get("model_vs_baseline") or {}).get("baseline_confidence")
                        row["blend_alpha"] = (row.get("model_vs_baseline") or {}).get("alpha")
                        row["selected_source"] = row.get("source")
                        head[str(horizon)] = row
                global_task_meta.update({
                    "applied": True,
                    "source": "global_task_lightgbm_locked_test_pass",
                    "manifest": self.global_task_lgbm_path,
                })
            except Exception as exc:
                global_task_meta["error"] = str(exc)
        elif self._global_task_lgbm_error:
            global_task_meta["error"] = self._global_task_lgbm_error
        enemy_region_transition_meta = {
            "enabled": bool(self.apply_enemy_region_transition),
            "applied": False,
            "source": "existing_enemy_region_head",
            "by_horizon": {},
        }
        if self.apply_enemy_region_transition and self._enemy_region_transitions:
            try:
                from forecast.temporal import REGION_LABELS
                region_index = (REGION_LABELS.index(baseline_now["enemy_region"])
                                if baseline_now.get("enemy_region") in REGION_LABELS else -1)
                mode_index = self._labels["global_mode"].index(baseline_now["global_mode"])
                for horizon, candidate in self._enemy_region_transitions.items():
                    key = str(horizon)
                    if key not in enemy_region:
                        continue
                    refined = candidate.predict(
                        base_region=region_index, side=self.team,
                        issued_at=issued, base_mode=mode_index)
                    row = dict(enemy_region[key])
                    row["neural_baseline"] = dict(row)
                    row.update(refined)
                    comparison = row.get("model_vs_baseline") or {}
                    row["model_top"] = row.get("top")
                    row["baseline_top"] = comparison.get("baseline_top")
                    row["model_confidence"] = row.get("confidence")
                    row["baseline_confidence"] = comparison.get("baseline_confidence")
                    row["blend_alpha"] = comparison.get("alpha_transition")
                    row["selected_source"] = row.get("source")
                    enemy_region[key] = row
                    enemy_region_transition_meta["by_horizon"][key] = {
                        "applied": True,
                        "source": row.get("source"),
                        "manifest": candidate.manifest_path,
                    }
                enemy_region_transition_meta["applied"] = bool(
                    enemy_region_transition_meta["by_horizon"])
                if enemy_region_transition_meta["applied"]:
                    enemy_region_transition_meta["source"] = "enemy_region_transition_locked_test_pass"
            except Exception as exc:
                enemy_region_transition_meta["error"] = str(exc)
        elif self._enemy_region_transition_error:
            enemy_region_transition_meta["error"] = self._enemy_region_transition_error
        entity_motion_candidate = None
        entity_motion_meta = {
            "enabled": bool(self.apply_entity_motion_lgbm),
            "applied": False,
            "source": "neural",
        }
        if self.apply_entity_motion_lgbm and self._entity_motion_lgbm is not None:
            try:
                entity_motion_candidate = self._entity_motion_lgbm.predict(
                    list(self._canonical_mode_lgbm_history))
                entity_motion_meta.update({
                    "applied": True,
                    "source": "entity_motion_lightgbm_locked_test_candidate",
                    "manifest": self.entity_motion_lgbm_path,
                })
            except Exception as exc:
                entity_motion_meta["error"] = str(exc)
        elif self._entity_motion_lgbm_error:
            entity_motion_meta["error"] = self._entity_motion_lgbm_error
        current_entity_context = {}
        current_entities = []
        for role_key in ("self_entities", "opponent_entities"):
            for entity in grouped[role_key].values():
                current_entities.append(entity)
                slot_key = f"{'self' if role_key == 'self_entities' else 'opponent'}:{entity.get('entity_type')}"
                current_entity_context.setdefault(slot_key, entity)
        # Keep an explicit current-facts section for downstream decision
        # making.  The forecast object itself is often the only artifact
        # handed to Tier2, so without this section the language model can
        # over-focus on future probabilities and underweight what is visible
        # now.  The input snapshot has already been visibility-masked by the
        # caller; we copy only operationally relevant fields here.
        current_state_units = []
        for entity in current_entities:
            state = entity.get('special_state') or {}
            row = {
                'entity_id': entity.get('entity_id'),
                'side': entity.get('role'),
                'entity_type': entity.get('entity_type'),
                'position': entity.get('position'),
                'yaw': entity.get('yaw'),
                'speed_mps': entity.get('speed_mps'),
                'hp': entity.get('hp'),
                'hp_ratio': entity.get('hp_ratio'),
                'life_state': entity.get('life_state'),
                'functional_zone': entity.get('functional_zone') or entity.get('current_functional_zone'),
                'canonical_zone': entity.get('canonical_zone'),
                'attack_status': state.get('attack_status'),
                'target': state.get('target'),
                'resource_phase': state.get('resource_phase'),
                'has_energy_unit': state.get('has_energy_unit'),
            }
            current_state_units.append({k: v for k, v in row.items()
                                        if v is not None})
        position_fusion_positions = None
        position_motion_vectors = None
        if self.apply_validation_position_fusion:
            previous_position_example = (self._position_history[-1]
                                         if self._position_history else None)
            position_fusion_positions, position_motion_vectors = self._causal_group_centroid_motion(
                example, previous_position_example)
            self._position_history.append(example)
        horizons = {}
        decode_current_positions = (position_fusion_positions
                                    if self.apply_validation_position_fusion
                                    else self._current_positions(self.history[-1]))
        for horizon in self._horizons:
            zone_logits = None
            canonical_zone_logits = None
            if "entity_zone_state" in outputs[horizon]:
                zone_logits = outputs[horizon]["entity_zone_state"][0].detach().cpu().numpy()
            if "entity_canonical_zone_state" in outputs[horizon]:
                canonical_zone_logits = outputs[horizon]["entity_canonical_zone_state"][0].detach().cpu().numpy()
            entity_state_values = outputs[horizon]["entity_state"][0].detach().cpu().numpy().copy()
            candidate_horizon = ((entity_motion_candidate or {}).get(str(horizon))
                                 if entity_motion_candidate is not None else None)
            if candidate_horizon is not None:
                # Use the sidecar's validation-selected position blend. The
                # neural presence/HP/count fields stay authoritative.
                def _logit(value):
                    value = min(1.0 - 1e-5, max(1e-5, float(value)))
                    return math.log(value / (1.0 - value))
                positions = np.asarray(candidate_horizon["positions"], dtype=np.float32)
                if positions.shape == (len(entity_state_values), 2):
                    for index, point in enumerate(positions):
                        entity_state_values[index, 1] = _logit(point[0])
                        entity_state_values[index, 2] = _logit(point[1])
                # Retain the categorical sidecar as an auditable candidate;
                # decode still applies the legal adjacency/speed projection.
                from worldstate.functional_zone import FUNCTIONAL_ZONE_LABELS
                from worldstate.canonical_zone import CANONICAL_ZONE_LABELS
                side_zones = candidate_horizon.get("functional_zones") or []
                side_canonical = candidate_horizon.get("canonical_zones") or []
                if side_zones:
                    zone_logits = np.full((len(side_zones), len(FUNCTIONAL_ZONE_LABELS)), -12.0, dtype=np.float32)
                    for index, label in enumerate(side_zones):
                        if label in FUNCTIONAL_ZONE_LABELS:
                            zone_logits[index, FUNCTIONAL_ZONE_LABELS.index(label)] = 12.0
                if side_canonical:
                    canonical_zone_logits = np.full((len(side_canonical), len(CANONICAL_ZONE_LABELS)), -12.0, dtype=np.float32)
                    for index, label in enumerate(side_canonical):
                        if label in CANONICAL_ZONE_LABELS:
                            canonical_zone_logits[index, CANONICAL_ZONE_LABELS.index(label)] = 12.0
            entities = self._decode_entities(
                entity_state_values,
                observing_side=self.team,
                current_positions=decode_current_positions,
                current_entity_context=current_entity_context,
                current_entities=current_entities,
                horizon_seconds=horizon,
                zone_logits=zone_logits,
                canonical_zone_logits=canonical_zone_logits,
                position_fusion_weights=(VALIDATION_POSITION_FUSION_WEIGHTS
                                         if self.apply_validation_position_fusion else None),
                position_motion_vectors=position_motion_vectors)
            if candidate_horizon is not None:
                # Keep the sidecar's zone decision visible to downstream
                # decision logic, while the decoder's actionable route stays
                # legal and speed constrained.
                side_zones = candidate_horizon.get("functional_zones") or []
                side_canonical = candidate_horizon.get("canonical_zones") or []
                from forecast.entity import _entity_speed_mps, _reachable_target_zone
                from worldstate.functional_zone import transition_record
                for index, ((group, entity_type), item) in enumerate(zip(ENTITY_SLOTS, entities.values())):
                    if index < len(side_zones):
                        item["entity_motion_lgbm_zone"] = side_zones[index]
                    if index < len(side_canonical):
                        item["entity_motion_lgbm_canonical_zone"] = side_canonical[index]
                    item["entity_motion_lgbm_position_applied"] = bool(
                        float(candidate_horizon.get("blend_alpha", 0.0)) > 0.0)
                    target_zone = (side_zones[index] if index < len(side_zones)
                                   else None)
                    from_zone = item.get("from_zone")
                    team = item.get("team")
                    if target_zone and from_zone and team in ("红", "蓝"):
                        from_xy = decode_current_positions.get(
                            f"{group}:{entity_type}")
                        speed = float(item.get("speed_mps") or
                                      _entity_speed_mps(
                                          entity_type,
                                          current_entity_context.get(
                                              f"{group}:{entity_type}", {}))[0])
                        raw_position = item.get("position") or {}
                        raw_xy = ((raw_position.get("x"), raw_position.get("y"))
                                  if isinstance(raw_position, dict) else raw_position)
                        chosen, projected, route_distance, travel_time, path = _reachable_target_zone(
                            entity_type, team, from_zone, target_zone,
                            raw_xy, from_xy, horizon, speed)
                        item["to_zone"] = chosen
                        movement = transition_record(
                            entity_type, team, from_zone, chosen,
                            probability=item.get("presence_probability"))
                        movement.update({
                            "raw_to_zone": target_zone,
                            "projection_applied": bool(projected),
                            "path_zones": list(path),
                            "path_hops": max(0, len(path) - 1),
                            "distance_to_target_m": route_distance,
                            "estimated_travel_time_s": travel_time,
                            "reachable_within_horizon": True,
                            "route_constraint": "entity_motion_lgbm_zone_with_speed_x_horizon",
                        })
                        item["movement"] = movement
            # The learned output is slot/type aggregated. Unknown/stale
            # tracks and unchanged positions are not evidence of death.
            suppress_destroyed_entity_forecasts(entities, current_entities)
            outlook = _derive_global_outlook(
                entities, global_mode[str(horizon)],
                opponent_trend[str(horizon)], enemy_region[str(horizon)],
                self.team, horizon_seconds=horizon)
            horizons[str(horizon)] = {
                "global_mode": global_mode[str(horizon)],
                "opponent_trend": opponent_trend[str(horizon)],
                "enemy_region": enemy_region[str(horizon)],
                "entities": entities,
                "global_outlook": outlook,
            }
        observable_hash = hashlib.sha256(
            repr(observable_snapshot).encode("utf-8")).hexdigest()
        output = {
            "schema_version": "temporal-entity-forecast-v1",
            "position_contract": {
                "zone_schema": "functional-zone-rmuc2026-v1",
                "canonical_zone_schema": "canonical-zone-rmuc2026-v1",
                "movement_field": "from_zone -> to_zone",
                "raw_zone_retained": True,
                "decoder_candidate": ("validation_causal_group_velocity_fusion_v1"
                                      if self.apply_validation_position_fusion
                                      else "legacy_reverse_only_correction"),
                "decoder_candidate_enabled": bool(self.apply_validation_position_fusion),
                "model_weight_by_horizon": ({f"{h}s": w for h, w in VALIDATION_POSITION_FUSION_WEIGHTS.items()}
                                             if self.apply_validation_position_fusion else None),
                "velocity_source": ("observable_group_centroid_xy_exact_3s_difference; yaw excluded"
                                    if self.apply_validation_position_fusion else None),
                "entity_motion_expert": ("locked_test_lightgbm_zone_and_short_position_sidecar"
                                          if entity_motion_meta.get("applied") else None),
            },
            "special_state_contract": {
                "schema": "specialized-unit-state-v1",
                "unknown_policy": "missing radar/sensor fields remain unknown",
                "inference_policy": "possible targets may use visible geometry and are marked as inferred",
            },
            "model_version": getattr(self._model, "architecture_name", "temporal_gru_entity_v1"),
            "calibration_version": getattr(self, "_calibration_version",
                                             "monotonic_horizon_v2_temperature"),
            "probability_fusion": (self._persistence_blend or {
                "schema": "none", "causal": True}),
            "team": self.team,
            "opponent_behavior_context": opponent_behavior_context,
            "issued_at": issued,
            "input_snapshot_hash": observable_hash,
            "causal_history_length": len(self.history),
            "current_state": {
                "issued_at": issued,
                "units": current_state_units,
                "visibility": "sensor_radar_masked_snapshot_only",
                "future_rows_used_at_issue_time": False,
            },
            "horizons": horizons,
            "global_mode_expert": canonical_mode_meta,
            "global_task_expert": global_task_meta,
            "enemy_region_transition_expert": enemy_region_transition_meta,
            "entity_motion_expert": entity_motion_meta,
        }

        if self.experimental_mode_xgb_path:
            candidate = {"status": "unavailable", "production_applied": False}
            try:
                if self._experimental_mode_xgb is None and self._experimental_mode_xgb_error is None:
                    from forecast.mode_change_xgb import ModeChangeXGBPredictor
                    self._experimental_mode_xgb = ModeChangeXGBPredictor(
                        self.experimental_mode_xgb_path,
                        persistence_alpha=self.experimental_mode_xgb_alpha)
                if self._experimental_mode_xgb_error is not None:
                    raise RuntimeError(self._experimental_mode_xgb_error)
                from forecast.mode_change_xgb import (build_latest_mode_change_features,
                                                       observable_snapshot_base_features)
                # Build the candidate's 20D frame summary from the same masked
                # snapshot given to this runtime; history is same match + side.
                prev_time = (self._experimental_mode_xgb_history[-1][0]
                             if self._experimental_mode_xgb_history else None)
                base, mode_idx = observable_snapshot_base_features(example, prev_time)
                self._experimental_mode_xgb_history.append((issued, base, mode_idx))
                hist = list(self._experimental_mode_xgb_history)
                x = build_latest_mode_change_features(
                    np.asarray([row[1] for row in hist], dtype=np.float32),
                    np.asarray([row[0] for row in hist], dtype=np.float64),
                    np.asarray([row[2] for row in hist], dtype=np.int64))
                probs = self._experimental_mode_xgb.predict_proba(
                    x, np.asarray([mode_idx], dtype=np.int64))[0]
                probability_map = {label: round(float(prob), 6)
                                   for label, prob in zip(self._labels["global_mode"], probs)}
                top = max(probability_map, key=probability_map.get)
                candidate = {
                    "status": "ok", "horizon_seconds": 30,
                    "top": top, "confidence": probability_map[top],
                    "probabilities": probability_map,
                    "feature_count": int(x.shape[0]),
                    "history_length": len(hist),
                    "persistence_alpha": self.experimental_mode_xgb_alpha,
                    "production_applied": False,
                }
            except Exception as exc:
                # Candidate failure must not affect the production forecast.
                candidate = {"status": "unavailable", "production_applied": False,
                             "reason": str(exc)}
                self._experimental_mode_xgb_error = str(exc)
            if self.apply_experimental_mode_xgb:
                candidate["production_applied"] = apply_mode_xgb_candidate(
                    horizons, candidate, team=self.team)
            output["experimental_candidates"] = {
                "global_mode_30s_xgb": candidate,
            }

        if self.adaptive_router is not None:
            # A persistence baseline must be made from fields available at
            # issue time.  Deployments may provide a complete distribution in
            # ``adaptive_baseline``; for convenience, scalar current labels
            # in ``current_labels`` are expanded to a one-hot distribution
            # over the model's visible classes.
            baseline = observable_snapshot.get("adaptive_baseline")
            if baseline is None:
                baseline = observable_snapshot.get("persistence_baseline")
            current_labels = (observable_snapshot.get("current_labels") or
                              observable_snapshot.get("current_state") or {})
            if baseline is None and not current_labels:
                # This is the same causal persistence label used by the
                # dataset builder.  It is computed from the just-observed
                # snapshot, never from the matured target.
                current_labels = self._baseline_labels(example)
            if isinstance(current_labels, dict):
                baseline = dict(baseline or {})
                for task, label in current_labels.items():
                    if task not in ("global_mode", "opponent_trend", "enemy_region"):
                        continue
                    baseline.setdefault(task, {})
                    for horizon, head in horizons.items():
                        if isinstance(label, dict):
                            current = label.get(str(horizon), label.get(int(horizon)))
                        else:
                            current = label
                        if current is not None:
                            labels = list((head.get(task) or {}).get("probabilities", {}))
                            if current in labels:
                                residual = ((1.0 - ADAPTIVE_PERSISTENCE_CONFIDENCE) /
                                            max(1, len(labels) - 1))
                                baseline[task][str(horizon)] = {
                                    name: (ADAPTIVE_PERSISTENCE_CONFIDENCE
                                           if name == current else residual)
                                    for name in labels
                                }
            output = self.adaptive_router.apply(output, baselines=baseline)
        self._register_observable_forecast(output, observable_snapshot)
        output["adaptive_feedback"] = {
            "mode": "observable_prequential" if self.auto_mature_feedback else "disabled",
            "latest_maturity": matured_feedback,
            "match_totals": self.feedback_status(),
        }
        return output

    def _mature_observable_feedback(self, observable_snapshot):
        """Score due forecasts from this tick's sensor/radar snapshot only."""
        self._last_observable_feedback = {
            "matured_forecast_records": 0,
            "matured_task_horizon_outcomes": 0,
        }
        if (not self.auto_mature_feedback or
                (self.adaptive_router is None and self.opponent_residual_calibrator is None)):
            return dict(self._last_observable_feedback)
        from forecast.temporal import HORIZONS
        from runtime.observable_feedback import feedback_labels_from_observable

        now = float(observable_snapshot.get("issued_at", 0.0))
        for forecast_id in list(self._observable_feedback_order):
            record = self._observable_feedback_pending.get(forecast_id)
            if record is None:
                continue
            due = [h for h in HORIZONS
                   if h not in record["scored"] and
                   now + 1e-6 >= record["issued_at"] + float(h)]
            if not due:
                continue
            truth = {task: {} for task in ("global_mode", "opponent_trend", "enemy_region")}
            for horizon in due:
                labels = feedback_labels_from_observable(
                    record["snapshot"], observable_snapshot, self.team, horizon)
                for task in truth:
                    if labels.get(task) is not None:
                        truth[task][str(horizon)] = labels[task]
            if self.adaptive_router is not None:
                result = self.adaptive_router.observe_outcome(
                    forecast_id, truth, matured_horizons=due)
                updates = int(result.get("updated", 0))
            else:
                updates = 0
            if (self.opponent_residual_calibrator is not None and 10 in due
                    and truth["opponent_trend"].get("10") is not None):
                calibration_result = self.opponent_residual_calibrator.observe(
                    record.get("opponent_trend_probabilities"),
                    record.get("opponent_behavior_context", "unknown"),
                    truth["opponent_trend"]["10"],
                )
                updates += int(bool(calibration_result.get("updated")))
            record["scored"].update(due)
            self._last_observable_feedback["matured_task_horizon_outcomes"] += updates
            if len(record["scored"]) == len(HORIZONS):
                self._observable_feedback_pending.pop(forecast_id, None)
                self._last_observable_feedback["matured_forecast_records"] += 1
        self._observable_feedback_counts["completed_forecast_records"] += (
            self._last_observable_feedback["matured_forecast_records"])
        self._observable_feedback_counts["matured_task_horizon_outcomes"] += (
            self._last_observable_feedback["matured_task_horizon_outcomes"])
        return dict(self._last_observable_feedback)

    def feedback_status(self):
        """Return match-local feedback counts without exposing hidden labels."""
        return {
            "enabled": bool(self.auto_mature_feedback),
            **self._observable_feedback_counts,
            "pending_forecast_records": len(self._observable_feedback_pending),
            "opponent_residual_samples": (
                self.opponent_residual_calibrator.sample_count("hold_or_mixed")
                if self.opponent_residual_calibrator is not None
                and hasattr(self.opponent_residual_calibrator, "sample_count") else 0),
            "last_tick": dict(self._last_observable_feedback),
        }

    def _register_observable_forecast(self, output, observable_snapshot):
        if (not self.auto_mature_feedback or
                (self.adaptive_router is None and self.opponent_residual_calibrator is None)):
            return
        forecast_id = ((output.get("adaptive_strategy") or {}).get("forecast_id")
                       if isinstance(output, dict) else None)
        if not forecast_id:
            issued = float(observable_snapshot.get("issued_at", 0.0))
            material = f"{issued}|{output.get('input_snapshot_hash', '')}"
            forecast_id = hashlib.sha256(material.encode("utf-8")).hexdigest()[:20]
        self._observable_feedback_pending[str(forecast_id)] = {
            "issued_at": float(observable_snapshot.get("issued_at", 0.0)),
            "snapshot": deepcopy(observable_snapshot),
            "scored": set(),
            "opponent_trend_probabilities": deepcopy(
                (((output.get("horizons") or {}).get("10") or {})
                 .get("opponent_trend") or {}).get("probabilities") or {}),
            "opponent_behavior_context": output.get("opponent_behavior_context", "unknown"),
        }
        self._observable_feedback_order.append(str(forecast_id))
        while len(self._observable_feedback_order) > 4096:
            old = self._observable_feedback_order.popleft()
            self._observable_feedback_pending.pop(old, None)

    def observe_outcome(self, forecast_id, truth, *, matured_horizons=None):
        """Forward only matured horizons to the optional adaptive router."""
        if self.adaptive_router is None:
            return {"forecast_id": str(forecast_id), "updated": 0,
                    "reason": "adaptive_router_disabled"}
        result = self.adaptive_router.observe_outcome(
            forecast_id, truth, matured_horizons=matured_horizons)
        record = self._observable_feedback_pending.get(str(forecast_id))
        if record is not None:
            if matured_horizons is None:
                record["scored"].update((10, 30, 60, 120))
            else:
                record["scored"].update(int(h) for h in matured_horizons)
            if len(record["scored"]) == 4:
                self._observable_feedback_pending.pop(str(forecast_id), None)
        return result


def resolve_temporal_forecaster(checkpoint="auto", *, device="auto", team="红",
                                context=12, adaptive_router=None,
                                auto_mature_feedback=None,
                                opponent_residual_calibrator=None,
                                experimental_mode_xgb_path="auto",
                                apply_experimental_mode_xgb=False,
                                apply_validation_position_fusion=False,
                                canonical_mode_lgbm_path="auto",
                                apply_canonical_mode_lgbm=True,
                                entity_motion_lgbm_path="auto",
                                apply_entity_motion_lgbm=True,
                                global_task_lgbm_path="auto",
                                apply_global_task_lgbm=True,
                                enemy_region_transition_path="auto",
                                apply_enemy_region_transition=True):
    """Create the temporal runtime or return ``None`` when disabled.

    The locked-test-passed canonical-side LightGBM expert is enabled for the
    global-mode head when its manifest is present. The older XGBoost artifact
    remains an optional experimental candidate.
    Online route adaptation is opt-in because current validation replay found
    that the available strategy-switching policy can reduce probability
    quality. Pass ``adaptive_router="auto"`` to enable it; due forecasts then
    mature automatically from sequential observable snapshots unless disabled.
    """
    if checkpoint in (None, "", "none", "null", "-"):
        return None
    if checkpoint == "auto":
        # Keep the verified canonical-side neural model for entity/trend/region
        # heads.  The locked-test-passed LightGBM expert refines only the
        # global-mode head; other optional candidates remain sidecars.
        path = (DEFAULT_CHECKPOINT if Path(DEFAULT_CHECKPOINT).exists()
                else FALLBACK_CHECKPOINT)
    elif checkpoint == "hybrid":
        path = HYBRID_CHECKPOINT
    else:
        path = str(checkpoint)
    if adaptive_router == "auto":
        from runtime.adaptive_strategy import AdaptiveStrategyRouter
        adaptive_router = AdaptiveStrategyRouter()
    if opponent_residual_calibrator == "auto":
        from runtime.adaptive_strategy import MatchLocalOpponentResidualCalibrator
        opponent_residual_calibrator = MatchLocalOpponentResidualCalibrator()
    if experimental_mode_xgb_path == "auto":
        experimental_mode_xgb_path = (MODE_XGB_CHECKPOINT
                                      if Path(MODE_XGB_CHECKPOINT).exists() else None)
    if canonical_mode_lgbm_path == "auto":
        canonical_mode_lgbm_path = (CANONICAL_MODE_LGBM_MANIFEST
                                    if Path(CANONICAL_MODE_LGBM_MANIFEST).exists() else None)
    if entity_motion_lgbm_path == "auto":
        entity_motion_lgbm_path = (ENTITY_MOTION_LGBM_MANIFEST
                                   if Path(ENTITY_MOTION_LGBM_MANIFEST).exists() else None)
    if global_task_lgbm_path == "auto":
        global_task_lgbm_path = (GLOBAL_TASK_LGBM_MANIFEST
                                 if Path(GLOBAL_TASK_LGBM_MANIFEST).exists() else None)
    if enemy_region_transition_path == "auto":
        enemy_region_transition_path = (ENEMY_REGION_TRANSITION_MANIFEST
                                        if Path(ENEMY_REGION_TRANSITION_MANIFEST).exists() else None)
    return TemporalForecastRuntime(path, device=device, team=team, context=context,
                                   adaptive_router=adaptive_router,
                                   auto_mature_feedback=auto_mature_feedback,
                                   opponent_residual_calibrator=opponent_residual_calibrator,
                                   experimental_mode_xgb_path=experimental_mode_xgb_path,
                                   apply_experimental_mode_xgb=apply_experimental_mode_xgb,
                                   apply_validation_position_fusion=apply_validation_position_fusion,
                                   canonical_mode_lgbm_path=canonical_mode_lgbm_path,
                                   apply_canonical_mode_lgbm=apply_canonical_mode_lgbm,
                                   entity_motion_lgbm_path=entity_motion_lgbm_path,
                                   apply_entity_motion_lgbm=apply_entity_motion_lgbm,
                                   global_task_lgbm_path=global_task_lgbm_path,
                                   apply_global_task_lgbm=apply_global_task_lgbm,
                                   enemy_region_transition_path=enemy_region_transition_path,
                                   apply_enemy_region_transition=apply_enemy_region_transition)
