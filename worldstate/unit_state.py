# -*- coding: utf-8 -*-
"""Unit-type-specific state sidecars for causal forecasts and narratives."""
from __future__ import annotations

import math
from typing import Any

from worldstate.functional_zone import (
    assign_functional_zone, candidate_moves, functional_zone_neighbors,
    zone_centroid,
)

_COMBAT_TYPES = {'英雄', '步兵3', '步兵4', '哨兵', '空中'}
_POSSIBLE_ATTACK_RANGE_M = {
    '英雄': 8.0,
    '步兵3': 6.0,
    '步兵4': 6.0,
    '哨兵': 7.0,
    '空中': 10.0,
}


def _position(entity):
    value = entity.get('position') if isinstance(entity, dict) else None
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        try:
            x, y = float(value[0]), float(value[1])
            if math.isfinite(x) and math.isfinite(y):
                return x, y
        except (TypeError, ValueError):
            pass
    return None


def _zone(entity):
    if isinstance(entity, dict) and entity.get('functional_zone'):
        return str(entity['functional_zone'])
    position = _position(entity or {})
    return assign_functional_zone(*position) if position is not None else '场地未知区'


def _same_team(entity, self_team):
    return entity.get('team') == self_team or entity.get('role') == 'self'


def _visible_enemies(entity, entities, self_team):
    result = []
    origin = _position(entity)
    # For an opponent slot, the relevant friendly team is the opponent's
    # team, not the observing side.  Falling back to the observing side keeps
    # synthetic/offline entities usable when they do not carry ``team``.
    entity_team = entity.get('team') or (self_team if entity.get('role') == 'self' else None)
    for candidate in entities or []:
        if not isinstance(candidate, dict) or candidate is entity:
            continue
        if entity_team is not None and candidate.get('team') == entity_team:
            continue
        if entity_team is None and _same_team(candidate, self_team):
            continue
        position = _position(candidate)
        if position is None:
            continue
        distance = math.hypot(position[0] - origin[0], position[1] - origin[1]) if origin else float('inf')
        result.append((distance, candidate))
    return sorted(result, key=lambda item: (item[0], str(item[1].get('entity_id', ''))))


def _target_record(candidate, distance=None, reason=None):
    if candidate is None:
        return None
    value = {
        'entity_id': candidate.get('entity_id'),
        'team': candidate.get('team'),
        'entity_type': candidate.get('entity_type'),
    }
    if distance is not None:
        value['distance_m'] = round(float(distance), 3)
    if reason:
        value['reason'] = reason
    return value


def _explicit_target(entity, entities):
    keys = ('target_entity_id', 'target_id', 'aim_target_id')
    ids = [entity.get(key) for key in keys if entity.get(key) is not None]
    id_set = {str(value) for value in ids}
    for candidate in entities or []:
        if str(candidate.get('entity_id')) in id_set:
            return candidate
    target_type = entity.get('target_type') or entity.get('aim_target_type')
    if target_type:
        for candidate in entities or []:
            if candidate.get('entity_type') == target_type and not _same_team(candidate, entity.get('team')):
                return candidate
    return None


def _attack_signal(entity):
    for key in ('is_attacking', 'attacking', 'firing', 'firing_inferred'):
        if isinstance(entity.get(key), bool):
            return entity[key]
    for key in ('dfire', 'ammo_delta', 'shots_delta'):
        value = entity.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0:
            return True
    # Heat is a retained/decaying telemetry value, not proof of a shot in the
    # current frame.  Treating ``small_heat > 0`` as attacking would turn
    # ordinary movement or a previous shot into a false direct assertion.
    return None


def _combat_state(entity, entities, self_team):
    enemies = _visible_enemies(entity, entities, self_team)
    explicit = _explicit_target(entity, entities)
    signal = _attack_signal(entity)
    attack_range = _POSSIBLE_ATTACK_RANGE_M.get(entity.get('entity_type'), 6.0)
    in_range = next((pair for pair in enemies if pair[0] <= attack_range), None)
    target_pair = (0.0, explicit) if explicit is not None else in_range
    # A visible enemy outside the conservative range is evidence that the
    # unit is not currently in an inferred attack window.  With no visible
    # enemy at all we cannot distinguish ``not attacking`` from ``unknown``.
    possible = True if target_pair else (False if enemies else None)
    possible_target = _target_record(
        target_pair[1], target_pair[0],
        'explicit_target' if explicit is not None else 'nearest_visible_enemy') if target_pair else None
    if signal is True:
        # A firing signal proves that the robot fired, not which nearby enemy
        # its projectile targeted. Only protocol target fields can establish
        # an attack target; geometry remains a candidate for guidance.
        status = 'attacking'
        attack_target = (_target_record(explicit, 0.0, 'explicit_target')
                         if explicit is not None else None)
    elif possible is True:
        status, attack_target = 'possible_attack', None
    elif signal is False:
        status, attack_target = 'not_attacking', None
    else:
        status, attack_target = 'unknown', None
    energy_flag = entity.get('attacking_energy_unit')
    if not isinstance(energy_flag, bool):
        # A nearby energy unit is not proof it was the object being attacked.
        target = explicit
        target_type = (target or {}).get('entity_type') if target else entity.get('target_type')
        energy_flag = True if isinstance(target_type, str) and ('能量' in target_type or '能源' in target_type) else None
    return {
        'attack_status': status,
        'is_attacking': True if status == 'attacking' else (False if status == 'not_attacking' else None),
        'possible_attack': possible,
        'attacking_target': attack_target,
        'possible_target': possible_target,
        'attacking_energy_unit': energy_flag,
        'evidence': 'direct_sensor_field' if signal is not None else ('visible_geometry_inference' if possible is True else 'unavailable'),
    }


def _engineer_state(entity, team, zone):
    own_supply = '红补给区' if team == '红' else '蓝补给区'
    has = entity.get('has_energy_unit')
    if not isinstance(has, bool):
        for key in ('carrying_energy_unit', 'carrying_ore', 'has_ore'):
            if isinstance(entity.get(key), bool):
                has = entity[key]
                break
    if has is None and isinstance(entity.get('ore_count'), (int, float)):
        has = entity['ore_count'] > 0
    exchanging = entity.get('is_exchanging')
    if not isinstance(exchanging, bool):
        for key in ('exchanging', 'delivering', 'exchanging_ore', 'at_exchange'):
            if isinstance(entity.get(key), bool):
                exchanging = entity[key]
                break
    collecting = entity.get('is_collecting_energy_unit')
    if not isinstance(collecting, bool):
        collecting = entity.get('collecting_energy_unit')
        if not isinstance(collecting, bool):
            collecting = None
    # The resource point can vary by mode/map. Do not invent its zone when
    # telemetry or the selected task does not identify it.
    target_zone = entity.get('resource_target_zone')
    # The return destination is different: the team's own supply zone is a
    # known, fixed map landmark. If carrying/exchanging is explicitly true,
    # it is safe to route toward that zone even when the mine/task is unknown.
    if target_zone is None and (has is True or exchanging is True):
        target_zone = own_supply
    if exchanging is True:
        status = 'exchanging'
        next_action = 'exchange_or_deliver'
    elif has is True:
        status = 'carrying_energy_unit' if zone != own_supply else 'at_supply_area_with_energy_unit'
        next_action = 'return_to_supply_area' if zone != own_supply else 'exchange_or_deliver'
    elif has is False:
        if collecting is True:
            status, next_action = 'collecting_energy_unit', 'continue_collection'
        elif target_zone and zone == target_zone:
            status, next_action = 'at_resource_area', 'collect_energy_unit'
        elif target_zone:
            status, next_action = 'going_to_resource_area', 'go_to_resource_area'
        else:
            status, next_action = 'unknown', 'unknown'
    else:
        status, next_action = 'unknown', 'unknown'
    route = {
        'from_zone': zone,
        'to_zone': target_zone,
        'description': (f'工程：{zone} → {target_zone}' if target_zone
                        else f'工程：{zone} → 目标区域未知'),
    }
    at_supply = zone == own_supply
    if exchanging is True or (at_supply and has is True):
        phase = 'exchanging'
        direction = 'exchange_or_deliver'
    elif has is True:
        phase = 'returning_to_exchange'
        direction = 'to_supply_area'
    elif has is False and collecting is True:
        phase = 'collecting_energy_unit'
        direction = 'collect_at_resource_area'
    elif has is False and target_zone and zone == target_zone:
        phase = 'collecting_energy_unit'
        direction = 'collect_at_resource_area'
    elif has is False and target_zone:
        phase = 'going_to_mine'
        direction = 'to_resource_area'
    else:
        phase = 'unknown'
        direction = 'unknown'
    return {
        'resource_status': status,
        'has_energy_unit': has,
        'has_mineral': has,
        'is_exchanging': exchanging,
        'is_collecting_energy_unit': collecting,
        'is_at_supply_area': at_supply,
        'resource_phase': phase,
        'resource_direction': direction,
        'resource_target_zone': target_zone,
        'resource_next_action': next_action,
        'resource_route': route,
        'evidence': 'direct_sensor_field' if has is not None else 'unavailable',
    }


def _health_ratio(entity):
    value = entity.get('hp_ratio')
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return max(0.0, min(1.0, float(value)))
    # Raw RM telemetry uses ``maxhp``; a few normalized callers use
    # ``max_hp``. Accept both so low-health guidance works on the actual feed.
    hp = entity.get('hp')
    maximum = entity.get('max_hp', entity.get('maxhp'))
    if (isinstance(hp, (int, float)) and not isinstance(hp, bool)
            and isinstance(maximum, (int, float)) and not isinstance(maximum, bool)
            and maximum > 0):
        return max(0.0, min(1.0, float(hp) / float(maximum)))
    return None


def _path_to_goal(unit, team, start, goal):
    """Shortest legal functional-zone path, used only when both ends are known."""
    if not start or not goal:
        return []
    queue = [(start, [start])]
    seen = {start}
    while queue:
        zone, path = queue.pop(0)
        if zone == goal:
            return path
        for nxt in candidate_moves(unit, team, zone):
            if nxt == zone or nxt in seen:
                continue
            seen.add(nxt)
            queue.append((nxt, path + [nxt]))
    return []


def _guidance_record(record, action_label, short_text):
    """Attach a stable machine label and concise evidence-aware user copy."""
    record['action_label'] = action_label
    record['short_text'] = short_text
    # This is the compact, one-glance contract consumed by the replay UI.
    # Keep it separate from the longer evidence and evaluation fields.
    actions = {
        'retreat_low_hp': record.get('action'),
        'continue_attack_visible_target': '目标仍可见且自身安全时继续压制',
        'cover_and_confirm_target': '留在当前区域确认目标，不单独追击',
        'return_resource_to_supply': record.get('action'),
        'continue_resource_route': record.get('action'),
        'guard_defense_zone': record.get('action'),
        'unit_destroyed': '停止行动建议和预测',
        'no_action_evidence': '暂不建议具体移动或攻击',
    }
    reasons = {
        'retreat_low_hp': '低血量，先降低暴露',
        'continue_attack_visible_target': '有开火记录；目标仍需确认',
        'cover_and_confirm_target': '只有距离推断，目标未确认',
        'return_resource_to_supply': '已确认携带资源，先完成兑换',
        'continue_resource_route': '资源任务状态已记录',
        'guard_defense_zone': '观测到明确守卫职责',
        'unit_destroyed': '单位已确认阵亡',
        'no_action_evidence': '任务或交战状态未确认',
    }
    record['display_summary'] = {
        'action': actions.get(action_label) or record.get('action') or '暂不建议具体移动或攻击',
        'reason': reasons.get(action_label) or record.get('action_reason') or '当前观测依据不足',
        'reassess_when': record.get('cancel_condition') or record.get('trigger_condition') or '关键状态发生变化',
    }
    return record


def _opponent_current_description(unit, zone, hp_ratio, state):
    """Describe enemy facts and inferences without upgrading either to commands."""
    parts = [f'敌方{unit}位于{zone}']
    if hp_ratio is not None:
        parts.append(f'血量约{hp_ratio:.0%}')

    attack = state.get('attack_status', 'unknown')
    target = state.get('attacking_target') or state.get('possible_target') or {}
    if attack == 'attacking':
        if state.get('attacking_target'):
            parts.append(f'有开火记录，传感器目标为{target.get("entity_type") or "已指定单位"}')
        else:
            parts.append('有开火记录，但攻击目标未确认')
    elif attack == 'possible_attack':
        if target:
            parts.append(f'附近约{target.get("distance_m", "未知")}米有{target.get("entity_type") or "可见单位"}，仅是可能目标')
        else:
            parts.append('可能处于交战状态，目标未知')
    elif attack == 'not_attacking':
        parts.append('当前未观测到开火')

    if unit == '工程':
        phase = state.get('resource_phase')
        phase_text = {
            'returning_to_exchange': '正携带资源返程',
            'collecting_energy_unit': '正在采集资源',
            'going_to_mine': '正前往已知资源目标区',
            'exchanging': '正在兑换资源',
        }.get(phase)
        if phase_text:
            parts.append(phase_text)
        elif state.get('has_energy_unit') is None:
            parts.append('是否持有资源未知')
    if unit == '空中':
        flight = state.get('flight_status')
        if flight and flight != 'unknown':
            parts.append(f'飞行状态：{flight}')
    if unit == '哨兵' and state.get('defense_evidence') == 'functional_zone_prior':
        parts.append('该区域只提示可能的防守职责，未确认防守任务')
    return '；'.join(parts) + '。'


def _opponent_state_forecast(unit, state):
    """Expose conservative state priors while explicitly leaving movement unknown."""
    horizons = {}
    for seconds in (10, 30):
        horizons[str(seconds)] = forecast_special_state(unit, state, seconds)
    return {
        'kind': 'probabilistic_unit_state_prior',
        'method': 'specialized_state_prior_v1',
        'horizons': horizons,
        'movement': {
            'status': 'unknown',
            'reason': '单兵状态侧车不含该单位的时序位置输出；不能由敌方当前位置推出移动方向',
        },
        'interpretation': '这是状态延续先验，不是已证实动作或已校准概率；后续需用回放观测核验。',
    }


def _decision_scorecard(record, action_label, unit, team, current_zone):
    """Separate action feasibility, execution evidence, outcome, and causality.

    The score is only a route-feasibility check against the functional-zone
    graph. It is never presented as a probability of success or decision
    accuracy. Replay data has no controller command acknowledgements.
    """
    action = record.get('action')
    if not action:
        feasibility = {'status': 'not_applicable', 'score': None,
                       'reason': 'no_action_recommendation'}
    else:
        goal = record.get('movement_goal_zone')
        path = record.get('path_zones') or []
        if goal is None:
            feasibility = {'status': 'not_required', 'score': None,
                           'reason': 'recommendation_does_not_request_zone_change'}
        elif not path:
            feasibility = {'status': 'unknown', 'score': None,
                           'reason': 'route_path_not_provided'}
        else:
            legal = path[0] == current_zone and path[-1] == goal
            for left, right in zip(path, path[1:]):
                legal = legal and right in candidate_moves(unit, team, left)
            legal = legal and (record.get('first_step_zone') == (path[1] if len(path) > 1 else current_zone))
            feasibility = {
                'status': 'scored' if legal else 'failed',
                'score': 100.0 if legal else 0.0,
                'metric': 'functional_zone_route_legality',
                'path_zones': list(path),
                'estimated_travel_time_s': None,
                'travel_time_status': 'unknown',
                'reason': ('all route transitions are legal' if legal
                           else 'route endpoints or transitions do not match the legal zone graph'),
            }

    outcome_metric = {
        'cover_and_confirm_target': ['current_zone_retention', 'own_firing_event', 'own_survival', 'own_hp_change'],
        'continue_attack_visible_target': ['own_firing_event'],
        'retreat_low_hp': ['progress_toward_safe_zone', 'own_survival', 'own_hp_change'],
        'return_resource_to_supply': ['resource_arrival', 'resource_exchange_completion'],
        'continue_resource_route': ['resource_arrival', 'resource_collection'],
        'guard_defense_zone': ['defense_zone_occupancy', 'facility_damage'],
    }.get(action_label, [])
    if action:
        outcome = {'status': 'pending', 'horizons_seconds': [3, 10],
                   'metrics': outcome_metric,
                   'interpretation': 'observed outcome only; not a measure of execution or causal benefit'}
    else:
        outcome = {'status': 'unknown', 'horizons_seconds': [3, 10],
                   'metrics': [], 'reason': 'no_action_recommendation'}
    if action_label in ('cover_and_confirm_target',):
        outcome['target_confirmation'] = {
            'status': 'unknown',
            'reason': 'future radar visibility/confirmation field is not available',
        }
    if action_label == 'continue_attack_visible_target':
        # Only a protocol-confirmed target can be linked to target HP/survival.
        # A geometrically nearest enemy is not proof of the projectile target.
        explicit_target = (record.get('evaluation') or {}).get('target_linkage') == 'explicit_sensor_target'
        if explicit_target:
            outcome['metrics'] += ['identified_target_hp_change', 'target_alive']
            outcome['target_linkage'] = 'explicit_sensor_target'
        else:
            outcome['target_linkage'] = 'unconfirmed_candidate_only'
            outcome['target_hp_metrics_status'] = 'not_scored_target_identity_not_confirmed'
    if action_label in ('return_resource_to_supply', 'continue_resource_route'):
        outcome['resource_completion'] = {
            'status': 'unknown',
            'reason': 'per-robot carrying/exchange telemetry is absent from historical rows',
        }
    execution = {
        'status': 'unknown' if action else 'not_applicable',
        'coverage': 0.0 if action else None,
        'reason': ('controller command/acknowledgement logs are unavailable'
                   if action else 'no action recommendation'),
        'observed_behavior_proxy_is_execution': False,
    }
    causal = {'status': 'not_identifiable' if action else 'not_applicable',
              'reason': ('replay does not intervene on control commands; outcome may be caused by other units/events'
                         if action else 'no action recommendation')}
    return {
        'feasibility': feasibility,
        'execution': execution,
        'consequence': outcome,
        'causal_benefit': causal,
    }


def derive_unit_guidance(entity: dict, special_state: dict | None = None,
                         entities: list[dict] | None = None,
                         self_team: str = '红') -> dict[str, Any]:
    """Evidence-linked, conditional suggestion for a friendly robot.

    This is advisory metadata, not an execution command. It separates the
    observed state, action/target, evidence, trigger and cancellation rule so
    later replay evaluation can score each part. Missing target zones remain
    unknown; all reported first steps come from the functional-zone graph.
    """
    entity = dict(entity or {})
    state = dict(special_state or {})
    team = entity.get('team') or self_team
    unit = entity.get('entity_type') or entity.get('type') or '未知'
    zone = state.get('functional_zone') or _zone(entity)
    hp_ratio = _health_ratio(entity)
    alive = hp_ratio is None or hp_ratio > 0
    def _record(record, action_label, short_text):
        # Stable subject key lets row-wise evaluators join advice to this exact
        # robot instead of accidentally pooling same-type units.
        record.setdefault('unit_ref', {
            'entity_id': entity.get('entity_id'),
            'team': team,
            'entity_type': unit,
        })
        goal = record.get('first_step_zone')
        position = _position(entity)
        center = zone_centroid(goal) if goal else None
        if goal == zone:
            distance = 0.0
        elif position is not None and center is not None:
            distance = round(math.hypot(center[0] - position[0], center[1] - position[1]), 2)
        else:
            distance = None
        record.setdefault('first_step_distance_m', distance)
        if 'movement_goal_zone' not in record:
            record['movement_goal_zone'] = record.get('target_zone')
        record['decision_scorecard'] = _decision_scorecard(
            record, action_label, unit, team, zone)
        if record.get('role') == 'opponent':
            record.setdefault('future_prediction', _opponent_state_forecast(unit, state))
            record['output_role'] = 'enemy_state_and_forecast'
        elif record.get('action'):
            record.setdefault('current_state_description', record.get('current_state'))
            reason = record.get('action_reason') or '当前观测到的信息支持重新评估该动作；理由细节见证据字段。'
            record['output_role'] = 'friendly_recommendation'
            record['recommendation'] = {
                'type': 'suggestion_not_prediction',
                'action': record['action'],
                'reason': reason,
                'trigger_condition': record.get('trigger_condition'),
                'cancel_condition': record.get('cancel_condition'),
            }
        else:
            record.setdefault('current_state_description', record.get('current_state'))
            record['output_role'] = 'friendly_state_only'
        return _guidance_record(record, action_label, short_text)
    evidence = []
    if hp_ratio is not None:
        evidence.append({'field': 'hp_ratio', 'value': round(hp_ratio, 4), 'source': 'current_observation'})
    evidence.append({'field': 'functional_zone', 'value': zone, 'source': 'current_position'})

    if not _same_team(entity, self_team):
        return _record({
            'applicable': False, 'role': 'opponent',
            'current_state': _opponent_current_description(unit, zone, hp_ratio, state),
            'current_state_description': _opponent_current_description(unit, zone, hp_ratio, state),
            'future_prediction': _opponent_state_forecast(unit, state),
            'action': None, 'target_zone': None, 'first_step_zone': None,
            'evidence': evidence, 'trigger_condition': None, 'cancel_condition': None,
            'evaluation': {'scorable': False, 'reason': 'enemy_forecast_not_own_action'},
        }, 'not_our_unit', '敌方单位只做状态观察与预测。')

    if not alive:
        return _record({
            'applicable': True, 'role': 'self', 'current_state': '已确认阵亡',
            'action': '停止行动建议并停止预测', 'target_zone': None, 'first_step_zone': None,
            'evidence': evidence,
            'trigger_condition': '观测到生命值大于0或明确复活事件',
            'cancel_condition': '复活后重新生成行动建议',
            'evaluation': {'scorable': True, 'metric': 'life_state', 'expected': 'destroyed'},
        }, 'unit_destroyed', '已确认阵亡；停止给出行动建议。')

    # A low-health retreat is only suggested if a legal own-side neighbor is
    # known. Prefer the one that shortens the graph path to the team's base.
    if hp_ratio is not None and hp_ratio < 0.30:
        base = '红基地' if team == '红' else '蓝基地'
        choices = [target for target in candidate_moves(unit, team, zone)
                   if target != zone and target.startswith(team)]
        ranked = []
        for target in choices:
            path = _path_to_goal(unit, team, target, base)
            if path:
                ranked.append((len(path), target, path))
        if ranked:
            _, target, path = min(ranked)
            return _record({
                'applicable': True, 'role': 'self', 'current_state': f'血量偏低（{hp_ratio:.0%}），位于{zone}',
                'action_reason': f'当前血量比例{hp_ratio:.0%}低于撤离触发阈值30%；选择图上可通行、且更靠近本方基地的首步区域。',
                'action': f'先向{target}撤离；需继续观察是否脱离威胁', 'target_zone': target,
                'movement_goal_zone': target,
                'first_step_zone': target, 'path_zones': [zone, target],
                'continuation_path_zones': path,
                'evidence': evidence + [{'field': 'legal_neighbor', 'value': target,
                                         'source': 'functional_zone_graph'}],
                'trigger_condition': '当前HP比例低于0.30',
                'cancel_condition': 'HP比例恢复到0.50以上，或目标相邻区不再可通行',
                'evaluation': {'scorable': True, 'metric': 'next_zone', 'expected': target,
                               'threshold': 'next observed functional zone equals first_step_zone'},
            }, 'retreat_low_hp',
                f'血量{hp_ratio:.0%}偏低；建议先向{target}撤离，后续再确认是否脱离威胁。')

    attack = state.get('attack_status')
    target = state.get('attacking_target') or state.get('possible_target')
    if attack == 'attacking' and target:
        target_id = target.get('entity_id')
        target_entity = (next((other for other in (entities or [])
                               if str(other.get('entity_id')) == str(target_id)), None)
                         if target_id is not None else None)
        target_zone = _zone(target_entity) if target_entity else None
        explicit_target = target.get('reason') == 'explicit_target'
        evidence.append({'field': 'attack_status', 'value': 'attacking',
                         'source': state.get('evidence', 'current_observation')})
        if target.get('entity_id') is not None:
            evidence.append({'field': 'target_entity_id', 'value': target.get('entity_id'),
                             'source': ('current_observation' if explicit_target
                                        else 'visible_geometry_inference')})
        action = '仅在目标仍可确认且自身安全时继续压制'
        current_state = f'正在攻击，当前位置{zone}'
        short_text = (f'当前有开火记录，目标{target.get("entity_type") or "可见单位"}仍需确认；'
                      '只在目标可见且自身安全时继续压制。')
        trigger = '目标仍可见且我方HP比例不低于0.30'
        if unit == '英雄':
            target_label = ('已指定目标' if explicit_target else '视野内候选目标')
            current_state = f'刚有42mm发弹记录，{target_label}为{target.get("entity_type") or "未知单位"}，当前位置{zone}'
            action = (f'继续跟踪{target_label}；再次开火前确认目标和自身安全')
            if explicit_target:
                short_text = (f'刚有42mm开火记录，传感器指定目标为{target.get("entity_type") or "未知单位"}；'
                              '继续跟踪，下一次开火前确认目标仍可见且自身安全。')
            else:
                short_text = ('刚有42mm开火记录；附近敌人只是候选目标，需继续跟踪并确认目标和自身安全，'
                              '下一次开火前再次核对。')
            trigger = f'{target_label}仍可见且我方HP比例不低于0.30；每次开火前重新确认'
        if explicit_target:
            action_reason = '当前存在开火记录和传感器指定目标；只有目标仍可见且自身安全时，才建议继续压制。'
        else:
            action_reason = '观测到开火信号，但附近单位只是几何候选目标，无法确认弹药指向；先继续跟踪并核实目标和自身安全。'
        return _record({
            'applicable': True, 'role': 'self', 'current_state': current_state,
            'action': action,
            'action_reason': action_reason,
            'target_unit_id': target.get('entity_id'), 'target_unit_type': target.get('entity_type'),
            'target_zone': target_zone, 'movement_goal_zone': None,
            'first_step_zone': zone, 'path_zones': [zone],
            'first_step_distance_m': 0.0,
            'target_distance_m': target.get('distance_m'),
            'evidence': evidence,
            'trigger_condition': trigger,
            'cancel_condition': '目标连续两次观测不可见、HP比例低于0.30或交战状态解除',
            'evaluation': {'scorable': target.get('entity_id') is not None,
                           'metric': 'identified_target_hp_and_survival',
                           'status': 'descriptive_if_exact_hp_available',
                           'expected_target_id': target.get('entity_id'),
                           'target_linkage': ('explicit_sensor_target' if explicit_target
                                              else 'visible_candidate_only'),
                           'window_seconds': [3, 10],
                           'confounded_by_other_attackers': True,
                           'visibility_and_command_execution': 'unknown'},
        }, 'continue_attack_visible_target', short_text)

    if attack == 'possible_attack':
        target = state.get('possible_target') or {}
        evidence.append({'field': 'possible_target', 'value': target,
                         'source': state.get('evidence', 'visible_geometry_inference')})
        return _record({
            'applicable': True, 'role': 'self', 'current_state': f'附近存在可能交战目标，当前位置{zone}',
            'action': '先留在当前区域确认目标，不单独追击',
            'action_reason': ('附近存在距离范围内的可能目标，但没有确认开火目标或攻击控制字段；'
                              '原地核验可避免把距离推断当成锁定目标。'),
            'target_unit_id': target.get('entity_id'), 'target_unit_type': target.get('entity_type'),
            # A possible target is not proof of usable cover or a movement
            # assignment.  This is a caution/confirmation reminder, not a
            # spatial maneuver recommendation.
            'target_zone': None, 'movement_goal_zone': None,
            'first_step_zone': zone, 'path_zones': [zone],
            'first_step_distance_m': 0.0,
            'target_distance_m': target.get('distance_m'),
            'evidence': evidence,
            'trigger_condition': '出现明确开火/弹药变化，或目标进入有效攻击范围',
            'cancel_condition': '目标确认离开视野，或传感器明确显示没有交战目标',
            'evaluation': {'scorable': False, 'metric': 'enemy_confirmation',
                           'status': 'unknown',
                           'reason': '仅有距离推断；当前没有足够的目标可见性/控制器开火字段',
                           'window_seconds': 3},
        }, 'cover_and_confirm_target',
            f'约{target.get("distance_m", "未知")}米处有疑似目标；先留在{zone}确认，不单独追击。')

    if unit == '工程':
        phase = state.get('resource_phase') or 'unknown'
        evidence.append({'field': 'resource_phase', 'value': phase,
                         'source': state.get('evidence', 'unknown')})
        if phase == 'returning_to_exchange':
            goal = '红补给区' if team == '红' else '蓝补给区'
            path = _path_to_goal(unit, team, zone, goal)
            target = path[1] if len(path) > 1 else (goal if zone == goal else None)
            return _record({
                'applicable': True, 'role': 'self', 'current_state': f'携带资源回兑换，当前位置{zone}',
                'action': '沿已知可通行路线返回本方补给区兑换资源', 'target_zone': goal,
                'action_reason': '传感器已确认携带资源；补给区是已知目的地，建议沿功能区图可达路线返程。',
                'first_step_zone': target, 'path_zones': path,
                'evidence': evidence,
                'trigger_condition': '持有资源为真且尚未完成兑换',
                'cancel_condition': '兑换完成、持有资源变为否，或回程路径被阻断',
                'evaluation': {'scorable': True, 'metric': 'reaches_target_zone',
                               'expected': goal, 'first_step_zone': target},
            }, 'return_resource_to_supply', f'已确认携带资源；建议沿可通行路线先到{target or goal}，再回{goal}兑换。')
        if phase in ('collecting_energy_unit', 'going_to_mine'):
            goal = state.get('resource_target_zone')
            path = _path_to_goal(unit, team, zone, goal) if goal else []
            target = path[1] if len(path) > 1 else (goal if zone == goal else None)
            return _record({
                'applicable': True, 'role': 'self',
                'current_state': ('正在采集资源' if phase == 'collecting_energy_unit'
                                  else '资源目标方向已知' if goal else '资源目标方向未知'),
                'action': '继续资源任务并避开已确认交战区',
                'action_reason': ('当前状态记录处于采集阶段' if phase == 'collecting_energy_unit'
                                  else '资源目标区已知' if goal
                                  else '资源目标区尚未确认，因此暂不建议盲目前往。'),
                'target_zone': goal, 'first_step_zone': target, 'path_zones': path,
                'evidence': evidence + ([{'field': 'resource_target_zone', 'value': goal,
                                          'source': 'task_or_sensor'}] if goal else []),
                'trigger_condition': '资源目标区明确且路线可达' if goal else '等待资源目标区或任务信号明确',
                'cancel_condition': '取得资源后切换返程，或路线受阻/确认遭遇威胁',
                'evaluation': {'scorable': bool(goal), 'metric': 'next_zone',
                               'expected': target, 'reason': None if goal else 'resource_target_zone_unknown'},
            }, 'continue_resource_route',
                f'{"正在采集资源" if phase == "collecting_energy_unit" else "资源目标已知"}；沿路线先到{target or goal}。')

    if (unit == '哨兵'
            and state.get('defense_status') in ('guarding_base', 'guarding_outpost', 'patrol')
            and state.get('defense_evidence') != 'functional_zone_prior'):
        defense = state['defense_status']
        role = {'guarding_base': '守住基地入口',
                'guarding_outpost': '守住前哨周边', 'patrol': '覆盖相邻己方通道'}[defense]
        evidence.append({'field': 'defense_status', 'value': defense,
                         'source': state.get('defense_evidence', 'functional_zone_prior')})
        return _record({
            'applicable': True, 'role': 'self', 'current_state': f'{role}，当前位置{zone}',
            'action': '保持防区，不离开当前区域单独追击',
            'action_reason': '防守状态有明确任务或传感器依据；当前建议是留守覆盖防区，避免单独追击。',
            'target_zone': zone, 'movement_goal_zone': zone,
            'first_step_zone': zone, 'path_zones': [zone],
            'first_step_distance_m': 0.0, 'evidence': evidence,
            'trigger_condition': '敌方进入本防区或附近相邻区域',
            'cancel_condition': '关键设施威胁解除且收到新的防守/支援任务',
            'evaluation': {'scorable': True, 'metric': 'zone_occupancy', 'expected': zone},
        }, 'guard_defense_zone', f'{role}；留在{zone}覆盖防区，敌人进入相邻区时再调整。')

    location_prior = (unit == '哨兵'
                      and state.get('defense_evidence') == 'functional_zone_prior'
                      and state.get('defense_role_prior') in ('guarding_base', 'guarding_outpost', 'patrol'))
    if location_prior:
        evidence.append({'field': 'defense_role_prior', 'value': state.get('defense_role_prior'),
                         'source': 'functional_zone_prior', 'interpretation': 'role_prior_only'})
    return _record({
        'applicable': False, 'role': 'self',
        'current_state': (f'位于{zone}；位置只能提示可能的防守职责' if location_prior
                          else f'位于{zone}，任务/交战状态尚未确认'),
        'action': None, 'target_zone': None, 'first_step_zone': None,
        'evidence': evidence,
        'trigger_condition': ('收到明确防守任务、关键设施告警或确认敌方威胁后重新评估'
                              if location_prior else
                              '确认可见敌方、收到明确任务信号，或观测到可核验的特殊状态后重新评估'),
        'cancel_condition': None,
        'evaluation': {'scorable': False, 'status': 'unknown',
                       'reason': ('location_prior_is_not_a_defense_order' if location_prior
                                  else 'no_action_supported_by_current_evidence')},
    }, 'no_action_evidence',
       ('当前位置仅提示可能承担防守职责；等明确任务或威胁后再建议行动。'
        if location_prior else '暂无足够证据支持具体移动或攻击；先确认任务或可见目标，再重新评估。'))


def derive_unit_special_state(entity: dict, entities: list[dict] | None = None,
                              self_team: str = '红') -> dict[str, Any]:
    """Derive an auditable unit-specific state sidecar from visible fields.

    Missing protocol fields remain ``None``/``unknown``. Opponent attack and
    mining states are never asserted from position alone; only possible targets
    may be inferred from visible geometry.
    """
    entity = dict(entity or {})
    entities = list(entities or [])
    team = entity.get('team', self_team)
    typ = entity.get('entity_type', entity.get('type', '未知'))
    zone = _zone(entity)
    result = {
        'functional_zone': zone,
        'entity_type': typ,
        'observability': 'self_sensor' if _same_team(entity, self_team) else 'opponent_radar',
    }
    if not _same_team(entity, self_team):
        # The radar contract exposes type/XY/HP and selected macro fields, but
        # does not expose these action-state measurements.  Keep the list
        # explicit so downstream narration and evaluation cannot mistake an
        # unknown for a negative observation.
        result['unknown_fields'] = [
            'firing', 'target_entity_id', 'has_energy_unit', 'is_exchanging',
            'z', 'yaw', 'power', 'heat', 'remaining_ammo',
        ]
    if typ == '哨兵':
        result.update(_combat_state(entity, entities, self_team))
        own_prefix = '红' if team == '红' else '蓝'
        if zone == f'{own_prefix}前哨区':
            defense_prior = 'guarding_outpost'
        elif zone == f'{own_prefix}基地':
            defense_prior = 'guarding_base'
        elif zone.startswith(own_prefix):
            defense_prior = 'patrol'
        else:
            defense_prior = 'unknown'
        # Location suggests a possible role but does not prove an issued
        # defense order or active guarding behavior.
        result.update({'defense_status': 'unknown',
                       'defense_role_prior': defense_prior,
                       'defense_target': _target_record(_explicit_target(entity, entities)),
                       'defense_evidence': 'functional_zone_prior'})
    elif typ in _COMBAT_TYPES:
        result.update(_combat_state(entity, entities, self_team))
        if typ == '空中':
            altitude = entity.get('z', entity.get('altitude'))
            if isinstance(altitude, (int, float)):
                flight = 'airborne' if float(altitude) > 0.2 else ('at_landing_pad' if '停机坪' in zone else 'grounded')
            else:
                flight = 'unknown'
            result.update({'flight_status': flight,
                           'flight_target': _target_record(_explicit_target(entity, entities)),
                           'flight_evidence': 'direct_sensor_field' if altitude is not None else 'unavailable'})
    elif typ == '工程':
        result.update(_engineer_state(entity, team, zone))
    elif typ in ('基地', '前哨站', '能量机关', '能量单元', '资源站', '补给区'):
        hp_ratio = entity.get('hp_ratio')
        if isinstance(hp_ratio, (int, float)):
            structure = 'destroyed' if hp_ratio <= 0 else ('damaged' if hp_ratio < 0.5 else 'active')
        else:
            structure = 'unknown'
        under_attack = entity.get('under_attack')
        if not isinstance(under_attack, bool):
            under_attack = None
        result.update({'structure_status': structure,
                       'under_attack': under_attack,
                       'interaction_status': ('under_attack' if under_attack is True
                                              else 'defend_or_attack'),
                       'objective_type': typ})
    else:
        result.update({'state': 'unknown', 'evidence': 'unsupported_unit_type'})
    # Keep unit advice beside (and explicitly grounded in) its observed state.
    # Downstream rendering may choose not to display this metadata, but it can
    # now evaluate the same contract instead of inventing a second heuristic.
    result['decision_guidance'] = derive_unit_guidance(
        entity, result, entities, self_team=self_team)
    return result



def _persistence_distribution(labels, top=None, horizon=10.0):
    """A transparent persistence prior, diffused as the horizon grows."""
    labels = list(labels)
    if not labels:
        return {}
    uniform = 1.0 / len(labels)
    if top not in labels:
        return {label: round(uniform, 6) for label in labels}
    persistence = max(0.0, min(0.8, 0.75 - 0.45 * (float(horizon) / 120.0)))
    rest = (1.0 - persistence) / max(1, len(labels) - 1)
    return {label: round(persistence if label == top else rest, 6) for label in labels}


def forecast_special_state(entity_type: str, current_state: dict | None,
                           horizon_seconds: float) -> dict[str, Any]:
    """Return a unit-specific probabilistic state prior.

    This is intentionally a calibrated persistence prior until specialized
    action labels are available. It never turns missing radar fields into a
    hard assertion.
    """
    current_state = dict(current_state or {})
    horizon = float(horizon_seconds)
    if entity_type == '空中':
        combat = forecast_special_state('英雄', current_state, horizon)
        labels = ('airborne', 'at_landing_pad', 'grounded', 'unknown')
        flight = _persistence_distribution(labels, current_state.get('flight_status'), horizon)
        return {
            'kind': 'air_combat',
            'attack_status_probabilities': combat['attack_status_probabilities'],
            'target': combat.get('target'),
            'attacking_energy_unit_probability': combat.get('attacking_energy_unit_probability'),
            'flight_status_probabilities': flight,
            'flight_target': current_state.get('flight_target'),
            'method': 'specialized_state_prior_v1',
            'source': current_state.get('observability', 'unavailable'),
        }
    if entity_type in _COMBAT_TYPES:
        labels = ('attacking', 'possible_attack', 'not_attacking', 'unknown')
        top = current_state.get('attack_status')
        probabilities = _persistence_distribution(labels, top, horizon)
        return {
            'kind': 'combat',
            'attack_status_probabilities': probabilities,
            'target': current_state.get('attacking_target') or current_state.get('possible_target'),
            'attacking_energy_unit_probability': (
                probabilities.get('attacking', 0.0)
                if current_state.get('attacking_energy_unit') is True else None),
            'method': 'specialized_state_prior_v1',
            'source': current_state.get('observability', 'unavailable'),
        }
    if entity_type == '工程':
        labels = ('carrying_energy_unit', 'at_supply_area_with_energy_unit',
                  'at_resource_area', 'going_to_resource_area',
                  'collecting_energy_unit', 'exchanging', 'unknown',
                  # Legacy states remain accepted so old calibration files
                  # and consumers keep working during migration.
                  'at_supply_area', 'going_to_supply_area', 'returning_to_supply_area')
        top = current_state.get('resource_status')
        return {
            'kind': 'resource',
            'resource_status_probabilities': _persistence_distribution(labels, top, horizon),
            'resource_phase_probabilities': _persistence_distribution(
                ('exchanging', 'collecting_energy_unit', 'going_to_mine',
                 'returning_to_exchange', 'unknown'),
                current_state.get('resource_phase'), horizon),
            'has_energy_unit_probability': (
                _persistence_distribution(('yes', 'no', 'unknown'),
                                          'yes' if current_state.get('has_energy_unit') is True
                                          else ('no' if current_state.get('has_energy_unit') is False else None),
                                          horizon)),
            'resource_target_zone': current_state.get('resource_target_zone'),
            'method': 'specialized_state_prior_v1',
            'source': current_state.get('observability', 'unavailable'),
        }
    if entity_type == '哨兵':
        labels = ('guarding_base', 'guarding_outpost', 'patrol', 'unknown')
        return {
            'kind': 'defense',
            'defense_status_probabilities': _persistence_distribution(
                labels, current_state.get('defense_status'), horizon),
            'defense_target': current_state.get('defense_target'),
            'method': 'specialized_state_prior_v1',
            'source': current_state.get('observability', 'unavailable'),
        }
    if entity_type == '空中':
        labels = ('airborne', 'at_landing_pad', 'grounded', 'unknown')
        return {
            'kind': 'flight',
            'flight_status_probabilities': _persistence_distribution(
                labels, current_state.get('flight_status'), horizon),
            'flight_target': current_state.get('flight_target'),
            'method': 'specialized_state_prior_v1',
            'source': current_state.get('observability', 'unavailable'),
        }
    if entity_type in ('基地', '前哨站', '能量机关', '能量单元', '资源站', '补给区'):
        labels = ('active', 'damaged', 'destroyed', 'unknown')
        return {
            'kind': 'structure',
            'structure_status_probabilities': _persistence_distribution(
                labels, current_state.get('structure_status'), horizon),
            'under_attack_probability': (
                1.0 if current_state.get('under_attack') is True else
                0.0 if current_state.get('under_attack') is False else None),
            'method': 'specialized_state_prior_v1',
            'source': current_state.get('observability', 'unavailable'),
        }
    return {
        'kind': 'generic',
        'state': 'unknown',
        'method': 'specialized_state_prior_v1',
        'source': current_state.get('observability', 'unavailable'),
    }
