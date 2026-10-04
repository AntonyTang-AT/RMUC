"""Evidence-grounded, layered situation and unit advice for RM matches."""
from __future__ import annotations

import math


def _fmt(value, default='未知'):
    return default if value is None or value == '' else str(value)


def _special_text(state):
    state = state or {}
    attack = state.get('attack_status')
    if attack == 'attacking':
        target = state.get('attacking_target') or {}
        target_name = target.get('entity_type') or target.get('objective_type')
        return f'正在攻击{target_name}' if target_name else '正在攻击（目标未确认）'
    if attack == 'possible_attack':
        target = state.get('possible_target') or {}
        target_name = target.get('entity_type') or target.get('objective_type')
        return f'可能攻击{target_name}' if target_name else '处于可能交战范围'
    phases = {
        'collecting_energy_unit': '在资源点取能量单元',
        'going_to_mine': '正在前往资源点',
        'returning_to_exchange': '携带资源返回兑换点',
        'exchanging': '正在兑换资源',
    }
    if state.get('resource_phase') in phases:
        return phases[state['resource_phase']]
    defenses = {'guarding_base': '守卫基地', 'guarding_outpost': '守卫前哨', 'patrol': '巡逻警戒'}
    if state.get('defense_status') in defenses:
        return defenses[state['defense_status']]
    if state.get('flight_status') == 'airborne':
        return '处于飞行状态'
    return ''


def _unit_status_lines(analysis, team):
    state = analysis.get('state') or {}
    own_team = state.get('team')
    locations = {str(item.get('entity_id')): item
                 for item in state.get('unit_locations') or []}
    units = []
    for item in state.get('visible_unit_states') or []:
        unit_team = item.get('team')
        relation = ('己方' if own_team is not None and unit_team == own_team
                    else '敌方' if own_team is not None and unit_team is not None
                    else '归属未知')
        unit_type = _fmt(item.get('type'), '单位')
        zone = _fmt(item.get('zone'), '位置未知')
        location = locations.get(str(item.get('entity_id')), {})
        ratio = location.get('hp_ratio', item.get('hp_ratio'))
        ratio_valid = (isinstance(ratio, (int, float)) and not isinstance(ratio, bool)
                       and math.isfinite(float(ratio)))
        if item.get('life_state') == 'destroyed' or (ratio_valid and float(ratio) <= 0):
            condition = '已阵亡'
        elif not ratio_valid:
            condition = '存活状态未知，当前仍可观测'
        elif float(ratio) < .3:
            condition = '血量危急'
        elif location.get('vulnerable'):
            condition = '处于易伤状态'
        else:
            condition = '仍在场'
        special = '' if condition == '已阵亡' else _special_text(item.get('state'))
        text = f'{relation}{unit_type}在{zone}，{condition}'
        if special:
            text += f'，{special}'
        units.append({
            'entity_id': item.get('entity_id'), 'team': unit_team,
            'relation': relation, 'entity_type': unit_type, 'zone': zone,
            'current_state': text, 'special_state': item.get('state') or {},
            'observability': item.get('observability', 'unavailable'),
            'state_evidence': (item.get('state') or {}).get('evidence', 'unavailable'),
            'life_state': ('destroyed' if condition == '已阵亡' else
                           'unknown' if condition == '存活状态未知，当前仍可观测' else 'active'),
            'hp_ratio': float(ratio) if ratio_valid else None,
        })
    return units


def _unit_decision(unit, zone_control, risk_data):
    """Return a conditional suggestion for own units; never a movement order."""
    if unit.get('life_state') == 'destroyed':
        return '无动作建议', '单位已阵亡，待确认复活后再纳入行动'
    state = unit.get('special_state') or {}
    if unit.get('hp_ratio') is not None and float(unit['hp_ratio']) < .3:
        return None, '血量偏低，但当前数据没有给出合法撤离首步；先不指定目的区，待路线信息明确后重算'
    if unit.get('entity_type') == '工程':
        phase = state.get('resource_phase')
        if phase == 'returning_to_exchange':
            return None, '已知处于资源返程阶段，但没有可核验的目的区和首步路线；待路线字段明确后再建议移动'
        if phase in ('collecting_energy_unit', 'going_to_mine'):
            return None, '资源阶段已知，但缺少明确资源目标区或首步路线；暂不指定移动方向'
        return None, '持有资源、目标区和兑换状态未确认；暂不指定移动动作'
    if unit.get('entity_type') == '哨兵':
        defense = state.get('defense_status')
        if defense == 'guarding_base':
            zone = unit.get('zone') or '当前位置'
            return f'留在{zone}覆盖基地入口；只响应进入防区的可见目标', f'观测到基地守卫职责，当前单位位于{zone}'
        if defense == 'guarding_outpost':
            zone = unit.get('zone') or '当前位置'
            return f'留在{zone}覆盖前哨；目标离开防区后再重新评估', f'观测到前哨守卫职责，当前单位位于{zone}'
        if defense == 'patrol':
            zone = unit.get('zone') or '当前位置'
            return f'在{zone}巡逻；发现可见威胁后再选择相邻支援区', f'观测到巡逻职责，尚无具体威胁目标'
        return None, '没有明确哨兵任务或威胁目标；暂不建议守点或追击'
    if state.get('attack_status') == 'attacking':
        target = state.get('attacking_target') or {}
        target_id = target.get('entity_id')
        target_type = target.get('entity_type') or '传感器指定目标'
        if target_id is not None:
            return (f'只在目标{target_type} #{target_id}仍可见且自身血量不低于30%时继续攻击',
                    '观测到开火记录和指定目标；每次开火前重新确认目标可见与自身血量')
        return None, '观测到开火记录，但没有传感器指定的攻击目标；不把附近敌人当成目标'
    contested = (zone_control or {}).get('contested_zones') or []
    if unit.get('zone') in contested:
        zone = unit.get('zone')
        return f'留在{zone}内掩护并等待友军；不跨区单独追击', f'当前功能区{zone}被标记为争夺区'
    if risk_data.get('enemy_intruders'):
        return None, '发现敌方进入我方防区，但未提供该机器人应去的合法首步；暂不指定移动'
    return None, f'当前位于{unit.get("zone") or "未知区域"}，没有明确任务、目标或路线；暂不建议移动或攻击'


def _forecast_analysis(analysis):
    context = analysis.get('forecast_context') or {}
    predictions = context.get('predictions') or []
    records = []
    for prediction in predictions:
        horizon = int(prediction.get('horizon_seconds') or 0)
        mode = (prediction.get('global_mode') or {}).get('top')
        trend = (prediction.get('opponent_trend') or {}).get('top')
        region = (prediction.get('enemy_region') or {}).get('top')
        mode_conf = float((prediction.get('global_mode') or {}).get('confidence') or 0)
        trend_conf = float((prediction.get('opponent_trend') or {}).get('confidence') or 0)
        region_conf = float((prediction.get('enemy_region') or {}).get('confidence') or 0)
        confidence = max(mode_conf, trend_conf, region_conf)
        outlook = prediction.get('outlook') or {}
        action = (outlook.get('recommended_actions') or [{}])[0] or {}
        if confidence < .45:
            interpretation = '置信度偏低，作为待观察方向，不据此单独调整阵型'
        elif horizon <= 10:
            interpretation = '可用于短时路线和局部交战准备，需和当前可见状态核对'
        elif horizon <= 30:
            interpretation = '用于准备局部控制变化，重点核对目标区域是否可达'
        else:
            interpretation = '只用于判断侧翼压力和总体方向，不按精确位置下达动作'
        records.append({
            'horizon_seconds': horizon, 'likely_mode': mode,
            'likely_enemy_trend': trend, 'likely_enemy_region': region,
            'confidence': round(confidence, 4),
            'mode_confidence': round(mode_conf, 4),
            'trend_confidence': round(trend_conf, 4),
            'region_confidence': round(region_conf, 4),
            'interpretation': interpretation,
            'predicted_action_reference': action.get('objective') or action.get('action'),
            'trigger': action.get('trigger'), 'fallback': action.get('fallback'),
        })
    return records


def _current_focus(own_units, enemy_units, state, risk_data, zone_control):
    """One plain-language fact per side; avoid echoing map labels and counts."""
    own_dead = next((u for u in own_units if u['life_state'] == 'destroyed'), None)
    enemy_dead = next((u for u in enemy_units if u['life_state'] == 'destroyed'), None)
    own_critical = next((u for u in own_units if u.get('hp_ratio') is not None
                         and 0 < float(u['hp_ratio']) < .3), None)
    enemy_intruder = next(iter(risk_data.get('enemy_intruders') or []), None)
    enemy_attack = next((u for u in enemy_units if '正在攻击' in u['current_state']), None)
    own_attack = next((u for u in own_units if '正在攻击' in u['current_state']), None)
    own_outpost = state.get('own_outpost_hp')
    own_base = state.get('own_base_hp')

    if own_dead:
        own = f"我方重点：{own_dead['entity_type']}（{own_dead['zone']}）已确认阵亡，停止对该单位预测"
    elif own_critical:
        own = f"我方重点：{own_critical['entity_type']}位于{own_critical['zone']}且血量危急，应先掩护"
    elif own_outpost is not None and float(own_outpost) <= .02:
        base_text = ('基地完好' if own_base is not None and float(own_base) >= .9
                     else '基地状态未知' if own_base is None
                     else '基地受损')
        own = f'我方重点：前哨已失守、{base_text}，应防止敌方继续深入'
    elif own_base is not None and float(own_base) < .5:
        own = '我方重点：基地受损，防守优先级上升'
    elif own_attack:
        own = f"我方重点：{own_attack['entity_type']}在{own_attack['zone']}正在交战，需确认目标仍可见"
    elif own_units:
        own = '我方重点：暂未发现明确的阵亡或危急单位'
    else:
        own = '我方重点：当前单位状态未知'

    if enemy_dead:
        enemy = f"敌方重点：{enemy_dead['entity_type']}（{enemy_dead['zone']}）已确认阵亡"
    elif enemy_intruder:
        enemy = f"敌方重点：{enemy_intruder.get('type') or '单位'}已进入我方防区的{enemy_intruder.get('zone') or '未知区域'}"
    elif enemy_attack:
        enemy = f"敌方重点：{enemy_attack['entity_type']}在{enemy_attack['zone']}正在攻击"
    elif state.get('enemy_trend') == 'press':
        enemy = '敌方重点：连续位置显示正在向我方推进'
    elif state.get('enemy_trend') == 'retreat':
        enemy = '敌方重点：连续位置显示正在回撤'
    elif int(state.get('opponent_push_count') or 0) > 0:
        enemy = '敌方重点：已有单位越过中线，但攻击目标和下一步路线尚未确认'
    elif enemy_units:
        enemy = '敌方重点：未确认明确攻势，攻击意图仍需观察'
    else:
        enemy = '敌方重点：当前可用信息不足'
    return own, enemy


def _plain_mode(mode):
    """Translate model labels into ordinary language for the prose summary."""
    return {
        '进攻': '我方目前更占主动',
        '防守': '我方目前更需要守住阵地',
        '均衡': '当前暂时看不出明显优势',
    }.get(mode, '当前整体局势尚不明确')


def _special_event_summary(unit_rows, risk_data, zone_control):
    """Summarize high-signal observed events without inventing hidden intent."""
    events = []
    if risk_data.get('enemy_intruders'):
        events.append('已观测到敌方单位进入我方半场，需要关注防线和设施安全')
    if risk_data.get('own_vulnerable_units'):
        events.append('我方有低血量或易受攻击的单位，需要掩护')
    contested = (zone_control or {}).get('contested_zones') or []
    if contested:
        events.append('敌我单位出现在同一争夺区域：' + '、'.join(map(str, contested[:2])))
    for unit in unit_rows:
        current = unit.get('current_state', '')
        if unit.get('life_state') == 'destroyed':
            events.append(f"{unit['relation']}{unit['entity_type']}已确认阵亡")
        elif '正在攻击' in current:
            events.append(f"{unit['relation']}{unit['entity_type']}在{unit['zone']}正在攻击")
        elif '可能攻击' in current:
            events.append(f"{unit['relation']}{unit['entity_type']}附近出现可能交战目标")
    return list(dict.fromkeys(events))[:4]


def _side_key_events(units):
    """Keep the highest-signal unit facts split by side in structured output."""
    result = []
    for unit in units:
        state = unit.get('special_state') or {}
        status = unit.get('current_state') or ''
        if unit.get('life_state') == 'destroyed':
            event = 'confirmed_destroyed'
        elif unit.get('hp_ratio') is not None and 0 < float(unit['hp_ratio']) < .3:
            event = 'critical_health'
        elif state.get('attack_status') == 'attacking':
            event = 'attacking'
        elif state.get('attack_status') == 'possible_attack':
            event = 'possible_contact'
        elif state.get('resource_phase') in ('returning_to_exchange', 'exchanging'):
            event = 'resource_return_or_exchange'
        elif state.get('resource_phase') in ('collecting_energy_unit', 'going_to_mine'):
            event = 'resource_collection'
        elif state.get('defense_status') in ('guarding_base', 'guarding_outpost', 'patrol'):
            event = 'defense_or_patrol_duty'
        else:
            continue
        result.append({
            'entity_id': unit.get('entity_id'),
            'entity_type': unit.get('entity_type'),
            'zone': unit.get('zone'),
            'event': event,
            'event_label': {
                'confirmed_destroyed': '已确认阵亡',
                'critical_health': '血量危急',
                'attacking': '观测到正在攻击',
                'possible_contact': '距离范围内疑似接敌，目标未确认',
                'resource_return_or_exchange': '携带资源返程或正在兑换',
                'resource_collection': '正在前往或采集资源',
                'defense_or_patrol_duty': '承担防守或巡逻职责',
            }[event],
            'description': status,
        })
        if len(result) >= 4:
            break
    return result


def _tactical_stage(unit_rows, state, risk_data, zone_control):
    """Name the most specific tactically useful phase supported by evidence.

    This is a rule-based summary, not a learned tactical classifier. In
    particular, activity alone is not called attrition: that requires outcome
    deltas which are not consistently available in the current snapshot.
    """
    states = [unit.get('special_state') or {} for unit in unit_rows
              if unit.get('life_state') != 'destroyed']
    destroyed = [unit for unit in unit_rows if unit.get('life_state') == 'destroyed']
    if destroyed:
        return {'label': '失去战力', 'basis': 'confirmed_destroyed_unit',
                'evidence': '当前帧确认有单位阵亡', 'confidence': 'high'}

    base_hp = state.get('own_base_hp')
    outpost_hp = state.get('own_outpost_hp')
    if isinstance(base_hp, (int, float)) and float(base_hp) < .35:
        return {'label': '基地应急防守', 'basis': 'observed_own_base_hp',
                'evidence': '当前己方基地血量低于35%', 'confidence': 'high'}
    if isinstance(outpost_hp, (int, float)) and float(outpost_hp) < .35:
        return {'label': '前哨稳固防守', 'basis': 'observed_own_outpost_hp',
                'evidence': '当前己方前哨血量低于35%', 'confidence': 'high'}

    friendly_targets = {}
    for unit in unit_rows:
        state_data = unit.get('special_state') or {}
        if unit.get('relation') == '己方' and state_data.get('attack_status') == 'attacking':
            target_id = (state_data.get('attacking_target') or {}).get('entity_id')
            if target_id is not None:
                friendly_targets[str(target_id)] = friendly_targets.get(str(target_id), 0) + 1
    if any(count >= 2 for count in friendly_targets.values()):
        return {'label': '集火压制',
                'basis': 'multiple_friendly_units_confirmed_same_target',
                'evidence': '当前观察到我方多台单位攻击同一目标',
                'confidence': 'high'}

    trend = state.get('enemy_trend')
    if trend == 'retreat':
        return {'label': '敌方回撤', 'basis': 'observed_enemy_position_trend',
                'evidence': '连续位置显示敌方正在回撤', 'confidence': 'medium'}

    resource_phases = {item.get('resource_phase') for item in states}
    if resource_phases.intersection({'collecting_energy_unit', 'going_to_mine'}):
        return {'label': '抢占资源', 'basis': 'observed_engineer_resource_phase',
                'evidence': '工程兵状态显示正在前往或采集资源点', 'confidence': 'medium'}
    if resource_phases.intersection({'returning_to_exchange', 'exchanging'}):
        return {'label': '资源回收', 'basis': 'observed_engineer_resource_phase',
                'evidence': '工程兵状态显示携带资源返回或正在兑换', 'confidence': 'medium'}

    contested = (zone_control or {}).get('contested_zones') or []
    if contested:
        return {'label': '争夺关键区', 'basis': 'friendly_and_enemy_share_functional_zone',
                'evidence': '敌我单位同时出现在' + '、'.join(map(str, contested[:2])),
                'confidence': 'high'}

    if risk_data.get('enemy_intruders') or trend == 'press':
        return {'label': '推进接敌', 'basis': 'observed_enemy_advance_or_intrusion',
                'evidence': ('当前帧发现敌方进入我方防区' if risk_data.get('enemy_intruders')
                             else '连续位置显示敌方正在向我方推进'),
                'confidence': 'medium'}

    active_attackers = [unit for unit in unit_rows
                        if unit.get('life_state') != 'destroyed'
                        and (unit.get('special_state') or {}).get('attack_status') == 'attacking']
    if active_attackers:
        return {'label': '接敌交战', 'basis': 'observed_attack_status',
                'evidence': '当前观测到单位正在攻击；未据此推断血量交换或战果',
                'confidence': 'medium'}

    possible_attackers = [unit for unit in unit_rows
                          if unit.get('life_state') != 'destroyed'
                          and (unit.get('special_state') or {}).get('attack_status')
                          == 'possible_attack']
    if possible_attackers:
        return {'label': '疑似接敌', 'basis': 'distance_inferred_possible_attack',
                'evidence': '单位进入可能交战距离；目标和实际开火尚未确认',
                'confidence': 'low'}

    defense_roles = {item.get('defense_status') for item in states}
    if 'guarding_base' in defense_roles:
        return {'label': '基地守卫', 'basis': 'observed_sentry_defense_role',
                'evidence': '哨兵状态显示正在守卫基地', 'confidence': 'medium'}
    if 'guarding_outpost' in defense_roles:
        return {'label': '前哨守卫', 'basis': 'observed_sentry_defense_role',
                'evidence': '哨兵状态显示正在守卫前哨', 'confidence': 'medium'}
    if 'patrol' in defense_roles:
        return {'label': '巡逻警戒', 'basis': 'observed_sentry_defense_role',
                'evidence': '哨兵状态显示正在巡逻；未据此推断存在敌方目标',
                'confidence': 'medium'}

    # Do not label a quiet or incomplete snapshot as a standoff, defense,
    # advantage, or attrition without temporal evidence.
    return {'label': '态势观察', 'basis': 'no_specific_tactical_event_confirmed',
            'evidence': '当前帧没有足够证据判定具体战术阶段', 'confidence': 'low'}


def _compact_forecast(forecasts, current_mode, current_trend, threshold=.65):
    candidates = []
    trend_names = {'press': '向我方推进', 'retreat': '回撤', 'hold': '维持当前位置',
                   'unknown': '未知'}
    for item in forecasts:
        horizon = item['horizon_seconds']
        if item.get('likely_mode') and item.get('likely_mode') != current_mode:
            confidence = item.get('mode_confidence', 0)
            if confidence >= threshold:
                candidates.append((confidence, horizon,
                                   f"约{horizon}秒后整体态势可能转为{item['likely_mode']}"))
        trend = item.get('likely_enemy_trend')
        if trend and trend != 'unknown':
            confidence = item.get('trend_confidence', 0)
            if confidence >= threshold:
                if trend == current_trend and trend in ('press', 'retreat', 'hold'):
                    continuation = {'press': '继续向我方推进', 'retreat': '继续回撤',
                                    'hold': '继续维持当前位置'}[trend]
                    statement = f'约{horizon}秒后敌方可能{continuation}'
                else:
                    statement = f"约{horizon}秒后敌方可能{trend_names.get(trend, trend)}"
                candidates.append((confidence, horizon, statement))
        region = item.get('likely_enemy_region')
        if region and item.get('region_confidence', 0) >= threshold:
            candidates.append((item['region_confidence'], horizon,
                               f"约{horizon}秒后敌方可能转向{region}"))
    if not candidates:
        if not forecasts:
            return '当前未接入时序预测，后续方向未知；请以接下来几帧的实际移动判断敌方是否推进或回撤。'
        return '模型虽有输出，但没有达到高置信变化门槛；敌方后续方向仍不确定，请以接下来几帧的实际移动为准。'
    confidence, _, statement = max(candidates, key=lambda row: (row[0], -row[1]))
    strength = '较高' if confidence >= .75 else '中等'
    return f'{statement}，置信度{strength}；这仍是候选判断，需等后续观测确认。'


def _brief_unit_side(units, relation):
    """Short display label for one side; full unit facts stay in evidence."""
    selected = [unit for unit in units if unit.get('relation') == relation]
    if not selected:
        return '暂无可见单位'
    ranked = []
    for unit in selected:
        state = unit.get('current_state') or ''
        zone = unit.get('zone')
        where = f'（{zone}）' if zone and zone != '位置未知' else ''
        if unit.get('life_state') == 'destroyed':
            rank, label = 0, f"{unit['entity_type']}阵亡{where}"
        elif unit.get('hp_ratio') is not None and 0 < float(unit['hp_ratio']) < .3:
            rank, label = 0, f"{unit['entity_type']}低血{where}"
        elif '正在攻击' in state:
            rank, label = 1, f"{unit['entity_type']}交战{where}"
        elif '可能攻击' in state or '可能交战范围' in state:
            rank, label = 2, f"{unit['entity_type']}疑似接敌{where}"
        elif any(word in state for word in ('采集', '持有资源', '兑换')):
            rank, label = 2, f"{unit['entity_type']}执行资源任务{where}"
        else:
            continue
        ranked.append((rank, label))
    details = []
    for _, label in sorted(ranked, key=lambda item: item[0]):
        if label not in details:
            details.append(label)
        if len(details) == 1:
            break
    if not details:
        active = next((unit for unit in selected if unit.get('life_state') != 'destroyed'), None)
        if active:
            location = (f"在{active['zone']}" if active.get('zone')
                        and active.get('zone') != '位置未知' else '在场')
            details.append(f"{active['entity_type']}{location}")
        else:
            details.append('暂无存活单位')
    return '、'.join(details)


def _brief_prediction(forecasts):
    if not forecasts:
        return '预测解析：方向与置信度未知，待后续核对。'
    # Explain each available horizon separately. A single best-overall label
    # hid useful near/mid/far differences; keep one strongest, task-matched
    # prediction per horizon so the brief stays compact and does not mix
    # confidence from unrelated forecast heads.
    trend_names = {
        'press': '敌方可能推进', '推进': '敌方可能推进',
        'retreat': '敌方可能回撤', '回撤': '敌方可能回撤',
        'hold': '敌方可能维持位置', '稳定': '敌方可能维持位置',
        'unknown': None, '未知': None,
    }
    mode_names = {'进攻': '整体态势可能偏进攻', '防守': '整体态势可能偏防守',
                  '均衡': '整体态势可能维持均衡'}
    horizon_uses = {
        10: '短时路线参考', 30: '局部控制准备',
        60: '中期态势参考', 120: '只看总体方向',
    }
    explanations = []
    for forecast in sorted(forecasts,
                           key=lambda row: int(row.get('horizon_seconds') or 0)):
        horizon = forecast.get('horizon_seconds')
        candidates = []
        for field, confidence_field, names, kind in (
                ('likely_enemy_trend', 'trend_confidence', trend_names, '趋势'),
                ('likely_enemy_region', 'region_confidence', {}, '区域'),
                ('likely_mode', 'mode_confidence', mode_names, '整体态势')):
            value = forecast.get(field)
            if value in (None, '', 'unknown', '未知'):
                continue
            confidence = float(forecast.get(confidence_field) or 0.0)
            label = names.get(value, f'敌方可能转向{value}' if kind == '区域' else str(value))
            if label:
                candidates.append((confidence, int(horizon or 0), kind, label))
        when = f'+{int(horizon)}秒' if horizon else '后续'
        if not candidates:
            explanations.append(f'{when}方向不确定')
            continue
        confidence, _, kind, label = max(candidates, key=lambda item: item[0])
        if confidence < .45:
            explanations.append(f'{when}{kind}置信度低，先观察')
            continue
        # The field type remains visible (trend / region / overall posture),
        # and its own probability is printed. Never borrow another head's
        # probability to make this statement sound more certain.
        use = next((text for limit, text in sorted(horizon_uses.items())
                    if int(horizon or 0) <= limit), '仅作远期趋势参考')
        explanations.append(
            f'{when}{label}（{kind}置信度{round(confidence * 100)}%，{use}）')
    if not explanations:
        return '预测解析：方向仍不确定，待后续核对。'
    use = '；'.join(explanations)
    return f'预测解析：时序模型：{use}；各项均为候选，需用实际位置和状态核验。'


def _brief_recommendation(first, posture, objective, trigger):
    """One short action and trigger; detailed rationale stays structured."""
    action = str(first.get('action') or '')
    summaries = {
        'stabilize_contested_zones': ('稳住争夺区', '确认目标后再投入'),
        'pressure_objective': ('压制关键设施', '暴露且路线可达时推进'),
        'resolve_resource_state': ('先确认资源状态', '持矿且路线通畅时返航'),
    }
    if action in summaries:
        objective_text, trigger_text = summaries[action]
    elif objective == '先守住基地入口，确认敌方路线后再组织拦截':
        objective_text, trigger_text = '守住基地入口', '确认敌方进入防区后拦截'
    elif objective == '先确认一个可见目标和可达路线，再集中推进':
        objective_text, trigger_text = '确认目标并集中推进', '目标可见且路线可达时推进'
    elif objective == '稳住基地入口和回防路线，暂不盲目追击':
        objective_text, trigger_text = '守住基地入口', '确认敌方进入防区后拦截'
    else:
        raw_objective = str(objective or '')
        objective_text = (raw_objective if len(raw_objective) <= 13 else
                          raw_objective.split('，', 1)[0].split('；', 1)[0])
        objective_text = objective_text or '稳住关键区域'
        raw_trigger = str(trigger or '')
        trigger_text = (raw_trigger if raw_trigger and len(raw_trigger) <= 14
                        else '条件满足后行动')
    # Preserve the recommendation's actual trigger in plain language. The
    # templates above are a fallback for older rows that did not store one.
    actual_trigger = str(first.get('trigger') or trigger or '')
    if actual_trigger:
        if '可确认的攻击目标' in actual_trigger:
            trigger_text = '确认目标后再投入'
        elif '暴露' in actual_trigger and ('路线' in actual_trigger or '通道' in actual_trigger):
            trigger_text = '目标暴露且路线可达时推进'
        elif '资源' in actual_trigger or '持矿' in actual_trigger:
            trigger_text = '确认持有资源且路线通畅后返航'
        else:
            trigger_text = actual_trigger.replace('连续观测确认', '连续观测到')
            trigger_text = trigger_text.split('，', 1)[0].split('；', 1)[0]
    if action in summaries:
        result = f'建议：{objective_text}；{trigger_text}。'
    else:
        result = f'建议：{objective_text}；{trigger_text}时执行。'
    return result


def _posture_evaluation(posture):
    """Return observable response proxies for a fine-grained posture label."""
    metrics = {
        'base_emergency_defense': ['己方基地血量变化', '敌方进入基地防区的单位数'],
        'deep_interception': ['敌方入侵单位数变化', '己方基地/前哨血量变化'],
        'outpost_defense': ['己方前哨血量变化', '前哨相邻区的敌方出现情况'],
        'contested_zone': ['争夺区控制方变化', '争夺区敌我单位出现情况'],
        'resource_cycle': ['工程资源阶段变化', '是否到达补给区并出现兑换'],
        'frontline_delay': ['敌方功能区推进变化', '敌方连续位置趋势'],
        'protect_vulnerable_units': ['低血/易伤单位的位置变化', '后续HP观测变化'],
        'controlled_pursuit': ['追击单位与友军支援距离', '是否仍沿合法路线移动'],
        'focused_pressure': ['目标设施HP变化', '压制单位的暴露和存活情况'],
        'consolidate_advantage': ['已控制区域是否保持', '关键通道的己方覆盖情况'],
        'layered_facility_defense': ['基地及前哨HP变化', '防区内敌方单位数变化'],
        'formation_and_patrol': ['己方关键通道覆盖情况', '敌方首次接触位置'],
    }
    code = posture.get('code')
    return {
        'category': code or posture.get('label') or 'unspecified',
        'observable_proxies': metrics.get(code, []),
        'window_seconds': [3, 10],
        'status': 'observed_response_proxy' if code in metrics else 'unknown',
        'causal_benefit_claimed': False,
        'limitation': '这些指标只用于核对局势和执行反馈，不证明姿态建议造成了战果。',
    }


def _display_unit_advice(action_label, action, reason, cancel_condition,
                         trigger_condition, short_text, action_reason=None,
                         objective_detail=None):
    """Compact unit advice for a one-glance UI; keep evidence and stop rules separate."""
    reasons = {
        'retreat_low_hp': '血量偏低，先降低暴露',
        'continue_attack_visible_target': '观测到开火，目标仍需确认',
        'cover_and_confirm_target': '只有距离推断，目标未确认',
        'return_resource_to_supply': '已确认携带资源，先完成兑换',
        'continue_resource_route': '资源任务状态已记录',
        'guard_defense_zone': '存在明确守卫职责',
        'unit_destroyed': '单位已阵亡',
        'no_action_evidence': '任务或交战状态未确认',
    }
    default_reassess = {
        'retreat_low_hp': '血量恢复至50%以上或路线受阻',
        'continue_attack_visible_target': '目标连续两次不可见、血量低于30%或交战解除',
        'cover_and_confirm_target': '目标离开视野或确认无目标',
        'return_resource_to_supply': '兑换完成、资源状态改变或路线受阻',
        'continue_resource_route': '取得资源、路线受阻或确认遭遇威胁',
        'guard_defense_zone': '设施威胁解除或收到新任务',
        'unit_destroyed': '确认复活后',
        'no_action_evidence': trigger_condition or '收到明确任务或确认可见目标',
    }
    default_trigger = {
        'retreat_low_hp': '当前HP比例低于30%',
        'continue_attack_visible_target': '目标仍可确认且自身HP比例不低于30%',
        'cover_and_confirm_target': '目标明确可见或出现明确开火记录',
        'return_resource_to_supply': '确认持有资源且尚未兑换',
        'continue_resource_route': '资源任务状态仍有效且路线可通行',
        'guard_defense_zone': '防守职责有效或可见威胁进入防区',
        'unit_destroyed': '观测到HP大于0或明确复活事件',
    }
    action_text = action or short_text or '暂不建议具体移动或攻击'
    if objective_detail and action_text != '暂不建议具体移动或攻击':
        # The narrative summary is what the user sees first. Preserve the
        # exact first step/target from unit_state instead of losing it in the
        # structured evidence drawer.
        if not all(part in action_text for part in objective_detail):
            missing = [part for part in objective_detail if part and part not in action_text]
            if missing:
                action_text = f'{action_text}；{"；".join(missing)}'
    if action_text.endswith('；风险升高时换位'):
        action_text = action_text.removesuffix('；风险升高时换位')
    fallback_reassess = (
        '目标连续两次不可见、血量低于30%或交战解除'
        if '压制' in action_text or '正在攻击' in str(reason or '') else
        '出现可见目标或明确推进信号'
        if '可见目标' in action_text or '推进信号' in action_text else
        '资源状态确认、路线受阻或出现明确威胁'
        if '资源' in action_text or '持矿' in str(reason or '') else
        '发现设施受威胁或收到新任务'
        if '巡逻' in action_text or '设施' in action_text or '守' in action_text else
        '任务、目标或威胁状态变化')
    return {
        'action': action_text,
        # Prefer the unit-state module's action-specific rationale when it is
        # present. The generic label map is only a fallback; otherwise an
        # important condition (for example, target identity is unconfirmed)
        # can disappear from the visible recommendation.
        'reason': reasons.get(action_label) or action_reason or reason or '当前观测依据不足',
        'trigger_when': trigger_condition or default_trigger.get(action_label) or '触发条件未知',
        'stop_when': cancel_condition or default_reassess.get(action_label) or '停止条件未知',
        'reassess_when': cancel_condition or default_reassess.get(action_label) or
                         trigger_condition or fallback_reassess,
    }


def _side_overview(units, side_name, state):
    active = [unit for unit in units if unit.get('life_state') != 'destroyed']
    dead = [unit for unit in units if unit.get('life_state') == 'destroyed']
    if active:
        roster = '、'.join(f"{name}{sum(1 for unit in active if unit['entity_type'] == name)}台"
                          for name in sorted({unit['entity_type'] for unit in active}))
        headline = f'{side_name}目前仍有{roster}在场'
    else:
        headline = f'{side_name}当前没有确认存活的机器人'
    zones = {}
    for unit in active:
        zones[unit['zone']] = zones.get(unit['zone'], 0) + 1
    if zones:
        common = sorted(zones.items(), key=lambda item: (-item[1], item[0]))[:3]
        places = '、'.join(zone for zone, _ in common)
        headline += f'，主要分布在{places}'
    if dead:
        headline += f'；另有{len(dead)}台已确认阵亡'

    own_prefix = 'own' if side_name == '我方' else 'opponent'
    base = state.get(f'{own_prefix}_base_hp')
    outpost = state.get(f'{own_prefix}_outpost_hp')
    def facility(value):
        if value is None:
            return '状态未知'
        if value <= .15:
            return '濒临失守'
        if value < .55:
            return '受损较重'
        if value < .9:
            return '已有损伤'
        return '基本完好'
    if base is not None or outpost is not None:
        headline += f"；基地{facility(base)}，前哨{facility(outpost)}"
    return headline


def build_global_narrative(analysis):
    """Produce the requested three-part global brief plus role-specific unit advice.

    Numbers and raw probabilities remain in the structured evidence. This text
    explains what was observed, what the model predicted, and what the team may
    do conditionally. It does not turn recommendations into executed commands.
    """
    analysis = dict(analysis or {})
    state = dict(analysis.get('state') or {})
    team = _fmt(state.get('team'), '己方')
    opponent = '蓝' if team == '红' else '红'
    # Missing mode output is not evidence of a balanced match. Keep the
    # narrative uncertain instead of silently turning absent input into a
    # tactical classification.
    mode = _fmt(analysis.get('current_mode'), '未知')
    posture = analysis.get('detailed_posture') or {}
    risk_data = analysis.get('risk_register') or state.get('risk_register') or {}
    zone_control = state.get('zone_control') or {}
    unit_rows = _unit_status_lines(analysis, team)
    own_units = [item for item in unit_rows if item['relation'] == '己方']
    enemy_units = [item for item in unit_rows if item['relation'] == '敌方']
    own_critical = [item['current_state'] for item in own_units
                    if item['life_state'] == 'destroyed' or item.get('hp_ratio') is not None
                    and float(item['hp_ratio']) < .3 or '正在攻击' in item['current_state']
                    or '资源点' in item['current_state'] or '兑换' in item['current_state']]
    enemy_critical = [item['current_state'] for item in enemy_units
                      if item['life_state'] == 'destroyed' or '正在攻击' in item['current_state']
                      or '可能攻击' in item['current_state'] or '进入' in item['zone']]
    own_desc, enemy_desc = _current_focus(own_units, enemy_units, state,
                                          risk_data, zone_control)
    highlights = []
    if state.get('control_posture') == 'contested':
        zones = zone_control.get('contested_zones') or []
        zone_text = '、'.join(zones[:3])
        highlights.append('关键区域正在交战' + (f'，重点关注{zone_text}' if zone_text else ''))
    if risk_data.get('enemy_intruders'):
        highlights.append('敌方单位已进入我方防区，优先关注拦截和设施安全')
    if risk_data.get('own_vulnerable_units'):
        highlights.append('己方有低血量或易伤单位，需要掩护或调整位置')
    destroyed = [item for item in unit_rows if item['life_state'] == 'destroyed']
    if destroyed:
        highlights.append('已确认阵亡：' + '、'.join(f"{x['relation']}{x['entity_type']}" for x in destroyed[:4]))
    if not highlights:
        highlights.append('尚无确认的突发事件；继续核对连续帧变化')

    forecasts = _forecast_analysis(analysis)
    forecast_summary = _compact_forecast(
        forecasts, mode, state.get('enemy_trend'), threshold=.65)

    recommendations = list(analysis.get('recommendations') or [])
    first = recommendations[0] if recommendations else {}
    tactical = _fmt(posture.get('label'), mode + '态势')
    if first.get('objective') or posture.get('objective'):
        objective = _fmt(posture.get('objective'), first.get('objective'))
        trigger = _fmt(posture.get('trigger') or first.get('trigger'), '触发条件未知')
    elif state.get('own_outpost_hp') is not None and float(state['own_outpost_hp']) <= .02:
        objective = '先守住基地入口，确认敌方路线后再组织拦截'
        trigger = '连续帧确认敌方向基地推进，或基地血量开始下降'
    elif mode == '进攻':
        objective = '先确认一个可见目标和可达路线，再集中推进'
        trigger = '目标连续可见且规划路径合法、预计十秒内可达'
    elif mode == '防守':
        objective = '稳住基地入口和回防路线，暂不盲目追击'
        trigger = '确认敌方进入防区或关键设施受击'
    elif mode == '均衡':
        objective = '保持当前区域控制，等敌方方向明确后再调整'
        trigger = '连续帧确认推进、回撤或目标交互'
    else:
        objective = '先核对双方位置、设施状态和连续移动，再决定推进或回防'
        trigger = '获得足够的连续观测并确认明确威胁或推进机会'
    # Keep the tactical focus supplied by the structured analysis when no
    # higher-priority, trigger-based recommendation is available.
    if not first and not posture.get('objective') and analysis.get('recommended_focus'):
        objective = str(analysis['recommended_focus'])
    trigger = _fmt(posture.get('trigger') or first.get('trigger'), trigger)
    fallback = _fmt(posture.get('fallback') or first.get('fallback'), '未知')
    balance = analysis.get('force_balance_guidance') or {}
    attack_guidance = _fmt(balance.get('attack'), '未知')
    defense_guidance = _fmt(balance.get('defense'), '未知')
    reserve_guidance = _fmt(balance.get('reserve'), '未知')
    if trigger == '触发条件未知':
        recommendation = f'建议：{objective}；先确认敌我变化后再行动。'
    else:
        recommendation = f'建议：{objective}；当{trigger}时执行。'

    unit_analyses = []
    for unit in unit_rows:
        row = dict(unit)
        if unit['relation'] == '敌方':
            raw_predictions = [entry for entry in
                               ((analysis.get('forecast_context') or {}).get('predictions') or [])
                               if int(entry.get('horizon_seconds') or 0) in (10, 30)]
            future = []
            same_type_count = sum(1 for other in enemy_units
                                  if other['entity_type'] == unit['entity_type'])
            for forecast in raw_predictions:
                for item in forecast.get('entities') or []:
                    if item.get('side') != 'opponent' or item.get('entity_type') != unit['entity_type']:
                        continue
                    predicted_id = item.get('entity_id') or item.get('id')
                    current_id = unit.get('entity_id')
                    # Never attribute one same-type robot's trajectory to all
                    # of its teammates. Fall back to type only when unambiguous.
                    if predicted_id is not None and current_id is not None:
                        if str(predicted_id) != str(current_id):
                            continue
                    elif same_type_count > 1:
                        continue
                    if item.get('forecast_suppressed') or item.get('life_state') == 'destroyed':
                        continue
                    target = item.get('to_zone')
                    if target:
                        probability = item.get('presence')
                        confidence = (f"，出现概率约{float(probability):.0%}"
                                      if isinstance(probability, (int, float)) else '')
                        future.append(
                            f"约{forecast.get('horizon_seconds')}秒后，模型候选路线"
                            f"{item.get('from_zone') or unit['zone']}到{target}{confidence}")
            row['analysis_role'] = 'enemy_forecast'
            row['output_type'] = 'prediction'
            row['future_prediction'] = ('；'.join(future[:2]) + '；仅是候选路线，到时核对'
                                        if future else '暂无可靠的单体路线预测；当前去向未知')
            row['current_status'] = row['current_state']
            row['future_state_predictions'] = [
                {'horizon_seconds': int(forecast.get('horizon_seconds') or 0),
                 'from_zone': item.get('from_zone') or unit['zone'],
                 'to_zone': item.get('to_zone'),
                 'presence_probability': item.get('presence'),
                 'status': 'candidate_route'}
                for forecast in raw_predictions
                for item in (forecast.get('entities') or [])
                if item.get('side') == 'opponent'
                and item.get('entity_type') == unit['entity_type']
                and ((item.get('entity_id') is not None and unit.get('entity_id') is not None
                      and str(item.get('entity_id')) == str(unit.get('entity_id')))
                     or ((item.get('entity_id') is None or unit.get('entity_id') is None)
                         and same_type_count <= 1))
                and item.get('to_zone')
                and not item.get('forecast_suppressed')
                and item.get('life_state') != 'destroyed'
            ]
            row['recommendation'] = None
            row['reason'] = None
            row['recommended_action'] = None
            row['recommendation_reason'] = None
        elif unit['relation'] == '己方':
            action, reason = _unit_decision(unit, zone_control, risk_data)
            guidance = unit.get('special_state', {}).get('decision_guidance') or {}
            observed_state = guidance.get('current_state') or unit.get('current_state')
            if (unit.get('life_state') == 'unknown'
                    and '存活状态未知，当前仍可观测' not in str(observed_state or '')):
                observed_state = ('存活状态未知，当前仍可观测；'
                                  + str(observed_state or unit.get('current_state') or '单位状态未知'))
            action_label = guidance.get('action_label')
            if not action_label and action:
                special = unit.get('special_state') or {}
                if unit.get('life_state') == 'destroyed':
                    action_label = 'unit_destroyed'
                elif unit.get('hp_ratio') is not None and float(unit['hp_ratio']) < .3:
                    action_label = 'retreat_low_hp'
                elif special.get('attack_status') == 'attacking':
                    action_label = 'continue_attack_visible_target'
                elif special.get('attack_status') == 'possible_attack':
                    action_label = 'cover_and_confirm_target'
                elif special.get('resource_phase') == 'returning_to_exchange':
                    action_label = 'return_resource_to_supply'
                elif special.get('resource_phase') in ('collecting_energy_unit', 'going_to_mine'):
                    action_label = 'continue_resource_route'
                elif special.get('defense_status') in ('guarding_base', 'guarding_outpost', 'patrol'):
                    action_label = 'guard_defense_zone'
            if action_label == 'no_action_evidence':
                action = None
                reason = guidance.get('short_text') or '当前观测不足以支持具体行动建议'
            elif guidance.get('applicable'):
                action = guidance.get('action') or action
                # Keep both what was seen and why it supports this specific
                # action. A state label alone is not a recommendation rationale.
                action_reason = guidance.get('action_reason')
                if observed_state and action_reason:
                    reason = f'依据当前观测：{observed_state}；建议理由：{action_reason}'
                elif action_reason:
                    reason = f'建议理由：{action_reason}'
                elif observed_state:
                    reason = f'依据当前观测：{observed_state}'
            first_step = guidance.get('first_step_zone')
            target_zone = guidance.get('movement_goal_zone')
            target_current_zone = guidance.get('target_zone')
            route_hint = (f'，先到{first_step}' if first_step and first_step != unit['zone'] else '')
            if target_zone and target_zone != first_step and target_zone != unit['zone']:
                route_hint += f'，目标区域{target_zone}'
            target_unit_id = guidance.get('target_unit_id')
            target_unit_type = guidance.get('target_unit_type')
            target_distance = guidance.get('target_distance_m')
            if target_unit_id is not None:
                target_hint = f'；对象：{target_unit_type or "可见单位"}（{target_unit_id}）'
                if isinstance(target_distance, (int, float)):
                    target_hint += f'，约{float(target_distance):.1f}米'
                if target_current_zone:
                    target_hint += f'，当前在{target_current_zone}'
            else:
                target_hint = ''
            first_step_distance = guidance.get('first_step_distance_m')
            distance_hint = (f'，首步约{float(first_step_distance):.1f}米'
                             if isinstance(first_step_distance, (int, float))
                             and first_step and first_step != unit['zone'] else '')
            unit_trigger = guidance.get('trigger_condition')
            unit_stop = guidance.get('cancel_condition')
            if action:
                action_text = f'建议：{action}{route_hint}{distance_hint}{target_hint}；依据：{reason}。'
            else:
                action_text = f'暂不给具体行动建议：{reason}。'
            advice_text = (f'现状：{observed_state or unit.get("current_state", "状态未知")}；'
                           f'{action_text}')
            if unit_trigger:
                advice_text += f'满足条件：{unit_trigger}。'
            if unit_stop:
                advice_text += f'停止条件：{unit_stop}。'
            row['future_prediction'] = None
            row['analysis_role'] = 'friendly_recommendation'
            row['output_type'] = 'recommendation'
            row['recommendation_type'] = ('conditional_advice' if action else 'insufficient_action_evidence')
            row['recommendation'] = action
            row['reason'] = reason
            row['current_status'] = observed_state or unit.get('current_state')
            row['recommended_action'] = action
            row['recommendation_reason'] = reason
            row['observed_state'] = observed_state
            row['advice_text'] = advice_text
            row['target_zone'] = target_zone
            row['target_current_zone'] = target_current_zone
            row['first_step_zone'] = first_step
            row['first_step_distance_m'] = first_step_distance
            row['target_unit_id'] = target_unit_id
            row['target_unit_type'] = target_unit_type
            row['target_distance_m'] = target_distance
            row['trigger_condition'] = unit_trigger
            row['cancel_condition'] = unit_stop
            row['decision_evaluation'] = guidance.get('decision_scorecard')
            row['action_metric_evaluation'] = guidance.get('evaluation')
            row['action_label'] = action_label
            row['short_action_text'] = guidance.get('short_text')
            display_advice = _display_unit_advice(
                action_label, action, reason, unit_stop, unit_trigger,
                guidance.get('short_text'), guidance.get('action_reason'),
                [part for part in (
                    f'首步到{first_step}' if first_step and first_step != unit['zone'] and str(first_step) not in str(action or '') else None,
                    f'目标区{target_zone}' if target_zone and target_zone not in (first_step, unit['zone']) and str(target_zone) not in str(action or '') else None,
                    f'约{float(first_step_distance):.1f}米' if isinstance(first_step_distance, (int, float)) and first_step and first_step != unit['zone'] else None,
                    f'对象{target_unit_type or "可见单位"}#{target_unit_id}' if target_unit_id is not None else None,
                    f'当前在{target_current_zone}' if target_unit_id is not None and target_current_zone else None,
                    f'约{float(target_distance):.1f}米' if target_unit_id is not None and isinstance(target_distance, (int, float)) else None,
                ) if part])
            row['unit_ref'] = guidance.get('unit_ref') or {
                'entity_id': unit.get('entity_id'), 'team': unit.get('team'),
                'entity_type': unit.get('entity_type'),
            }
            row['display_advice'] = display_advice
            if action:
                # The replay's collapsed line is an actionable summary. Keep
                # the rationale and stop/failure rules in structured fields
                # for the expandable detail view.
                row['advice_text'] = (f"现在：{display_advice['action']}；"
                                      f"触发：{display_advice['trigger_when']}。")
            else:
                row['advice_text'] = (f"现在：暂不建议具体移动或攻击；"
                                      f"原因：{display_advice['reason']}；"
                                      f"再评估：{display_advice['trigger_when']}。")
        else:
            row['analysis_role'] = 'unknown_affiliation'
            row['output_type'] = 'unavailable'
            row['future_prediction'] = None
            row['current_status'] = row['current_state']
            row['future_state_predictions'] = []
            row['recommended_action'] = None
            row['recommendation_reason'] = None
            row['recommendation'] = None
            row['reason'] = '单位所属方未知，不生成我方动作建议或敌方预测'
        unit_analyses.append(row)

    special_events = _special_event_summary(unit_rows, risk_data, zone_control)
    tactical_stage = _tactical_stage(unit_rows, state, risk_data, zone_control)
    current = {
        'own_side': own_desc, 'opponent_side': enemy_desc,
        'side_key_events': {
            'friendly': _side_key_events(own_units),
            'opponent': _side_key_events(enemy_units),
        },
        'special_events': special_events,
        'tactical_stage': tactical_stage,
        'posture_plain_language': _plain_mode(mode),
        'source': analysis.get('source', 'unknown'),
        'limitation': '位置和生命状态来自当前回放观测；可能攻击属于可见几何推断，无法确认的任务、目标与意图保持未知',
    }
    prediction = {
        'summary': forecast_summary,
        'horizons': forecasts,
        'unit_predictions': [
            {'entity_id': item.get('entity_id'), 'entity_type': item.get('entity_type'),
             'current_state': item.get('current_state'),
             'future_prediction': item.get('future_prediction')}
            for item in unit_analyses if item['relation'] == '敌方'
        ],
    }
    # A fixed, measurable outcome rubric. Feasibility is marked unknown unless
    # target/path/ETA evidence exists; observed response/outcome is not causal
    # attribution and must not be reported as strategy benefit.
    decision_evaluation = {
        'feasibility': {
            'fields': ['target_zone', 'path_zones', 'estimated_travel_time_s'],
            'rule': 'legal_path_and_eta_within_10s',
            'status': ('unknown' if not (first.get('target_zone') or first.get('path_zones')
                                             or first.get('estimated_travel_time_s'))
                       else 'evaluate_from_route_and_eta'),
        },
        'execution_response': {
            'window_seconds': 3,
            'fields': ['first_step_zone', 'observed_action'],
            'rule': 'check_whether_observed_action_matches_recommendation',
        },
        'target_outcome': {
            'window_seconds': [3, 10],
            'fields': ['hp_delta', 'objective_control_delta', 'resource_phase_or_exchange'],
            'rule': 'report_observed_change_only',
        },
        'causal_benefit_claimed': False,
        'limitation': '匹配动作或目标变化只说明执行/结果，不证明建议造成了收益',
    }
    sections = {
        'current_situation': current,
        'forecast_analysis': prediction,
        'overall_recommendation': {
            'detailed_posture': tactical, 'objective': objective,
            'attack': attack_guidance, 'defense': defense_guidance,
            'reserve': reserve_guidance, 'trigger': trigger, 'fallback': fallback,
            'layers': {
                'strategic': {
                    'posture': tactical,
                    'objective': objective,
                    'evidence': posture.get('evidence_conditions') or [],
                    'source': posture.get('source', 'evidence_rule_or_fallback'),
                    'evaluation': _posture_evaluation(posture),
                },
                'tactical': {
                    'attack': attack_guidance,
                    'defense': defense_guidance,
                    'reserve': reserve_guidance,
                },
                'immediate': {'trigger': trigger, 'fallback': fallback},
            },
            'text': recommendation, 'evaluation': decision_evaluation,
        },
        'unit_analysis': unit_analyses,
    }
    # Keep the visualization prose short. Full entity, location, forecast,
    # recommendation, trigger, and evaluation fields remain in `sections`.
    contested = [str(zone) for zone in (zone_control.get('contested_zones') or []) if zone]
    if risk_data.get('enemy_intruders'):
        intruder_zone = str(risk_data['enemy_intruders'][0].get('zone') or '我方防区')
        focus = f'敌方进入{intruder_zone}'
    elif (isinstance(state.get('own_base_hp'), (int, float))
          and float(state['own_base_hp']) < .35):
        focus = '我方基地血量危急'
    elif (isinstance(state.get('own_outpost_hp'), (int, float))
          and float(state['own_outpost_hp']) < .35):
        focus = '我方前哨血量危急'
    elif risk_data.get('own_vulnerable_units'):
        focus = '我方有低血单位'
    elif contested:
        focus = f'争夺{contested[0]}'
    else:
        focus = tactical_stage['label'] if tactical_stage['label'] != '态势观察' else '具体战术阶段未确认'
    current_text = (f'现况：{focus}；我方{_brief_unit_side(unit_rows, "己方")}；'
                    f'敌方{_brief_unit_side(unit_rows, "敌方")}。')
    forecast_text = _brief_prediction(forecasts)
    display_recommendation = _brief_recommendation(first, posture, objective, trigger)
    compact_text = f'{current_text}\n{forecast_text}\n{display_recommendation}'
    return {
        'schema_version': 'global-narrative-v2',
        'source': 'structured_evidence_template',
        'language': 'zh-CN',
        'sections': sections,
        'description': current_text,
        'forecast_analysis': forecast_text,
        'recommendation': display_recommendation,
        'paragraph': compact_text,
        'decision_evaluation': decision_evaluation,
        'confidence': round(float(analysis.get('confidence') or 0.0), 4),
        'evidence': [str(item) for item in (analysis.get('evidence') or []) if item],
    }


__all__ = ['build_global_narrative']
