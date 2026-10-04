# -*- coding: utf-8 -*-
"""语义态势层近似（考核原图 WorldState → 语义态势）。

数据源为 SQLite 回放 WorldState，非相机/激光真融合；source=replay_approx。
模块级无 torch。
"""
from __future__ import annotations

from collections import Counter
import math

from worldstate.functional_zone import assign_functional_zone_normalized
from worldstate.unit_state import derive_unit_special_state


def _building_hp(w, team, btype):
    for b in (w.get('buildings') or {}).values():
        if b.get('team') == team and b.get('type') == btype:
            return float(b.get('hp_ratio') or 0.0)
    return 1.0


def _robots(w, team):
    return [r for r in (w.get('robots') or {}).values() if r.get('team') == team]


def _alive(robot):
    hp = robot.get('hp_ratio')
    if (isinstance(hp, (int, float)) and not isinstance(hp, bool)
            and math.isfinite(float(hp))):
        return float(hp) > 0.0
    # Missing/invalid HP means this observed robot remains a candidate in
    # battlefield counts; it does not prove that the robot is alive or dead.
    return True


def _life_state_from_hp(hp):
    if (isinstance(hp, (int, float)) and not isinstance(hp, bool)
            and math.isfinite(float(hp))):
        return 'active' if float(hp) > 0.0 else 'destroyed'
    return 'unknown'


def _active(robots):
    return [robot for robot in robots if _alive(robot)]


def _mean_x(robots):
    xs = [(r.get('norm_xy') or [0.5, 0.5])[0] for r in _active(robots) if r.get('norm_xy')]
    if not xs:
        return None
    return sum(xs) / len(xs)


def _enemy_in_our_half(w, team='红'):
    opp = '蓝' if team == '红' else '红'
    for r in _active(_robots(w, opp)):
        nx = (r.get('norm_xy') or [0.5, 0.5])[0]
        if team == '红' and nx <= 0.5:
            return True
        if team == '蓝' and nx >= 0.5:
            return True
    return False


def _types_set(robots):
    return {r.get('type') for r in _active(robots) if r.get('type')}


def _mean_hp(robots):
    # Surviving-robot condition is easier to interpret when confirmed dead
    # robots are reported separately instead of dragging the mean toward 0.
    values = [float(r['hp_ratio']) for r in _active(robots)
              if isinstance(r.get('hp_ratio'), (int, float))
              and not isinstance(r.get('hp_ratio'), bool)
              and math.isfinite(float(r['hp_ratio']))]
    return (sum(values) / len(values)) if values else None


def _centroid_xy(robots):
    points = [r.get('norm_xy') for r in _active(robots)
              if isinstance(r.get('norm_xy'), (list, tuple)) and len(r.get('norm_xy')) >= 2]
    if not points:
        return None
    return [round(sum(float(p[0]) for p in points) / len(points), 4),
            round(sum(float(p[1]) for p in points) / len(points), 4)]


def _functional_zone(robot):
    xy = robot.get('norm_xy') or [0.5, 0.5]
    if not isinstance(xy, (list, tuple)) or len(xy) < 2:
        return '场地未知区'
    return assign_functional_zone_normalized(float(xy[0]), float(xy[1]))


def _zone_counts(robots):
    counts = Counter(_functional_zone(robot) for robot in _active(robots))
    return {zone: int(counts[zone]) for zone in sorted(counts)}


def _unit_locations(robots):
    values = []
    for robot in sorted(robots, key=lambda item: (str(item.get('type')), str(item.get('team')))):
        hp_ratio = robot.get('hp_ratio')
        values.append({
            'entity_id': robot.get('entity_id'),
            'team': robot.get('team'),
            'type': robot.get('type'),
            'zone': _functional_zone(robot),
            'hp_ratio': round(float(hp_ratio), 4) if isinstance(hp_ratio, (int, float))
                        and not isinstance(hp_ratio, bool) and math.isfinite(float(hp_ratio)) else None,
            'vulnerable': bool(robot.get('vuln')),
            'life_state': _life_state_from_hp(hp_ratio),
        })
    return values


def _push_count(robots, team):
    """Count mobile units across the midline from ``team``'s perspective."""
    count = 0
    for robot in _active(robots):
        if robot.get('type') in ('基地', '前哨站'):
            continue
        xy = robot.get('norm_xy') or [0.5, 0.5]
        if team == '红' and float(xy[0]) > 0.5:
            count += 1
        elif team == '蓝' and float(xy[0]) < 0.5:
            count += 1
    return count


def _special_entities(w, team):
    """Build unit-state sidecars from the visible worldstate fields."""
    all_robots = list((w.get('robots') or {}).values())
    entities = []
    for robot in all_robots:
        item = dict(robot)
        item['entity_type'] = robot.get('type')
        item['entity_id'] = robot.get('entity_id') or robot.get('id')
        item['role'] = 'self' if robot.get('team') == team else 'opponent'
        item['life_state'] = _life_state_from_hp(robot.get('hp_ratio'))
        if 'position' not in item:
            xy = robot.get('norm_xy')
            if isinstance(xy, (list, tuple)) and len(xy) >= 2:
                item['position'] = [float(xy[0]) * 28.0, float(xy[1]) * 15.0]
        if robot.get('dfire') is not None and 'firing' not in item:
            item['firing'] = bool(float(robot.get('dfire') or 0.0) > 0.0)
        entities.append(item)
    states = []
    for entity in entities:
        state = derive_unit_special_state(entity, entities, self_team=team)
        states.append({
            'entity_id': entity.get('entity_id'),
            'team': entity.get('team'),
            'type': entity.get('entity_type'),
            'zone': state.get('functional_zone'),
            'observability': state.get('observability'),
            'life_state': entity.get('life_state'),
            'state': state,
        })
    return states


def _special_state_summary(states, team):
    # A last-frame firing/resource flag on a destroyed unit is stale state,
    # not a current tactical event.
    own = [item for item in states if item.get('team') == team
           and item.get('life_state') != 'destroyed']
    opp = [item for item in states if item.get('team') != team
           and item.get('life_state') != 'destroyed']
    def count(items, predicate):
        return sum(1 for item in items if predicate(item.get('state') or {}))
    return {
        'own_attacking': count(own, lambda s: s.get('attack_status') == 'attacking'),
        'own_possible_attack': count(own, lambda s: s.get('attack_status') == 'possible_attack'),
        'opponent_attacking': count(opp, lambda s: s.get('attack_status') == 'attacking'),
        'opponent_possible_attack': count(opp, lambda s: s.get('attack_status') == 'possible_attack'),
        'own_engineer_phases': dict(Counter(
            s.get('resource_phase', 'unknown') for item in own
            for s in [item.get('state') or {}] if item.get('type') == '工程')),
        'opponent_engineer_phases': dict(Counter(
            s.get('resource_phase', 'unknown') for item in opp
            for s in [item.get('state') or {}] if item.get('type') == '工程')),
        'own_defense_states': dict(Counter(
            s.get('defense_status', 'unknown') for item in own
            for s in [item.get('state') or {}] if item.get('type') == '哨兵')),
        'opponent_defense_states': dict(Counter(
            s.get('defense_status', 'unknown') for item in opp
            for s in [item.get('state') or {}] if item.get('type') == '哨兵')),
    }


def _zone_control(own, enemies):
    own_counts = Counter(_functional_zone(robot) for robot in _active(own))
    opp_counts = Counter(_functional_zone(robot) for robot in _active(enemies))
    zones = sorted(set(own_counts) | set(opp_counts))
    control = {}
    for zone in zones:
        own_n, opp_n = own_counts.get(zone, 0), opp_counts.get(zone, 0)
        if own_n and opp_n:
            posture = 'contested'
        elif own_n:
            posture = 'own_control'
        else:
            posture = 'opponent_control'
        control[zone] = {
            'posture': posture,
            'own_units': own_n,
            'opponent_units': opp_n,
        }
    contested = [zone for zone, value in control.items()
                 if value['posture'] == 'contested']
    return {
        'by_zone': control,
        'contested_zones': contested,
        'own_controlled_zones': [zone for zone, value in control.items()
                                 if value['posture'] == 'own_control'],
        'opponent_controlled_zones': [zone for zone, value in control.items()
                                      if value['posture'] == 'opponent_control'],
    }


def _risk_register(w, team, own, enemies, situation):
    own_base = _building_hp(w, team, '基地')
    own_outpost = _building_hp(w, team, '前哨站')
    opp = '蓝' if team == '红' else '红'
    enemy_intruders = []
    for robot in _active(enemies):
        xy = robot.get('norm_xy') or [0.5, 0.5]
        in_our_half = float(xy[0]) <= 0.5 if team == '红' else float(xy[0]) >= 0.5
        if in_our_half:
            enemy_intruders.append({
                'type': robot.get('type'),
                'zone': _functional_zone(robot),
                'entity_id': robot.get('entity_id'),
            })
    vulnerable = [{
        'type': robot.get('type'),
        'zone': _functional_zone(robot),
        'entity_id': robot.get('entity_id'),
    } for robot in _active(own) if float(robot.get('hp_ratio') or 0.0) < 0.35 or robot.get('vuln')]
    risks = []
    if own_base < 0.5:
        risks.append({'type': 'base_integrity', 'level': 'high', 'action': 'protect_base'})
    if own_outpost < 0.5:
        risks.append({'type': 'outpost_integrity', 'level': 'medium', 'action': 'protect_outpost'})
    if enemy_intruders:
        risks.append({'type': 'enemy_intrusion', 'level': 'high' if own_base < 0.75 else 'medium',
                      'action': 'intercept_intruders', 'units': enemy_intruders})
    if vulnerable:
        risks.append({'type': 'own_vulnerable_units', 'level': 'medium',
                      'action': 'screen_vulnerable_units', 'units': vulnerable})
    if situation.get('enemy_trend') == 'press':
        risks.append({'type': 'opponent_pressure', 'level': 'medium', 'action': 'hold_defensive_line'})
    return {
        'items': risks,
        'enemy_intruders': enemy_intruders,
        'own_vulnerable_units': vulnerable,
        'opponent_objective_status': {
            'base_hp': round(_building_hp(w, opp, '基地'), 4),
            'outpost_hp': round(_building_hp(w, opp, '前哨站'), 4),
        },
    }


def _recommendations(mode, threat_level, zone_control, risks, special_summary,
                     situation, team):
    recommendations = []
    if threat_level == '高' or risks.get('enemy_intruders'):
        recommendations.append({
            'priority': 'P0', 'action': 'intercept_and_protect',
            'objective': '先阻断进入己方半场的单位，再保护基地和前哨站',
            'trigger': '敌方继续向己方深区推进或建筑风险升高',
            'fallback': '收缩到基地侧相邻功能区',
        })
    if zone_control.get('contested_zones'):
        recommendations.append({
            'priority': 'P1', 'action': 'stabilize_contested_zones',
            'objective': '在争夺区形成局部人数和火力优势',
            'zones': list(zone_control['contested_zones']),
            'trigger': '争夺区出现可确认的攻击目标',
            'fallback': '保持相邻道路控制，不跨区追击',
        })
    if mode == '进攻' and not risks.get('enemy_intruders'):
        recommendations.append({
            'priority': 'P1', 'action': 'pressure_objective',
            'objective': '沿相邻道路压迫敌方前哨或能量机关，保留回撤路线',
            'trigger': '敌方前哨或能量机关暴露且我方前排未处于高风险',
            'fallback': '转为区域控制并等待下一次可见目标',
        })
    engineer_unknown = sum(special_summary.get('own_engineer_phases', {}).get(k, 0)
                           for k in ('unknown',))
    if engineer_unknown:
        recommendations.append({
            'priority': 'P2', 'action': 'resolve_resource_state',
            'objective': '补充工程持矿与兑换事件，避免误把工程当作普通战斗单位',
            'trigger': '工程进入补给区或资源区附近',
            'fallback': '只按当前位置和相邻功能区预测移动',
        })
    if not recommendations:
        recommendations.append({
            'priority': 'P2', 'action': 'maintain_control',
            'objective': '保持当前阵型，等待可验证的敌方意图变化',
            'trigger': '敌方出现明确推进、回撤或建筑交互事件',
            'fallback': '回到最近己方控制区',
        })
    return recommendations


def _detailed_posture(mode, threat, risks, zone_control, special_summary,
                      situation, own_base, own_outpost, control_posture):
    """Split the former three broad postures into an evidence-triggered task."""
    intruders = risks.get('enemy_intruders') or []
    own_risks = risks.get('own_vulnerable_units') or []
    if own_base < .35:
        code, label, objective, evidence = ('base_emergency_defense', '基地应急防守',
            '停止分散推进，先拦截进入基地方向的敌人并保护基地', ['己方基地血量低于35%'])
    elif intruders:
        code, label, objective, evidence = ('deep_interception', '纵深拦截',
            '在基地与前哨的相邻防区拦截入侵单位，避免单车追击', ['发现敌方单位进入己方防区'])
    elif own_outpost < .35:
        code, label, objective, evidence = ('outpost_defense', '前哨稳固防守',
            '压住前哨附近通道，保护设施并保留回撤路线', ['己方前哨站血量低于35%'])
    elif zone_control.get('contested_zones'):
        code, label, objective, evidence = ('contested_zone', '关键区域争夺',
            '围绕争夺区域集中协同，避免把力量拆成多条弱线', ['敌我单位同时出现在争夺区域'])
    elif any(int(special_summary.get('own_engineer_phases', {}).get(key, 0)) > 0
             for key in ('collecting_energy_unit', 'going_to_mine',
                         'returning_to_exchange', 'exchanging')):
        code, label, objective, evidence = ('resource_cycle', '资源节奏保障',
            '确保工程往返路线安全，再把战斗力量用于关键通道', ['工程正在执行资源采集或兑换'])
    elif situation.get('enemy_trend') == 'press':
        code, label, objective, evidence = ('frontline_delay', '前沿阻滞',
            '在前沿相邻区域减缓敌方推进，同时留出回撤空间', ['连续位置显示敌方正在压进'])
    elif own_risks:
        code, label, objective, evidence = ('protect_vulnerable_units', '脆弱单位掩护',
            '先为低血量单位提供掩护，再决定是否继续推进', ['己方存在低血量或易受攻击单位'])
    elif situation.get('enemy_trend') == 'retreat' and mode == '进攻':
        code, label, objective, evidence = ('controlled_pursuit', '控制性追击',
            '沿可达路线跟进，但不脱离支援范围', ['敌方在回撤且我方态势允许推进'])
    elif mode == '进攻':
        code, label, objective, evidence = ('focused_pressure', '单目标集中压制',
            '选一个可确认的前哨或目标集中施压，并保留防守兵力', ['当前粗粒度态势为进攻'])
    elif control_posture == 'own_advantage':
        code, label, objective, evidence = ('consolidate_advantage', '优势巩固与有序推进',
            '先巩固已控制区域，再沿可通行路线逐步推进', ['我方越线单位多于敌方且无更高优先级风险'])
    elif mode == '防守':
        code, label, objective, evidence = ('layered_facility_defense', '关键设施分层防护',
            '覆盖基地和前哨入口，减少无收益的远距离追击', ['当前粗粒度态势为防守'])
    else:
        code, label, objective, evidence = ('formation_and_patrol', '阵型稳固与机动巡逻',
            '维持关键通道控制，等待明确接触后再局部增援', ['暂无触发优先级更高的特殊态势'])
    return {
        'code': code, 'label': label, 'objective': objective,
        'evidence_conditions': evidence,
        'trigger': '连续观测确认目标、推进方向或关键设施受击',
        'fallback': '退回最近己方可控区域，保留基地方向的防护力量',
        'source': 'evidence_rule_derived_posture',
    }


def derive_semantic_situation(w, team='红', prev_w=None):
    """从图 1 语义态势层字段做回放近似。"""
    w = w or {}
    base_hp = _building_hp(w, team, '基地')
    out_hp = _building_hp(w, team, '前哨站')
    enemy_press = _enemy_in_our_half(w, team=team)
    base_threat = bool(base_hp < 0.5 or out_hp < 0.25 or enemy_press)

    opp = '蓝' if team == '红' else '红'
    cur_x = _mean_x(_robots(w, opp))
    enemy_trend = 'unknown'
    if prev_w is not None and cur_x is not None:
        prev_x = _mean_x(_robots(prev_w, opp))
        if prev_x is not None:
            dx = cur_x - prev_x
            # 红视角：敌方 x 减小 = 向我方压迫
            if team == '红':
                if dx < -0.02:
                    enemy_trend = 'press'
                elif dx > 0.02:
                    enemy_trend = 'retreat'
                else:
                    enemy_trend = 'hold'
            else:
                if dx > 0.02:
                    enemy_trend = 'press'
                elif dx < -0.02:
                    enemy_trend = 'retreat'
                else:
                    enemy_trend = 'hold'

    our = _robots(w, team)
    in_enemy = any(
        ((r.get('norm_xy') or [0.5, 0.5])[0] > 0.5) if team == '红'
        else ((r.get('norm_xy') or [0.5, 0.5])[0] < 0.5)
        for r in our
    )
    opp_out_alive = _building_hp(w, opp, '前哨站') > 0.02
    reachable = bool(in_enemy or opp_out_alive)
    reachable_note = 'ally_in_enemy_half' if in_enemy else (
        'opp_outpost_alive' if opp_out_alive else 'limited')

    lost_target = False
    lost_types = []
    if prev_w is not None:
        prev_types = _types_set(_robots(prev_w, opp))
        cur_types = _types_set(_robots(w, opp))
        lost_types = sorted(prev_types - cur_types)
        lost_target = bool(lost_types)

    result = {
        'base_threat': base_threat,
        'enemy_trend': enemy_trend,
        'reachable': reachable,
        'reachable_note': reachable_note,
        'lost_target': lost_target,
        'lost_types': lost_types,
        'source': 'replay_approx',
    }
    return result


def derive_global_analysis(w, team='红', prev_w=None, mode=None,
                           opponent_intent=None, sentry_state=None):
    """Return a detailed, auditable global-situation sidecar.

    This is deliberately a present-state evidence summary.  It does not use
    future frames and does not pretend that a replay-derived state is a sensor
    measurement.  The three horizon views tell a downstream LLM what level of
    detail is appropriate at 10/30--60/120 seconds.
    """
    w = w or {}
    opp = '蓝' if team == '红' else '红'
    own = _robots(w, team)
    enemies = _robots(w, opp)
    own_push = _push_count(own, team)
    opp_push = _push_count(enemies, opp)
    own_base = _building_hp(w, team, '基地')
    opp_base = _building_hp(w, opp, '基地')
    own_outpost = _building_hp(w, team, '前哨站')
    opp_outpost = _building_hp(w, opp, '前哨站')
    situ = derive_semantic_situation(w, team=team, prev_w=prev_w)
    score = float(own_push - opp_push) + 2.0 * (opp_base - own_base)
    if mode not in ('进攻', '均衡', '防守'):
        mode = '防守' if score <= -1.0 else ('进攻' if score >= 1.0 else '均衡')

    threat_score = (2.0 if own_base < 0.5 else 0.0)
    threat_score += (1.0 if own_outpost < 0.25 else 0.0)
    threat_score += (1.0 if situ.get('enemy_trend') == 'press' else 0.0)
    threat_score += min(2.0, max(0.0, float(opp_push - own_push)))
    threat_level = '高' if threat_score >= 3.0 else ('中' if threat_score >= 1.0 else '低')
    confidence = min(0.95, 0.45 + 0.08 * len(own + enemies)
                    + (0.12 if situ.get('enemy_trend') != 'unknown' else 0.0))

    evidence = []
    if opp_push > own_push:
        evidence.append(f'敌方越过中线单位 {opp_push} 个，高于我方 {own_push} 个')
    elif own_push > opp_push:
        evidence.append(f'我方越线单位 {own_push} 个，高于敌方 {opp_push} 个')
    else:
        evidence.append(f'双方越线单位相当（我方 {own_push}、敌方 {opp_push}）')
    own_zone_text = '、'.join(f'{zone}×{count}' for zone, count in _zone_counts(own).items())
    opp_zone_text = '、'.join(f'{zone}×{count}' for zone, count in _zone_counts(enemies).items())
    if own_zone_text:
        evidence.append('我方功能区分布：' + own_zone_text)
    if opp_zone_text:
        evidence.append('敌方功能区分布：' + opp_zone_text)
    if own_base < 0.5 or own_outpost < 0.25:
        evidence.append(f'我方建筑受损（基地 {own_base:.2f}，前哨站 {own_outpost:.2f}）')
    if opp_base < 0.5 or opp_outpost < 0.25:
        evidence.append(f'敌方建筑受损（基地 {opp_base:.2f}，前哨站 {opp_outpost:.2f}）')
    if situ.get('enemy_trend') not in (None, 'unknown', 'hold'):
        evidence.append('敌方当前趋势：' + ('压迫' if situ['enemy_trend'] == 'press' else '回撤'))
    if not evidence:
        evidence.append('当前没有足够的单帧优势证据，按均衡态势处理')

    if threat_level == '高':
        focus = '优先保护己方低血量建筑与关键单位，压制敌方进入我方半场'
    elif mode == '进攻':
        focus = '保持推进并集中火力于敌方前哨站或基地'
    elif mode == '防守':
        focus = '收缩阵型，优先处理敌方前排和入侵单位'
    else:
        focus = '保持区域控制，等待敌方意图证据后再扩大投入'

    intent = None
    if opponent_intent:
        intent = {
            'top': opponent_intent.get('top'),
            'probabilities': dict(opponent_intent.get('probs') or {}),
        }
    special_entities = _special_entities(w, team)
    special_summary = _special_state_summary(special_entities, team)
    zone_control = _zone_control(own, enemies)
    risk_register = _risk_register(w, team, own, enemies, situ)
    if zone_control['contested_zones']:
        control_posture = 'contested'
    elif own_push > opp_push:
        control_posture = 'own_advantage'
    elif opp_push > own_push:
        control_posture = 'opponent_advantage'
    else:
        control_posture = 'balanced'
    recommendations = _recommendations(
        mode, threat_level, zone_control, risk_register,
        special_summary, situ, team)
    detailed_posture = _detailed_posture(
        mode, threat_level, risk_register, zone_control, special_summary,
        situ, own_base, own_outpost, control_posture)
    own_mean_hp = _mean_hp(own)
    opponent_mean_hp = _mean_hp(enemies)
    result = {
        'schema_version': 'global-analysis-v1',
        'source': 'replay_approx',
        'current_mode': mode,
        'confidence': round(float(confidence), 4),
        'threat_level': threat_level,
        'evidence': evidence,
        'state': {
            'team': team,
            'time_s': round(float(w.get('t') or 0.0), 3),
            'own_push_count': own_push,
            'opponent_push_count': opp_push,
            'push_margin': own_push - opp_push,
            'own_mean_hp': round(own_mean_hp, 4) if own_mean_hp is not None else None,
            'opponent_mean_hp': round(opponent_mean_hp, 4) if opponent_mean_hp is not None else None,
            'own_base_hp': round(own_base, 4),
            'opponent_base_hp': round(opp_base, 4),
            'own_outpost_hp': round(own_outpost, 4),
            'opponent_outpost_hp': round(opp_outpost, 4),
            'opponent_centroid': _centroid_xy(enemies),
            'own_functional_zones': _zone_counts(own),
            'opponent_functional_zones': _zone_counts(enemies),
            'unit_locations': _unit_locations(own + enemies),
            'enemy_trend': situ.get('enemy_trend'),
            'control_posture': control_posture,
            'zone_control': zone_control,
            'risk_register': risk_register,
            'special_state_summary': special_summary,
            'visible_unit_states': special_entities,
            'economy': {
                'own_coins': (w.get('team', {}).get(team) or {}).get('coins'),
                'opponent_coins': (w.get('team', {}).get(opp) or {}).get('coins'),
                'own_engineer_state': special_summary.get('own_engineer_phases', {}),
                'opponent_engineer_state': special_summary.get('opponent_engineer_phases', {}),
            },
        },
        'opponent_intent': intent,
        'sentry_state': sentry_state,
        'recommended_focus': focus,
        'detailed_posture': detailed_posture,
        'force_balance_guidance': {
            'attack': ('把主攻力量集中到一个可确认目标，沿可达通道推进；'
                       '进攻类单位负责压制，工程兵优先完成资源循环。'),
            'defense': ('保留哨兵和至少一条回防路线覆盖基地、前哨；'
                        '有入侵或设施受击时先处理防区威胁。'),
            'reserve': ('让机动单位留在争夺区的相邻区域，满足触发条件再支援；'
                        '避免预备力量过早追入敌方深区。'),
            'allocation_source': 'role_and_risk_rules_not_learned_allocation_head',
        },
        'recommendations': recommendations,
        'decision_gates': [
            '先确认目标属于当前可见范围，再决定是否从可能攻击升级为正在攻击',
            '工程状态未知时不把资源路线当作确定事实',
            '跨越功能区必须经过相邻图，远期预测保留回撤触发条件',
        ],
        'horizon_views': {
            '10s': {'specificity': 'high', 'focus': focus,
                    'track': ['建筑血量', '敌我越线单位', '敌方趋势', '关键单位血量']},
            '30_60s': {'specificity': 'medium', 'focus': '观察区域控制、阵型推进深度与建筑交换结果',
                       'track': ['区域占用', '双方推进深度', '前哨站状态', '敌方意图概率']},
            '120s': {'specificity': 'coarse', 'focus': '判断总体攻防方向与基地/前哨站风险，不给出单单位精确位置',
                     'track': ['攻防方向', '建筑存活风险', '敌方总体位置区域']},
        },
    }
    # Natural language is an auditable sidecar. It verbalizes only the
    # structured evidence above and never changes the strict decision schema.
    from worldstate.narrative import build_global_narrative
    result['narrative'] = build_global_narrative(result)
    return result


def format_semantic_line(sem):
    """写入 situation_digest 的短行。"""
    if not sem:
        return ''
    bt = 1 if sem.get('base_threat') else 0
    lt = 1 if sem.get('lost_target') else 0
    return (
        f"语义: base_threat={bt} enemy_trend={sem.get('enemy_trend', 'unknown')} "
        f"reachable={1 if sem.get('reachable') else 0} lost_target={lt}"
    )
