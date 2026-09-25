# -*- coding: utf-8 -*-
"""语义态势层近似（考核原图 WorldState → 语义态势）。

数据源为 SQLite 回放 WorldState，非相机/激光真融合；source=replay_approx。
模块级无 torch。
"""
from __future__ import annotations


def _building_hp(w, team, btype):
    for b in (w.get('buildings') or {}).values():
        if b.get('team') == team and b.get('type') == btype:
            return float(b.get('hp_ratio') or 0.0)
    return 1.0


def _robots(w, team):
    return [r for r in (w.get('robots') or {}).values() if r.get('team') == team]


def _mean_x(robots):
    xs = [(r.get('norm_xy') or [0.5, 0.5])[0] for r in robots if r.get('norm_xy')]
    if not xs:
        return None
    return sum(xs) / len(xs)


def _enemy_in_our_half(w, team='红'):
    opp = '蓝' if team == '红' else '红'
    for r in _robots(w, opp):
        nx = (r.get('norm_xy') or [0.5, 0.5])[0]
        if team == '红' and nx <= 0.5:
            return True
        if team == '蓝' and nx >= 0.5:
            return True
    return False


def _types_set(robots):
    return {r.get('type') for r in robots if r.get('type')}


def _mean_hp(robots):
    values = [float(r.get('hp_ratio') or 0.0) for r in robots]
    return (sum(values) / len(values)) if values else 1.0


def _centroid_xy(robots):
    points = [r.get('norm_xy') for r in robots
              if isinstance(r.get('norm_xy'), (list, tuple)) and len(r.get('norm_xy')) >= 2]
    if not points:
        return None
    return [round(sum(float(p[0]) for p in points) / len(points), 4),
            round(sum(float(p[1]) for p in points) / len(points), 4)]


def _push_count(robots, team):
    """Count mobile units across the midline from ``team``'s perspective."""
    count = 0
    for robot in robots:
        if robot.get('type') in ('基地', '前哨站'):
            continue
        xy = robot.get('norm_xy') or [0.5, 0.5]
        if team == '红' and float(xy[0]) > 0.5:
            count += 1
        elif team == '蓝' and float(xy[0]) < 0.5:
            count += 1
    return count


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

    return {
        'base_threat': base_threat,
        'enemy_trend': enemy_trend,
        'reachable': reachable,
        'reachable_note': reachable_note,
        'lost_target': lost_target,
        'lost_types': lost_types,
        'source': 'replay_approx',
    }


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
    return {
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
            'own_mean_hp': round(_mean_hp(own), 4),
            'opponent_mean_hp': round(_mean_hp(enemies), 4),
            'own_base_hp': round(own_base, 4),
            'opponent_base_hp': round(opp_base, 4),
            'own_outpost_hp': round(own_outpost, 4),
            'opponent_outpost_hp': round(opp_outpost, 4),
            'opponent_centroid': _centroid_xy(enemies),
            'enemy_trend': situ.get('enemy_trend'),
        },
        'opponent_intent': intent,
        'sentry_state': sentry_state,
        'recommended_focus': focus,
        'horizon_views': {
            '10s': {'specificity': 'high', 'focus': focus,
                    'track': ['建筑血量', '敌我越线单位', '敌方趋势', '关键单位血量']},
            '30_60s': {'specificity': 'medium', 'focus': '观察区域控制、阵型推进深度与建筑交换结果',
                       'track': ['区域占用', '双方推进深度', '前哨站状态', '敌方意图概率']},
            '120s': {'specificity': 'coarse', 'focus': '判断总体攻防方向与基地/前哨站风险，不给出单单位精确位置',
                     'track': ['攻防方向', '建筑存活风险', '敌方总体位置区域']},
        },
    }


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
