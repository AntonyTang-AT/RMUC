# -*- coding: utf-8 -*-
"""延时分层路由：Tier0 规则 / Tier1 快启发·小模型·RL / Tier2 微调大模型。

首期仅规则复杂度门控；返回 decision_meta（不替代主决策内容）。
"""
from __future__ import annotations

TIER0 = 'tier0_rule'
TIER1 = 'tier1_fast'
TIER2 = 'tier2_llm'


def _building_hp(w, team, btype):
    for b in (w.get('buildings') or {}).values():
        if b.get('team') == team and b.get('type') == btype:
            return float(b.get('hp_ratio') or 0.0)
    return 1.0


def _outpost_alive(w, team):
    return _building_hp(w, team, '前哨站') > 0.02


def complexity_features(w, mode_history=None, our_team='红'):
    """提取路由特征。mode_history: 近几帧 global_mode 列表（可选）。"""
    opp = '蓝' if our_team == '红' else '红'
    rb = _building_hp(w, our_team, '基地')
    ob = _building_hp(w, opp, '基地')
    base_gap = abs(rb - ob)
    our_out = _outpost_alive(w, our_team)
    opp_out = _outpost_alive(w, opp)
    # 关键建筑低压
    critical_low = (rb < 0.25) or (ob < 0.25) or (our_out and _building_hp(w, our_team, '前哨站') < 0.2)

    hist = list(mode_history or [])
    mode_jitter = 0
    if len(hist) >= 2:
        mode_jitter = sum(1 for a, b in zip(hist, hist[1:]) if a != b)

    robots = list((w.get('robots') or {}).values())
    n_low_hp = sum(1 for r in robots if float(r.get('hp_ratio') or 1.0) < 0.35)

    return {
        'base_gap': round(base_gap, 4),
        'our_base': round(rb, 4),
        'opp_base': round(ob, 4),
        'our_outpost_alive': our_out,
        'opp_outpost_alive': opp_out,
        'critical_low': bool(critical_low),
        'mode_jitter': int(mode_jitter),
        'n_low_hp': int(n_low_hp),
    }


# 默认可校准阈值（由 _work/route_calibration.py 全量对照后更新）
# 关键发现：主线数据 base_gap 中位≈0、p75≈0，thr=0.12 会把 ~98% 帧打成 need_llm。
# 学生在 gap<0.05 上仍有 ~0.74 acc，故关闭纯 gap 门控，改由关键低压/抖动/多单位危急触发 Tier2。
DEFAULT_THRESHOLDS = {
    'base_gap_thr': 0.0,
    'use_critical_low': True,
    'mode_jitter_thr': 3,
    'use_multi_low_hp': True,
    'multi_low_hp_gap': 0.25,
    'high_conflict': 0.45,
    # 学生 softmax 置信：高于此优先 Tier1；低于此且局势复杂才标 need_llm
    'student_conf_thr': 0.55,
}


def route_tier(w, mode_history=None, our_team='红', conflict_rate=0.0, thresholds=None,
               student_conf=None):
    """返回路由 meta。

    质量优先的轻量策略：
    - 危急建筑 → 仍可标 need_llm（稀有）
    - 学生高置信 → Tier1（不因 gap≈0 全进 LLM）
    - 学生低置信 + 复杂 → need_llm；低置信 + 简单 → Tier0/规则回退（由调用方处理）
    """
    thr = dict(DEFAULT_THRESHOLDS)
    if thresholds:
        thr.update(thresholds)
    feat = complexity_features(w, mode_history=mode_history, our_team=our_team)
    conflict_rate = float(conflict_rate or 0.0)
    stu_thr = float(thr.get('student_conf_thr', 0.55))

    hard = False
    reasons = []
    if feat['base_gap'] < float(thr['base_gap_thr']):
        hard = True
        reasons.append('base_gap_tight')
    if thr.get('use_critical_low', True) and feat['critical_low']:
        hard = True
        reasons.append('critical_building_low')
    if feat['mode_jitter'] >= int(thr['mode_jitter_thr']):
        hard = True
        reasons.append('mode_jitter')
    if conflict_rate >= float(thr['high_conflict']):
        hard = True
        reasons.append('high_conflict')
    if (thr.get('use_multi_low_hp', True)
            and feat['n_low_hp'] >= 3
            and feat['base_gap'] < float(thr['multi_low_hp_gap'])):
        hard = True
        reasons.append('multi_unit_critical')

    stu_conf = None if student_conf is None else float(student_conf)
    need_llm = False
    if hard:
        # 危急：若学生很有把握仍可留 Tier1；否则要 LLM
        if stu_conf is None or stu_conf < stu_thr:
            need_llm = True
        else:
            reasons.append('student_override_hard')
    elif stu_conf is not None and stu_conf < stu_thr:
        # 非危急但学生没把握 → 不强制 LLM（太慢），标记低置信供规则回退
        reasons.append('student_low_conf')

    if need_llm:
        tier = TIER2
        conf = 0.55
    elif stu_conf is not None and stu_conf >= stu_thr:
        tier = TIER1
        conf = round(min(0.95, stu_conf), 3)
    else:
        # 无学生或低置信：快路径仍 Tier1 槽位，执行器由调用方选规则
        tier = TIER1
        conf = round(0.5 if stu_conf is None else stu_conf, 3)

    return {
        'tier': tier,
        'need_llm': need_llm,
        'confidence': conf,
        'reasons': reasons,
        'features': feat,
        'student_conf': stu_conf,
        'prefer_rule': bool(stu_conf is not None and stu_conf < stu_thr and not need_llm),
        'thresholds': {k: thr[k] for k in DEFAULT_THRESHOLDS},
    }


def attach_decision_meta(decision, route, opponent_intent=None, semantic_situation=None,
                         sentry_state=None, strategic_intent=None,
                         global_analysis=None, temporal_forecast=None):
    """在决策旁路挂 decision_meta / opponent_intent / semantic_situation。"""
    d = dict(decision) if decision else {}
    meta = {
        'tier': route.get('tier'),
        'need_llm': bool(route.get('need_llm')),
        'confidence': route.get('confidence'),
        'reasons': list(route.get('reasons') or []),
    }
    if sentry_state is not None:
        meta['sentry_state'] = sentry_state
    if strategic_intent is not None:
        meta['strategic_intent'] = strategic_intent
    d['decision_meta'] = meta
    if opponent_intent is not None:
        d['opponent_intent'] = {
            'probs': dict(opponent_intent.get('probs') or {}),
            'top': opponent_intent.get('top'),
            'source': opponent_intent.get('source'),
        }
    if semantic_situation is not None:
        d['semantic_situation'] = dict(semantic_situation)
    if global_analysis is not None:
        d['global_analysis'] = dict(global_analysis)
    if temporal_forecast is not None:
        d['temporal_forecast'] = dict(temporal_forecast)
    return d


def strip_sidecar_for_validate(decision):
    """validate 前去掉旁路字段。"""
    if not isinstance(decision, dict):
        return decision
    d = dict(decision)
    d.pop('decision_meta', None)
    d.pop('opponent_intent', None)
    d.pop('semantic_situation', None)
    d.pop('global_analysis', None)
    d.pop('temporal_forecast', None)
    d.pop('_tier2_generation', None)
    return d
