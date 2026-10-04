# -*- coding: utf-8 -*-
"""延时分层路由单测（无 torch）。"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from labels import router as rt
from schema import layered as sl


def _w(our_base=0.9, opp_base=0.3, our_out=1.0, n_low=0):
    robots = {}
    for i in range(4):
        robots[f'r{i}'] = {
            'team': '红', 'type': '英雄',
            'hp_ratio': 0.2 if i < n_low else 0.9,
            'norm_xy': [0.3, 0.5], 'dfire': 0,
        }
    buildings = {
        'rb': {'team': '红', 'type': '基地', 'hp_ratio': our_base},
        'bb': {'team': '蓝', 'type': '基地', 'hp_ratio': opp_base},
        'ro': {'team': '红', 'type': '前哨站', 'hp_ratio': our_out},
        'bo': {'team': '蓝', 'type': '前哨站', 'hp_ratio': 1.0},
    }
    return {'t': 100.0, 'robots': robots, 'buildings': buildings, 'team': {}}


def test_large_gap_is_tier1():
    # 默认已关闭纯 gap 门控；无 critical 时应 Tier1
    r = rt.route_tier(_w(our_base=0.95, opp_base=0.35), student_conf=0.8)
    assert r['tier'] == rt.TIER1
    assert r['need_llm'] is False
    assert r['confidence'] >= 0.6


def test_student_high_conf_overrides_hard():
    r = rt.route_tier(_w(our_base=0.2, opp_base=0.9), student_conf=0.9)
    assert r['need_llm'] is False
    assert 'student_override_hard' in r['reasons']


def test_student_low_conf_prefers_rule_not_llm():
    r = rt.route_tier(_w(our_base=0.8, opp_base=0.8), student_conf=0.4)
    assert r['need_llm'] is False
    assert r.get('prefer_rule') is True


def test_tight_gap_is_tier2():
    # 显式恢复旧 gap 阈值时，胶着应进 Tier2
    r = rt.route_tier(
        _w(our_base=0.55, opp_base=0.5),
        thresholds={'base_gap_thr': 0.12})
    assert r['need_llm'] is True
    assert r['tier'] == rt.TIER2
    assert 'base_gap_tight' in r['reasons']


def test_default_tied_bases_go_tier1_without_critical():
    """校准后：双方基地血接近且无危急 → Tier1（不再因 gap≈0 全进 LLM）。"""
    r = rt.route_tier(_w(our_base=0.8, opp_base=0.8))
    assert r['need_llm'] is False
    assert r['tier'] == rt.TIER1


def test_critical_building_triggers_llm():
    r = rt.route_tier(_w(our_base=0.2, opp_base=0.9))
    assert r['need_llm'] is True
    assert 'critical_building_low' in r['reasons']


def test_mode_jitter_triggers_llm():
    hist = ['进攻', '防守', '进攻', '防守', '均衡']
    r = rt.route_tier(_w(our_base=0.8, opp_base=0.4), mode_history=hist)
    assert r['need_llm'] is True
    assert 'mode_jitter' in r['reasons']


def test_high_conflict_triggers_llm():
    r = rt.route_tier(_w(our_base=0.8, opp_base=0.4), conflict_rate=0.5)
    assert r['need_llm'] is True
    assert 'high_conflict' in r['reasons']


def test_attach_and_strip_sidecar():
    d = sl.build_empty_decision(0.0)
    route = rt.route_tier(_w())
    opp = {'probs': {'压前哨': 1.0}, 'top': '压前哨', 'source': 'delta_s+game_boost'}
    # pad probs for realism
    from labels import opp_intent as oi
    opp['probs'] = {i: (1.0 if i == '压前哨' else 0.0) for i in oi.INTENTS}
    enriched = rt.attach_decision_meta(
        d, route, opponent_intent=opp,
        semantic_situation={'base_threat': False, 'source': 'replay_approx'},
        sentry_state='cruise', strategic_intent='StrategicPatrol')
    assert 'decision_meta' in enriched and 'opponent_intent' in enriched
    assert enriched['decision_meta']['sentry_state'] == 'cruise'
    assert 'semantic_situation' in enriched
    assert enriched['opponent_intent']['top'] == '压前哨'
    clean = rt.strip_sidecar_for_validate(enriched)
    assert 'decision_meta' not in clean and 'opponent_intent' not in clean
    assert 'semantic_situation' not in clean
    assert sl.validate(clean)[0]
    # 旁路字段存在时 validate 仍应通过（忽略未知键）
    assert sl.validate(enriched)[0]
