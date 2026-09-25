# -*- coding: utf-8 -*-
"""runtime TieredDecider 无 torch 路径单测（假学生 + NullTier2）。"""
from schema import layered as sl
from labels import router as rt
from runtime.decide import TieredDecider
from runtime.tier2 import NullTier2, resolve_tier2_backend


class _FakeStudent:
    def __init__(self, mode='均衡', conf=0.9):
        self.mode = mode
        self.conf = conf

    def predict_proba(self, obs):
        return self.mode, self.conf, {self.mode: self.conf}


def _dp(t=10.0, our_base_hp=5000.0, opp_base_hp=5000.0):
    robots = {
        '7': {'type': '哨兵', 'team': '红', 'x': 900.0, 'y': 0.0, 'hp': 1000.0, 'maxhp': 1000.0,
              'small_heat': 0.0, 'big_heat': 0.0, 'power': 0.0, 'ammo17': 0.0, 'ammo42': 0.0, 'vuln': 0,
              'team_coins_total': 0.0, 'team_coins_left': 0.0},
        '1': {'type': '英雄', 'team': '红', 'x': 850.0, 'y': -50.0, 'hp': 2000.0, 'maxhp': 2000.0,
              'small_heat': 0.0, 'big_heat': 0.0, 'power': 0.0, 'ammo17': 0.0, 'ammo42': 0.0, 'vuln': 0,
              'team_coins_total': 0.0, 'team_coins_left': 0.0},
        '100': {'type': '英雄', 'team': '蓝', 'x': 100.0, 'y': 0.0, 'hp': 2000.0, 'maxhp': 2000.0,
                'small_heat': 0.0, 'big_heat': 0.0, 'power': 0.0, 'ammo17': 0.0, 'ammo42': 0.0, 'vuln': 0,
                'team_coins_total': 0.0, 'team_coins_left': 0.0},
        'b1': {'type': '基地', 'team': '红', 'x': 0.0, 'y': 0.0, 'hp': our_base_hp, 'maxhp': 5000.0,
               'small_heat': 0.0, 'big_heat': 0.0, 'power': 0.0, 'ammo17': 0.0, 'ammo42': 0.0, 'vuln': 0,
               'team_coins_total': 0.0, 'team_coins_left': 0.0},
        'b2': {'type': '基地', 'team': '蓝', 'x': 1200.0, 'y': 0.0, 'hp': opp_base_hp, 'maxhp': 5000.0,
               'small_heat': 0.0, 'big_heat': 0.0, 'power': 0.0, 'ammo17': 0.0, 'ammo42': 0.0, 'vuln': 0,
               'team_coins_total': 0.0, 'team_coins_left': 0.0},
        'o1': {'type': '前哨站', 'team': '红', 'x': 100.0, 'y': 0.0, 'hp': 1500.0, 'maxhp': 1500.0,
               'small_heat': 0.0, 'big_heat': 0.0, 'power': 0.0, 'ammo17': 0.0, 'ammo42': 0.0, 'vuln': 0,
               'team_coins_total': 0.0, 'team_coins_left': 0.0},
        'o2': {'type': '前哨站', 'team': '蓝', 'x': 1100.0, 'y': 0.0, 'hp': 1500.0, 'maxhp': 1500.0,
               'small_heat': 0.0, 'big_heat': 0.0, 'power': 0.0, 'ammo17': 0.0, 'ammo42': 0.0, 'vuln': 0,
               'team_coins_total': 0.0, 'team_coins_left': 0.0},
    }
    return {'t': float(t), 'robots': robots,
            'team': {'红': {'coins_total': 0.0, 'coins_left': 0.0},
                     '蓝': {'coins_total': 0.0, 'coins_left': 0.0}}}


def _bounds():
    return {'xmin': -500.0, 'xmax': 1400.0, 'ymin': -500.0, 'ymax': 500.0}


def test_resolve_tier2_none_is_null():
    b = resolve_tier2_backend(adapter='none', device='auto')
    assert isinstance(b, NullTier2)
    assert b.available() is False
    assert b.generate_one({}, {}) is None


def test_null_tier2_fallback_on_need_llm(monkeypatch):
    """危急帧 need_llm → NullTier2 → tier2_fallback_rule，决策合法。"""
    dp = _dp(our_base_hp=800.0, opp_base_hp=5000.0)  # hp_ratio=0.16 → critical
    stu = _FakeStudent(mode='进攻', conf=0.2)
    import rl.mode_features as mf
    monkeypatch.setattr(mf, 'pack_obs', lambda dp, bounds, team='红': [0.0] * 40)
    decider = TieredDecider(mode_student=stu, tier2_backend=NullTier2(), team='红')
    d = decider.decide(dp, _bounds(), base_decision=sl.build_empty_decision(dp['t']))
    meta = d['decision_meta']
    assert meta.get('need_llm') is True
    assert meta.get('mode_source') == 'tier2_fallback_rule'
    assert meta.get('tier2_attempted') is True
    assert meta.get('tier2_ok') is False
    ok, errs = sl.validate(rt.strip_sidecar_for_validate(d))
    assert ok, errs
    assert decider.stats['n_tier2_fallback'] == 1


def test_high_conf_student_tier1(monkeypatch):
    dp = _dp(our_base_hp=5000.0, opp_base_hp=5000.0)
    stu = _FakeStudent(mode='进攻', conf=0.9)
    import rl.mode_features as mf
    monkeypatch.setattr(mf, 'pack_obs', lambda dp, bounds, team='红': [0.0] * 40)
    decider = TieredDecider(mode_student=stu, tier2_backend=NullTier2(), team='红')
    d = decider.decide(dp, _bounds(), base_decision=sl.build_empty_decision(dp['t']))
    meta = d['decision_meta']
    assert meta.get('need_llm') is False
    assert 'student' in meta.get('mode_source', '')
    assert meta.get('sentry_state') in ('def', 'hitbuilding', 'cruise')
    assert meta.get('strategic_intent', '').startswith('Strategic')
    assert 'semantic_situation' in d
    assert d['semantic_situation'].get('source') == 'replay_approx'
    assert d['global_analysis']['schema_version'] == 'global-analysis-v1'
    assert d['decision_meta']['global_analysis_schema'] == 'global-analysis-v1'
    ok, errs = sl.validate(rt.strip_sidecar_for_validate(d))
    assert ok, errs


def test_decision_request_adapter():
    from runtime.decision_request import to_decision_request, from_decision_request
    from tree import global_mode as gm
    d = sl.build_empty_decision(1.0)
    d['global_mode'] = '进攻'
    d['decision_meta'] = {
        'sentry_state': 'hitbuilding',
        'strategic_intent': gm.sentry_to_strategic('hitbuilding'),
        'tier': 'tier1_fast',
        'need_llm': False,
        'confidence': 0.8,
        'reasons': [],
    }
    d['semantic_situation'] = {'base_threat': False, 'source': 'replay_approx'}
    d['global_analysis'] = {'schema_version': 'global-analysis-v1'}
    req = to_decision_request(d)
    assert req['interface'] == 'DecisionRequest'
    assert req['sentry_state'] == 'hitbuilding'
    assert req['strategic_intent'] == 'StrategicAttackOutpost'
    assert req['global_analysis']['schema_version'] == 'global-analysis-v1'
    core = from_decision_request(req)
    ok, errs = sl.validate(core)
    assert ok, errs
