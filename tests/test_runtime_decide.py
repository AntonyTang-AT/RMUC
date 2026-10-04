# -*- coding: utf-8 -*-
"""runtime TieredDecider 无 torch 路径单测（假学生 + NullTier2）。"""
from schema import layered as sl
from labels import router as rt
from runtime.decide import TieredDecider, _project_tier2_to_visible_state
from runtime.tier2 import NullTier2, resolve_tier2_backend
from runtime.single_executor import SingleExecutorArbiter
from data.radar_visibility_replay import PROVIDER_ID


class _FakeStudent:
    def __init__(self, mode='均衡', conf=0.9):
        self.mode = mode
        self.conf = conf

    def predict_proba(self, obs):
        return self.mode, self.conf, {self.mode: self.conf}


class _FakeTemporal:
    def __init__(self):
        self.calls = 0

    def reset(self):
        self.calls = 0

    def predict(self, snapshot):
        self.calls += 1
        return {
            'schema_version': 'temporal-entity-forecast-v1',
            'horizons': {
                '10': {'global_mode': {'top': '进攻', 'confidence': 0.9}},
                '30': {'global_mode': {'top': '均衡', 'confidence': 0.6}},
                '60': {'global_mode': {'top': '均衡', 'confidence': 0.5}},
                '120': {'global_mode': {'top': '防守', 'confidence': 0.4}},
            },
        }


def _dp(t=10.0, our_base_hp=5000.0, opp_base_hp=5000.0, opp_hero_x=100.0):
    robots = {
        '7': {'type': '哨兵', 'team': '红', 'x': 900.0, 'y': 0.0, 'hp': 1000.0, 'maxhp': 1000.0,
              'small_heat': 0.0, 'big_heat': 0.0, 'power': 0.0, 'ammo17': 0.0, 'ammo42': 0.0, 'vuln': 0,
              'team_coins_total': 0.0, 'team_coins_left': 0.0},
        '1': {'type': '英雄', 'team': '红', 'x': 850.0, 'y': -50.0, 'hp': 2000.0, 'maxhp': 2000.0,
              'small_heat': 0.0, 'big_heat': 0.0, 'power': 0.0, 'ammo17': 0.0, 'ammo42': 0.0, 'vuln': 0,
              'team_coins_total': 0.0, 'team_coins_left': 0.0},
        '100': {'type': '英雄', 'team': '蓝', 'x': opp_hero_x, 'y': 0.0, 'hp': 2000.0, 'maxhp': 2000.0,
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


class _VisibleTier2:
    def __init__(self, decision):
        self.decision = decision
        self.visible_input = None

    def generate_visible_one(self, visible_input, forecast_context=None):
        self.visible_input = visible_input
        return self.decision


def test_visible_tier2_without_observable_snapshot_must_fall_back(monkeypatch):
    """A deployable Tier2 backend never receives a full replay point as fallback input."""
    import runtime.decide as decide_module

    generated = sl.build_empty_decision(0.0)
    backend = _VisibleTier2(generated)
    monkeypatch.setattr(decide_module.rt, 'route_tier', lambda *a, **kw: {'need_llm': True})
    monkeypatch.setattr(decide_module, '_project_tier2_to_visible_state',
                        lambda decision, visible: (decision, []))
    monkeypatch.setattr(decide_module, 'audit_tier2_decision',
                        lambda *a, **kw: {'accepted': True, 'reasons': [],
                                          'forecast_disagreements': []})
    decider = TieredDecider(tier2_backend=backend, team='红')

    out = decider.decide(_dp(0.0), _bounds(), observable_snapshot=None)

    assert backend.visible_input is None
    assert out['decision_meta']['mode_source'] == 'tier2_fallback_rule'
    assert out['decision_meta']['tier2_input_contract'] == 'missing_visible_input_rule_fallback'
    assert out['decision_meta']['tier2_ok'] is False


def test_visible_tier2_schema_failure_must_fall_back(monkeypatch):
    import runtime.decide as decide_module

    dp = _dp(0.0)
    snapshot = {
        'visibility_provider': PROVIDER_ID, 'self_team': '红', 'issued_at': 0.0,
        'entities': {
            '红:7': {'entity_id': '红:7', 'team': '红', 'entity_type': '哨兵',
                     'role': 'self', 'source': 'sensor', 'observability_state': 'current',
                     'observed_at': 0.0, 'received_at': 0.0, 'position': [900.0, 0.0],
                     'hp': 1000.0, 'maxhp': 1000.0},
            '蓝:100': {'entity_id': '蓝:100', 'team': '蓝', 'entity_type': '工程',
                       'role': 'opponent', 'source': 'radar', 'observability_state': 'current',
                       'observed_at': 0.0, 'received_at': 0.0, 'position': [100.0, 0.0],
                       'hp': 2000.0},
        },
    }
    backend = _VisibleTier2({'not': 'a layered decision'})
    monkeypatch.setattr(decide_module.rt, 'route_tier', lambda *a, **kw: {'need_llm': True})
    decider = TieredDecider(tier2_backend=backend, team='红')

    out = decider.decide(dp, _bounds(), observable_snapshot=snapshot)

    assert backend.visible_input is not None
    assert out['decision_meta']['mode_source'] == 'tier2_fallback_rule'
    assert out['decision_meta']['tier2_guard']['accepted'] is False
    assert out['decision_meta']['tier2_guard']['reasons'] == ['generation_or_schema_failure']


def test_tier2_uses_visibility_snapshot_and_never_passes_full_decision_point(monkeypatch):
    from schema import layered as sl
    import runtime.decide as decide_module

    dp = _dp(0.0)
    bounds = _bounds()
    snapshot = {
        'visibility_provider': PROVIDER_ID, 'self_team': '红', 'issued_at': 0.0,
        'entities': {
            '红:7': {'entity_id': '红:7', 'team': '红', 'entity_type': '哨兵',
                     'role': 'self', 'source': 'sensor', 'observability_state': 'current',
                     'observed_at': 0.0, 'received_at': 0.0, 'position': [900.0, 0.0],
                     'hp': 1000.0, 'maxhp': 1000.0, 'small_heat': 0.0},
            '蓝:100': {'entity_id': '蓝:100', 'team': '蓝', 'entity_type': '工程',
                       'role': 'opponent', 'source': 'radar', 'observability_state': 'current',
                       'observed_at': 0.0, 'received_at': 0.0, 'position': [100.0, 0.0],
                       'hp': 2000.0},
        },
    }
    generated = sl.build_empty_decision(0.0)
    generated['units']['哨兵'] = {'target': '敌方工程', 'move': '坚守', 'fire': False, 'risk': '低'}
    backend = _VisibleTier2(generated)
    monkeypatch.setattr(decide_module.rt, 'route_tier', lambda *a, **kw: {'need_llm': True})
    decider = TieredDecider(tier2_backend=backend, team='红')

    out = decider.decide(dp, bounds, observable_snapshot=snapshot)

    assert backend.visible_input is not None
    assert backend.visible_input['visibility_policy'] == 'rmuc_2026_referee_radar_protocol_v1'
    assert 'power' not in backend.visible_input['issue_state']['robots']['蓝:100']
    assert out['decision_meta']['tier2_input_contract'] == 'rmuc_2026_referee_radar_protocol_v1'


def test_tier2_projects_destroyed_own_slot_before_the_safety_guard():
    robots = {
        f'红:{unit_type}': {
            'team': '红', 'type': unit_type, 'life_state': 'current',
            'observability_state': 'current', 'hp_ratio': 1.0,
        }
        for unit_type in sl.UNIT_CLASSES if unit_type != '步兵3'
    }
    robots['红:步兵3'] = {
        'team': '红', 'type': '步兵3', 'life_state': 'destroyed',
        'observability_state': 'current', 'hp_ratio': 0.0,
    }
    visible_input = {
        'perspective': '红',
        'issue_state': {'robots': robots},
    }
    decision = sl.build_empty_decision(0.0)
    decision['units']['步兵3'] = {
        'target': '敌方英雄', 'move': '推进', 'fire': True, 'risk': '高',
    }
    projected, overridden = _project_tier2_to_visible_state(decision, visible_input)
    assert overridden == ['步兵3']
    assert projected['units']['步兵3'] == {
        'target': '无', 'move': '无', 'fire': False, 'risk': '低',
    }


def test_tier2_contract_repair_metadata_is_preserved(monkeypatch):
    import runtime.decide as decide_module

    dp = _dp(0.0)
    def snapshot(t):
        return {
            'visibility_provider': PROVIDER_ID, 'self_team': '红', 'issued_at': t,
            'entities': {},
        }
    decision = sl.build_empty_decision(0.0)
    decision['_tier2_generation'] = {
        'strict_parse_valid': False,
        'contract_repairs': ['missing_unit_idle:英雄'],
        'parse_error': 'schema_invalid',
        'contract_error': 'schema_repaired',
    }
    backend = _VisibleTier2(decision)
    monkeypatch.setattr(decide_module.rt, 'route_tier', lambda *a, **kw: {'need_llm': True})
    decider = TieredDecider(tier2_backend=backend, team='红')

    out = decider.decide(dp, _bounds(), observable_snapshot=snapshot(0.0))

    assert decider.stats['n_tier2_contract_repaired'] == 1
    meta = out['decision_meta']['tier2_generation']
    assert meta['strict_parse_valid'] is False
    assert meta['contract_repairs'] == ['missing_unit_idle:英雄']
    assert '_tier2_generation' not in out
    assert '_tier2_generation' not in rt.strip_sidecar_for_validate(out)


def test_tier2_repair_budget_opens_circuit_and_falls_back(monkeypatch):
    import runtime.decide as decide_module

    def snapshot(t):
        return {
            'visibility_provider': PROVIDER_ID, 'self_team': '红', 'issued_at': t,
            'entities': {},
        }
    decision = sl.build_empty_decision(0.0)
    decision['_tier2_generation'] = {
        'strict_parse_valid': False,
        'contract_repairs': ['missing_unit_idle:英雄'],
        'parse_error': 'schema_invalid',
        'contract_error': 'schema_repaired',
    }
    backend = _VisibleTier2(decision)
    monkeypatch.setattr(decide_module.rt, 'route_tier', lambda *a, **kw: {'need_llm': True})
    decider = TieredDecider(
        tier2_backend=backend, team='红', tier2_repair_rate_limit=0.25,
        tier2_repair_min_samples=2)

    first = decider.decide(_dp(0.0), _bounds(), observable_snapshot=snapshot(0.0))
    second = decider.decide(_dp(3.0), _bounds(), observable_snapshot=snapshot(3.0))
    third = decider.decide(_dp(6.0), _bounds(), observable_snapshot=snapshot(6.0))

    assert first['decision_meta']['mode_source'] == 'tier2_llm'
    assert second['decision_meta']['mode_source'] == 'tier2_repair_budget_fallback'
    assert third['decision_meta']['mode_source'] == 'tier2_repair_budget_fallback'
    assert decider.stats['n_tier2_repair_budget_fallback'] == 2
    assert decider.stats['n_tier2_repair_observed'] == 2
    assert backend.visible_input is not None


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


def test_decider_exposes_state_machine_and_behavior_tree_trace():
    dp = _dp(our_base_hp=800.0, opp_base_hp=5000.0)
    decider = TieredDecider(team='红')
    decision = decider.decide(dp, _bounds(), base_decision=sl.build_empty_decision(dp['t']))
    meta = decision['decision_meta']
    assert meta['sentry_state'] == 'def'
    assert meta['strategic_intent'] == 'StrategicDefendBase'
    assert meta['sentry_state_machine']['event'] in ('base_threat', 'safety_override')
    assert meta['sentry_behavior_tree']['priority_level'] in (
        'T0_system_safety', 'T1_task_coordination')


def test_behavior_tree_arbitrates_llm_attack_when_base_defense_is_active(monkeypatch):
    import runtime.decide as decide_module

    generated = sl.build_empty_decision(0.0)
    generated['global_mode'] = '进攻'
    for action in generated['units'].values():
        action.update({'target': '敌方前哨站', 'move': '推进', 'fire': True, 'risk': '高'})
    backend = _VisibleTier2(generated)
    monkeypatch.setattr(decide_module.rt, 'route_tier', lambda *a, **kw: {'need_llm': True})
    monkeypatch.setattr(decide_module, '_project_tier2_to_visible_state',
                        lambda decision, visible: (decision, []))
    monkeypatch.setattr(decide_module, 'audit_tier2_decision',
                        lambda *a, **kw: {'accepted': True, 'reasons': [],
                                          'forecast_disagreements': []})
    decider = TieredDecider(tier2_backend=backend, team='红')

    decision = decider.decide(_dp(our_base_hp=1000.0), _bounds(),
                              observable_snapshot={
                                  'visibility_provider': PROVIDER_ID,
                                  'self_team': '红', 'issued_at': 10.0,
                                  'entities': {
                                      '红:7': {'entity_id': '红:7', 'team': '红',
                                               'entity_type': '哨兵', 'role': 'self',
                                               'source': 'sensor', 'observability_state': 'current',
                                               'observed_at': 10.0, 'received_at': 10.0,
                                               'position': [900.0, 0.0], 'hp': 1000.0,
                                               'maxhp': 1000.0, 'small_heat': 0.0},
                                  },
                              })

    assert decision['global_mode'] == '防守'
    assert decision['units']['哨兵']['move'] == '回防'
    trace = decision['decision_meta']['behavior_arbitration']
    assert trace['applied'] is True
    assert trace['before']['global_mode'] == '进攻'
    assert trace['after']['global_mode'] == '防守'


def test_decision_request_mainline_dispatches_once_through_executor():
    observed = []
    arbiter = SingleExecutorArbiter()
    arbiter.register('shadow', lambda request: observed.append(request) or {
        'ack_status': 'shadow_recorded', 'executed': False,
    })
    decider = TieredDecider(team='红', as_decision_request=True,
                            executor_arbiter=arbiter)

    request = decider.decide(_dp(0.0), _bounds(),
                             base_decision=sl.build_empty_decision(0.0))

    assert request['interface'] == 'DecisionRequest'
    assert request['decision_meta']['executor_dispatch']['executor'] == 'shadow'
    assert request['decision_meta']['executor_dispatch']['controller_response']['executed'] is False
    assert len(observed) == 1
    assert len(arbiter.history) == 1


def test_temporal_forecast_is_attached_and_can_hint_mode():
    temporal = _FakeTemporal()
    decider = TieredDecider(temporal_forecaster=temporal, team='红')
    snapshot = {
        'issued_at': 10.0, 'self_team': '红', 'entities': {},
    }
    d = decider.decide(_dp(10.0, opp_base_hp=5000.0, opp_hero_x=1100.0), _bounds(),
                       base_decision=sl.build_empty_decision(10.0),
                       observable_snapshot=snapshot)
    assert temporal.calls == 1
    assert d['global_mode'] == '进攻'
    assert d['temporal_forecast']['schema_version'] == 'temporal-entity-forecast-v1'
    assert d['decision_meta']['temporal_forecast_attached'] is True
    clean = rt.strip_sidecar_for_validate(d)
    assert 'temporal_forecast' not in clean
