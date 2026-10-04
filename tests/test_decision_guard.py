import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from runtime.decision_guard import audit_tier2_decision
from runtime.decide import TieredDecider
from schema import layered as sl


def _world(hp=1.0):
    return {'robots': {
        'r1': {'team': '红', 'type': '哨兵', 'hp_ratio': hp},
        'b1': {'team': '蓝', 'type': '英雄', 'hp_ratio': 1.0},
    }}


def _decision(fire=False, target='无', mode='均衡'):
    d = sl.build_empty_decision(0.0)
    d['global_mode'] = mode
    d['units']['哨兵'] = {'target': target, 'move': '坚守', 'fire': fire, 'risk': '低'}
    return d


def test_guard_rejects_fire_without_target():
    out = audit_tier2_decision(_decision(fire=True), _world())
    assert out['accepted'] is False
    assert 'fire_without_target:哨兵' in out['reasons']


def test_guard_rejects_action_for_confirmed_destroyed_unit():
    out = audit_tier2_decision(_decision(target='敌方英雄'), _world(hp=0.0))
    assert out['accepted'] is False
    assert 'destroyed_unit_action:哨兵' in out['reasons']


def test_guard_allows_required_inert_slot_for_confirmed_destroyed_unit():
    decision = sl.build_empty_decision(0.0)
    out = audit_tier2_decision(decision, _world(hp=0.0))
    assert out['accepted'] is True


def test_guard_records_forecast_disagreement_without_rejecting():
    forecast = {'predictions': [
        {'horizon_seconds': 10,
         'global_mode': {'top': '进攻', 'confidence': 0.9}},
    ]}
    out = audit_tier2_decision(_decision(mode='防守'), _world(), forecast)
    assert out['accepted'] is True
    assert out['forecast_disagreements'][0]['horizon_seconds'] == 10


class _BadTier2:
    def generate_one(self, dp, bounds, forecast_context=None):
        return _decision(fire=True)


def test_tiered_decider_falls_back_when_guard_rejects():
    dp = {
        't': 0.0,
        'robots': {
            'r1': {'team': '红', 'type': '哨兵', 'x': 0.0, 'y': 0.0,
                   'hp': 100.0, 'maxhp': 100.0, 'small_heat': 0.0, 'big_heat': 0.0,
                   'power': 0.0, 'ammo17': 0.0, 'ammo42': 0.0, 'vuln': 0,
                   'dfire': 0.0},
            'rb': {'team': '红', 'type': '基地', 'x': 0.0, 'y': 0.0,
                   'hp': 100.0, 'maxhp': 1000.0, 'small_heat': 0.0, 'big_heat': 0.0,
                   'power': 0.0, 'ammo17': 0.0, 'ammo42': 0.0, 'vuln': 0,
                   'dfire': 0.0},
            'bb': {'team': '蓝', 'type': '基地', 'x': 100.0, 'y': 0.0,
                   'hp': 1000.0, 'maxhp': 1000.0, 'small_heat': 0.0, 'big_heat': 0.0,
                   'power': 0.0, 'ammo17': 0.0, 'ammo42': 0.0, 'vuln': 0,
                   'dfire': 0.0},
        },
        'team': {'红': {'coins_left': 0.0}, '蓝': {'coins_left': 0.0}},
    }
    bounds = {'xmin': -100.0, 'xmax': 100.0, 'ymin': -100.0, 'ymax': 100.0}
    decider = TieredDecider(tier2_backend=_BadTier2(), team='红')
    out = decider.decide(dp, bounds)
    assert decider.stats['n_tier2_guard_reject'] == 1
    assert out['decision_meta']['mode_source'] == 'tier2_fallback_rule'
    assert out['decision_meta']['tier2_guard']['accepted'] is False
