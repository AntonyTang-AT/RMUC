# tests/test_llm_m3.py
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import run_llm_m3
from schema import layered as sl

def _world():
    # 一张红方哨兵(中线处)的世界状态
    return {'t': 0.0,
            'robots': {'7': {'type': '哨兵', 'team': '红', 'norm_xy': [0.5, 0.5],
                             'hp_ratio': 1.0, 'vuln': 0, 'dfire': 0.0}},
            'buildings': {'b': {'team': '红', 'hp_ratio': 1.0}}}

def test_llm_state_to_intent_valid():
    assert run_llm_m3.llm_state_to_intent('def')['move'] == '回防'
    assert run_llm_m3.llm_state_to_intent('hitbuilding')['fire'] is True
    assert run_llm_m3.llm_state_to_intent('cruise')['target'] == '无'
    assert run_llm_m3.llm_state_to_intent('unknown')['move'] == '游击'  # 未知回退 cruise

def test_as_decision_uses_llm_state_and_is_valid():
    d = run_llm_m3._as_decision(_world(), 'def')
    assert d['units']['哨兵']['move'] == '回防'
    ok, errs = sl.validate(d)
    assert ok, errs
    d2 = run_llm_m3._as_decision(_world(), 'cruise')
    assert d2['units']['哨兵']['move'] == '游击'
    assert sl.validate(d2)[0] is True

def test_assess_reports_agreement_and_sentry_valid():
    rule = [sl.build_empty_decision(0.0)] * 3
    llm = [sl.build_empty_decision(0.0)] * 3
    for d in (rule + llm):
        d['global_mode'] = '均衡'
    for d in llm:
        d['units']['哨兵']['move'] = '游击'
    for d in rule:
        d['units']['哨兵']['move'] = '推进'   # 与 llm 不一致 -> agreement < 1
    rep = run_llm_m3.assess(llm, rule)
    assert rep['llm_sentry_valid'] == 1.0
    assert rep['compliance_rate'] == 1.0
    assert rep['agreement'] == 0.0           # 恰各不相同(匀速) -> 0.0

def test_assess_per_class_nonempty():
    rule = [sl.build_empty_decision(0.0)] * 3
    llm = [sl.build_empty_decision(0.0)] * 3
    for d in (rule + llm):
        d['global_mode'] = '均衡'
    for i, d in enumerate(llm):
        d['units']['哨兵']['move'] = '推进' if i % 2 else '游击'
    for i, d in enumerate(rule):
        d['units']['哨兵']['move'] = '推进'
    rep = run_llm_m3.assess(llm, rule)
    assert rep['per_class'], 'per_class must not be empty'
    assert set(rep['per_class'].keys()) <= {'推进', '回防', '游击'}
