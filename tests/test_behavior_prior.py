# tests/test_behavior_prior.py
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from tree import behavior_prior as bp

def test_state_sig_encodes():
    ws = {'robots': {7: {'team': '红', 'zone': 'Z12'}}, 'global_mode': '进攻'}
    assert bp.state_sig(ws, '进攻') == '进攻|Z12'

def test_mine_finds_frequent_path():
    sigA, sigB = ('均衡|Z01'), ('进攻|Z12')
    # 反复出现 A->进攻 (decision), A->防守, A->进攻 占据多数
    traj = [[(sigA, 'cruise'), (sigB, 'hitbuilding'), (sigA, 'cruise')],
            [(sigA, 'cruise'), (sigB, 'cruise'), (sigA, 'def')],
            [(sigA, 'cruise'), (sigB, 'cruise')]]
    prior = bp.mine(traj, k=3)
    # A 的 top 决策应是 cruise (出现次数多)
    a_rules = [p for p in prior if p['sig'] == sigA]
    assert a_rules and a_rules[0]['decision'] == 'cruise'
    assert a_rules[0]['count'] >= 2

def test_build_prior_context():
    prior = [{'sig': '进攻|Z12', 'decision': 'hitbuilding', 'count': 3, 'freq': 1.0}]
    ctx = bp.build_prior_context(prior, '进攻|Z12')
    assert any('hitbuilding' in line for line in ctx)
