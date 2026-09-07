# tests/test_behavior.py
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from fastloop import behavior as bh

def _snap(state='hitbuilding', hp=0.15, vuln=0):
    return {'robots': {7: {'type':'哨兵','team':'红','zone':'Z00','hp_ratio':hp,'vuln':vuln}}, 'bot_state': state}

def test_feasible_low_hp_blocks_attack():
    r = bh.Rules(min_hp_to_attack=0.2)
    assert bh.feasible('hitbuilding', _snap(hp=0.15), r) is False
    assert bh.feasible('def', _snap(hp=0.15), r) is True

def test_feasible_vulnerable_blocks_attack():
    r = bh.Rules()
    assert bh.feasible('hitbuilding', _snap(hp=1.0, vuln=1), r) is False

def test_decide_fast_returns_action():
    r = bh.Rules()
    a = bh.decide_fast(_snap('hitbuilding', hp=1.0), r)
    assert a['state'] in bh.SENTRY_STATES or a['state'] in ('def', 'hitbuilding', 'cruise')
    assert a['target'] is not None
