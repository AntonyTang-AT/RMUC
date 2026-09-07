# tests/test_prompt.py
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from model import prompt

def test_build_prompt_contains_worldstate_and_states():
    ws = {'t': 0.0, 'team': {'红': {'coins': 100}}, 'robots': {7: {'type':'哨兵','team':'红','zone':'Z00','norm_xy':[0.1,0.2],'hp_ratio':1.0,'vuln':0,'dfire':0}}, 'buildings': {}}
    p = prompt.build_prompt(ws)
    assert '哨兵' in p and 'def' in p and 'hitbuilding' in p and 'cruise' in p
    assert 'Z00' in p

def test_parse_decision_recovers_state():
    o = prompt.parse_decision('{"state": "hitbuilding"}')
    assert o['state'] == 'hitbuilding'

def test_parse_decision_falls_back():
    o = prompt.parse_decision('???')   # 非法 JSON
    assert o['state'] == 'cruise'
