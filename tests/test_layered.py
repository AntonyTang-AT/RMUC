# tests/test_layered.py
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from schema import layered as sl

def _valid():
    d = sl.build_empty_decision(12.0)
    d['global_mode'] = '进攻'
    d['units']['哨兵'] = {'target': '敌方前哨站', 'move': '推进', 'fire': True, 'risk': '中'}
    d['units']['英雄'] = {'target': '敌方英雄', 'move': '游击', 'fire': False, 'risk': '低'}
    return d

def test_constants():
    assert sl.GLOBAL_MODES == ('进攻', '均衡', '防守')
    assert len(sl.UNIT_CLASSES) == 6
    assert '哨兵' in sl.UNIT_CLASSES

def test_validate_valid():
    ok, errs = sl.validate(_valid())
    assert ok is True and errs == []

def test_validate_bad_global_mode():
    d = _valid(); d['global_mode'] = '跑路'
    ok, errs = sl.validate(d)
    assert ok is False and any('global_mode' in e for e in errs)

def test_validate_missing_unit_class():
    d = _valid(); del d['units']['英雄']
    ok, errs = sl.validate(d)
    assert ok is False and any('英雄' in e for e in errs)

def test_validate_bad_target():
    d = _valid(); d['units']['哨兵']['target'] = '敌方坦克'
    ok, errs = sl.validate(d)
    assert ok is False and any('target' in e for e in errs)

def test_validate_bad_fire_type():
    d = _valid(); d['units']['哨兵']['fire'] = 'yes'
    ok, errs = sl.validate(d)
    assert ok is False

def test_conflict_rate():
    a = _valid()
    b = _valid(); b['global_mode'] = '防守'
    c = dict(b)   # same as b -> 仅 a->b 这一对变化
    r = sl.conflict_rate([a, b, c])
    assert 0.0 <= r <= 1.0
    assert abs(r - 1/2) < 1e-9  # 两对相邻中恰好一对变化
