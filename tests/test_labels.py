# tests/test_labels.py
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import config
from data import db, features
from worldstate import zone
from labels import sentry_labels as sl

def _dps():
    con = db.connect()
    rows = db.timeseries_for(con, config.GAME_REF)
    dps = features.build_decision_points(rows, config.GAME_REF)
    con.close()
    return dps

def test_label_sentry_returns_valid():
    dps = _dps()
    b = features.field_bounds(dps)
    lbl = sl.label_sentry(dps[0], b, team='红', sentry_rid=7)
    assert lbl in sl.SENTRY_STATES

def test_label_series_grows_and_valid():
    dps = _dps()
    b = features.field_bounds(dps)
    series = sl.label_sentry_series(dps[:20], b, team='红', sentry_rid=7)
    assert len(series) == min(20, len(dps))
    assert all(s[1] in sl.SENTRY_STATES for s in series)

def test_smooth_sequence_removes_flicker():
    raw = [(float(i), ('cruise' if i % 2 == 0 else 'def')) for i in range(10)]
    out = sl.smooth_sequence(raw, min_dur=5.0)
    assert len(out) < len(raw)                          # single-frame flickers merged
    assert out[0][0] == raw[0][0]                       # output still anchors the start time
    assert [s for s, _ in out] == sorted(s for s, _ in out)  # segments stay time-ordered

def test_enemy_in_half_ignores_buildings():
    # 敌方建筑(x=0)不应触发 def; 敌方单位进入己方半场才触发
    import os, sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from data.features import BUILDING_TYPES
    from labels import sentry_labels as sl
    b = {'xmin': -10.0, 'xmax': 30.0, 'ymin': 0.0, 'ymax': 10.0}
    dp = {
        't': 0.0, 'team': {'红': {}, '蓝': {}},
        'robots': {
            7: {'type': '哨兵', 'team': '红', 'x': 5.0},
            110: {'type': '基地', 'team': '蓝', 'x': 0.0},
            111: {'type': '前哨站', 'team': '蓝', 'x': 0.0},
        },
    }
    assert sl._enemy_in_half(dp, b, '红', dp['robots'][7]) is False
    dp['robots'][103] = {'type': '步兵3', 'team': '蓝', 'x': 0.0}
    assert sl._enemy_in_half(dp, b, '红', dp['robots'][7]) is True


# --------------------------------------------------------------------------
# label_sentry branch tests (synthetic dp, no DB / no torch)
# bounds xmin/xmax from ref => midfield = (xmin+xmax)/2 = 12.6335
# --------------------------------------------------------------------------
_B = {'xmin': -2.844, 'xmax': 28.111}   # midfield = 12.6335

def _dp(sentry_x, dfire=0.0, ammo17=0.0, blues=()):
    robots = {7: {'type': '哨兵', 'team': '红', 'x': sentry_x,
                  'dfire': dfire, 'ammo17': ammo17}}
    for i, x in enumerate(blues):
        robots[101 + i] = {'type': '步兵3', 'team': '蓝', 'x': x}
    return {'t': 0.0, 'team': {'红': {}, '蓝': {}}, 'robots': robots}

def test_label_sentry_hitbuilding_when_push_and_recent_fire():
    # sentry past midline (x=20.0 > mid) with recent dfire>0 -> hitbuilding
    dp = _dp(sentry_x=20.0, dfire=5.0)
    assert sl.label_sentry(dp, _B, team='红', sentry_rid=7) == 'hitbuilding'

def test_label_sentry_def_when_enemy_in_our_half():
    # sentry not pushing (x=5.0 < mid) but blue infantry at x=3.0 (< mid) -> def
    dp = _dp(sentry_x=5.0, blues=(3.0,))
    assert sl.label_sentry(dp, _B, team='红', sentry_rid=7) == 'def'

def test_label_sentry_cruise_when_idle():
    # sentry not pushing (x=5.0) and no enemy in our half (blue at x=20.0 > mid) -> cruise
    dp = _dp(sentry_x=5.0, blues=(20.0,))
    assert sl.label_sentry(dp, _B, team='红', sentry_rid=7) == 'cruise'

def test_label_sentry_hitbuilding_requires_recent_fire_not_ever_fired():
    # REGRESSION (Finding A): cumulative ammo17 must NOT trigger hitbuilding;
    # only recent dfire does. ammo17=999 (ever-fired) + pushing should NOT be 'hitbuilding'.
    dp = _dp(sentry_x=20.0, dfire=0.0, ammo17=999.0)
    assert sl.label_sentry(dp, _B, team='红', sentry_rid=7) != 'hitbuilding'
