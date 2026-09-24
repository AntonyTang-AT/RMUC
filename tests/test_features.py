# tests/test_features.py
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import config
from data import db, features

def test_build_decision_points_structure():
    con = db.connect()
    rows = db.timeseries_for(con, config.GAME_REF)
    dps = features.build_decision_points(rows, config.GAME_REF, step=3.0, trend=15.0)
    con.close()
    assert len(dps) > 50
    dp = dps[0]
    assert dp['t'] == min(row['时刻秒'] for row in rows)
    assert 'robots' in dp and 'team' in dp
    # 红哨兵 robot_id=7 应在
    assert 7 in dp['robots']
    r = dp['robots'][7]
    for k in ('type','team','hp','maxhp','x','y','dx','dy','dhp','dfire'):
        assert k in r

def test_decision_point_t_spacing():
    con = db.connect()
    rows = db.timeseries_for(con, config.GAME_REF)
    dps = features.build_decision_points(rows, config.GAME_REF, step=3.0, trend=15.0)
    con.close()
    ts = [d['t'] for d in dps]
    assert all(abs(b - a - 3.0) < 1e-6 for a, b in zip(ts, ts[1:]))

def test_field_bounds():
    con = db.connect()
    rows = db.timeseries_for(con, config.GAME_REF)
    dps = features.build_decision_points(rows, config.GAME_REF)
    b = features.field_bounds(dps)
    con.close()
    assert b['xmin'] < b['xmax'] and b['ymin'] < b['ymax']
