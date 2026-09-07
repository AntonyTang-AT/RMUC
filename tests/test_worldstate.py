# tests/test_worldstate.py
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import config
from data import db, features
from worldstate import worldstate as ws

def test_normalize_xy_clamps():
    b = {'xmin': 0.0, 'xmax': 10.0, 'ymin': 0.0, 'ymax': 10.0}
    nx, ny = ws.normalize_xy(5.0, 5.0, b)
    assert 0.0 <= nx <= 1.0 and 0.0 <= ny <= 1.0
    nx2, _ = ws.normalize_xy(-100.0, 5.0, b)
    assert nx2 == 0.0

def test_build_worldstate_shape():
    con = db.connect()
    rows = db.timeseries_for(con, config.GAME_REF)
    dps = features.build_decision_points(rows, config.GAME_REF)
    con.close()
    b = features.field_bounds(dps)
    w = ws.build_worldstate(dps[0], b)
    assert 't' in w and 'robots' in w and 'team' in w
    s7 = w['robots'][7]
    assert s7['zone'] in ws.ZONE_LABELS
    assert len(s7['norm_xy']) == 2
    assert 0.0 <= s7['hp_ratio'] <= 1.0
    assert s7['team'] == '红'
