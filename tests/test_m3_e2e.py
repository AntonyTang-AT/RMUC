# tests/test_m3_e2e.py
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import config
from data import db, features
from worldstate import worldstate as ws
import run_m3

def test_compute_decision_series_valid():
    con = db.connect()
    rows = db.timeseries_for(con, config.GAME_REF)
    dps = features.build_decision_points(rows, config.GAME_REF)
    con.close()
    b = features.field_bounds(dps)
    decisions, prior, sigs = run_m3.compute_decision_series(dps, b)
    assert len(decisions) == len(sigs)
    from schema import layered as sl
    for d in decisions:
        ok, errs = sl.validate(d)
        assert ok, errs
    # 合规率应为 1.0 (我们生成时即保证合法)
    assert run_m3.compliance_rate(decisions) == 1.0

def test_prior_mines_downstream_move():
    con = db.connect()
    rows = db.timeseries_for(con, config.GAME_REF)
    dps = features.build_decision_points(rows, config.GAME_REF)
    con.close()
    b = features.field_bounds(dps)
    decisions, prior, sigs = run_m3.compute_decision_series(dps, b)
    from schema import layered as sl
    assert all(p['decision'] in sl.MOVES for p in prior)   # 挖掘的是单位 move, 非 self-referential global_mode
