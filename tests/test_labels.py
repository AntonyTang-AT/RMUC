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
