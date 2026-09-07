# tests/test_db.py
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import config
from data import db

def test_connect_and_meta():
    con = db.connect()
    meta = db.match_meta(con, config.GAME_REF)
    assert meta is not None
    assert meta['game_id'] == config.GAME_REF
    assert meta['红方学校'] and meta['蓝方学校']
    con.close()

def test_timeseries_nonempty():
    con = db.connect()
    rows = db.timeseries_for(con, config.GAME_REF)
    assert len(rows) > 1000
    con.close()

def test_events_for_nonempty():
    con = db.connect()
    rows = db.events_for(con, config.GAME_REF)
    assert len(rows) > 0
    con.close()
