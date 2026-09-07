import sqlite3
from config import DB_PATH

def connect():
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    return con

def timeseries_for(con, game_id):
    return con.execute("SELECT * FROM timeseries WHERE game_id=?", (game_id,)).fetchall()

def events_for(con, game_id):
    return con.execute("SELECT * FROM events WHERE game_id=?", (game_id,)).fetchall()

def match_meta(con, game_id):
    return con.execute("SELECT * FROM matches WHERE game_id=?", (game_id,)).fetchone()
