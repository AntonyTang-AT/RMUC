# tests/test_make_rollout_records.py
import numpy as np
from finetune import make_dataset as md
from schema import layered as sl


def _dp():
    def rob(rid, typ, team, x, hp, maxhp):
        return {'type': typ, 'team': team, 'x': x, 'y': 50.0, 'hp': hp, 'maxhp': maxhp,
                'power': 0.0, 'small_heat': 0.0, 'big_heat': 0.0, 'vuln': 0.0,
                'ammo17': 0.0, 'ammo42': 0.0, 'dfire': 0.0, 'yaw': 0.0, 'alive': 1}
    return {'t': 0.0, 'robots': {
        'r1': rob('r1', '哨兵', '红', 30.0, 300.0, 400.0),
        'r2': rob('r2', '英雄', '红', 28.0, 200.0, 240.0),
        'b1': rob('b1', '步兵3', '蓝', 70.0, 275.0, 275.0)}, 'team': {'红': {'coins_left': 1000}, '蓝': {'coins_left': 1000}}}


BOUNDS = {'xmin': 0.0, 'xmax': 100.0, 'ymin': 0.0, 'ymax': 100.0}


def test_records_from_rollout_shape_and_mode():
    import json
    dps = [_dp(), _dp()]
    fused = [{'global_mode': '均衡', 'units': {}} for _ in dps]
    win_model = (np.zeros(7), np.zeros(1))
    recs = md.records_from_rollout(dps, BOUNDS, fused, win_model, T=2, team='红')
    assert len(recs) == len(dps)
    for r in recs:
        assert 'instruction' in r and 'completion' in r
        d = json.loads(r['completion'])
        assert d['global_mode'] in sl.GLOBAL_MODES
        assert set(d['units']) == set(sl.UNIT_CLASSES)
