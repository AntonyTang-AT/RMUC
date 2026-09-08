# tests/test_make_rollout_steps.py
import numpy as np
from finetune import make_rollout_dataset as mrd


def test_label_consistency_identical_game_agrees():
    # 用同一 win_model 对同一批 dps 标两次不同 T; 恒等态势下标签应相近
    from labels import rollout_label as rrl
    from eval import win_proba as wp
    from worldstate import worldstate as ws
    dps = [_dp(), _dp(), _dp()]
    b = BOUNDS
    ws_list = [ws.build_worldstate(d, b) for d in dps]
    win_model = (np.zeros(7), np.zeros(1))
    d1 = rrl.relabel_decisions_rollout(ws_list, dps, b, [{}]*3, win_model, T=2)[1]
    d2 = rrl.relabel_decisions_rollout(ws_list, dps, b, [{}]*3, win_model, T=5)[1]
    agree = mrd.label_consistency(dps, b, win_model, [2, 5], '红')
    assert set(agree) == {2, 5}
    assert 0.0 <= agree[2] <= 1.0


def test_outcome_consistency_shape():
    labels = ['进攻', '防守', '进攻', '均衡']
    scores = {'进攻': 0.8, '防守': 0.2, '均衡': 0.5}
    res = mrd.outcome_consistency(
        [labels[0], labels[1], labels[2], labels[3]],
        [{m: scores[m] for m in ('进攻', '均衡', '防守')} for _ in labels], '红')
    msb = res['mean_score_by_label']
    assert '进攻' in msb and '防守' in msb
    assert res['op_vs_def_gap'] == round(msb['进攻'] - msb['防守'], 4)


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
