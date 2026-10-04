# -*- coding: utf-8 -*-
"""rl.forward 前向动力学纯逻辑单测(无 torch, 离线可跑)。"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from rl import forward as fd


def _w(team='红', cls='英雄'):
    """最简 worldstate: 一个单位 + 一个建筑。"""
    return {'robots': {0: {'type': cls, 'team': team, 'norm_xy': [0.3, 0.5],
                           'hp_ratio': 0.8, 'heat': 0.1, 'dfire': 0.0}},
            'buildings': {0: {'type': '基地', 'team': '红', 'hp_ratio': 1.0}},
            'team': {'红': {'coins': 1.0}, '蓝': {'coins': 1.0}}}


def _dec(move='推进', fire=False):
    return {'units': {'英雄': {'move': move, 'fire': fire}}}


def test_state_features_dim():
    assert fd.state_features(_w()).shape == (fd.D_STATE,)


def test_action_features_dim_and_move_onehot():
    a = fd.action_features(_dec('推进'))
    assert a.shape == (fd.D_ACTION,)
    cls0 = fd.MOVES.index('推进')
    assert a[cls0] == 1.0
    # 6 类各占一个 move 独热 → 6 个 1; 其余(fire 槽 + 其他 move 槽)为 0
    assert a.sum() == len(fd.UNIT_CLASSES)


def test_design_matrix_dim():
    assert fd.design_matrix(_w(), _dec()).shape == (fd.D_IN,)


def test_fit_predict_roundtrip_synthetic_linear():
    # 合成"真"线性动力学 Y = A@X + b, 用 ridge 学回 -> 训练集内近似精确
    rng = np.random.RandomState(0)
    n, din, dout = 200, fd.D_IN, fd.D_STATE
    X = rng.randn(n, din)
    A = rng.randn(dout, din) * 0.3
    b = rng.randn(dout) * 0.2
    Y = X @ A.T + b
    m = fd.ForwardDynamics(l2=1e-6).fit(X, Y)
    assert np.sqrt(np.mean((m.predict(X) - Y) ** 2)) < 1e-6


def test_fit_generalizes_off_distribution():
    # 模型应在"训练分布附近"泛化, 而不是噪声: 用低 L2 + 干净线性, 留出集 RMSE 远小于输出方差
    rng = np.random.RandomState(1)
    n, din, dout = 300, fd.D_IN, fd.D_STATE
    X = rng.randn(n, din)
    A = rng.randn(dout, din)
    Y = X @ A.T
    m = fd.ForwardDynamics(l2=1e-8).fit(X, Y)
    Xv = rng.randn(50, din)
    Yv = Xv @ A.T
    out_var = float(np.mean(np.var(Yv, axis=0)))
    err = float(np.mean((m.predict(Xv) - Yv) ** 2))
    assert err < out_var * 1e-2, (err, out_var)


def test_next_state_vector_length_and_stable():
    m = fd.ForwardDynamics(l2=1e-4)
    X = np.zeros((10, fd.D_IN)); Y = np.zeros((10, fd.D_STATE))
    m.fit(X, Y)
    v = m.next_state_vector(_w(), _dec())
    assert v.shape == (fd.D_STATE,) and np.all(np.isfinite(v))


def test_worldstate_from_vec_structure():
    v = fd.state_features(_w('红', '英雄'))
    w2 = fd.worldstate_from_vec(v)
    robots = [r for r in w2['robots'].values()]
    assert len(robots) == 1 and robots[0]['type'] == '英雄' and robots[0]['team'] == '红'
    assert abs(robots[0]['norm_xy'][0] - 0.3) < 1e-9
    assert abs(w2['team']['红']['coins'] - 1.0) < 1e-9


def test_rmse_and_capture():
    pred = np.array([[1.0, 3.0], [2.0, 4.0]])
    real = np.array([[1.0, 4.0], [2.0, 5.0]])
    nochange = np.array([[0.0, 2.0], [0.0, 2.0]])
    assert abs(fd.rmse(pred, real) - np.sqrt(0.5)) < 1e-6
    rep = fd.realism_report(pred, real, nochange)
    assert rep['capture'] is not None and rep['capture'] > 0
