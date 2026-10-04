# -*- coding: utf-8 -*-
"""mode_student shapes 单测；无 torch 则 skip。"""
import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

torch = pytest.importorskip('torch')

from rl import mode_student as ms
from rl import mode_features as mf


def test_net_forward_shape():
    net = ms._build_net('cpu')
    x = torch.zeros(4, mf.OBS_DIM)
    y = net(x)
    assert tuple(y.shape) == (4, 3)


def test_train_and_predict_roundtrip(tmp_path):
    X = np.random.randn(64, mf.OBS_DIM).astype(np.float32)
    y = np.random.randint(0, 3, size=(64,), dtype=np.int64)
    stu, stats = ms.train_mode_student(X, y, epochs=2, batch=16, device='cpu')
    assert stats['n'] == 64
    mode = stu.predict_mode(X[0])
    assert mode in mf.MODES
    out = tmp_path / 'stu'
    stu.save(str(out))
    stu2 = ms.ModeStudent.load(str(out), device='cpu')
    assert stu2.predict_mode(X[0]) == mode


def test_macro_f1_perfect():
    y = np.array([0, 1, 2, 0, 1, 2])
    assert abs(ms.macro_f1(y, y) - 1.0) < 1e-6


def test_smoother_majority():
    from rl.mode_student import ModeSmoother
    s = ModeSmoother(3)
    assert s.push('进攻') == '进攻'
    assert s.push('防守') in ('进攻', '防守')
    assert s.push('进攻') == '进攻'


def test_predict_proba_sums():
    X = np.random.randn(8, mf.OBS_DIM).astype(np.float32)
    y = np.random.randint(0, 3, size=(8,), dtype=np.int64)
    stu, _ = ms.train_mode_student(X, y, epochs=1, batch=8, device='cpu')
    mode, conf, probs = stu.predict_proba(X[0])
    assert mode in mf.MODES
    assert 0.0 < conf <= 1.0
    assert abs(sum(probs.values()) - 1.0) < 1e-3
