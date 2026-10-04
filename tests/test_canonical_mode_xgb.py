import numpy as np

from eval.train_canonical_mode_xgb import _features, _fused


def test_canonical_xgb_features_use_latest_frame_and_causal_difference():
    x = np.zeros((2, 3, 232), dtype=np.float32)
    x[:, -2, 0] = 0.2
    x[:, -1, 0] = 0.4
    base = np.array([0, 2], dtype=np.int64)
    issued = np.array([3.0, 30.0], dtype=np.float32)
    out = _features(x, base, issued)
    assert out.shape == (2, 468)
    assert np.allclose(out[:, 0], 0.4)
    assert np.allclose(out[:, 232], 0.2)
    assert np.allclose(out[:, 464:467], np.eye(3, dtype=np.float32)[base])


def test_fused_probabilities_keep_current_mode_as_persistence_prior():
    p = np.array([[0.1, 0.8, 0.1], [0.8, 0.1, 0.1]], dtype=np.float32)
    base = np.array([0, 2], dtype=np.int64)
    fused = _fused(p, base, 0.0)
    assert np.argmax(fused, axis=1).tolist() == [0, 2]
    assert np.allclose(fused.sum(axis=1), 1.0)
