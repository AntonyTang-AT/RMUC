import numpy as np
import pytest

from forecast.mode_change_xgb import (
    FEATURE_DIM,
    ModeChangeXGBPredictor,
    build_mode_change_features,
)
from forecast.temporal import MODES


def _sample():
    base = np.zeros((5, 20), dtype=np.float32)
    base[:, 2] = [0.0, 0.1, 0.2, 0.8, 0.9]
    base[:, 3] = [0.0, 0.0, 0.1, 0.1, 0.1]
    base[:, 4] = [0.0, 0.0, 0.0, 0.2, 0.2]
    base[:, 5] = [1.0, 1.0, 0.9, 0.8, 0.7]
    base[:, 6] = [-0.5, -0.25, 0.25, 1.1, 1.2]
    base[:, 7] = [2.0, 2.0, 3.0, 4.0, 4.0]
    base[:, 8] = [2.0, 1.0, 1.0, 1.0, 0.0]
    times = np.array([0, 3, 9, 18, 30], dtype=np.float64)
    modes = np.array([0, 0, 1, 2, 2], dtype=np.int64)
    groups = np.array(["gA-red", "gA-red", "gA-red", "gB-blue", "gA-red"])
    return base, times, modes, groups


def test_causal_feature_builder_shape_and_values():
    base, times, modes, groups = _sample()
    x = build_mode_change_features(base, times, modes, groups)
    assert x.shape == (5, FEATURE_DIM) == (5, 42)
    # Row 2 uses the earliest observation in its six-second lookback window
    # (t=3), while row 0 has no prior observation.
    assert x[0, 20 + 2 + len(MODES)] == 0.0
    assert x[2, 20 + 2 + len(MODES)] == pytest.approx((0.25 - (-0.25)) * 8)
    # Time-in-mode is reset on a mode transition and group isolation applies.
    assert x[2, -1] == 0.0
    assert x[3, -1] == 0.0
    assert x[4, -1] == 0.0  # mode 2 began at this issue time


def test_future_rows_and_other_groups_cannot_change_existing_features():
    base, times, modes, groups = _sample()
    original = build_mode_change_features(base, times, modes, groups)
    extra_base = np.full((1, 20), 123.0, dtype=np.float32)
    extended = build_mode_change_features(
        np.concatenate((base, extra_base)),
        np.concatenate((times, [400.0])),
        np.concatenate((modes, [1])),
        np.concatenate((groups, ["gA-red"])),
    )
    np.testing.assert_array_equal(original, extended[:len(base)])


def test_tuple_group_keys_are_supported_for_game_and_side_streams():
    base, times, modes, _ = _sample()
    groups = [(1, "红"), (1, "红"), (1, "红"), (2, "蓝"), (1, "红")]
    x = build_mode_change_features(base, times, modes, groups)
    np.testing.assert_array_equal(x, build_mode_change_features(base, times, modes,
                                                                 np.asarray(groups, dtype=object)))


def test_invalid_or_noncausal_streams_are_rejected():
    base, times, modes, groups = _sample()
    with pytest.raises(ValueError, match="unique"):
        build_mode_change_features(base, np.array([0, 3, 3, 18, 30]), modes, groups)
    with pytest.raises(ValueError, match="shape"):
        build_mode_change_features(base[:, :19], times, modes, groups)


def test_saved_probe_artifact_loads_on_cpu_and_returns_valid_probabilities():
    from pathlib import Path

    path = Path("out_m2/mode_change_xgb_probe/report.model.ubj")
    if not path.exists():
        pytest.skip("exploratory XGBoost artifact is not present")
    predictor = ModeChangeXGBPredictor(path)
    x = np.zeros((2, FEATURE_DIM), dtype=np.float32)
    probs = predictor.predict_proba(x, np.array([0, 1]))
    assert probs.shape == (2, len(MODES))
    assert np.isfinite(probs).all()
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-5)
