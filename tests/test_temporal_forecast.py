import numpy as np
import torch

from forecast.model import HORIZON_POOL, TemporalForecastNet
from forecast.calibration import enforce_monotonic_horizon_confidence
from forecast.temporal import FEATURE_DIM, HORIZON_PROFILES, feature_vector, target_labels
from forecast.entity import ENTITY_SLOTS, ENTITY_TARGET_DIM


def _example():
    return {
        "game_id": 1, "side": "红", "issued_at": 10.0,
        "observable": {
            "self_entities": {
                "红:1": {"entity_type": "英雄", "position": [2, 7], "hp": 100, "maxhp": 100,
                         "team_coins_total": 500, "team_coins_left": 400},
                "红:10": {"entity_type": "基地", "hp": 5000, "maxhp": 5000},
            },
            "opponent_entities": {
                "蓝:101": {"entity_type": "英雄", "position": [23, 7], "hp": 100},
            },
        },
        "targets": {"10": {"target_at": 20, "censored": False, "truth": {
            "entities": {
                "红:1": {"team": "红", "entity_type": "英雄", "position": [15, 7], "hp": 100, "maxhp": 100},
                "红:10": {"team": "红", "entity_type": "基地", "position": [0, 0], "hp": 5000, "maxhp": 5000},
                "蓝:101": {"team": "蓝", "entity_type": "英雄", "position": [18, 7], "hp": 100, "maxhp": 100},
            }
        }}}
    }


def test_feature_dimension_and_causal_target():
    row = _example()
    assert feature_vector(row).shape == (FEATURE_DIM,)
    labels = target_labels(row, 10)
    assert labels["opponent_trend"] == "推进"
    assert labels["enemy_region"] == "x2y1"


def test_temporal_model_heads():
    model = TemporalForecastNet()
    outputs = model(torch.zeros((2, 4, FEATURE_DIM), dtype=torch.float32))
    assert set(outputs) == {10, 30, 60, 120}
    assert outputs[10]["global_mode"].shape == (2, 3)
    assert outputs[120]["enemy_region"].shape == (2, 12)
    assert HORIZON_POOL == {10: 1, 30: 2, 60: 3, 120: 6}
    assert [HORIZON_PROFILES[h]["detail_level"] for h in (10, 30, 60, 120)] == [
        "high", "medium_high", "medium", "coarse"
    ]


def test_horizon_calibration_widens_overconfident_farther_distribution():
    calibrated = enforce_monotonic_horizon_confidence({
        10: [0.60, 0.25, 0.15],
        30: [0.80, 0.10, 0.10],
        60: [0.70, 0.20, 0.10],
        120: [0.90, 0.05, 0.05],
    })
    maxima = [max(calibrated[h]) for h in (10, 30, 60, 120)]
    assert all(a >= b - 1e-6 for a, b in zip(maxima, maxima[1:]))
    assert all(abs(sum(calibrated[h]) - 1.0) < 1e-6 for h in (10, 30, 60, 120))


def test_temperature_calibration_flattens_far_horizon():
    from forecast.calibration import calibrate_horizon_probabilities
    calibrated = calibrate_horizon_probabilities({
        10: [0.9, 0.05, 0.05], 30: [0.9, 0.05, 0.05],
        60: [0.9, 0.05, 0.05], 120: [0.9, 0.05, 0.05],
    })
    assert max(calibrated[120]) <= max(calibrated[10]) + 1e-6
    assert calibrated[120][1] > calibrated[10][1]


def test_runtime_classification_softmaxes_logits_before_calibration_and_blending():
    from runtime.temporal_forecast import TemporalForecastRuntime

    logits = {h: torch.tensor([2.0, -1.0, 0.5]) for h in (10, 30, 60, 120)}
    result = TemporalForecastRuntime._classification(
        logits, ('进攻', '均衡', '防守'), lambda values: values)
    expected = torch.softmax(logits[10], dim=0).numpy()
    observed = np.array([result['10']['probabilities'][name]
                         for name in ('进攻', '均衡', '防守')])
    assert np.allclose(observed, expected, atol=1e-6)
    assert np.isclose(observed.sum(), 1.0)

    prior = {'global_mode': {str(h): '防守' for h in (10, 30, 60, 120)}}
    fused = TemporalForecastRuntime._classification(
        logits, ('进攻', '均衡', '防守'), lambda values: values,
        baseline_labels=prior, task='global_mode',
        blend_config={'alphas': {f'global_mode@{h}s': 0.0 for h in (10, 30, 60, 120)}})
    assert fused['30']['top'] == '防守'
    assert fused['30']['probabilities']['防守'] == 0.98


def test_temporal_model_can_emit_entity_slots():
    model = TemporalForecastNet(entity_slots=len(ENTITY_SLOTS))
    outputs = model(torch.zeros(2, 12, FEATURE_DIM))
    assert outputs[10]["entity_state"].shape == (2, len(ENTITY_SLOTS), ENTITY_TARGET_DIM)


def test_validation_position_fusion_runtime_is_opt_in_and_reversible():
    from runtime.temporal_forecast import (TemporalForecastRuntime,
                                           VALIDATION_POSITION_FUSION_WEIGHTS)

    default = TemporalForecastRuntime("unused")
    candidate = TemporalForecastRuntime("unused", apply_validation_position_fusion=True)
    assert default.apply_validation_position_fusion is False
    assert candidate.apply_validation_position_fusion is True
    assert VALIDATION_POSITION_FUSION_WEIGHTS == {10: .25, 30: .75, 60: .75, 120: 1.0}
