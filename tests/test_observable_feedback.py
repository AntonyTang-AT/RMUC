import pytest

from runtime.observable_feedback import feedback_labels_from_observable


def _snap(time, own_x, enemy_x, *, enemy_state="current", hidden_trend="回撤"):
    return {
        "issued_at": float(time),
        # Deliberately conflicting oracle data: feedback must ignore it.
        "targets": {"10": {"truth": {"opponent_trend": hidden_trend}}},
        "entities": {
            "red:engineer": {
                "team": "红", "role": "self", "entity_type": "工程",
                "source": "sensor", "observability_state": "current",
                "observed_at": float(time), "position": [own_x, 7.5],
                "hp": 200, "life_state": "active",
            },
            "blue:engineer": {
                "team": "蓝", "role": "opponent", "entity_type": "工程",
                "source": "radar", "observability_state": enemy_state,
                "observed_at": float(time), "position": [enemy_x, 7.5],
                "hp": 200, "life_state": "active",
            },
        },
    }


def test_matured_feedback_uses_only_visible_state_and_side_relative_trend():
    issue = _snap(0, 3.0, 20.0, hidden_trend="回撤")
    matured = _snap(10, 3.2, 18.0, hidden_trend="回撤")

    labels = feedback_labels_from_observable(issue, matured, "红", 10)

    assert labels["opponent_trend"] == "推进"
    assert labels["enemy_region"] == "x2y1"
    assert labels["global_mode"] in {"进攻", "均衡", "防守"}


def test_feedback_rejects_observations_before_forecast_maturity():
    with pytest.raises(ValueError, match="has not matured"):
        feedback_labels_from_observable(_snap(0, 3, 20), _snap(9.99, 3, 19),
                                        "红", 10)


def test_unobserved_enemy_is_not_fabricated_as_a_feedback_label():
    labels = feedback_labels_from_observable(
        _snap(0, 3, 20), _snap(10, 3, 18, enemy_state="stale"), "红", 10)
    assert labels == {"global_mode": None,
                      "opponent_trend": None,
                      "enemy_region": None}


def test_future_dated_sensor_or_radar_rows_are_excluded():
    issue = _snap(0, 3, 20)
    matured = _snap(10, 3, 18)
    matured["entities"]["blue:engineer"]["observed_at"] = 11
    matured["entities"]["blue:engineer"]["received_at"] = 11

    labels = feedback_labels_from_observable(issue, matured, "红", 10)

    assert labels["global_mode"] is None
    assert labels["enemy_region"] is None
    assert labels["opponent_trend"] is None


def test_runtime_matures_pending_forecasts_automatically_from_next_snapshot():
    from runtime.temporal_forecast import TemporalForecastRuntime

    class Router:
        def __init__(self):
            self.calls = []

        def observe_outcome(self, forecast_id, truth, *, matured_horizons=None):
            self.calls.append((forecast_id, truth, list(matured_horizons or [])))
            return {"updated": sum(bool(values) for values in truth.values())}

        def reset(self):
            self.calls.clear()

    router = Router()
    runtime = TemporalForecastRuntime("unused", team="红", adaptive_router=router)
    runtime._observable_feedback_pending["f0"] = {
        "issued_at": 0.0, "snapshot": _snap(0, 3, 20), "scored": set()}
    runtime._observable_feedback_order.append("f0")

    update = runtime._mature_observable_feedback(_snap(10, 3, 18))

    assert router.calls[0][0] == "f0"
    assert router.calls[0][1]["opponent_trend"]["10"] == "推进"
    assert router.calls[0][2] == [10]
    assert update["matured_task_horizon_outcomes"] == 3
    assert runtime.feedback_status()["pending_forecast_records"] == 1

    runtime.reset()
    assert runtime.feedback_status()["pending_forecast_records"] == 0
    assert runtime.feedback_status()["completed_forecast_records"] == 0
    assert router.calls == []


def test_match_local_residual_calibrator_is_context_gated_and_resettable():
    from runtime.adaptive_strategy import MatchLocalOpponentResidualCalibrator

    calibrator = MatchLocalOpponentResidualCalibrator(window=4, prior_strength=2)
    probabilities = {"推进": 0.2, "回撤": 0.5, "停留": 0.3}
    untouched, gated = calibrator.apply(probabilities, "advance")
    assert untouched == probabilities
    assert gated["applied"] is False

    result = calibrator.observe(probabilities, "hold_or_mixed", "推进")
    assert result == {"updated": True, "samples": 1}
    calibrated, meta = calibrator.apply(probabilities, "hold_or_mixed")
    assert meta["applied"] is True
    assert calibrated["推进"] > probabilities["推进"]
    assert sum(calibrated.values()) == pytest.approx(1.0)

    calibrator.reset()
    reset_output, reset_meta = calibrator.apply(probabilities, "hold_or_mixed")
    assert reset_output == probabilities
    assert reset_meta["reason"] == "warmup"


def test_runtime_calibrator_uses_only_matured_visible_feedback_and_issued_probs():
    from runtime.adaptive_strategy import MatchLocalOpponentResidualCalibrator
    from runtime.temporal_forecast import TemporalForecastRuntime

    calibrator = MatchLocalOpponentResidualCalibrator()
    runtime = TemporalForecastRuntime(
        "unused", team="红", opponent_residual_calibrator=calibrator)
    issued_probabilities = {"推进": 0.2, "回撤": 0.5, "停留": 0.3}
    runtime._observable_feedback_pending["f0"] = {
        "issued_at": 0.0,
        "snapshot": _snap(0, 3, 20, hidden_trend="回撤"),
        "scored": set(),
        "opponent_trend_probabilities": issued_probabilities,
        "opponent_behavior_context": "hold_or_mixed",
    }
    runtime._observable_feedback_order.append("f0")

    update = runtime._mature_observable_feedback(
        _snap(10, 3, 18, hidden_trend="回撤"))

    assert update["matured_task_horizon_outcomes"] == 1
    assert calibrator.sample_count("hold_or_mixed") == 1
    # Visible motion says advance; deliberately conflicting hidden target is ignored.
    calibrated, meta = calibrator.apply(issued_probabilities, "hold_or_mixed")
    assert meta["applied"] is True
    assert calibrated["推进"] > issued_probabilities["推进"]
    assert runtime.feedback_status()["enabled"] is True

    runtime.reset()
    assert calibrator.sample_count("hold_or_mixed") == 0


def test_runtime_behavior_context_uses_side_relative_visible_motion():
    from runtime.temporal_forecast import TemporalForecastRuntime

    runtime = TemporalForecastRuntime("unused", team="红")
    previous = {
        "issued_at": 0.0,
        "observable": {"opponent_entities": {
            "blue:engineer": {
                "entity_id": "blue:engineer", "entity_type": "工程",
                "observability_state": "current", "source": "radar",
                "position": [0.52, 0.50], "life_state": "active",
            },
        }},
    }
    runtime._behavior_observable_history.append(previous)
    current = {
        "issued_at": 3.0,
        "observable": {"opponent_entities": {
            "blue:engineer": {
                "entity_id": "blue:engineer", "entity_type": "工程",
                "observability_state": "current", "source": "radar",
                "position": [0.50, 0.50], "life_state": "active",
            },
        }},
    }
    assert runtime._opponent_behavior_context(current) == "advance"
