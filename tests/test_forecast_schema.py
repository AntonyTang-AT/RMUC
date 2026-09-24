"""Contract tests for archived probability forecasts."""

import math

import pytest

from eval.forecast_schema import validate_forecast


def forecast(**changes):
    row = {
        "prediction_id": "p1",
        "game_id": 1,
        "side": "红",
        "issued_at": 10,
        "valid_at": 20,
        "horizon": 10,
        "task": "mode",
        "entity_or_region": "global",
        "target_schema_version": "v1",
        "input_snapshot_hash": "abc123",
        "visibility_summary": {"radar_tracks": 1},
        "model_version": "fixture",
        "calibration_version": "none",
        "labels": ["攻", "守"],
        "probabilities": {"攻": 0.7, "守": 0.3},
    }
    row.update(changes)
    return row


def test_accepts_complete_categorical_forecast():
    assert validate_forecast(forecast()) == (True, [])


@pytest.mark.parametrize("field", [
    "prediction_id", "game_id", "side", "issued_at", "valid_at", "horizon",
    "task", "entity_or_region", "target_schema_version", "input_snapshot_hash",
    "visibility_summary", "model_version", "calibration_version",
])
def test_rejects_missing_required_field(field):
    row = forecast()
    del row[field]
    ok, errors = validate_forecast(row)
    assert not ok and any(field in error for error in errors)


def test_rejects_probability_vector_with_wrong_sum():
    ok, errors = validate_forecast(forecast(probabilities={"攻": 0.6, "守": 0.6}))
    assert not ok and any("sum" in error for error in errors)


def test_rejects_non_red_blue_side_and_nonstandard_horizon():
    assert not validate_forecast(forecast(side="绿"))[0]
    assert not validate_forecast(forecast(valid_at=25, horizon=15))[0]


def test_rejects_unhashable_labels_without_raising():
    ok, errors = validate_forecast(forecast(labels=["攻", ["守"]]))
    assert not ok and any("labels" in error for error in errors)


def test_requires_target_identity_and_calibration_version():
    row = forecast()
    del row["entity_or_region"]
    del row["calibration_version"]
    ok, errors = validate_forecast(row)
    assert not ok
    assert any("entity_or_region" in error for error in errors)
    assert any("calibration_version" in error for error in errors)


def test_extreme_integer_values_are_reported_invalid_without_raising():
    huge = 10 ** 1000
    assert not validate_forecast(forecast(horizon=huge))[0]
    row = forecast(task="position", quantiles={huge: 2.0}, bounds=[0.0, 3.0])
    del row["labels"]
    del row["probabilities"]
    assert not validate_forecast(row)[0]


def test_rejects_two_competing_categorical_distribution_fields():
    row = forecast(probability_distribution={"攻": 0.1, "守": 0.9})
    assert not validate_forecast(row)[0]


@pytest.mark.parametrize("probabilities", [
    {"攻": 1.0},
    {"攻": -0.1, "守": 1.1},
    {"攻": math.nan, "守": 0.3},
    {"攻": math.inf, "守": 0.3},
    {"攻": True, "守": 0.0},
])
def test_rejects_incomplete_or_nonfinite_probability_vectors(probabilities):
    assert not validate_forecast(forecast(probabilities=probabilities))[0]


def test_rejects_undeclared_probability_label():
    assert not validate_forecast(forecast(probabilities={"攻": 0.5, "守": 0.3, "撤": 0.2}))[0]


def test_rejects_malformed_metadata_and_time_order():
    assert not validate_forecast(forecast(visibility_summary=[]))[0]
    assert not validate_forecast(forecast(valid_at=9))[0]
    assert not validate_forecast(forecast(horizon=9))[0]


def test_accepts_ordered_finite_continuous_quantiles():
    row = forecast(task="position", quantiles={"0.1": 2.0, "0.5": 4.0, "0.9": 7.0}, bounds=[0.0, 10.0])
    del row["labels"]
    del row["probabilities"]
    assert validate_forecast(row) == (True, [])


@pytest.mark.parametrize("quantiles,bounds", [
    ({"0.1": 4.0, "0.5": 2.0}, [0.0, 10.0]),
    ({"0.9": 7.0, "0.1": 2.0}, [0.0, 10.0]),
    ({"-0.1": 2.0, "0.5": 4.0}, [0.0, 10.0]),
    ({"0.1": math.nan, "0.5": 4.0}, [0.0, 10.0]),
    ({"0.1": 2.0, "0.5": 4.0}, [0.0, math.inf]),
    ({"0.1": 2.0, "0.5": 4.0}, [5.0, 1.0]),
])
def test_rejects_malformed_continuous_quantiles_or_bounds(quantiles, bounds):
    row = forecast(task="position", quantiles=quantiles, bounds=bounds)
    del row["labels"]
    del row["probabilities"]
    assert not validate_forecast(row)[0]


def test_rejects_non_mapping_row_without_raising():
    ok, errors = validate_forecast(None)
    assert not ok and errors
