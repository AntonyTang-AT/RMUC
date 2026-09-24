"""Append-only forecast archive tests."""

import copy
import json

import pytest

from eval.prediction_ledger import append_prediction, load_predictions, mature_predictions
from test_forecast_schema import forecast


def test_append_prediction_preserves_archived_distribution(tmp_path):
    path = tmp_path / "predictions.jsonl"
    row = forecast(probabilities={"攻": 0.7, "守": 0.3})
    append_prediction(path, row)
    row["probabilities"]["攻"] = 0.1
    archived = load_predictions(path)
    assert archived[0]["probabilities"] == {"攻": 0.7, "守": 0.3}
    assert archived[0]["calibration_version"] == "none"


def test_append_prediction_adds_one_line_without_overwriting(tmp_path):
    path = tmp_path / "predictions.jsonl"
    append_prediction(path, forecast())
    append_prediction(path, forecast(prediction_id="p2"))
    assert [row["prediction_id"] for row in load_predictions(path)] == ["p1", "p2"]
    assert len(path.read_text(encoding="utf-8").splitlines()) == 2


def test_rejects_invalid_prediction_without_writing(tmp_path):
    path = tmp_path / "predictions.jsonl"
    with pytest.raises(ValueError, match="probabilities"):
        append_prediction(path, forecast(probabilities={"攻": 0.6, "守": 0.6}))
    assert not path.exists()


def test_future_prediction_is_not_mature():
    row = forecast(issued_at=10, valid_at=20)
    assert mature_predictions([row], now=19) == []
    assert mature_predictions([row], now=20) == [row]


def test_maturity_does_not_change_archived_probabilities_or_calibration():
    rows = [forecast(), forecast(prediction_id="p2", valid_at=30, horizon=20)]
    original = copy.deepcopy(rows)
    mature = mature_predictions(rows, now=20)
    assert mature == [original[0]]
    assert rows == original
    mature[0]["probabilities"]["攻"] = 0.2
    assert rows == original


def test_load_preserves_malformed_archived_record_for_diagnostics(tmp_path):
    path = tmp_path / "predictions.jsonl"
    path.write_text(json.dumps({"prediction_id": "bad", "valid_at": 5}) + "\n", encoding="utf-8")
    assert load_predictions(path) == [{"prediction_id": "bad", "valid_at": 5}]
