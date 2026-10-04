from __future__ import annotations

import pytest

from eval.audit_validation_global_horizon import (
    HISTORICAL_REPORT,
    assert_new_output_path,
    contract_consistent_pair,
    exact_second,
)


def test_issue_and_future_modes_use_same_explicit_contract():
    dead_push = {
        "entity_id": "红:步兵3", "team": "红", "entity_type": "步兵3",
        "position": [22.0, 7.0], "hp": 0.0, "maxhp": 300.0,
    }
    enemy = {
        "entity_id": "蓝:步兵3", "team": "蓝", "entity_type": "步兵3",
        "position": [20.0, 7.0], "hp": 100.0, "maxhp": 300.0,
    }
    red_base = {
        "entity_id": "红:基地", "team": "红", "entity_type": "基地",
        "position": [1.0, 7.0], "hp": 5000.0, "maxhp": 5000.0,
    }
    blue_base = {
        "entity_id": "蓝:基地", "team": "蓝", "entity_type": "基地",
        "position": [27.0, 7.0], "hp": 5000.0, "maxhp": 5000.0,
    }
    pair = contract_consistent_pair(
        [dead_push, red_base], [enemy, blue_base],
        [dead_push, red_base], [enemy, blue_base], "红")

    # Legacy counts the zero-HP push unit in both frames; death-aware excludes
    # it in both frames. Each paired comparison uses one consistent contract.
    assert pair["legacy"][0] == pair["legacy"][1]
    assert pair["death_aware"][0] == pair["death_aware"][1]
    assert pair["legacy"][0] != pair["death_aware"][0]


def test_exact_second_refuses_rounded_nearest_timestamp():
    assert exact_second(10.0, key="issued_at") == 10
    with pytest.raises(ValueError, match="not an exact integer second"):
        exact_second(10.5, key="issued_at")


def test_audit_refuses_historical_report_destination(tmp_path):
    fresh = assert_new_output_path(tmp_path / "new-report.json")
    assert fresh.name == "new-report.json"
    with pytest.raises(ValueError, match="historical mixed-contract report"):
        assert_new_output_path(HISTORICAL_REPORT)
