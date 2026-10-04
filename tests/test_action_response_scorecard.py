from eval.audit_all_history_action_response import confirmed_target_id_for_scoring


def test_only_sensor_confirmed_target_id_is_scorable():
    explicit = {
        "target_unit_id": "blue:4",
        "evaluation": {"target_linkage": "explicit_sensor_target"},
    }
    geometric_candidate = {
        "target_unit_id": "blue:4",
        "evaluation": {"target_linkage": "visible_candidate_only"},
    }
    unknown = {"target_unit_id": "blue:4", "evaluation": {}}

    assert confirmed_target_id_for_scoring(explicit) == "blue:4"
    assert confirmed_target_id_for_scoring(geometric_candidate) is None
    assert confirmed_target_id_for_scoring(unknown) is None
    assert confirmed_target_id_for_scoring(None) is None
