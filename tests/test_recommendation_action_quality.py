from eval.audit_recommendation_action_quality import _paired_action


def test_paired_action_bootstrap_weights_issued_recommendations_not_unit_type_means():
    rows = [
        {'game_id': 1, 'side': '红', 'unit_type': '步兵3', 'goal_n': 100,
         'goal_hits': 80, 'stay_goal_hits': 100},
        {'game_id': 1, 'side': '红', 'unit_type': '哨兵', 'goal_n': 1,
         'goal_hits': 0, 'stay_goal_hits': 1},
        {'game_id': 1, 'side': '蓝', 'unit_type': '步兵3', 'goal_n': 100,
         'goal_hits': 90, 'stay_goal_hits': 100},
    ]

    result = _paired_action(rows, 'goal_hits')

    # Red uses recommendation-level pooling: (80 - 101) / 101. The rare
    # sentinel sample must not receive the same weight as 100 infantry rows.
    assert abs(result['by_side']['红']['mean_delta'] - (-21 / 101)) < 1e-12
    # Only games containing both colors enter the red/blue-balanced estimate.
    assert abs(result['red_blue_equal_game_paired_bootstrap']['mean_delta'] -
               (((-21 / 101) + (-10 / 100)) / 2)) < 1e-12
