from eval.audit_decision_metric_validity import summarize


def test_metric_validity_requires_rows_and_games_for_all_modes():
    rows = []
    for mode in ('进攻', '防守', '均衡'):
        for game_id in range(1, 4):
            rows.append({
                'game_id': game_id,
                'side': '红',
                'expected_global_mode': mode,
                'label_stable_next_5': True,
            })
    result = summarize(rows, min_mode_rows=3, min_mode_games=3)
    assert result['红']['mode_row_coverage_pass'] is True
    assert result['红']['mode_game_coverage_pass'] is True
    assert result['红']['label_stability_next_5']['coverage'] == 1.0
    assert result['蓝']['mode_row_coverage_pass'] is False
