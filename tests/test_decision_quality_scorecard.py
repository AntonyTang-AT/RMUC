from eval.decision_quality_scorecard import (_bootstrap, _paired_game_delta,
                                             _expected_mode_counts, _hdq_by_expected_mode,
                                             _label_stable_next, _quality, _response_for_move,
                                             _weighted_hdq)
from eval.evaluate_visible_lora_context import _sample_decision_points
import pytest


def test_bootstrap_reports_complete_game_summary():
    result = _bootstrap([0.2, 0.4, 0.6])
    assert result['n_games'] == 3
    assert result['mean'] == pytest.approx(0.4)
    assert len(result['ci95']) == 2


def test_movement_response_is_mirrored_for_blue():
    current = {'position': [8.0, 0.0]}
    future = {'position': [9.0, 0.0]}
    assert _response_for_move('推进', '红', current, future) == 1.0
    assert _response_for_move('推进', '蓝', current, future) == 0.0


def test_hold_response_accepts_small_displacement():
    current = {'position': [8.0, 0.0]}
    future = {'position': [8.1, 0.0]}
    assert _response_for_move('坚守', '红', current, future) == 1.0


def test_paired_delta_compares_identical_game_rows():
    rows = [
        {'game_id': 1, 'side': '红', 'candidate': .8, 'baseline': .5},
        {'game_id': 1, 'side': '红', 'candidate': .6, 'baseline': .5},
        {'game_id': 2, 'side': '红', 'candidate': .4, 'baseline': .6},
    ]
    result = _paired_game_delta(rows, 'candidate', 'baseline')
    assert result['n_games'] == 2
    assert result['mean'] == pytest.approx(0.0)


def test_weighted_hdq_blends_global_and_unit_components():
    assert _weighted_hdq(1.0, 0.5, 0.4) == pytest.approx(0.7)
    assert _weighted_hdq(None, 0.5, 0.4) is None
    assert _weighted_hdq(1.0, None, 0.4) is None


def test_expected_mode_counts_keeps_absent_modes_explicit():
    counts = _expected_mode_counts([
        {'expected_global_mode': '进攻'},
        {'expected_global_mode': '进攻'},
        {'expected_global_mode': '防守'},
    ])
    assert counts == {'进攻': 2, '防守': 1, '均衡': 0}


def test_hdq_by_expected_mode_keeps_game_level_delta_groups():
    rows = [
        {'game_id': 1, 'side': '红', 'expected_global_mode': '进攻',
         'end_to_end_hdq': .8, 'baseline_hdq': .6},
        {'game_id': 2, 'side': '红', 'expected_global_mode': '防守',
         'end_to_end_hdq': .6, 'baseline_hdq': .5},
    ]
    result = _hdq_by_expected_mode(rows)
    assert result['进攻']['n_rows'] == 1
    assert result['进攻']['end_to_end_hdq_delta_vs_legacy_rule']['n_games'] == 1
    assert result['均衡']['end_to_end_hdq_delta_vs_legacy_rule']['n_games'] == 0


def test_label_stability_requires_all_next_five_labels_to_match():
    labels = ['进攻', '进攻', '进攻', '进攻', '进攻', '进攻', '防守']
    assert _label_stable_next(labels, 0) is True
    assert _label_stable_next(labels, 1) is False
    assert _label_stable_next(labels, 2) is None


def test_uniform_sampling_covers_early_middle_and_late_points():
    points = [{'t': float(index)} for index in range(10)]
    chosen = _sample_decision_points(points, 3, sampling='uniform')
    assert [row['t'] for row in chosen] == [0.0, 4.0, 9.0]


def test_quality_excludes_unseen_unit_from_action_denominator_and_checks_safety():
    world = {
        't': 1.0,
        'robots': {
            '蓝:1': {'entity_id': '蓝:1', 'team': '蓝', 'type': '英雄',
                     'hp_ratio': 1.0, 'norm_xy': [0.2, 0.5], 'dfire': 0.0},
            '蓝:2': {'entity_id': '蓝:2', 'team': '蓝', 'type': '工程',
                     'hp_ratio': 1.0, 'norm_xy': [0.2, 0.5], 'dfire': 0.0},
        }, 'buildings': {},
    }
    issue_state = {'robots': {
        '蓝:2': {'entity_id': '蓝:2', 'team': '蓝', 'type': '工程',
                 'observability_state': 'current', 'hp_ratio': 1.0,
                 'firing_inferred': False},
    }}
    decision = {'t': 1.0, 'global_mode': '均衡',
                'units': {name: {'target': '无', 'move': '无', 'fire': False, 'risk': '低'}
                          for name in ('英雄', '工程', '步兵3', '步兵4', '空中', '哨兵')}}
    result = _quality(decision, world, issue_state, '蓝', '均衡', '均衡')
    assert result['observable_unit_slots'] == 1
    assert result['hidden_slot_safety'] == pytest.approx(0.0)
