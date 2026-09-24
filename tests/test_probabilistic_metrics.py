import math

import pytest

from eval.probabilistic_metrics import (
    aggregate_game_scores,
    aggregate_sides,
    bootstrap_game_clusters,
    calibration_bins,
    cluster_bootstrap_side_delta,
    crps_ensemble,
    energy_score,
    multiclass_brier,
    multiclass_log_loss,
)


def test_multiclass_scores_match_hand_computed_single_and_batch_values():
    labels = ['攻', '守']
    assert multiclass_brier('攻', {'攻': .8, '守': .2}, labels) == pytest.approx(.08)
    assert multiclass_brier(['攻', '守'], [
        {'攻': .8, '守': .2}, {'攻': .4, '守': .6},
    ], labels) == pytest.approx((.08 + .32) / 2)
    assert multiclass_log_loss('攻', {'攻': .8, '守': .2}, labels) == pytest.approx(-math.log(.8))
    assert multiclass_log_loss(['攻', '守'], [
        {'攻': .8, '守': .2}, {'攻': .4, '守': .6},
    ], labels) == pytest.approx(-math.log(.8 * .6) / 2)


def test_multiclass_scores_reject_invalid_categories_and_distributions():
    for fn in (multiclass_brier, multiclass_log_loss):
        with pytest.raises(ValueError):
            fn('unknown', {'攻': .8, '守': .2}, ['攻', '守'])
        with pytest.raises(ValueError):
            fn('攻', {'攻': .8}, ['攻', '守'])
        with pytest.raises(ValueError):
            fn('攻', {'攻': .8, '守': .8}, ['攻', '守'])
        with pytest.raises(ValueError):
            fn('攻', {'攻': float('nan'), '守': .2}, ['攻', '守'])


def test_log_loss_clips_only_zero_probability_for_logarithm():
    assert multiclass_log_loss('攻', {'攻': 0, '守': 1}, ['攻', '守']) == pytest.approx(-math.log(1e-15))
    assert multiclass_log_loss('攻', {'攻': 1, '守': 0}, ['攻', '守']) == 0


def test_empirical_crps_and_energy_score_use_pairwise_dispersion():
    # X={0,2}, y=1: E|X-y|=1, E|X-X'|/2=.5.
    assert crps_ensemble(1, [0, 2]) == pytest.approx(.5)
    # X={(0,0),(2,0)}, y=(1,0) reduces to the same value.
    assert energy_score([1, 0], [[0, 0], [2, 0]]) == pytest.approx(.5)
    assert crps_ensemble(3, [3]) == 0


def test_crps_rejects_boolean_observation():
    with pytest.raises(ValueError):
        crps_ensemble(True, [0, 1])


def test_side_macro_is_equal_weight_and_micro_uses_point_counts():
    scores = aggregate_sides({'红': {'score': .2, 'n': 100}, '蓝': {'score': .8, 'n': 10}})
    assert scores['red_view']['score'] == .2
    assert scores['blue_view']['score'] == .8
    assert scores['side_macro_average']['score'] == pytest.approx(.5)
    assert scores['pooled_micro']['score'] == pytest.approx(28 / 110)
    assert scores['pooled_micro']['n_points'] == 110


def test_game_aggregation_weights_games_within_side_then_sides_equally():
    result = aggregate_game_scores({
        'g1': {'红': {'score': .2, 'n': 100}, '蓝': {'score': .8, 'n': 10}},
        'g2': {'红': {'score': .4, 'n': 20}, '蓝': {'score': .6, 'n': 20}},
    })
    assert set(result['sides']) == {'红', '蓝'}
    assert result['sides']['红']['score'] == pytest.approx(28 / 120)
    assert result['sides']['蓝']['score'] == pytest.approx(20 / 30)
    assert result['side_macro_average']['score'] == pytest.approx(((28 / 120) + (20 / 30)) / 2)
    assert result['pooled_micro']['score'] == pytest.approx(48 / 150)
    assert result['pooled_micro']['n_points'] == 150


def test_bootstrap_draws_entire_games_with_red_blue_views_together():
    rows = {
        'g1': [{'game_id': 'g1', 'side': '红', 'score': .1}, {'game_id': 'g1', 'side': '蓝', 'score': .2}],
        'g2': [{'game_id': 'g2', 'side': '红', 'score': .3}, {'game_id': 'g2', 'side': '蓝', 'score': .4}],
    }
    draws = bootstrap_game_clusters(rows, n_bootstrap=20, seed=0)
    assert len(draws) == 20
    assert all(len(draw) == 2 for draw in draws)
    assert all(all({r['side'] for r in group} == {'红', '蓝'} for group in draw) for draw in draws)
    assert draws == bootstrap_game_clusters(rows, n_bootstrap=20, seed=0)


def test_clustered_delta_keeps_side_pairing_and_returns_interval():
    rows = {
        'g1': [{'side': '红', 'score': 1, 'n': 1}, {'side': '蓝', 'score': 3, 'n': 1}],
        'g2': [{'side': '红', 'score': 2, 'n': 1}, {'side': '蓝', 'score': 4, 'n': 1}],
    }
    result = cluster_bootstrap_side_delta(rows, metric='score', n_bootstrap=40, seed=7)
    assert result['point_estimate'] == pytest.approx(-2)
    assert result['ci_low'] <= result['point_estimate'] <= result['ci_high']
    assert result['n_games'] == 2
    assert result['n_bootstrap'] == 40


def test_calibration_bins_include_counts_and_observed_frequencies():
    bins = calibration_bins([.1, .4, .8, .9], [0, 1, 0, 1], n_bins=2)
    assert [b['count'] for b in bins] == [2, 2]
    assert bins[0]['mean_confidence'] == pytest.approx(.25)
    assert bins[0]['observed_frequency'] == pytest.approx(.5)
