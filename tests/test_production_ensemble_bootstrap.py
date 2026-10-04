import numpy as np

from eval.evaluate_production_ensemble import _paired_game_bootstrap


def test_paired_game_bootstrap_requires_whole_game_red_blue_evidence():
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int64)
    production = np.array([
        [.9, .1], [.1, .9], [.9, .1], [.1, .9],
        [.9, .1], [.1, .9], [.9, .1], [.1, .9],
    ])
    persistence = np.array([
        [.1, .9], [.9, .1], [.1, .9], [.9, .1],
        [.1, .9], [.9, .1], [.1, .9], [.9, .1],
    ])
    sides = np.array(['红', '红', '蓝', '蓝', '红', '红', '蓝', '蓝'])
    game_ids = np.array([10, 10, 10, 10, 11, 11, 11, 11], dtype=np.int64)

    result = _paired_game_bootstrap(
        y, production, persistence, sides, game_ids, ('a', 'b'),
        replicates=80, seed=7)

    assert result['available'] is True
    assert result['n_complete_games'] == 2
    assert result['metrics']['accuracy']['ci_excludes_zero_in_favor_of_production'] is True
    assert result['metrics']['macro_f1']['ci_excludes_zero_in_favor_of_production'] is True
    assert result['metrics']['brier']['ci_excludes_zero_in_favor_of_production'] is True
