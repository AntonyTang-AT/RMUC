import numpy as np
import pytest
from eval import win_proba as wp
from rl import reward_learn as rl_le


def _mk(red_base, blue_base, red_hp=1.0, blue_hp=1.0, red_alive=6, blue_alive=6):
    """最小红方 worldstate(直接符合 build_worldstate 输出 schema)。"""
    def rob(typ, team, hp_ratio, x):
        return {'type': typ, 'team': team, 'hp_ratio': hp_ratio, 'norm_xy': [x, 0.5],
                'dfire': 0.0, 'ammo17': 0.0, 'ammo42': 0.0}
    robots = {}
    for i, typ in enumerate(['英雄', '工程', '步兵3', '步兵4', '空中', '哨兵']):
        if i < red_alive:
            robots[f'r{i}'] = rob(typ, '红', red_hp, 0.3)
        if i < blue_alive:
            robots[f'b{i}'] = rob(typ, '蓝', blue_hp, 0.7)
    buildings = {
        'rb': {'type': '基地', 'team': '红', 'hp_ratio': red_base},
        'bb': {'type': '基地', 'team': '蓝', 'hp_ratio': blue_base},
        'ro': {'type': '前哨站', 'team': '红', 'hp_ratio': 1.0},
        'bo': {'type': '前哨站', 'team': '蓝', 'hp_ratio': 1.0},
    }
    return {'t': 0.0, 'team': {'红': {'coins': 1000}, '蓝': {'coins': 1000}},
            'robots': robots, 'buildings': buildings}


def test_window_features_monotonic_with_base_hp():
    good = _mk(red_base=1.0, blue_base=0.2)   # 敌基地残血 -> red_base_atk 高
    bad = _mk(red_base=0.1, blue_base=1.0)    # 己基地残血 -> red_base_def 低
    f_good = wp.window_features([good], '红')
    f_bad = wp.window_features([bad], '红')
    assert len(f_good) == 7
    i_base_atk = 4   # WIN_FEATURES 里 base_atk 是第 5 位(索引不假设顺序, 用字典查)
    # 用列名直接断言: 敌方基地掉血 -> base_atk 分量更大
    names = wp.WIN_FEATURES
    ia = names.index('base_atk'); ib = names.index('base_def')
    assert f_good[ia] > f_bad[ia]           # 打敌方基地更有利
    assert f_good[ib] > f_bad[ib]           # 己方基地满血更有利(base_def 更大)


def test_fit_predict_correlates_and_cv_finite():
    rng = np.random.default_rng(1)
    n = 120
    X = rng.normal(size=(n, 7))
    w_true = np.array([1.2, -0.8, 0.5, 0.0, 2.0, -1.0, 0.3])
    z = X @ w_true + 0.3 * rng.normal(size=n)
    y = (1.0 / (1.0 + np.exp(-z)) > 0.5).astype(float)
    gids = np.repeat(np.arange(20), 6)      # 20 局 × 6 点

    w, b, meta = wp.fit(X, y, seed=0)
    p = wp.predict_proba(X, w, b)
    assert 0.7 < rl_le.auc(y, p) <= 1.0
    assert meta['n'] == n and meta['n_pos'] > 0

    cv = wp.grouped_cv(X, y, gids, k=5, seed=0)
    assert 0.5 <= cv['cv_auc'] <= 1.0
    assert cv['n_folds'] == 5
