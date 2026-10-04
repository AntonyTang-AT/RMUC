# -*- coding: utf-8 -*-
"""outcome-based backtest 纯函数单测(无 db / torch / transformers)。

只测可独立验证的逻辑:
  - cv_auc_win_model: 合成可分特征 -> held-out AUC 应高; 纯随机 -> 应≈0.5。
  - per_frame_fav / fav_from_worldstate: 有利度对"红方更胜势"单调。
  - fav_by_decision / alignment_gap: def 应落在低有利度(负 gap), 纯分组求和正确。
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from eval import outcome_backtest as ob
from rl import reward as rw
from worldstate import worldstate as ws


# ---- 复用 test_reward 的合成决策点 ----
def _dp(t, sentry_x=900.0, enum_x=100.0, sentry_hp=1000.0, own_base_hp=1.0, en_base_hp=1.0):
    robots = {
        '7': {'type': '哨兵', 'team': '红', 'x': sentry_x, 'y': 0.0, 'hp': sentry_hp, 'maxhp': 1000.0,
              'small_heat': 0.0, 'big_heat': 0.0, 'power': 0.0, 'ammo17': 0.0, 'ammo42': 0.0,
              'vuln': 0, 'team_coins_total': 0.0, 'team_coins_left': 0.0},
        '100': {'type': '工程', 'team': '蓝', 'x': enum_x, 'y': 0.0, 'hp': 2000.0, 'maxhp': 2000.0,
                'small_heat': 0.0, 'big_heat': 0.0, 'power': 0.0, 'ammo17': 0.0, 'ammo42': 0.0,
                'vuln': 0, 'team_coins_total': 0.0, 'team_coins_left': 0.0},
        'b1': {'type': '前哨站', 'team': '红', 'x': 0.0, 'y': 0.0, 'hp': 1500.0, 'maxhp': 1500.0,
               'small_heat': 0.0, 'big_heat': 0.0, 'power': 0.0, 'ammo17': 0.0, 'ammo42': 0.0,
               'vuln': 0, 'team_coins_total': 0.0, 'team_coins_left': 0.0},
        'b2': {'type': '基地', 'team': '红', 'x': 0.0, 'y': 0.0, 'hp': own_base_hp * 5000.0, 'maxhp': 5000.0,
               'small_heat': 0.0, 'big_heat': 0.0, 'power': 0.0, 'ammo17': 0.0, 'ammo42': 0.0,
               'vuln': 0, 'team_coins_total': 0.0, 'team_coins_left': 0.0},
        'b3': {'type': '基地', 'team': '蓝', 'x': 0.0, 'y': 0.0, 'hp': en_base_hp * 5000.0, 'maxhp': 5000.0,
               'small_heat': 0.0, 'big_heat': 0.0, 'power': 0.0, 'ammo17': 0.0, 'ammo42': 0.0,
               'vuln': 0, 'team_coins_total': 0.0, 'team_coins_left': 0.0},
    }
    return {'t': float(t), 'robots': robots,
            'team': {'红': {'coins_total': 0.0, 'coins_left': 0.0}, '蓝': {'coins_total': 0.0, 'coins_left': 0.0}}}


def _bounds():
    return {'xmin': -500.0, 'xmax': 1400.0, 'ymin': -500.0, 'ymax': 500.0}


def _w(dp):
    return ws.build_worldstate(dp, _bounds())


# -------- A. 交叉验证 --------
def test_cv_separable_high_auc():
    # base_atk 越强越易胜 -> 可分, held-out AUC 高
    rng = np.random.default_rng(1)
    n, d = 200, 5
    base_atk = rng.uniform(0, 1, n)
    y = (base_atk + rng.normal(0, 0.2, n) > 0.5).astype(float)
    X = np.column_stack([base_atk, rng.uniform(0, 1, (n, d - 1))])
    res = ob.cv_auc_win_model(X, y, k=5, seed=1)
    assert res['cv_auc'] > 0.85
    assert res['n_folds'] >= 4                       # 有类缺失的 fold 被跳过, 至少 4 个有效


def test_cv_random_near_half():
    # 纯随机标签 -> AUC≈0.5
    rng = np.random.default_rng(2)
    n = 120
    X = rng.normal(0, 1, (n, 4))
    y = (rng.uniform(0, 1, n) > 0.5).astype(float)
    res = ob.cv_auc_win_model(X, y, k=5, seed=2)
    assert 0.35 < res['cv_auc'] < 0.65


# -------- B. 有利度 + 决策条件 --------
def test_fav_monotone_with_red_advantage():
    # 红方基地满血 vs 红方基地残血 => 满血有利度更高
    fav_full = ob.fav_from_worldstate(_w(_dp(0.0, own_base_hp=1.0)), '红')
    fav_low = ob.fav_from_worldstate(_w(_dp(0.0, own_base_hp=0.2)), '红')
    assert fav_full > fav_low
    # 敌方基地被打残 => 对红方更胜势
    fav_enhurt = ob.fav_from_worldstate(_w(_dp(0.0, en_base_hp=0.1)), '红')
    assert fav_enhurt > fav_full - 1e-9


def test_per_frame_fav_uses_win_features_only():
    comp = rw.reward_components(_w(_dp(0.0)), '红')
    fav = ob.per_frame_fav(comp)
    assert fav == sum(comp[k] for k in ob.WIN_FEATURES)
    # 与全家福不同: 只算胜相关子集
    assert set(ob.WIN_FEATURES) <= set(rw.feat_names())
    assert 'ego_surv' not in ob.WIN_FEATURES and 'heat' not in ob.WIN_FEATURES


def test_fav_by_decision_groups_and_gap():
    # 人为构造: def 落在低有利度, cruise 落在高有利度
    pairs = [('def', 0.1), ('def', 0.3), ('cruise', 0.8), ('cruise', 0.9), ('hitbuilding', 0.7)]
    by = ob.fav_by_decision(pairs)
    assert by['def']['count'] == 2 and by['cruise']['count'] == 2
    assert abs(by['def']['mean_fav'] - 0.2) < 1e-9
    gap = ob.alignment_gap(pairs)
    assert gap is not None and gap < 0          # def 有利度低于 cruise -> 负 gap 合理
    # 没有 def 时 gap 为 None
    assert ob.alignment_gap([('cruise', 0.5)]) is None
