# -*- coding: utf-8 -*-
"""M2 奖励学权重 + 部分可观测 离线测试(无 torch/transformers)。"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from rl import reward as rw
from rl import reward_learn as rl_le
from rl import env as renv
from worldstate import worldstate as ws


def _dp(t, sentry_x=900.0, enum_x=100.0, sentry_hp=1000.0):
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
    }
    return {'t': float(t), 'robots': robots,
            'team': {'红': {'coins_total': 0.0, 'coins_left': 0.0}, '蓝': {'coins_total': 0.0, 'coins_left': 0.0}}}


def _bounds():
    return {'xmin': -500.0, 'xmax': 1400.0, 'ymin': -500.0, 'ymax': 500.0}


def _w(dp):
    return ws.build_worldstate(dp, _bounds())


# -------- reward 组件 --------
def test_reward_components_keys_and_sign():
    w = _w(_dp(0.0))
    c = rw.reward_components(w, '红')
    assert set(c.keys()) == set(rw.feat_names())
    # 满状态: 哨兵满血存活、敌方建筑满血 -> outpost_def/base_def/ego 为正, 攻击为0
    assert c['ego_surv'] == 1.0 and c['ego_hp'] == 1.0
    assert c['outpost_def'] == 1.0 and c['base_def'] == 1.0
    assert c['outpost_atk'] == 0.0 and c['base_atk'] == 0.0


def test_compute_reward_weighted():
    w = _w(_dp(0.0))
    val = rw.compute_reward(w, rw.DEFAULT_CONF, '红')
    assert isinstance(val, float)
    # 无权重项(空 conf) -> 0
    assert rw.compute_reward(w, {}, '红') == 0.0


def test_reward_features_sums_over_states():
    ws_list = [_w(_dp(0.0)), _w(_dp(3.0))]
    acc = rw.reward_features(ws_list, '红')
    single = rw.reward_features([_w(_dp(0.0))], '红')
    for k in acc:
        assert acc[k] == single[k] + single[k]  # 两个相同状态 -> 两倍


# -------- 逻辑回归 --------
def test_fit_logistic_separates_by_base_atk():
    # 构造: base_atk(打爆敌方基地) 越高, 胜率越高
    rng = np.random.default_rng(0)
    n = 200
    base_atk = rng.uniform(0, 1, n)
    # 敌方基地血越低(1-base_atk 越高)胜面越大; 加噪声
    y = (base_atk + rng.normal(0, 0.2, n) > 0.5).astype(float)
    X = np.stack([base_atk, rng.uniform(0, 1, n), rng.uniform(0, 1, n)], axis=1)
    w_un, b_un, _ = rl_le.fit_logistic(X, y, l2=1e-3)
    p = rl_le.predict_proba(X, w_un, b_un)
    a = rl_le.auc(y, p)
    assert a > 0.9                      # 特征能区分胜负
    assert w_un[0] > 0                  # base_atk 权重应为正


# -------- 部分可观测 M2Env --------
def test_m2env_obs_dim_matches_and_visibility():
    # 哨兵在 x=900, 敌方工程在 x=100(远) -> 不可见
    e = renv.M2Env([_dp(0.0)], _bounds(), vis_radius=8.0)
    obs = e.obs('哨兵')
    assert obs.shape == (e.obs_dim,)
    assert len(e.action_meanings) == 12
    # 敌方工程块 5 维, 最后是 vis; 远->vis=0, 位置隐藏为中心
    fe = e._enemy_feat(e.ws[0], '工程', e._ego_sentry_xy(e.ws[0]))
    assert fe[4] == 0.0
    assert fe[1] == 0.5 and fe[2] == 0.5
    # 半径放大到覆盖全部 -> 可见, 位置保留
    e2 = renv.M2Env([_dp(0.0)], _bounds(), vis_radius=2000.0)
    fe2 = e2._enemy_feat(e2.ws[0], '工程', e2._ego_sentry_xy(e2.ws[0]))
    assert fe2[4] == 1.0
    assert [fe2[1], fe2[2]] == _w(_dp(0.0))['robots']['100']['norm_xy']


# -------- learned 权重 -> 可用 reward_conf 映射 --------
def test_weights_json_to_conf_rebase_and_zero_collinear():
    base = rw.DEFAULT_CONF
    rows = [{'feature': 'base_atk', 'learned': 0.001, 'effect': 2.0},
            {'feature': 'kill', 'learned': 0.002, 'effect': 1.0},
            {'feature': 'ego_hp', 'learned': 0.001, 'effect': 0.3},
            {'feature': 'econ', 'learned': 0.0005, 'effect': 0.2}]
    conf = rl_le.weights_json_to_conf(rows)
    # 共线特征(death/dmg_take 未用于逻辑回归)必须清零, 避免与 kill/dmg_deal 双重计数
    assert conf['death'] == 0.0 and conf['dmg_take'] == 0.0
    # 相对重要性(按 effect)保留: base_atk > kill > ego_hp > econ
    assert conf['base_atk'] > conf['kill'] > conf['ego_hp'] > conf['econ']
    # 量纲回正到 DEFAULT_CONF: 非零权的均值幅度落在 hand 幅度的一半~两倍之间(可比 per-step 奖励)
    used = [r['feature'] for r in rows]
    lmean = float(np.mean([abs(conf[k]) for k in used]))
    bmean = float(np.mean([abs(base[k]) for k in used]))
    assert 0.5 * bmean < lmean < 2.0 * bmean


def test_weights_json_to_conf_from_file(tmp_path):
    import json
    rows = [{'feature': 'econ', 'learned': 0.001, 'effect': 3.0},    # 经济重要性高
            {'feature': 'base_atk', 'learned': 0.001, 'effect': 0.5}]  # 打基地重要性低
    p = tmp_path / 'w.json'
    p.write_text(json.dumps({'weights': rows}), encoding='utf-8')
    conf = rl_le.weights_json_to_conf(str(p))
    # 相对重要性(按 effect)保留: econ > base_atk
    assert conf['econ'] > conf['base_atk']
    assert conf['econ'] > rw.DEFAULT_CONF['econ']           # 高经济重要性 -> 高于默认
    assert conf['base_atk'] < rw.DEFAULT_CONF['base_atk']   # 低打基地重要性 -> 低于默认
    assert conf['kill'] == 0.0                              # 未用特征清零
