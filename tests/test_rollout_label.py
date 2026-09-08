import numpy as np
import pytest
from schema import layered as sl
from labels import rollout_label as rl
from tree import global_mode as gm


def _dp():
    """最小红方哨兵 + 蓝方步兵决策点(字段与 features.build_decision_points 一致)。"""
    def rob(rid, typ, team, x, hp, maxhp):
        return {'type': typ, 'team': team, 'x': x, 'y': 50.0, 'hp': hp, 'maxhp': maxhp,
                'power': 0.0, 'small_heat': 0.0, 'big_heat': 0.0, 'vuln': 0.0,
                'ammo17': 0.0, 'ammo42': 0.0, 'dfire': 0.0, 'yaw': 0.0, 'alive': 1}
    robots = {
        'r1': rob('r1', '哨兵', '红', 30.0, 300.0, 400.0),
        'r2': rob('r2', '英雄', '红', 28.0, 200.0, 240.0),
        'b1': rob('b1', '步兵3', '蓝', 70.0, 275.0, 275.0),
    }
    team = {'红': {'coins_left': 1000}, '蓝': {'coins_left': 1000}}
    return {'t': 0.0, 'robots': robots, 'team': team}


BOUNDS = {'xmin': 0.0, 'xmax': 100.0, 'ymin': 0.0, 'ymax': 100.0}


def test_enemy_best_response_adversarial():
    # 对抗性应对: 我方行固定, 敌方挑最小化我方 payoff 的列。
    # PAYOFF_3 = [[-0.5,0.3,0.8],[0,0,0.2],[-0.3,-0.5,0.5]]
    # 进攻(row0): min(-0.5,0.3,0.8)=-0.5 -> 列0=进攻
    # 均衡(row1): min(0,0,0.2)=0 -> 列0=进攻
    # 防守(row2): min(-0.3,-0.5,0.5)=-0.5 -> 列1=均衡
    assert rl.enemy_best_response('进攻') == '进攻'
    assert rl.enemy_best_response('均衡') == '进攻'
    assert rl.enemy_best_response('防守') == '均衡'


def test_commit_actions_legal_and_fire_bool():
    dp = _dp()
    from worldstate import worldstate as ws
    w = ws.build_worldstate(dp, BOUNDS)
    acts = rl.commit_actions(w, '进攻', '红')
    assert set(acts) == set(sl.UNIT_CLASSES)
    for cls, a in acts.items():
        assert a['move'] in sl.MOVES
        assert isinstance(a['fire'], bool)


def test_rollout_once_move_changes_position():
    from worldstate import worldstate as ws
    dt = 3.0
    x0 = _sentry_x(ws.build_worldstate(_dp(), BOUNDS))      # norm_xy (0-1) 基准
    # 我方哨兵推演推进(x 应增大, 红主场低x 推敌=+x); 全程 norm 对 norm
    traj_push = rl.rollout_once(_dp(), BOUNDS, '进攻', '红', T=3, dt=dt, params=None)
    assert len(traj_push) == 4
    x_push = _sentry_x(traj_push[-1])
    # 回防(模式=防守)应让 x 减小
    traj_def = rl.rollout_once(_dp(), BOUNDS, '防守', '红', T=3, dt=dt, params=None)
    x_def = _sentry_x(traj_def[-1])
    assert x_push > x0
    assert x_def < x0
    # 敌方单位也参与: 我方攻时敌方最优应对='防守'->蓝回防(BLUE 主场高x, 回防=+x), 位置改变
    b0 = ws.build_worldstate(_dp(), BOUNDS)['robots']['b1']['norm_xy'][0]
    b_end = traj_push[-1]['robots']['b1']['norm_xy'][0]
    assert abs(b_end - b0) > 1e-6


def _sentry_x(w):
    for r in w['robots'].values():
        if r['type'] == '哨兵' and r['team'] == '红':
            return r['norm_xy'][0]
    return None


def test_relabel_decisions_rollout_shape_and_legal():
    from worldstate import worldstate as ws
    from eval import win_proba as wp
    dps = [_dp(), _dp()]
    ws_list = [ws.build_worldstate(d, BOUNDS) for d in dps]
    fused = [{'global_mode': '进攻', 'units': {}} for _ in dps]   # 占位
    # 假 win_model: 常数 0.5 -> argmax 应取第一个模式(进攻)
    win_model = (np.zeros(7), np.zeros(1))   # w=0,b=0 -> p=0.5 全等
    dec, labels, scores = rl.relabel_decisions_rollout(ws_list, dps, BOUNDS, fused, win_model, T=2, team='红')
    assert len(dec) == len(dps) and len(labels) == len(dps) and len(scores) == len(dps)
    assert all(l in sl.GLOBAL_MODES for l in labels)
    for d in dec:
        assert d['global_mode'] in sl.GLOBAL_MODES
        assert set(d['units']) == set(sl.UNIT_CLASSES)
    # 常数模型 -> 每个模式得分全 0.5, argmax 取 GLOBAL_MODES 第一个='进攻'
    assert all(l == '进攻' for l in labels)


def test_step_robots_enemy_fire_flag():
    """R-E6: enemy_fire=False 关返回敌火(防双算)。红方 action-fire 打蓝方掉血;
    而蓝方返回敌火被抑制 -> 红方自身零伤。对照组 enemy_fire=True: 红方经返回敌火掉血(证明测试确能区分标志)。"""
    from rl import physics as ph
    def mk():
        return {'r1': {'type': '哨兵', 'team': '红', 'x': 40.0, 'y': 50.0, 'hp': 300.0, 'maxhp': 400.0,
                       'power': 0.0, 'small_heat': 0.0, 'big_heat': 0.0, 'vuln': 0.0,
                       'ammo17': 0.0, 'ammo42': 0.0, 'dfire': 0.0, 'yaw': 0.0, 'alive': 1},
                'b1': {'type': '步兵3', 'team': '蓝', 'x': 46.0, 'y': 50.0, 'hp': 275.0, 'maxhp': 275.0,
                       'power': 0.0, 'small_heat': 0.0, 'big_heat': 0.0, 'vuln': 0.0,
                       'ammo17': 0.0, 'ammo42': 0.0, 'dfire': 0.0, 'yaw': 0.0, 'alive': 1}}
    act = {'哨兵': {'move': '无', 'fire': True}}
    off = ph.step_robots(mk(), '红', act, dt=3.0, enemy_fire=False)
    assert off['b1']['hp'] < 275.0   # 红方 action-fire 命中蓝方(合法) -> 蓝方掉血
    assert off['r1']['hp'] == 300.0  # 无返回敌火 -> 红方自身零伤
    on = ph.step_robots(mk(), '红', act, dt=3.0, enemy_fire=True)
    assert on['r1']['hp'] < 300.0    # 对照组: 蓝方返回敌火 -> 红方掉血, 证明标志可区分
