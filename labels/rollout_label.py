# -*- coding: utf-8 -*-
"""反事实最优承诺标注器: 用"符号前向 + 敌方博弈应对 + 单点胜负机"给每个决策点标最优 global_mode。

什么是"最优承诺": 对决策点, 我方枚举三个候选 global_mode(进攻/均衡/防守), 每个模式提交给一个
T 帧符号推演; 敌方按"对我方提交模式的最优应对(best_response)"行动; 终点用单点胜负机(win_proba)
打"红方胜算"分; 取分最高的模式 = 该点标签。即"若我方宣布 m, 对手最优应 e, 预期胜率 Z -> 选 Z 最大"。

同构于 labels/strategy_label.relabel_decision_series: 返回的 decisions 与 fused_decisions 等长,
global_mode 被替换为 rollout 最优标签, units 由 derive_unit_intent 重派生(整体自洽)。
纯 numpy, 无 torch; 单测用合成决策点, 无 data.db。
"""
import numpy as np
import config
from schema import layered as sl
from tree import global_mode as gm
from rl import physics as ph
from game import game_theory as gt
from worldstate import worldstate as ws
from data.features import BUILDING_TYPES
from eval import win_proba as wp

# 全局模式与 PAYOFF_3 行/列对齐
GLOBAL_MODES = list(sl.GLOBAL_MODES)
PAYOFF_3 = gt.PAYOFF_3


def enemy_best_response(our_mode, matrix=PAYOFF_3):
    """敌方对我方提交纯策略 our_mode 的**对抗性**最优应对模式: 选让"我方 payoff 最小"的敌方模式。
    我方行固定为 i, 敌方在列 j 中挑最小 PAYOFF_3[i][j] —— 敌方损我方(而非早期 argmax 版本反而帮我方)。"""
    i = GLOBAL_MODES.index(our_mode)
    row_i = matrix[i]           # 我方行固定 i: 每个敌方选择 j 下的我方 payoff
    return GLOBAL_MODES[min(range(3), key=lambda j: row_i[j])]


def worldstate_from_live(live, dp_t, bounds):
    """由物理态 live(step_robots 输出) 重建 worldstate; 排除阵亡单位, 保留建筑。"""
    robots = {}
    for rid, r in live.items():
        if r['type'] in BUILDING_TYPES:
            robots[rid] = r
        elif r.get('hp', 0.0) > 0.0:
            robots[rid] = r
    dp = {'t': dp_t['t'], 'robots': robots, 'team': dp_t['team']}
    return ws.build_worldstate(dp, bounds)


def commit_actions(w, mode, team='红'):
    """team 侧六单位类 -> derive_unit_intent 的 (move, fire, target)。缺失类补 '无'/False/'无'。
    target 供 rl.physics 做**意图化目标选择**(攻击追建筑/防守只打缠斗)。"""
    cls_by_team = {r['type']: r for r in w['robots'].values() if r['team'] == team}
    acts = {}
    for cls in sl.UNIT_CLASSES:
        unit = cls_by_team.get(cls)
        if unit is None:
            acts[cls] = {'move': '无', 'fire': False, 'target': '无'}
        else:
            it = gm.derive_unit_intent(unit, w, mode)
            acts[cls] = {'move': it['move'], 'fire': bool(it['fire']), 'target': it['target']}
    return acts


def rollout_once(dp0, bounds, my_mode, team='红', T=5, dt=None, params=None):
    """以 dp0 为起点, 我方按 my_mode、敌方按 enemy_best_response(my_mode) 推演 T 帧。
    每帧: 由当前 live 重建 worldstate -> 派生双方单位动作 -> 我方 step、敌方 step。
    返回 [w0...wT](T+1 个 worldstate)。承诺的是"模式", 单位级动作每帧随 hp/位置重派生。"""
    dt = dt if dt is not None else config.DF_STEP
    my_team = team
    en_team = '蓝' if team == '红' else '红'
    live = {rid: dict(r) for rid, r in dp0['robots'].items()}
    e_m = enemy_best_response(my_mode)
    traj = [worldstate_from_live(live, dp0, bounds)]
    for _ in range(T):
        w = worldstate_from_live(live, dp0, bounds)
        my_act = commit_actions(w, my_mode, my_team)
        en_act = commit_actions(w, e_m, en_team)
        live = ph.step_robots(live, my_team, my_act, dt, params, enemy_fire=False)
        live = ph.step_robots(live, en_team, en_act, dt, params, enemy_fire=False)
        traj.append(worldstate_from_live(live, dp0, bounds))
    return traj


def relabel_decisions_rollout(ws_list, dps, bounds, fused_decisions, win_model,
                              T=5, team='红'):
    """逐决策点: 三模式各自 rollout -> win_proba 打分 -> argmax 标签 -> 派生产出 decisions。
    win_model=(w,b) 来自 eval/win_proba.fit。返回 (decisions, labels, scores);
    decisions 与 fused_decisions 等长, global_mode+units 被 rollout 标签替换。"""
    w_m, b_m = win_model
    grid = []
    for i, dp in enumerate(dps):
        gm_scores = {}
        for m in GLOBAL_MODES:
            traj = rollout_once(dp, bounds, m, team, T)
            feats = np.array([wp.window_features(traj, team)], float)
            gm_scores[m] = float(wp.predict_proba(feats, w_m, b_m)[0])
        lbl = max(gm_scores, key=gm_scores.get)
        w = ws_list[i] if i < len(ws_list) else ws.build_worldstate(dp, bounds)
        units = {}
        for cls in sl.UNIT_CLASSES:
            unit = next((r for r in w['robots'].values()
                         if r['team'] == team and r['type'] == cls), None)
            if unit is None:
                units[cls] = {'target': '无', 'move': '无', 'fire': False, 'risk': '低'}
            else:
                units[cls] = gm.derive_unit_intent(unit, w, lbl)
        grid.append({'global_mode': lbl, 'units': units})
    decisions = grid
    labels = [d['global_mode'] for d in grid]
    scores = [{m: round(v, 4) for m, v in gm_scores.items()} for gm_scores in
              [_scores(dps, bounds, win_model, i, T, team) for i in range(len(dps))]]
    return decisions, labels, scores


def _scores(dps, bounds, win_model, i, T, team):
    w_m, b_m = win_model
    out = {}
    for m in GLOBAL_MODES:
        traj = rollout_once(dps[i], bounds, m, team, T)
        feats = np.array([wp.window_features(traj, team)], float)
        out[m] = round(float(wp.predict_proba(feats, w_m, b_m)[0]), 4)
    return out
