# -*- coding: utf-8 -*-
"""前向动力学 f(state_t, action_t) -> state_{t+1} —— B1: 从回放数据学, 复用语义状态。

为什么: rl/env.py 的 step() 之前只是 `t+=1` 跳到下一个真实决策点(replay-bandit), 动作不改未来,
导致 RL"收敛太快、质量≈规则"。前向动力学把环境变成**真 MDP**: 改变动作 -> 预测值 → 未来后果
进入价值, 覆盖的差距(vis_radius 部分观测 / zone 语义)与原观测层无关, 只是"下一步从哪来"变了。

B1 诚实前提(离线强化学习天花板): 数据里每个状态 t 只有**一个实际动作**; 本模型以 (state,
*我方*动作) 为输入、预测下一状态, 隐式学的是"当敌方按**平均行为**时"的条件转移动力学。
能对"我方动作"做反事实(改我方动作→不同未来), 但**不能**预测"敌方换策略"的后果 —— 那需要 B2
符号物理模拟。短期 3s 步的位移/掉血近似线性, 故先做**线性 ridge 基线**, 非线性(MLP)留作升级。

纯 numpy(无 torch), 单测离线可跑。向量布局可逆(可还原回 worldstate, 供 reward/obs 复用)。
"""
import numpy as np
from schema import layered as sl

UNIT_CLASSES = sl.UNIT_CLASSES
MOVES = sl.MOVES
TEAMS = ('红', '蓝')

SLOT = 6                                       # 每(类,队)槽: present,x,y,hp,heat,dfire
D_STATE = 2 * len(UNIT_CLASSES) * SLOT + 2 + 2  # 6类×2队×6=72 + buildings2 + coins2 = 76
D_ACTION = len(UNIT_CLASSES) * (len(MOVES) + 1)  # 6类×(6 move 独热 + fire) = 42
D_IN = D_STATE + D_ACTION


def _slot(w, cls, team):
    for r in w.get('robots', {}).values():
        if r.get('type') == cls and r.get('team') == team:
            return [1.0, r['norm_xy'][0], r['norm_xy'][1],
                    r['hp_ratio'], r.get('heat', 0.0), r.get('dfire', 0.0)]
    return [0.0, 0.5, 0.5, 0.0, 0.0, 0.0]


def state_features(w):
    """worldstate -> 固定 76 维向量。"""
    out = []
    for team in TEAMS:
        for cls in UNIT_CLASSES:
            out += _slot(w, cls, team)
    # 每队建筑最低血量(base/outpost 里最脆的)"
    red_hp, blue_hp = [1.0], [1.0]
    for b in w.get('buildings', {}).values():
        (red_hp if b.get('team') == '红' else blue_hp).append(b.get('hp_ratio', 1.0))
    out += [min(red_hp), min(blue_hp)]
    tm = w.get('team', {})
    out += [tm.get('红', {}).get('coins', 0.0), tm.get('蓝', {}).get('coins', 0.0)]
    return np.array(out, dtype=float)


def action_features(dec, team='红'):
    """决策 dict -> (6类 × 7) 42 维: 我方每类的 move 独热 + fire。"""
    out = []
    units = dec.get('units', {})
    for cls in UNIT_CLASSES:
        mv = units.get(cls, {}).get('move', '无')
        me = [0.0] * len(MOVES)
        me[MOVES.index(mv) if mv in MOVES else MOVES.index('无')] = 1.0
        fire = 1.0 if units.get(cls, {}).get('fire', False) else 0.0
        out += me + [fire]
    return np.array(out, dtype=float)


def design_matrix(w, dec, team='红'):
    return np.concatenate([state_features(w), action_features(dec, team)])


def worldstate_from_vec(v, team='红'):
    """76 维向量 -> 近似 worldstate(供 reward/obs 复用)。"""
    robots, buildings = {}, {}
    idx, rid = 0, 0
    for t2 in TEAMS:
        for cls in UNIT_CLASSES:
            present, nx, ny, hp, heat, dfire = v[idx:idx + 6]
            idx += 6
            if present > 0.5:
                robots[rid] = {'type': cls, 'team': t2, 'zone': '', 'zone_idx': [0, 0], 'depth': '',
                               'norm_xy': [float(nx), float(ny)], 'hp_ratio': float(hp),
                               'heat': float(heat), 'dfire': float(dfire)}
                rid += 1
    b_red, b_blue = v[idx:idx + 2]
    idx += 2
    buildings[0] = {'type': '基地', 'team': '红', 'hp_ratio': float(min(1.0, max(0.0, b_red)))}
    buildings[1] = {'type': '基地', 'team': '蓝', 'hp_ratio': float(min(1.0, max(0.0, b_blue)))}
    c_red, c_blue = v[idx:idx + 2]
    return {'robots': robots, 'buildings': buildings,
            'team': {'红': {'coins': float(c_red)}, '蓝': {'coins': float(c_blue)}}}


class ForwardDynamics:
    """线性(ridge)前向动力学: X=[state⊕action] -> Y=next_state。closed-form, 确定性, 快。

    特征/目标异量纲(coins≈1e3 vs 位置≈0.5), 闭式解在未归一化 Gram 上 λ 会失效, 故
    fit 时对 X、Y 逐维 z-score(存均方差, predict 反变换)。"""

    def __init__(self, l2=1.0, delta=False):
        self.l2 = l2
        self.delta = delta   # True: 学 next-current 的增量(聚焦动作边际效应), 输出再加回当前状态
        self.theta = None
        self.bias = None
        self.x_mean = None
        self.x_std = None
        self.y_mean = None
        self.y_std = None

    def _scale(self, X):
        return (X - self.x_mean) / self.x_std

    def fit(self, X, Y):
        X = np.asarray(X, dtype=float)
        Y = np.asarray(Y, dtype=float)
        if self.delta:
            Y = Y - X[:, :D_STATE]   # X 前 D_STATE 列=当前状态
        self.x_mean = X.mean(axis=0)
        self.x_std = X.std(axis=0) + 1e-9
        self.y_mean = Y.mean(axis=0)
        self.y_std = Y.std(axis=0) + 1e-9
        Xs = (X - self.x_mean) / self.x_std
        Ys = (Y - self.y_mean) / self.y_std
        n = Xs.shape[0]
        Xb = np.hstack([Xs, np.ones((n, 1))])
        A = Xb.T @ Xb + self.l2 * np.eye(Xb.shape[1])
        B = Xb.T @ Ys
        W = np.linalg.solve(A, B)
        self.theta = W[:-1]
        self.bias = W[-1]
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        if self.theta is None:
            raise RuntimeError('ForwardDynamics not fit yet; call fit() first')
        single = X.ndim == 1
        if single:
            X = X[None, :]
        Zs = self._scale(X)
        pred_std = Zs @ self.theta + self.bias
        out = pred_std * self.y_std + self.y_mean
        if self.delta:
            out = out + X[:, :D_STATE]
        return out[0] if single else out

    def next_state_vector(self, w, dec, team='红'):
        return self.predict(design_matrix(w, dec, team))


def rmse(pred, real):
    pred = np.asarray(pred, dtype=float)
    real = np.asarray(real, dtype=float)
    return float(np.sqrt(np.mean((pred - real) ** 2)))


def realism_report(pred, real, nochange):
    """对比预测 vs 无变化基线, 报告整体 + 位置/血量分量 RMSE。"""
    out = {
        'rmse_forward': round(rmse(pred, real), 5),
        'rmse_nochange_baseline': round(rmse(nochange, real), 5),
        'capture': None,   # 1 - rmse_forward/rmse_nochange ; 越小越好
    }
    nz = out['rmse_nochange_baseline']
    out['capture'] = round(out['rmse_forward'] / nz, 4) if nz > 1e-9 else None
    # 位置分量(每槽 [present,x,y]) -> index 1,2; 每 6 槽一个
    pos_idx = [i * 6 + 1 for i in range(len(real[0]) // 6)] + [i * 6 + 2 for i in range(len(real[0]) // 6)]
    if pos_idx:
        out['rmse_pos_forward'] = round(rmse(pred[:, pos_idx], real[:, pos_idx]), 5)
        out['rmse_pos_nochange'] = round(rmse(nochange[:, pos_idx], real[:, pos_idx]), 5)
    hp_idx = [i * 6 + 3 for i in range(len(real[0]) // 6)]
    if hp_idx:
        out['rmse_hp_forward'] = round(rmse(pred[:, hp_idx], real[:, hp_idx]), 5)
        out['rmse_hp_nochange'] = round(rmse(nochange[:, hp_idx], real[:, hp_idx]), 5)
    return out
