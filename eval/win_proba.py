# -*- coding: utf-8 -*-
"""M2 单点胜负机: 用"真实回放窗口"的胜势反馈累加 -> 该局是否红胜 的判模型。

为什么另训(不能直接用 reward_learn 整局权重): reward_learn 的 P(win) 是**整局累加**特征
(幅值几十)学出; 喂进 15-30s 的短窗外推会塌到先验。这里特征=**逐决策点的短窗累加反馈**(T 帧),
标签=该局红方是否胜, 用**按局分组 CV** 验证(防同局点泄漏)。
用途(见 labels/rollout_label): 给反事实最优承诺 rollouts 的终点打"红方胜算"分。
纯 numpy, 无 torch; 单测离线可跑。
"""
import numpy as np

from rl import reward_learn as rl_le
from rl import reward as rw
from eval.outcome_backtest import WIN_FEATURES


def window_features(worldstates, team='红'):
    """对 worldstate 列表累加 reward_components-> 取 WIN_FEATURES 定长向量。"""
    acc = {k: 0.0 for k in rw.DEFAULT_CONF}
    for w in worldstates:
        comp = rw.reward_components(w, team)
        for k in acc:
            acc[k] += comp[k]
    return [float(acc[k]) for k in WIN_FEATURES]


def fit(X, y, seed=0, l2=1e-2):
    X = np.asarray(X, float); y = np.asarray(y, float)
    w, b, _ = rl_le.fit_logistic(X, y, l2=l2, seed=seed)
    return w, b, {'n': int(len(y)), 'n_pos': int(float(y.sum())),
                  'l2': l2, 'seed': seed}


def predict_proba(X, w, b):
    return rl_le.predict_proba(np.asarray(X, float), w, b)


def grouped_cv(X, y, gids, k=5, seed=0, l2=1e-2):
    """按 game_id 分组的 K 折 held-out AUC/acc。只有同局点聚到同一折才不泄漏。"""
    X = np.asarray(X, float); y = np.asarray(y, float); gids = np.asarray(gids)
    uniq = np.unique(gids)
    if len(uniq) < 2 * k or len(np.unique(y)) < 2:
        return {'cv_auc': float('nan'), 'cv_acc': float('nan'), 'fold_aucs': [],
                'n_folds': 0, 'n': int(len(y))}
    perm = np.random.default_rng(seed).permutation(len(uniq))
    folds = np.array_split(uniq[perm], min(k, len(uniq)))
    aucs, accs = [], []
    for te_g in folds:
        te = np.isin(gids, te_g)
        tr = ~te
        if len(np.unique(y[te])) < 2 or len(np.unique(y[tr])) < 2:
            continue
        w_, b_, _ = rl_le.fit_logistic(X[tr], y[tr], l2=l2)
        p = rl_le.predict_proba(X[te], w_, b_)
        aucs.append(rl_le.auc(y[te], p))
        accs.append(float(((p >= 0.5).astype(int) == y[te]).mean()))
    return {'cv_auc': float(np.mean(aucs)) if aucs else float('nan'),
            'cv_acc': float(np.mean(accs)) if accs else float('nan'),
            'fold_aucs': [round(float(a), 4) for a in aucs],
            'n_folds': len(aucs), 'n': int(len(y)), 'k': k, 'seed': seed}
