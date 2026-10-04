# -*- coding: utf-8 -*-
"""M2 奖励权重学习: 从对局结果(胜负)反推各整形项的**数据驱动**权重。

思路(镜像 RMUC-OfflineRL reward_model.py, 无仿真/混合质量数据/约千条轨迹标签 ->
推荐用可解释逻辑回归而非经典 IRL):
  每局(取红方) 把未加权组件在整局累加 -> 特征向量 X;
  用 matches.胜利(红=1) 做标签 y;
  岭逻辑回归  P(win) = sigmoid(w · X + b)  学出每项目标权重;
  与手调 DEFAULT_CONF 对比, 并用 AUC 衡量区分度。

纯 numpy(手写 GD + AUC), 单测可跑。只读 sqlite, 不改库。

用法:
  python -m rl.reward_learn --db data.db --limit-games 120 --out _work/rl_weights.json
"""
import os, sys, json, math, sqlite3, argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

from rl import reward as rw
from data import db, features
from worldstate import worldstate as ws

# 去共线特征集(drop 掉 = 精确负相关的 death / dmg_take, 保留净增益)
LR_FEATURES = ['outpost_atk', 'base_atk', 'kill', 'dmg_deal',
               'outpost_def', 'base_def', 'ego_surv', 'ego_hp', 'heat', 'econ']
WINNER_COL = 8  # matches.胜利 列(位置索引, 避免中文列名编码问题)


# ---- 纯 numpy 岭逻辑回归 ----
def fit_logistic(X, y, l2=1e-2, lr=0.2, iters=4000, seed=0):
    """标准化后 GD 训练, 返回 (w_unstd, b, scaler)。"""
    rng = np.random.default_rng(seed)
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    n, d = X.shape
    mu = X.mean(0)
    sd = X.std(0) + 1e-9
    Xs = (X - mu) / sd
    w = np.zeros(d)
    b = 0.0
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-(Xs @ w + b)))
        g = Xs.T @ (p - y) / n + l2 * w
        gb = (p - y).mean()
        w -= lr * g
        b -= lr * gb
    # 映射回原尺度: w/b =?  w·Xs+b = w·(X-mu)/sd + b = X·(w/sd) + (b - mu·w/sd)
    w_un = w / sd
    b_un = b - (mu * w_un).sum()
    return w_un, b_un, (mu, sd)


def predict_proba(X, w, b):
    z = X @ w + b
    return 1.0 / (1.0 + np.exp(-z))


def auc(y, score):
    order = np.argsort(score)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(score) + 1)
    pos = np.asarray(y) == 1
    n_pos, n_neg = pos.sum(), (~pos).sum()
    if n_pos == 0 or n_neg == 0:
        return float('nan')
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def weights_json_to_conf(path_or_rows, base_conf=None):
    """把 reward_learn 学到的权重 JSON 映射成 RL 可直接用的 per-step reward_conf。

    为什么不能直接用 learned 原始系数: reward_learn 的特征是**整局累加**尺度(幅值几十),
    learned 系数≈1e-3; 而 RL 的 reward(t)=Σ conf_i*comp_i(t) 里 comp_i(t) 是 per-step 归一言值(≤1),
    直接喂 learned 会让加权和塌成 ~0, 被 action_coherence(±0.15)/终端(±1) 淹没——等于没学。

    处理: (1) 用**效应量** `effect=w*std`(该特征+1std 对胜算的对数几率增量)作奖励权重, 它量纲自洽、
    反映跨特征重要性; (2) 按 base_conf(默认 DEFAULT_CONF)在所用特征上的均值幅度**回正** scale,
    使 per-step 奖励幅度与手调可比; (3) 未用于逻辑回归的共线特征(death/dmg_take)置 0, 避免与
    kill/dmg_deal 双重计数。返回完整 conf dict(仅 used 特征被 learned 覆盖)。"""
    import json
    import os
    base_conf = dict(base_conf or rw.DEFAULT_CONF)
    if isinstance(path_or_rows, (str, os.PathLike)):
        with open(path_or_rows, encoding='utf-8') as f:
            rows = json.load(f)['weights']
    else:
        rows = list(path_or_rows)
    avail = [r for r in rows if r['feature'] in base_conf]
    if not avail:
        return base_conf
    learned = {r['feature']: r.get('effect', r['learned']) for r in avail}
    base_abs = [abs(base_conf[r['feature']]) for r in avail]
    lrn_abs = [abs(learned[r['feature']]) for r in avail]
    b_mean = sum(base_abs) / len(base_abs)
    l_mean = sum(lrn_abs) / len(lrn_abs)
    scale = b_mean / l_mean if l_mean > 1e-12 else 1.0
    conf = {k: 0.0 for k in base_conf}
    for r in avail:
        conf[r['feature']] = learned[r['feature']] * scale
    return conf


def build_features(con, game_id):
    """单局红方未加权组件累加向量(仅 LR_FEATURES)。"""
    rows = db.timeseries_for(con, game_id)
    if not rows:
        return None
    dps = features.build_decision_points(rows, game_id)
    if len(dps) < 1:
        return None
    b = features.field_bounds(dps)
    ws_list = [ws.build_worldstate(dp, b) for dp in dps]
    acc = rw.reward_features(ws_list, '红')
    return np.array([acc[k] for k in LR_FEATURES], float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--db', default=db.DB_PATH)
    ap.add_argument('--l2', type=float, default=1e-2)
    ap.add_argument('--limit-games', type=int, default=0, help='0=全部; 否则只取前 N 局(验证加速)')
    ap.add_argument('--out', default='_work/rl_weights.json')
    args = ap.parse_args()

    con = sqlite3.connect(args.db)
    # 需 row_factory=sqlite3.Row(db.connect 已设), 否则 timeseries_for 返回 tuple 无法按列名索引
    con.row_factory = sqlite3.Row
    gids = [r[0] for r in con.execute('SELECT game_id FROM matches ORDER BY game_id')]
    if args.limit_games:
        gids = gids[:args.limit_games]

    X, y, used = [], [], []
    for gid in gids:
        v = build_features(con, gid)
        if v is None:
            continue
        meta = db.match_meta(con, gid)
        winv = meta[WINNER_COL]
        winner_is_red = ('e7baa2' in (winv or '').encode('utf-8').hex() if winv else False)
        # winv 是 '红'(e7baa2) 或 '蓝'(e8939d); 无法识别的跳过(应不会)
        if winv is None:
            continue
        X.append(v)
        y.append(1.0 if winner_is_red else 0.0)
        used.append(gid)
    con.close()
    if not X:
        print('NO_DATA'); return 1
    X = np.array(X); y = np.array(y)

    w_un, b_un, (mu, sd) = fit_logistic(X, y, l2=args.l2)
    probs = predict_proba(X, w_un, b_un)
    a = auc(y, probs)
    acc = ((probs >= 0.5).astype(int) == y).mean()

    # 与手调对比。learned 是"逐局累加特征"的原始权重(幅值受特征尺度影响, 很小),
    # 不能直接和手调比绝对值; 用 effect = w * std(该特征+1标准差对胜算的对数几率增量) 跨特征比重要性。
    hand = rw.DEFAULT_CONF
    rows = []
    for i, k in enumerate(LR_FEATURES):
        rows.append({'feature': k,
                     'hand': hand.get(k, 0.0),
                     'learned': round(float(w_un[i]), 5),
                     'effect': round(float(w_un[i] * sd[i]), 4)})
    rows.sort(key=lambda r: -abs(r['effect']))

    result = {
        'n_games': len(y), 'n_pos(红胜)': int(y.sum()),
        'auc': round(a, 4), 'accuracy': round(float(acc), 4),
        'intercept': round(float(b_un), 4), 'l2': args.l2,
        'weights': rows,
        'note': ('learned=原始尺度权重(数值很小因特征是逐局累加, 幅值几十); '
                 'effect=标准化效应量(w*std, 该特征+1标准差对胜算的对数几率增量, 可跨特征比重要性). '
                 '正值越强越易胜. 按 |effect| 降序.'),
    }
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps({'auc': round(a, 4), 'acc': round(float(acc), 4), 'n': len(y),
                      'type': 'SAVE', 'out': args.out}, ensure_ascii=False))
    for row in rows:
        print(f"  {row['feature']:<12} hand={row['hand']:>5.2f}  learned={row['learned']:>8.5f}  "
              f"effect={row['effect']:>+7.4f}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
