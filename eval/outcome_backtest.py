# -*- coding: utf-8 -*-
"""M2 面向对局结果(真实胜负)的决策质量回测 outcome-based backtest。

为什么做: "胜率才是最终指标"。之前的 accuracy / agreement_with_rule 是"模型 vs 规则标签一致",
而规则标签本身就是遥测规则的产物 -> 循环, 不能证明"决策对不对"。这个模块把最终指标锚到**真实胜负**。

方法: 用 reward_learn 的"特征 -> 谁赢"模型当可靠 scorer。但要可信, 必须先过 **交叉验证**:
否则是在拿训练集预测它自己拟合过的标签(又一次循环)。所以核心两件事:

  A. cv_win_model(): K折验证 "per-game reward_features -> winner" 的 held-out AUC。
     证明"状态特征真的携带胜负信号、不是过拟合"。这是整个"胜率指标"的可靠地基。
  B. fav_by_decision(): 对每决策点, 用**胜相关组件**加权出此刻红方有利度 fav,
     看**哨兵决策**是否在恰当的有利度区间被发出:
        def(守) 应落在低有利度(己方受威胁), 进攻应落在高有利度。
     输出 {state: mean_fav, count} 表 + def 与 cruise 的有利度差(负=合理)。

★ 诚实前提(重要): 这是**关联性/条件有利度**度量, 不是"决策导致了胜利"的因果度量。
我们的 env.step 是 replay 环境, 动作不改变未来状态(见 rl/env.py step: t+=1 跳到下一个真实决策点)。
所以本模块给出的"def 落在低有利度"只说明**决策发生在对的红方胜势区间**, 不说明**是决策赢的**。
要因果只能走前向动力学(路线 B, 已推迟)。
★ 决策来源(默认): 用 3 状态规则标签 sl.label_sentry(dp,bounds,team='红',sentry_rid=7) 作为
"被评估的决策"(无 GPU/transformers, 含 hitbuilding, 对齐 brief 的三状态)。
gm.derive_sentry_state 已对齐瞬时三态(def/hitbuilding/cruise)，可与 sentry_labels 对照；
本文件默认仍用 sentry_labels（基于原始 dp 坐标），避免与 worldstate 归一化口径混用。
本文件是"替换成 LLM/融合决策"的模板: 只需把决策序列喂进 fav_by_decision。

纯 numpy, 只读 sqlite, 单测可跑(用合成 worldstate, 不碰 torch/transformers)。

用法:
  python -m eval.outcome_backtest --db data.db --limit-games 120 --out _work/outcome_backtest.json
"""
import os, sys, json, sqlite3, argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

from data import db, features
from worldstate import worldstate as ws
from rl import reward as rw
from rl import reward_learn as rl_le
from labels import sentry_labels as sl

# reward_learn 证明的"胜相关"组件(ego_surv/ego_hp/heat 在数据上不判别胜负, 已剔除)
WIN_FEATURES = ['base_atk', 'base_def', 'outpost_def', 'kill', 'outpost_atk', 'dmg_deal', 'econ']

# 决策语义: 被评估的哨兵宏决策(与 brief 三状态一致)
DECISIONS = ('def', 'hitbuilding', 'cruise')


# ---------- A. 纯 numpy 交叉验证: 特征 -> 胜负 ----------
def cv_auc_win_model(X, y, k=5, seed=0, l2=1e-2):
    """K折 held-out AUC/acc。X=per-game reward_features, y=红方是否胜。返回 dict。"""
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    n = len(y)
    if n < 2 * k or len(np.unique(y)) < 2:
        return {'cv_auc': float('nan'), 'cv_acc': float('nan'), 'fold_aucs': [], 'n': n}
    idx = np.random.default_rng(seed).permutation(n)
    folds = np.array_split(idx, min(k, n))
    aucs, accs = [], []
    for i in range(len(folds)):
        te = folds[i]
        tr = np.concatenate([folds[j] for j in range(len(folds)) if j != i])
        if len(np.unique(y[te])) < 2 or len(np.unique(y[tr])) < 2:
            continue
        w, b, _ = rl_le.fit_logistic(X[tr], y[tr], l2=l2)
        p = rl_le.predict_proba(X[te], w, b)
        aucs.append(rl_le.auc(y[te], p))
        accs.append(float(((p >= 0.5).astype(int) == y[te]).mean()))
    return {'cv_auc': float(np.mean(aucs)) if aucs else float('nan'),
            'cv_acc': float(np.mean(accs)) if accs else float('nan'),
            'fold_aucs': [round(float(a), 4) for a in aucs],
            'n_folds': len(aucs), 'n': n, 'l2': l2}


# ---------- B. 有利度 + 决策条件 ----------
def per_frame_fav(components):
    """红方此刻"胜势有利度" = 胜相关组件按说明正权求和(组件本身已归一化到 ~[-1,1])。"""
    return float(sum(components.get(k, 0.0) for k in WIN_FEATURES))


def fav_from_worldstate(w, team='红'):
    """对一个 worldstate 算红方有利度。"""
    return per_frame_fav(rw.reward_components(w, team))


def fav_by_decision(pairs):
    """pairs: [(decision_state, fav_float)] -> {state: {mean_fav, count}}。纯函数, 可单测。"""
    agg = {}
    for state, fav in pairs:
        agg.setdefault(state, []).append(fav)
    return {s: {'mean_fav': float(np.mean(v)) if v else float('nan'), 'count': len(v)}
            for s, v in agg.items()}


def alignment_gap(pairs):
    """def 与 cruise 的有利度差。负=def 落在更低有利度(受威胁才守), 即合理。"""
    by = fav_by_decision(pairs)
    if 'def' in by and 'cruise' in by and by['def']['count'] and by['cruise']['count']:
        return by['def']['mean_fav'] - by['cruise']['mean_fav']
    return None


# ---------- 数据装载(只读 db) ----------
def _load_game_features(con, gid):
    """单局红方 reward_features(LR_FEATURES 顺序) -> np.array; 无法构build -> None。"""
    rows = db.timeseries_for(con, gid)
    if not rows:
        return None
    dps = features.build_decision_points(rows, gid)
    if not dps:
        return None
    b = features.field_bounds(dps)
    ws_list = [ws.build_worldstate(dp, b) for dp in dps]
    acc = rw.reward_features(ws_list, '红')
    return np.array([acc[k] for k in rl_le.LR_FEATURES], float)


def _game_winner_red(con, gid):
    meta = db.match_meta(con, gid)
    winv = meta[rl_le.WINNER_COL] if meta else None
    if winv is None:
        return None
    return True if ('e7baa2' in (winv or '').encode('utf-8').hex()) else False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--db', default=db.DB_PATH)
    ap.add_argument('--limit-games', type=int, default=0, help='0=全部; 否则只取前 N 局')
    ap.add_argument('--k', type=int, default=5)
    ap.add_argument('--l2', type=float, default=1e-2)
    ap.add_argument('--out', default='_work/outcome_backtest.json')
    args = ap.parse_args()

    con = sqlite3.connect(args.db)
    con.row_factory = sqlite3.Row
    gids = [r[0] for r in con.execute('SELECT game_id FROM matches ORDER BY game_id')]
    if args.limit_games:
        gids = gids[:args.limit_games]

    # A. 特征 -> 胜负 的 CV 可靠性
    X, y, used = [], [], []
    for gid in gids:
        v = _load_game_features(con, gid)
        wred = _game_winner_red(con, gid)
        if v is None or wred is None:
            continue
        X.append(v)
        y.append(1.0 if wred else 0.0)
        used.append(gid)
    cv = cv_auc_win_model(X, y, k=args.k, l2=args.l2) if len(X) >= 2 * args.k else {
        'cv_auc': float('nan'), 'cv_acc': float('nan'), 'fold_aucs': [], 'n': len(X)}

    # B. 决策条件有利度(规则头 + 合成 worldstate, 无 GPU)
    #    为控制运行时长, 默认扫全部 used 局; 太慢可 --limit。复用同一 con。
    pair_counter = {}
    for gid in used:
        rows = db.timeseries_for(con, gid)
        if not rows:
            continue
        dps = features.build_decision_points(rows, gid)
        if not dps:
            continue
        b = features.field_bounds(dps)
        for dp in dps:
            w = ws.build_worldstate(dp, b)
            # 用 3 状态规则标签 label_sentry 作为"被评估的决策"(含 hitbuilding, 对齐 brief)。
            # 不用 gm.derive_sentry_state: 它只映射全局模式 -> 2 状态, 遗漏 hitbuilding。
            decision = sl.label_sentry(dp, b, team='红', sentry_rid=7)
            if decision not in DECISIONS:
                continue
            fav = fav_from_worldstate(w)
            pair_counter.setdefault(decision, []).append(fav)

    result = {
        'cv_win_model': cv,
        'caveat': ('这是关联性/条件有利度度量, 不是因果。env.step 是 replay 环境, 动作不改未来状态, '
                   '故"def 落在低有利度"只说明决策发生在对的胜势区间, 不说明是决策赢的。'
                   '要因果需前向动力学(路线 B)。'),
        'mean_fav_by_decision': {s: {'mean_fav': round(float(np.mean(v)), 4), 'count': len(v)}
                                 for s, v in pair_counter.items() if v},
        'alignment_gap_def_minus_cruise': None,
    }
    if 'def' in pair_counter and 'cruise' in pair_counter and pair_counter['def'] and pair_counter['cruise']:
        result['alignment_gap_def_minus_cruise'] = round(
            float(np.mean(pair_counter['def']) - np.mean(pair_counter['cruise'])), 4)
    result['n_games_for_cv'] = len(y)
    result['n_decision_points'] = int(sum(len(v) for v in pair_counter.values()))
    con.close()

    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps({'type': 'SAVE', 'out': args.out,
                      'cv': cv, 'align_gap': result['alignment_gap_def_minus_cruise'],
                      'n_dps': result['n_decision_points']}, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    sys.exit(main())
