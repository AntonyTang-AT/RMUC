# -*- coding: utf-8 -*-
"""在真实对局上评估前向动力学真实度(B1 落地)。纯 numpy/CPU, 不吃 GPU。

度量"f(state_t, action_t) -> state_{t+1}"能多贴近真实下一帧, 分三层:
  1) 全局 capture = RMSE_fwd / RMSE_nochange, 越接近 0 越真实, ~1 白噪声, >1 比"原样不动"更差。
  2) 逐维分量: 位置(26)/血量(13) 各给 前向 / 无变化 两组 RMSE, 并统计"前向胜过无变化"的维数。
  3) ★反事实方向: 同一状态下把某单位 move 改成 推进/回防, 看在 敌半场 方向上的预测位移 dx
     是否符号正确(推进应推向敌方=+x(红)/-x(蓝), 回防应相反)。这是真正对 RL 有用的信号,
     与绝对 RMSE(被敌方行为/复活的不可约噪声淹没)无关。

我方动作通道 = run_m2.rule_head 每时刻决策。换 action 即改变"若我方这么打, 下一步如何"。

用法:
  python -m eval.forward_eval --train G... --eval G... --delta --unit 英雄 --pair 推进 回防 --out _work/forward_eval.json
"""
import os, sys, json, argparse

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.chdir(REPO)

import numpy as np
from data import db, features
from worldstate import worldstate as ws
from rl import forward as fd
from schema import layered as sl

POS_IDX = [i for i in range(fd.D_STATE) if i % 6 in (1, 2)]      # 各槽 x,y
HP_IDX = [i for i in range(fd.D_STATE) if i % 6 == 3]            # 各槽 hp


def load_game(gid):
    con = db.connect()
    rows = db.timeseries_for(con, gid)
    dps = features.build_decision_points(rows, gid)
    b = features.field_bounds(dps)
    wsl = [ws.build_worldstate(dp, b) for dp in dps]
    decs = None
    try:
        import run_m2
        decs = run_m2.rule_head(dps, b)
    finally:
        con.close()
    return wsl, decs


def build_xy(wsl, decs, team='红'):
    X, Y = [], []
    for i in range(len(wsl) - 1):
        X.append(fd.design_matrix(wsl[i], decs[i], team))
        Y.append(fd.state_features(wsl[i + 1]))
    return np.array(X), np.array(Y)


def _game_metrics(model, wsl, decs, team):
    preds, reals, nochange = [], [], []
    for i in range(len(wsl) - 1):
        preds.append(model.next_state_vector(wsl[i], decs[i], team))
        reals.append(fd.state_features(wsl[i + 1]))
        nochange.append(fd.state_features(wsl[i]))
    preds, reals, nochange = np.array(preds), np.array(reals), np.array(nochange)
    rf = np.sqrt(((reals - preds) ** 2).mean(0))
    rn = np.sqrt(((reals - nochange) ** 2).mean(0))
    return {
        'n': int(len(reals)),
        'rmse_fwd': round(float(fd.rmse(preds, reals)), 4),
        'rmse_nochange': round(float(fd.rmse(nochange, reals)), 4),
        'capture': round(float(fd.rmse(preds, reals) / fd.rmse(nochange, reals)), 3),
        'pos_rmse_fwd': round(float(rf[POS_IDX].mean()), 4),
        'pos_rmse_nochange': round(float(rn[POS_IDX].mean()), 4),
        'hp_rmse_fwd': round(float(rf[HP_IDX].mean()), 4),
        'hp_rmse_nochange': round(float(rn[HP_IDX].mean()), 4),
        'dims_fwd_better_nochange': int((rn > rf).sum()),
    }


def _counterfactual_dir(model, wsl, team, unit, move_a, move_b):
    """同状态下仅改 unit 的 move, 测其在敌方方向上的预测位移 dx 的方向是否符合 move 语义。"""
    slot = sl.UNIT_CLASSES.index(unit) if unit in sl.UNIT_CLASSES else 0
    x_idx = slot * fd.SLOT + 1            # uniform slot layout 红在前
    sign = 1.0 if team == '红' else -1.0  # 红主场低x: 推敌=+x; 蓝镜像=-x
    mean_dx = {mv: [] for mv in (move_a, move_b)}
    for i in range(0, len(wsl), 3):
        w0 = wsl[i]
        s0 = fd.state_features(w0)
        for mv in (move_a, move_b):
            dec = {'units': {c: {'move': '无', 'fire': False} for c in sl.UNIT_CLASSES},
                   'global_mode': '进攻'}
            dec['units'][unit] = {'move': mv, 'fire': False}
            dx = (model.next_state_vector(w0, dec)[x_idx] - s0[x_idx]) * sign
            mean_dx[mv].append(float(dx))
    scale = np.median(np.abs(mean_dx[move_a])) + 1e-9
    d_a = float(np.mean(mean_dx[move_a]))
    d_b = float(np.mean(mean_dx[move_b]))
    return {
        'unit': unit, 'team': team,
        f'dx[{move_a}]': round(d_a, 4),
        f'dx[{move_b}]': round(d_b, 4),
        'push_more_forward': round((d_a - d_b) / scale, 3),   # >0 表示 move_a 更像推进
        'sign_ok': (d_a - d_b) > 0 if move_a in ('推进', '绕行', '游击') else None,
    }


def evaluate(train, evals, team, l2, delta, unit, pair):
    X, Y = [], []
    for gid in train:
        wsl, decs = load_game(gid)
        if decs is None or len(wsl) < 2:
            print('skip train', gid, flush=True)
            continue
        x, y = build_xy(wsl, decs, team)
        X.append(x); Y.append(y)
    X = np.vstack(X); Y = np.vstack(Y)
    print(f'TRAIN n={len(X)} d={X.shape[1]}->{Y.shape[1]} l2={l2} delta={delta}', flush=True)
    model = fd.ForwardDynamics(l2=l2, delta=delta).fit(X, Y)
    out = {'d_state': fd.D_STATE, 'd_action': fd.D_ACTION, 'train_n': int(len(X)),
           'team': team, 'l2': l2, 'delta': delta, 'games': {}, 'counterfactual': {}}
    for gid in evals:
        wsl, decs = load_game(gid)
        if decs is None or len(wsl) < 2:
            print('skip eval', gid, flush=True)
            continue
        rep = _game_metrics(model, wsl, decs, team)
        out['games'][str(gid)] = rep
        out['counterfactual'][str(gid)] = _counterfactual_dir(model, wsl, team, unit, *pair)
        print(json.dumps({'game_id': gid, **rep, 'cf': out['counterfactual'][str(gid)]},
                         ensure_ascii=False), flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train', nargs='+', type=int, required=True)
    ap.add_argument('--eval', nargs='+', type=int, required=True)
    ap.add_argument('--team', default='红')
    ap.add_argument('--l2', type=float, default=1.0)
    ap.add_argument('--delta', action='store_true', help='预测增量而非绝对值')
    ap.add_argument('--unit', default='英雄')
    ap.add_argument('--pair', nargs=2, default=['推进', '回防'])
    ap.add_argument('--out', default='_work/forward_eval.json')
    args = ap.parse_args()
    out = evaluate(args.train, args.eval, args.team, args.l2, args.delta, args.unit, args.pair)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print('SAVE', args.out, flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
