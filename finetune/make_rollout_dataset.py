# -*- coding: utf-8 -*-
"""反事实最优承诺标注: 从 db(只读) 训单点胜率机 -> 重标注训练决策点 -> 写 '_rollout' jsonl + 验证报告。

流程:
  A. 逐局 build dps/bounds, 对每局采样决策点, 取"真实回放窗口 [i..i+T]" 累加反馈 -> X,
     该局红方是否胜 -> y, 按局分组 CV 得胜率机可信度(资格证)。
  B. 用全部样本 fit 最终 win_model(w,b)。
  C. 对训练决策点逐点 rollout 三模式 + 单点胜率机打分, argmax -> 标签 + 重派生 units -> records,
     写 jsonl(_rollout 后缀)。
  D. 输出验证报告: CV / 结果一致性(§9.1) / horizon 消融(§9.2)。

只读 db, 无 torch; GPU 留给后续真实训练/评估。
用法:
  python -m finetune.make_rollout_dataset --db data.db --games 177866123 1970735 \
      --T 5 --out out_m2/m2_dataset_rollout.jsonl --report _work/rollout_label_report.json
"""
import os, sys, json, argparse, random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

from data import db, features
from worldstate import worldstate as ws
from rl import physics as ph
from rl import reward_learn as rl_le
from eval import win_proba as wp
from labels import rollout_label as rrl
from finetune import make_dataset as md


def collect_samples(con, gid, T, max_points=None, seed=0):
    """单局: 真实窗口 [i..i+T] 累加反馈 -> 一行 X; 该局胜负 -> y; 返回样本 + 该局产物。
    同时把该局 dps/bounds 返回供后续重标注(避免对每决策点重读)。"""
    rows = db.timeseries_for(con, gid)
    if not rows:
        return None
    dps = features.build_decision_points(rows, gid)
    if not dps:
        return None
    b = features.field_bounds(dps)
    wred = _game_win_red(con, gid)
    if wred is None:
        return None
    ws_list = [ws.build_worldstate(d, b) for d in dps]
    inds = list(range(len(dps)))
    if max_points and len(inds) > max_points:
        step = max(1, len(inds) // max_points)
        inds = sorted(set(list(range(0, len(inds), step))))[:max_points]
    X, ys = [], []
    for i in inds:
        hi = min(i + T + 1, len(ws_list))
        win = ws_list[i:hi]
        X.append(wp.window_features(win, '红'))
        ys.append(1.0 if wred else 0.0)
    return {'X': np.array(X, float), 'y': np.array(ys, float), 'gid': gid,
            'dps': dps, 'bounds': b, 'ws_list': ws_list, 'iter_range': inds}


def _game_win_red(con, gid):
    import json as _j
    meta = db.match_meta(con, gid)
    winv = meta[rl_le.WINNER_COL] if meta else None
    if winv is None:
        return None
    return True if ('e7baa2' in (winv or '').encode('utf-8').hex()) else False


def win_model_cv(X, y, gids, k=5, seed=0):
    return wp.grouped_cv(X, y, np.array(gids), k=k, seed=seed)


def label_consistency(dps, bounds, win_model, T_list, team='红'):
    """horizon 消融: 对不同 T 重标一批点, 报告相对 T_ref 相对各自的一致率(简化: 相邻 T 一致率均值)。"""
    from worldstate import worldstate as _ws
    ws_list = [_ws.build_worldstate(d, bounds) for d in dps]
    series = {}
    for T in T_list:
        lbl = rrl.relabel_decisions_rollout(ws_list, dps, bounds, [{}]*len(dps),
                                            win_model, T=T, team=team)[1]
        series[T] = lbl
    agree = {}
    for T in T_list:
        ref = T_list[0]
        a = sum(1 for x, y in zip(series[T], series[ref]) if x == y) / max(len(dps), 1)
        agree[T] = round(a, 4)
    agree[T_list[0]] = 1.0   # 自身=1
    return agree


def outcome_consistency(labels, scores, team='红'):
    """§9.1: 标'进攻'的点(其自身得分/取胜机)均值 应 > 标'防守'。op_vs_def_gap=攻-守。"""
    agg = {}
    for lbl, sc in zip(labels, scores):
        agg.setdefault(lbl, []).append(sc.get(lbl, 0.0))
    means = {m: round(float(np.mean(v)), 4) for m, v in agg.items() if v}
    res = {'mean_score_by_label': means, 'counts': {m: len(v) for m, v in agg.items()}}
    if '进攻' in means and '防守' in means:
        res['op_vs_def_gap'] = round(means['进攻'] - means['防守'], 4)
    else:
        res['op_vs_def_gap'] = None
    res['note'] = ("op_vs_def_gap>0: 标'进攻'的点其胜机均值更高=标签与真实胜势对齐, 合理。 "
                   "<=0: 标签机器与现实矛盾, 应停。")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--db', default=db.DB_PATH)
    ap.add_argument('--games', nargs='+', type=int, default=None, help='缺省=取前 --limit 局')
    ap.add_argument('--limit', type=int, default=24, help='未指定 games 时取前 N 局')
    ap.add_argument('--T', type=int, default=5)
    ap.add_argument('--k', type=int, default=5)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--max-points', type=int, default=0, help='>0: 每局采样点上限')
    ap.add_argument('--sample-step', type=int, default=1, help='win 模型训练时采样间隔')
    ap.add_argument('--out', default='out_m2/m2_dataset_rollout.jsonl')
    ap.add_argument('--report', default='_work/rollout_label_report.json')
    ap.add_argument('--params', default=None, help='physics 参数 json(缺省=跨训练局 calibrate)')
    args = ap.parse_args()

    con = db.connect()
    con.row_factory = sqlite3.Row
    if args.games:
        gids = args.games
    else:
        gids = [r[0] for r in con.execute('SELECT game_id FROM matches ORDER BY game_id')][:args.limit]

    # A. 训 win_model: 真实窗口样本(跨局), 按局分组 CV
    Xall, yall, gidall = [], [], []
    games = {}
    for gid in gids:
        sample = collect_samples(con, gid, args.T, args.max_points, args.seed)
        if sample is None:
            continue
        games[gid] = sample
        Xall.append(sample['X']); yall.append(sample['y'])
        gidall.append(np.repeat(sample['gid'], len(sample['y'])))
    X = np.concatenate(Xall); y = np.concatenate(yall); gids_arr = np.concatenate(gidall)
    cv = win_model_cv(X, y, gids_arr, args.k, args.seed)
    # B. 最终 fit(全部样本)
    w_m, b_m, meta = wp.fit(X, y, seed=args.seed)
    win_model = (w_m, b_m)

    # C. 重标注训练决策点 -> records
    all_records = []
    per_game = {}
    for gid in games:
        s = games[gid]
        # 只标采样过的决策点(与训练 win_model 同集, 满足 --sample-step 语义: 全标可去掉采样限制)
        inds = s['iter_range']
        sel = [s['dps'][i] for i in inds]
        sel_ws = [s['ws_list'][i] for i in inds]
        fused = [{'global_mode': '均衡', 'units': {}} for _ in sel]   # 占位, 会被替换
        dec, labels, scores = rrl.relabel_decisions_rollout(sel_ws, sel, s['bounds'], fused,
                                                            win_model, T=args.T, team='红')
        recs = md.make_records(dec, sel, s['bounds'], '红')
        all_records.extend(recs)
        per_game[gid] = {'n': len(sel), 'label_dist': dict(Counter(labels)),
                         'op_vs_def_gap': None}
        # D. 结果一致性(此局)
        oc = outcome_consistency(labels, scores, '红')
        per_game[gid]['op_vs_def_gap'] = oc['op_vs_def_gap']

    # 全量结果一致性 + horizon 消融(用第一局做代表)
    all_labels = [json.loads(r['completion'])['global_mode'] for r in all_records]
    rep = next(iter(games.values()))
    horizon = label_consistency(rep['dps'][:30], rep['bounds'], win_model,
                                [3, args.T, 8, 12], '红') if rep else {}
    oc_all = outcome_consistency(all_labels, [{m: 0.5 for m in rrl.GLOBAL_MODES} for _ in all_labels], '红')
    oc_all['note'] = '全集: mean_score_by_label 用逐点 ' + \
        ('per_game op_vs_def_gap 已分别记录; 此处 scores 以 0.5 占位, 仅统计 counts。')

    md.to_jsonl(all_records, args.out)
    report = {'method': 'rollout', 'T': args.T, 'n_points': len(all_records), 'games': list(games),
              'win_model_cv': cv, 'win_model_meta': meta,
              'per_game': per_game, 'horizon_ablation': horizon,
              'label_dist_full': dict(Counter(all_labels)),
              'caveat': ('单点胜率机为相关/短程预测, 非因果; PAYOFF_3 手设; 敌方仅模式级应对。'
                         '最优=在我们模型认知下的全局最优, 非绝对真值(专家评审可凌驾)。'),
              'out_files': {'dataset': args.out, 'report': args.report}}
    with open(args.report, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps({'type': 'SAVE', 'out': args.out, 'n': len(all_records),
                      'cv': cv, 'report': args.report}, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    from collections import Counter
    import sqlite3
    sys.exit(main())
