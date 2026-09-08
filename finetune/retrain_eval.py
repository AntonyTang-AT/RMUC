# -*- coding: utf-8 -*-
"""M2 LLM 自动重训+评估 (Track 1) 单趟运行器: rebalance -> QLoRA 重训 -> 留出评估 -> global_mode 胜势对齐。

回到底层问题(见 spec): Plan A 只把标签从"行为锚定"升格为"最优锚定", 但没验证用这套标签训出的
LLM 是否真的更好。本脚本把 "重平衡 → train_lora → eval_model → global_mode_align_gap" 固化成一条
单趟、可重放的流水线; `--both` 对 **lookahead / rollout 两种标签源**各训一版做 head-to-head,
回答 "rollout 标签是否更优"(spec §9.4)。

流程(每趟, 单趟一次):
  1. 读训练数据集 jsonl(records)
  2. rebalance_records(攻45/守30/均25, 只增不删, 均衡作默认锚)
  3. train_lora(rebalanced, --out-dir, epochs=1, lr=2e-4, ...) -> adapter_path + meta
  4. eval_model(adapter_path, eval_dps, eval_bounds, return_predictions=True)
       -> legal_rate / conflict_rate / agreement_with_rule / 逐点(决策, 状态)
  5. global_mode_align_gap(modes, states, '红') -> gap = mean_fav(进攻) − mean_fav(防守)
  6. 写 report json(含 win_model CV 地基)

评估集两行共用(保证头部可比): 用 m2_report.json 的 eval_games, 只读 data.db 构 eval_dps/eval_bounds。
训练源: lookahead <- out_m2/m2_dataset.jsonl; rollout <- out_m2/m2_dataset_rollout_planb.jsonl。
只读 db; torch/transformers 在 train_lora/eval_model 内惰性 import; 不 commit out_m*/_work 产物。

用法:
  python -m finetune.retrain_eval --both --report out_m2/retrain_eval_report.json
  python -m finetune.retrain_eval --method lookahead --report _work/retrain_eval_lookahead.json
"""
import os, sys, json, argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data import db, features
from finetune import make_dataset as md
from finetune import lora_train as lt
from finetune import eval as feval
from eval import decision_gap as dgap
from eval import outcome_backtest as ob

# 重平衡目标(与 track1 spec §2 一致): 攻45 守30 均25, 只增不删
REBALANCE = {'进攻': 45, '防守': 30, '均衡': 25}


def load_records(path):
    """读 jsonl(records) -> list[{'instruction','completion'}]。"""
    recs = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    return recs


def subsample(points, n):
    """对决策点做确定性**均匀**子采样(n 点), 两行共用同一批 -> 对照公平。n<=0 -> 原样。"""
    if n <= 0 or len(points) <= n:
        return points
    import numpy as np
    idx = np.linspace(0, len(points) - 1, n).astype(int)
    return [points[i] for i in idx]


def build_eval(con, gids):
    """对 eval_games 读只读 db -> (eval_dps, eval_bounds)。data.db 决不改写。"""
    dps_all = []
    bounds = None
    for gid in gids:
        rows = db.timeseries_for(con, gid)
        if not rows:
            continue
        gdps = features.build_decision_points(rows, gid)
        if not gdps:
            continue
        b = features.field_bounds(gdps)
        if bounds is None:
            bounds = b
        dps_all.extend(gdps)
    return dps_all, bounds


def win_cv(con, gids, k=5, l2=1e-2):
    """"状态携带胜负信号"的地基: 按局 reward_features -> 红方是否胜, K 折 held-out AUC。
    在 dataset_games 上报(24 局 > 2k, 可算); eval_games 仅 4 局不够 K 折。"""
    X, y = [], []
    for gid in gids:
        v = ob._load_game_features(con, gid)
        wred = ob._game_winner_red(con, gid)
        if v is None or wred is None:
            continue
        X.append(v)
        y.append(1.0 if wred else 0.0)
    return ob.cv_auc_win_model(X, y, k=k, l2=l2) if len(X) >= 2 * k else {
        'cv_auc': float('nan'), 'cv_acc': float('nan'), 'fold_aucs': [], 'n': len(X)}


def merge_delta(by_method):
    """对照 delta(纯函数, 可单测): rollout 是否 ≥ lookahead(spec §4.2 承重判据)。"""
    if not ({'lookahead', 'rollout'} <= set(by_method)):
        return {}
    la = by_method['lookahead']['global_mode_align_gap']
    ro = by_method['rollout']['global_mode_align_gap']
    return {
        'gap_lookahead': la['gap'], 'gap_rollout': ro['gap'],
        'gap_delta_rollout_minus_lookahead': (
            round(ro['gap'] - la['gap'], 4)
            if (la['gap'] is not None and ro['gap'] is not None) else None),
        'rollout_gap_ge_lookahead': dgap.gap_lte(ro, la['gap']),
        'legal_lookahead': by_method['lookahead']['eval']['legal_rate'],
        'legal_rollout': by_method['rollout']['eval']['legal_rate'],
    }


def run_one(method, records_path, out_dir, eval_dps, eval_bounds, device,
            epochs, lr, legal_floor, max_new):
    """一趟: 读数据 -> 重平衡 -> 训 -> 评 -> gap。返回该行 dict。"""
    records = load_records(records_path)
    rebal = md.rebalance_records(records, REBALANCE)
    adapter_path, train_stats = lt.train_lora(rebal, out_dir=out_dir, device=device,
                                              epochs=epochs, lr=lr)
    ev = feval.eval_model(adapter_path, eval_dps, eval_bounds, device=device,
                          max_new=max_new, return_predictions=True)
    preds = ev.get('predictions', [])
    states = ev.get('states', [])
    modes = [d.get('global_mode', '均衡') for d in preds]
    gap = dgap.global_mode_align_gap(modes, states, '红')
    return {'method': method,
            'source': records_path,
            'n_train_raw': len(records),
            'n_train_rebalanced': len(rebal),
            'train_stats': train_stats,
            'adapter': adapter_path,
            'eval': ev,
            'global_mode_align_gap': gap,
            'legal_pass': ev['legal_rate'] >= legal_floor}


def main():
    ap = argparse.ArgumentParser(description='M2 LLM 自动重训+评估(单趟/对照)')
    ap.add_argument('--both', action='store_true', help='跑 lookahead + rollout 两趟并合并对照')
    ap.add_argument('--method', choices=['lookahead', 'rollout'],
                    help='单独跑某一种(与 --both 二选一)')
    ap.add_argument('--lookahead-source', default='out_m2/m2_dataset.jsonl')
    ap.add_argument('--rollout-source', default='out_m2/m2_dataset_rollout_planb.jsonl')
    ap.add_argument('--out-dir-lookahead', default='out_m2/finetuned_lookahead')
    ap.add_argument('--out-dir-rollout', default='out_m2/finetuned_rollout')
    ap.add_argument('--report', default='out_m2/retrain_eval_report.json')
    ap.add_argument('--m2-report', default='out_m2/m2_report.json',
                    help='内含 eval_games/dataset_games')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--epochs', type=int, default=1)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--legal-floor', type=float, default=0.8,
                    help='legal_rate 门控(默认 0.8); 跌破置 legal_pass=False')
    ap.add_argument('--max-new', type=int, default=200)
    ap.add_argument('--max-eval-points', type=int, default=160,
                    help='评估点均匀子采样上限(0=全部); 两行共用同一批, 保证对照公平')
    args = ap.parse_args()

    # 评估集 + win-model 地基(只读 db)
    with open(args.m2_report, encoding='utf-8') as f:
        m2r = json.load(f)
    eval_games = m2r.get('eval_games', [])
    dataset_games = m2r.get('dataset_games', [])
    con = db.connect()
    eval_dps, eval_bounds = build_eval(con, eval_games)
    eval_dps = subsample(eval_dps, args.max_eval_points)
    wcv = win_cv(con, dataset_games) if dataset_games else {
        'cv_auc': float('nan'), 'cv_acc': float('nan'), 'fold_aucs': [], 'n': 0}
    con.close()
    if not eval_dps:
        print('Evaluation decision points empty — abort.')
        return 1

    runs = []
    if args.both:
        for method in ('lookahead', 'rollout'):
            src = args.lookahead_source if method == 'lookahead' else args.rollout_source
            outdir = args.out_dir_lookahead if method == 'lookahead' else args.out_dir_rollout
            if not os.path.exists(src):
                print(f'skip {method}: source missing {src}')
                continue
            runs.append(run_one(method, src, outdir, eval_dps, eval_bounds, args.device,
                                args.epochs, args.lr, args.legal_floor, args.max_new))
    else:
        method = args.method or 'lookahead'
        src = (args.lookahead_source if method == 'lookahead' else args.rollout_source)
        outdir = (args.out_dir_lookahead if method == 'lookahead' else args.out_dir_rollout)
        runs.append(run_one(method, src, outdir, eval_dps, eval_bounds, args.device,
                            args.epochs, args.lr, args.legal_floor, args.max_new))

    # 合并对照 delta
    delta = merge_delta({r['method']: r for r in runs})

    report = {'pipeline': 'retrain_eval', 'device': args.device, 'epochs': args.epochs,
              'lr': args.lr, 'eval_games': eval_games, 'n_eval_dps': len(eval_dps),
              'win_model_cv': wcv, 'rows': runs, 'delta': delta,
              'caveat': ('global_mode_align_gap 是关联性/条件有利度, 非因果; 训练源两行共用同一批训练决策点, '
                         '仅标签方法不同; 评估集两行共用(留出 4 局)。rollout_gap_ge_lookahead=True 即'
                         ' "rollout 标签训出的模型把进攻放对" (spec §4.2 承重判据)。')}
    os.makedirs(os.path.dirname(args.report) or '.', exist_ok=True)
    with open(args.report, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps({'type': 'SAVE', 'report': args.report,
                      'rows': [{r['method']: {'legal': r['eval']['legal_rate'],
                                              'gap': r['global_mode_align_gap']['gap'],
                                              'n_train': r['n_train_rebalanced']}}
                               for r in runs], 'delta': delta}, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    sys.exit(main())
