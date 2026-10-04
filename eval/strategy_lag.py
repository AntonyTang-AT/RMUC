# -*- coding: utf-8 -*-
"""在真实对局上跑"回看标签 + 滞后一致率"(Q2 落地)。无 GPU。

预测通道 = 规则头 run_m2.rule_head 的 global_mode(确定性、数据条件) —— 作为"预测"占位;
标签通道 = labels.strategy_label.strategy_series(前向窗口[位置推进+建筑掉血]推断的短时策略)。
两者对齐后算 lag_curve: 度量"几秒前(t-δ)预测是否仍准确"的衰减, 用于校准。
未来把 规则头 换成 LLM 决策(重训后模型)即为"LLM 真值靠回看标签校准"。

用法:
  python -m eval.strategy_lag --games 1779323658229 --delta-frames 5 --team 红
"""
import os, sys, json, argparse

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.chdir(REPO)

from data import db, features
from worldstate import worldstate as ws
from labels import strategy_label as slb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--games', nargs='+', type=int, required=True)
    ap.add_argument('--delta-frames', type=int, default=5, help='回看窗口帧数(每帧=DF_STEP秒)')
    ap.add_argument('--margin', type=float, default=slb.DEF_MARGIN, help='攻防判定 margin(偏差调优: 大->均衡多, 小->极端多)')
    ap.add_argument('--sweep-margins', default=None,
                    help='逗号分隔的 margin 列表, 只打印各 margin 下回看标签分布(偏差调优), 不问一致率')
    ap.add_argument('--team', default='红')
    ap.add_argument('--out', default='_work/strategy_lag.json')
    args = ap.parse_args()

    import config
    con = db.connect()
    results = []
    for gid in args.games:
        rows = db.timeseries_for(con, gid)
        dps = features.build_decision_points(rows, gid)
        b = features.field_bounds(dps)
        ws_list = [ws.build_worldstate(dp, b) for dp in dps]
        # 偏差调优模式: 只打印回看标签在若干 margin 下的分布
        if args.sweep_margins:
            margins = [float(x) for x in args.sweep_margins.split(',')]
            for m in margins:
                lab = slb.strategy_series(ws_list, delta=args.delta_frames, team=args.team, margin=m)
                print(f'  gid={gid} margin={m}: {_mode_dist(lab)}', flush=True)
            continue
        # 预测通道: 规则头 global_mode
        import run_m2
        decs = run_m2.rule_head(dps, b)
        preds = [d.get('global_mode', '均衡') for d in decs]
        labels = slb.strategy_series(ws_list, delta=args.delta_frames, team=args.team, margin=args.margin)
        curve = slb.lag_curve(preds, labels, horizons=(0, 1, 2, 3, 5, 8))
        n = len(labels)
        seconds_per_frame = config.DF_STEP
        results.append({
            'game_id': gid, 'n_points': n, 'delta_frames': args.delta_frames,
            'delta_sec': args.delta_frames * seconds_per_frame, 'team': args.team,
            'mode_dist_label': _mode_dist(labels),
            'lag_curve': curve,
            'method': ('标签=前向窗口[推送+建筑掉血]回看策略; 预测通道=规则头 global_mode;'
                       ' horizon=k 帧前预测 vs 当前标签 一致率'),
        })
        print(json.dumps(results[-1], ensure_ascii=False), flush=True)
    con.close()
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print('SAVE', args.out, flush=True)
    return 0


def _mode_dist(labels):
    d = {}
    for x in labels:
        d[x] = d.get(x, 0) + 1
    return d


if __name__ == '__main__':
    sys.exit(main())
