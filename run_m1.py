# -*- coding: utf-8 -*-
import os, argparse, json, time
import config
from data import db, features
from worldstate import zone, worldstate as ws
from labels import sentry_labels as sl
from fastloop import behavior as bh
from eval import metrics
from tree import global_mode as gm

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--games', nargs='*', type=int, default=None)
    ap.add_argument('--out', default='out_m1')
    ap.add_argument('--no-llm', action='store_true', help='仅评估规则基线(快环),不加载LLM')
    ap.add_argument('--rule-baseline', action='store_true', help='诚实规则基线: 用 derive_sentry_state(非 LLM/非真值)提议, 作下限参照')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    con = db.connect()
    gids = args.games or [config.GAME_REF]
    y_true, y_pred, latencies, samples = [], [], [], []
    rules = bh.Rules()
    model = tok = None
    if not (args.no_llm or args.rule_baseline):
        model, tok = _load()
    for gid in gids:
        rows = db.timeseries_for(con, gid)
        dps = features.build_decision_points(rows, gid)
        b = features.field_bounds(dps)
        series = sl.label_sentry_series(dps, b, team='红', sentry_rid=7)
        for dp, (t, state) in zip(dps, series):
            w = ws.build_worldstate(dp, b)
            if args.rule_baseline:
                pred = bh.decide_fast({'robots': w['robots'], 'bot_state': gm.derive_sentry_state(w)}, rules)
                p = pred['state']
            elif args.no_llm:
                pred = bh.decide_fast({'robots': w['robots'], 'bot_state': state}, rules)
                p = pred['state']
            else:
                t0 = time.time()
                res = _llm_decide(model, tok, w)
                latencies.append(time.time() - t0)
                pred = bh.decide_fast({'robots': w['robots'], 'bot_state': res['state']}, rules)
                p = pred['state']
            y_true.append(state); y_pred.append(p)
            samples.append({'t': t, 'true': state, 'pred': p})
    con.close()
    legal = metrics.legal_rate([{'state': s} for s in y_pred])
    report = {
        'mode': ('rule_baseline' if args.rule_baseline
                 else ('rule_baseline_smoke' if args.no_llm else 'llm_slowloop')),
        'games': gids, 'n': len(samples),
        'accuracy': metrics.accuracy(y_true, y_pred),
        'f1_macro': metrics.f1_macro(y_true, y_pred),
        'legal_rate': legal,
        'per_class': metrics.per_class(y_true, y_pred),
        'mean_latency_s': (sum(latencies)/len(latencies) if latencies else None),
        'confusion': metrics.confusion(y_true, y_pred, list(sl.SENTRY_STATES)),
    }
    with open(os.path.join(args.out, 'report.json'), 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.no_llm:
        print("[WARN] --no-llm: rule-baseline smoke (feeds true label as proposal; acc≈1.0 is an infra/fast-loop-identity signal, NOT a model-quality claim).")

def _load():
    from model import predict
    return predict.load_model('Qwen/Qwen2.5-1.5B-Instruct', quantize=True)

def _llm_decide(model, tok, w):
    from model import predict
    return predict.llm_decide(model, tok, w)

if __name__ == '__main__':
    main()
