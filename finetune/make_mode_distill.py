# -*- coding: utf-8 -*-
"""从主线局构建 Tier1 mode 蒸馏集：lookahead 教师 mode + 40 维 obs + need_llm。"""
from __future__ import annotations

import json
import os
from collections import Counter

from schema import layered as sl
from worldstate import worldstate as ws
from labels import strategy_label as slb
from labels import router as rt
from rl import mode_features as mf


def build_teacher_modes(dps, bounds, lookahead_delta=5, margin=None, team='红'):
    """规则头 + lookahead 重标 → list[mode]。"""
    import run_m3
    decisions, _, _ = run_m3.compute_decision_series(dps, bounds)
    if lookahead_delta and lookahead_delta > 0:
        ws_list = [ws.build_worldstate(dp, bounds) for dp in dps]
        marg = margin if margin is not None else slb.DEF_MARGIN
        decisions, labels = slb.relabel_decision_series(
            ws_list, decisions, lookahead_delta, team=team, margin=marg)
        return [d.get('global_mode', '均衡') for d in decisions], decisions, labels
    return [d.get('global_mode', '均衡') for d in decisions], decisions, None


def samples_for_game(gid, dps, bounds, lookahead_delta=5, margin=None, team='红',
                     conflict_rate=0.0):
    modes, decisions, _ = build_teacher_modes(
        dps, bounds, lookahead_delta=lookahead_delta, margin=margin, team=team)
    rows = []
    mode_hist = []
    for dp, mode, d in zip(dps, modes, decisions):
        row = mf.sample_row(
            dp, bounds, teacher_mode=mode, mode_history=mode_hist,
            conflict_rate=conflict_rate, team=team, game_id=gid)
        rows.append(row)
        mode_hist.append(mode)
        if len(mode_hist) > 5:
            mode_hist = mode_hist[-5:]
    return rows, decisions


def build_mode_distill(game_ids, load_decision_points, out_path,
                       lookahead_delta=5, margin=None, team='红', limit=None):
    """扫多局写 jsonl。load_decision_points(gid, limit=None)->(dps,bounds)。"""
    all_rows = []
    for i, gid in enumerate(game_ids):
        dps, b = load_decision_points(gid, limit)
        if not dps:
            continue
        rows, _ = samples_for_game(
            gid, dps, b, lookahead_delta=lookahead_delta, margin=margin, team=team)
        all_rows.extend(rows)
        print(f'[mode_distill] game {i + 1}/{len(game_ids)} gid={gid} n={len(rows)}',
              flush=True)
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        for r in all_rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    modes = Counter(r['mode'] for r in all_rows)
    n_tier1 = sum(1 for r in all_rows if not r['need_llm'])
    summary = {
        'path': out_path,
        'n': len(all_rows),
        'n_tier1': n_tier1,
        'need_llm_rate': round(1.0 - n_tier1 / max(len(all_rows), 1), 4),
        'mode_dist': {k: modes[k] for k in sl.GLOBAL_MODES},
        'game_ids': list(game_ids),
        'lookahead_delta': lookahead_delta,
    }
    print('mode_distill written:', json.dumps(summary, ensure_ascii=False), flush=True)
    return all_rows, summary


def main():
    import argparse
    import run_m2 as m2
    from data import db

    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='out_m2/mode_distill.jsonl')
    ap.add_argument('--train-games', type=int, default=3)
    ap.add_argument('--dataset-games', type=int, default=24)
    ap.add_argument('--lookahead-delta', type=int, default=5)
    ap.add_argument('--limit', type=int, default=None)
    args = ap.parse_args()

    con = db.connect()
    gids = m2.list_game_ids(con)
    con.close()
    dataset_ids = gids[args.train_games:args.train_games + args.dataset_games]
    build_mode_distill(
        dataset_ids, m2.load_decision_points, args.out,
        lookahead_delta=args.lookahead_delta, limit=args.limit)


if __name__ == '__main__':
    main()
