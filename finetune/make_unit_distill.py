# -*- coding: utf-8 -*-
"""从融合/lookahead 决策构建哨兵单位蒸馏 jsonl（脚手架，本机可不训）。

用法:
  python -c "from finetune.make_unit_distill import build_and_write; ..."
"""
from __future__ import annotations

import json
import os

from rl import unit_features as uf
from labels import opp_intent as oi
from worldstate import worldstate as ws
from finetune import make_mode_distill as mmd


def samples_for_game(gid, dps, bounds, lookahead_delta=5, team='红'):
    modes, decisions, _ = mmd.build_teacher_modes(
        dps, bounds, lookahead_delta=lookahead_delta, team=team)
    tracker = oi.GameIntentTracker()
    prev = None
    rows = []
    for dp, d in zip(dps, decisions):
        w = ws.build_worldstate(dp, bounds)
        opp = oi.infer_opponent_intent(w, prev, our_team=team, tracker=tracker)
        # 教师单位已在 decisions 中；若为空则用 refill
        if not (d.get('units') or {}).get(uf.UNIT_FOCUS):
            from rl import mode_features as mf
            d = mf.refill_units_for_mode(d, w, d.get('global_mode', '均衡'),
                                         team=team, opp_intent=opp)
        rows.append(uf.sample_sentry_row(
            dp, bounds, d, opp_intent=opp, team=team, game_id=gid))
        prev = w
    return rows


def write_jsonl(rows, path):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    return path


def build_and_write(game_loader, gids, out_path, lookahead_delta=5, limit=None):
    """game_loader(gid) -> (dps, bounds)。"""
    all_rows = []
    for gid in gids:
        dps, b = game_loader(gid)
        if limit:
            dps = dps[:limit]
        all_rows.extend(samples_for_game(gid, dps, b, lookahead_delta=lookahead_delta))
    write_jsonl(all_rows, out_path)
    return {'n': len(all_rows), 'path': out_path, 'feat_dim': uf.FEAT_DIM}
