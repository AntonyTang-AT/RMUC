# -*- coding: utf-8 -*-
"""M2 数据集: 融合决策 + worldstate -> instruction/completion 对, 供 QLoRA 蒸馏。
无 torch(除 to_hf_dataset 惰性 import datasets)。"""
import json
from schema import layered as sl
from worldstate import worldstate as ws

SYSTEM = ("你是一台比赛策略决策哨兵。根据双方当前世界状态, 输出一组分层宏观决策"
          "(global_mode + 英雄/工程/步兵3/步兵4/空中/哨兵 六类单位的 target/move/fire/risk)。")


def situation_digest(w):
    """把 worldstate 压成紧凑中文局势摘要(供 instruction)。"""
    lines = [f"t={w['t']:.0f}"]
    for tag in ('红', '蓝'):
        grp = [r for r in w['robots'].values() if r['team'] == tag]
        cells = []
        for r in grp:
            cells.append(f"{r['type']} hp{r['hp_ratio']:.2f} xy({r['norm_xy'][0]:.2f},{r['norm_xy'][1]:.2f})"
                         f"dfire{r.get('dfire', 0):.0f}")
        lines.append(f"{tag}: " + (" | ".join(cells) if cells else "无"))
    rb = lb = 1.0
    for b in w['buildings'].values():
        if b['type'] == '基地':
            if b['team'] == '红':
                rb = b['hp_ratio']
            else:
                lb = b['hp_ratio']
    lines.append(f"基地血 红{rb:.2f} 蓝{lb:.2f}")
    return "局势: " + " | ".join(lines)


def decision_to_json(d):
    return json.dumps(d, ensure_ascii=False, separators=(',', ':'))


def make_records(fused_decisions, dps, bounds, team='红'):
    """fused_decisions 与 dps 等长; 返回 list[{'instruction','completion'}]。"""
    records = []
    for d, dp in zip(fused_decisions, dps):
        w = ws.build_worldstate(dp, bounds)
        records.append({'instruction': situation_digest(w), 'completion': decision_to_json(d)})
    return records


def to_jsonl(records, path):
    with open(path, 'w', encoding='utf-8') as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    return path


def to_hf_dataset(records):
    from datasets import Dataset
    return Dataset.from_list(records)
