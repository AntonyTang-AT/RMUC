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


GLOBAL_MODES = ('进攻', '均衡', '防守')


def record_mode(rec):
    """从 completion(decision JSON) 提取 global_mode; 缺省 '均衡'。"""
    return json.loads(rec['completion']).get('global_mode', '均衡')


def rebalance_records(records, target_ratios=None, seed=0):
    """按 global_mode 上采样缺类, 使语料分布逼近 target_ratios(相对占比)。只增不删。

    以占比最大类为锚(原样保留), 其余类复制样本到其相对目标量; 已超目标的类保留全部
    (不降采样, 守住"只增不删")。复制按 pool 轮转以摊薄重复, 最后 seeded 洗牌去时序偏置。
    target_ratios: {mode: 相对权重}; 缺省 {进攻:45, 防守:30, 均衡:25}, 防止 '均衡' 被淹没。"""
    from collections import defaultdict
    import random
    tr = target_ratios or {'进攻': 45, '防守': 30, '均衡': 25}
    grouped = defaultdict(list)
    for r in records:
        grouped[record_mode(r)].append(r)
    anchor = max(tr, key=tr.get)
    anchor_cnt = len(grouped.get(anchor, [])) or 1
    rnd = random.Random(seed)
    out = list(grouped.get(anchor, []))
    for m in GLOBAL_MODES:
        if m == anchor:
            continue
        pool = grouped.get(m, [])
        if not pool:
            continue
        target = anchor_cnt * tr[m] / tr[anchor]
        if target <= len(pool):
            out.extend(pool)            # 已超目标: 保留全部
        else:
            k = int(target // len(pool))
            out.extend(pool * k)
            extra = int(round(target)) - k * len(pool)
            if extra > 0:
                out.extend(pool[:extra])
    rnd.shuffle(out)
    return out


def to_hf_dataset(records):
    from datasets import Dataset
    return Dataset.from_list(records)


def records_from_rollout(dps, bounds, fused_decisions, win_model, T=5, team='红'):
    """用反事实最优承诺标签生成训练 records(instruction/completion)。

    内部走 labels.rollout_label.relabel_decisions_rollout: 终把每个决策点的 global_mode 换成
    rollout 最优标签、units 由 derive_unit_intent 重派生, 再复用 make_records 生成 对。
    用途: 产出带 '_rollout' 后缀的训练 jsonl(比其他标注法更权威, 代价是更慢、需先训 win_model)。
    """
    from worldstate import worldstate as ws
    from labels import rollout_label as rrl
    ws_list = [ws.build_worldstate(dp, bounds) for dp in dps]
    dec, _labels, _scores = rrl.relabel_decisions_rollout(ws_list, dps, bounds, fused_decisions,
                                                          win_model, T=T, team=team)
    return make_records(dec, dps, bounds, team)
