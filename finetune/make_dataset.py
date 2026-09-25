# -*- coding: utf-8 -*-
"""M2 数据集: 融合决策 + worldstate -> instruction/completion 对, 供 QLoRA 蒸馏。
无 torch(除 to_hf_dataset 惰性 import datasets)。"""
import json
from schema import layered as sl
from worldstate import worldstate as ws

SYSTEM = ("你是一台比赛策略决策哨兵。根据双方当前世界状态, 输出一组分层宏观决策"
          "(global_mode + 英雄/工程/步兵3/步兵4/空中/哨兵 六类单位的 target/move/fire/risk)。")


def situation_digest(w, opponent_intent=None, semantic_situation=None,
                     global_analysis=None):
    """把 worldstate 压成紧凑中文局势摘要(供 instruction)。
    可选注入敌方意图概率行与语义态势行（Tier1 快路径产出）。"""
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
    if opponent_intent:
        from labels import opp_intent as oi
        intent_line = oi.format_intent_line(opponent_intent)
        if intent_line:
            lines.append(intent_line)
    if semantic_situation:
        from worldstate import semantic as sem
        sem_line = sem.format_semantic_line(semantic_situation)
        if sem_line:
            lines.append(sem_line)
    if global_analysis:
        state = global_analysis.get('state') or {}
        lines.append(
            '全局分析: mode={mode} threat={threat} confidence={conf:.2f} '
            'push_margin={margin} enemy_trend={trend}'.format(
                mode=global_analysis.get('current_mode', '均衡'),
                threat=global_analysis.get('threat_level', '未知'),
                conf=float(global_analysis.get('confidence') or 0.0),
                margin=int(state.get('push_margin') or 0),
                trend=state.get('enemy_trend', 'unknown')))
        evidence = '；'.join(str(x) for x in (global_analysis.get('evidence') or [])[:3])
        if evidence:
            lines.append('证据: ' + evidence)
        horizons = global_analysis.get('horizon_views') or {}
        if horizons:
            lines.append('时间尺度: 10s高细节；30-60s中等粒度；120s只判断总体攻防方向')
    return "局势: " + " | ".join(lines)


def decision_to_json(d):
    return json.dumps(d, ensure_ascii=False, separators=(',', ':'))


def make_records(fused_decisions, dps, bounds, team='红', attach_intent=True,
                 keep_sidecars_in_completion=False):
    """fused_decisions 与 dps 等长; 返回 list[{'instruction','completion'}]。
    attach_intent: 逐步推断敌方意图并写入 digest。
    keep_sidecars_in_completion: False(默认)时 completion 只保留主 schema，
    意图以输入条件为主，缩短生成负担；True 时旁路字段一并写入 completion。
    """
    from labels import opp_intent as oi
    from labels import router as rt
    from worldstate import semantic as semmod
    from tree import global_mode as gm
    records = []
    tracker = oi.GameIntentTracker()
    prev_w = None
    mode_hist = []
    for d, dp in zip(fused_decisions, dps):
        w = ws.build_worldstate(dp, bounds)
        opp = None
        if attach_intent:
            opp = oi.infer_opponent_intent(w, prev_w, our_team=team, tracker=tracker)
        situ = semmod.derive_semantic_situation(w, team=team, prev_w=prev_w)
        sentry = gm.derive_sentry_state(w, team=team)
        analysis = semmod.derive_global_analysis(
            w, team=team, prev_w=prev_w, mode=d.get('global_mode'),
            opponent_intent=opp, sentry_state=sentry)
        route = rt.route_tier(w, mode_history=mode_hist, our_team=team)
        d_out = rt.attach_decision_meta(
            d, route, opponent_intent=opp, semantic_situation=situ,
            sentry_state=sentry, strategic_intent=gm.sentry_to_strategic(sentry),
            global_analysis=analysis)
        if not keep_sidecars_in_completion:
            d_out = rt.strip_sidecar_for_validate(d_out)
        mode_hist.append(d.get('global_mode', '均衡'))
        if len(mode_hist) > 5:
            mode_hist = mode_hist[-5:]
        records.append({
            'instruction': situation_digest(
                w, opponent_intent=opp, semantic_situation=situ,
                global_analysis=analysis),
            'completion': decision_to_json(d_out),
        })
        prev_w = w
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


def rebalance_records(records, target_ratios=None, seed=0, max_upsample=3):
    """按 global_mode 上采样缺类, 使语料分布逼近 target_ratios(相对占比)。只增不删。

    以"实际占比相对目标占比最超配"的类为锚(原样保留), 其余类复制样本到其相对目标量; 已超
    目标的类保留全部(不降采样, 守住"只增不删")。锚类取 actual_count/target_ratio 最大者——
    它是当前相对目标最"过剩"的类, 不删它、把它对齐到自己的目标占比, 其余类按比例补齐。这样无论
    源分布偏向哪个类(防守多数/均衡多数/进攻多数), 都能把语料凑到 target_ratios(修 bug: 原实现
    锚定 max(target_ratio), 当目标比例最大的类恰是稀有类时配平完全失效)。

    max_upsample: 每类每个样本最多复制次数上限(默认 3), 禁止无上限 10× 复制过拟合。
    复制按 pool 轮转以摊薄重复, 最后 seeded 洗牌去时序偏置。
    target_ratios: {mode: 相对权重}; 缺省 {进攻:45, 防守:30, 均衡:25}, 防'防守'或'均衡'被淹没。"""
    from collections import defaultdict
    import random
    tr = target_ratios or {'进攻': 45, '防守': 30, '均衡': 25}
    max_up = max(1, int(max_upsample))
    grouped = defaultdict(list)
    for r in records:
        grouped[record_mode(r)].append(r)
    # 锚类 = actual_count/target_ratio 最大(最超配); 排除目标占比为 0 或空类
    cands = [m for m in grouped if tr.get(m)]
    if not cands:
        return list(records)
    anchor = max(cands, key=lambda m: len(grouped[m]) / tr[m])
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
        cap = max_up * len(pool)
        target_capped = min(float(target), float(cap))
        if target_capped <= len(pool):
            out.extend(pool)            # 已超目标: 保留全部
        else:
            k = int(target_capped // len(pool))
            out.extend(pool * k)
            extra = int(round(target_capped)) - k * len(pool)
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
