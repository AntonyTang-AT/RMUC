# -*- coding: utf-8 -*-
"""M2 数据集: 融合决策 + worldstate -> instruction/completion 对, 供 QLoRA 蒸馏。
无 torch(除 to_hf_dataset 惰性 import datasets)。"""
import json
from schema import layered as sl
from worldstate import worldstate as ws

SYSTEM = ("你是一台比赛策略决策哨兵。根据双方当前可见世界状态和预测参考信息，输出一组分层宏观决策"
          "(global_mode + 英雄/工程/步兵3/步兵4/空中/哨兵 六类单位的 target/move/fire/risk)。"
          "输入中若提供全局分析，按现状描述（分敌我并突出特殊事件/位置/状态）、预测解析、整体建议三部分理解；"
          "逐单位区分敌方状态与未来预测、我方状态与建议动作及理由。预测仅供参考，不能覆盖当前事实。"
          "global_mode仍按合法三类输出，细分战术方向通过单位target/move/fire/risk表达；"
          "risk是封闭枚举，只能输出高、中、低；target=无且move=无时risk必须为低，禁止输出risk=无；"
          "短期只选当前或可达路线上的目标，中长期只参考区域和总体方向。未知字段保持未知，不要频繁切换。"
          "预测参考会标出模型原始判断、持久性基线和最终融合结果：当前可见事实优先，"
          "低置信度预测只能作为风险提示；建议动作必须引用可见单位、区域、速度、血量或明确预测字段，"
          "不能补写输入中没有的目标和状态。")


def situation_digest(w, opponent_intent=None, semantic_situation=None,
                     global_analysis=None, forecast_context=None):
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
        narrative = global_analysis.get('narrative') or {}
        if narrative.get('description'):
            lines.append('自然语言态势: ' + str(narrative['description']))
        sections = narrative.get('sections') or {}
        current = sections.get('current_situation') or {}
        if current:
            lines.append('现状（己方）: ' + str(current.get('own_side') or '未知'))
            lines.append('现状（敌方）: ' + str(current.get('opponent_side') or '未知'))
            events = current.get('special_events') or []
            if events:
                lines.append('特殊事件: ' + '；'.join(str(x) for x in events[:4]))
        forecast_analysis = sections.get('forecast_analysis') or {}
        if forecast_analysis.get('summary'):
            lines.append('预测解析: ' + str(forecast_analysis['summary']))
        overall = sections.get('overall_recommendation') or {}
        if overall.get('text'):
            lines.append('整体建议（仅作建议）: ' + str(overall['text']))
        unit_rows = sections.get('unit_analysis') or []
        if unit_rows:
            unit_lines = []
            for unit in unit_rows[:8]:
                if unit.get('relation') == '敌方':
                    unit_lines.append(str(unit.get('current_state', '')) + '；未来：' +
                                      str(unit.get('future_prediction') or '未知'))
                else:
                    unit_lines.append(str(unit.get('current_state', '')) + '；建议：' +
                                      str(unit.get('recommendation') or '未知') + '；理由：' +
                                      str(unit.get('reason') or '未知'))
            lines.append('单兵分析（敌方预测/我方建议）: ' + ' | '.join(unit_lines))
        if narrative.get('recommendation'):
            lines.append('自然语言建议: ' + str(narrative['recommendation']))
    if forecast_context:
        from worldstate.forecast_context import format_forecast_context
        # Give the formatter enough room to retain probability heads and
        # model/persistence comparisons before LoRA prompt compaction.
        forecast_text = format_forecast_context(forecast_context, max_chars=6000)
        if forecast_text:
            lines.append(forecast_text.strip())
    return "局势: " + " | ".join(lines)


def decision_to_json(d):
    return json.dumps(d, ensure_ascii=False, separators=(',', ':'))


def make_records(fused_decisions, dps, bounds, team='红', attach_intent=True,
                 keep_sidecars_in_completion=False, temporal_forecasts=None,
                 forecast_contexts=None):
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
    if temporal_forecasts is not None and forecast_contexts is not None:
        raise ValueError('temporal_forecasts and forecast_contexts are mutually exclusive')
    if temporal_forecasts is not None:
        from worldstate.forecast_context import build_forecast_context
        forecast_contexts = [build_forecast_context(item) for item in temporal_forecasts]
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
        forecast_context = None
        if forecast_contexts is not None and len(records) < len(forecast_contexts):
            forecast_context = forecast_contexts[len(records)]
        if forecast_context:
            # Keep the three-part analysis and the compact forecast reference
            # aligned. Without this, the prompt contained forecasts only as a
            # trailing JSON blob while its natural-language analysis claimed
            # no temporal predictions were available.
            analysis['forecast_context'] = forecast_context
            from worldstate.narrative import build_global_narrative
            analysis['narrative'] = build_global_narrative(analysis)
        records.append({
            'instruction': situation_digest(
                w, opponent_intent=opp, semantic_situation=situ,
                global_analysis=analysis, forecast_context=forecast_context),
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
