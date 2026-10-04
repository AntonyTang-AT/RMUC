# -*- coding: utf-8 -*-
"""敌方下一步意图概率（Tier1 快路径）。

不反演完整决策树：输出离散意图分布 P(i)，并用本局 top 意图频次抬升。
纯 numpy-free，无 torch。
"""
from __future__ import annotations

INTENTS = ('压前哨', '拆基地', '盯哨兵', '换血前排', '回防收缩', '摸资源/中立')
_DEFAULT_ALPHA = 0.5


def _building_hp(w, team, btype):
    for b in (w.get('buildings') or {}).values():
        if b.get('team') == team and b.get('type') == btype:
            return float(b.get('hp_ratio') or 0.0)
    return 1.0


def _robots(w, team):
    return [r for r in (w.get('robots') or {}).values() if r.get('team') == team]


def _mean_xy(robots):
    if not robots:
        return None
    xs = [r['norm_xy'][0] for r in robots if r.get('norm_xy')]
    ys = [r['norm_xy'][1] for r in robots if r.get('norm_xy')]
    if not xs:
        return None
    return (sum(xs) / len(xs), sum(ys) / len(ys))


def _sentry(robots):
    for r in robots:
        if r.get('type') == '哨兵':
            return r
    return None


def _frontline(robots):
    return [r for r in robots if r.get('type') in ('英雄', '步兵3', '步兵4')]


def prior_probs(t=0.0):
    """开局偏摸资源/压前哨；中后期趋均匀。"""
    early = 1.0 if (t is None or t < 60.0) else (0.4 if t < 180.0 else 0.0)
    base = {i: 1.0 for i in INTENTS}
    base['摸资源/中立'] += 0.8 * early
    base['压前哨'] += 0.6 * early
    return _normalize(base)


def likelihood_scores(w_prev, w_curr, our_team='红'):
    """由 Δs 打分：敌方行为证据 → 各意图非负分数（未归一化）。"""
    opp = '蓝' if our_team == '红' else '红'
    scores = {i: 0.15 for i in INTENTS}  # 平滑地板
    if w_curr is None:
        return scores
    if w_prev is None:
        w_prev = w_curr

    # 敌前哨掉血 → 压前哨
    d_out = _building_hp(w_prev, opp, '前哨站') - _building_hp(w_curr, opp, '前哨站')
    if d_out > 0.002:
        scores['压前哨'] += min(2.0, d_out * 25.0)

    # 敌基地掉血 → 拆基地
    d_base = _building_hp(w_prev, opp, '基地') - _building_hp(w_curr, opp, '基地')
    if d_base > 0.002:
        scores['拆基地'] += min(2.5, d_base * 30.0)

    # 我方哨兵承伤 / 敌哨兵逼近 → 盯哨兵
    our_s = _sentry(_robots(w_curr, our_team))
    prev_s = _sentry(_robots(w_prev, our_team))
    if our_s and prev_s:
        dmg = float(prev_s.get('hp_ratio') or 0) - float(our_s.get('hp_ratio') or 0)
        if dmg > 0.01:
            scores['盯哨兵'] += min(2.0, dmg * 20.0)
    opp_s = _sentry(_robots(w_curr, opp))
    if our_s and opp_s and our_s.get('norm_xy') and opp_s.get('norm_xy'):
        dx = our_s['norm_xy'][0] - opp_s['norm_xy'][0]
        dy = our_s['norm_xy'][1] - opp_s['norm_xy'][1]
        dist = (dx * dx + dy * dy) ** 0.5
        if dist < 0.25:
            scores['盯哨兵'] += 0.8

    # 双方前排互相掉血 → 换血前排
    def _front_hp(w, team):
        rs = _frontline(_robots(w, team))
        if not rs:
            return None
        return sum(float(r.get('hp_ratio') or 0) for r in rs) / len(rs)

    fh_o0, fh_o1 = _front_hp(w_prev, opp), _front_hp(w_curr, opp)
    fh_u0, fh_u1 = _front_hp(w_prev, our_team), _front_hp(w_curr, our_team)
    if None not in (fh_o0, fh_o1, fh_u0, fh_u1):
        if (fh_o0 - fh_o1) > 0.02 and (fh_u0 - fh_u1) > 0.02:
            scores['换血前排'] += min(2.0, ((fh_o0 - fh_o1) + (fh_u0 - fh_u1)) * 12.0)

    # 敌方质心回撤（对红：norm_x 变小；对蓝：norm_x 变大）→ 回防收缩
    c0 = _mean_xy(_robots(w_prev, opp))
    c1 = _mean_xy(_robots(w_curr, opp))
    if c0 and c1:
        if opp == '蓝':
            retreat = c1[0] - c0[0]  # 蓝往右（己方）撤
        else:
            retreat = c0[0] - c1[0]  # 红往左撤
        if retreat > 0.03:
            scores['回防收缩'] += min(1.5, retreat * 15.0)

    # 中立区活动 / 工程活跃且建筑伤害弱 → 摸资源
    eng = [r for r in _robots(w_curr, opp) if r.get('type') == '工程']
    if eng and d_out < 0.001 and d_base < 0.001:
        scores['摸资源/中立'] += 0.5
        for e in eng:
            xy = e.get('norm_xy') or [0.5, 0.5]
            if 0.35 <= xy[0] <= 0.65:
                scores['摸资源/中立'] += 0.4

    return scores


def apply_game_boost(probs, counts, alpha=_DEFAULT_ALPHA):
    """本局意图记录抬升: P' ∝ P * (1 + α * c_i / Σc)。"""
    counts = counts or {}
    total = float(sum(max(0, int(counts.get(i, 0))) for i in INTENTS))
    out = {}
    for i in INTENTS:
        c = max(0, int(counts.get(i, 0)))
        mult = 1.0 + (alpha * c / total if total > 0 else 0.0)
        out[i] = float(probs.get(i, 0.0)) * mult
    return _normalize(out)


def _normalize(d):
    s = sum(max(0.0, float(v)) for v in d.values()) or 1.0
    return {k: round(max(0.0, float(d.get(k, 0.0))) / s, 4) for k in INTENTS}


def combine_prior_likelihood(prior, scores):
    """P ∝ prior * (ε + score)。"""
    out = {}
    for i in INTENTS:
        out[i] = float(prior.get(i, 0.0)) * (0.05 + float(scores.get(i, 0.0)))
    return _normalize(out)


class GameIntentTracker:
    """局级 top 意图频次缓冲。"""

    def __init__(self):
        self.counts = {i: 0 for i in INTENTS}

    def observe(self, top_intent):
        if top_intent in self.counts:
            self.counts[top_intent] += 1

    def as_dict(self):
        return dict(self.counts)


def infer_opponent_intent(w_curr, w_prev=None, our_team='红', tracker=None, alpha=_DEFAULT_ALPHA):
    """主入口：返回 opponent_intent 旁路结构。"""
    t = float((w_curr or {}).get('t') or 0.0)
    prior = prior_probs(t)
    scores = likelihood_scores(w_prev, w_curr, our_team=our_team)
    probs = combine_prior_likelihood(prior, scores)
    counts = tracker.as_dict() if tracker is not None else {}
    probs = apply_game_boost(probs, counts, alpha=alpha)
    top = max(probs, key=probs.get)
    if tracker is not None:
        tracker.observe(top)
    return {
        'probs': probs,
        'top': top,
        'source': 'delta_s+game_boost',
        'counts': dict(counts),
    }


def format_intent_line(opp_intent):
    """写入 digest 的一行。"""
    if not opp_intent or not opp_intent.get('probs'):
        return ''
    probs = opp_intent['probs']
    ordered = sorted(probs.items(), key=lambda kv: -kv[1])
    parts = [f'{k}{v:.2f}' for k, v in ordered[:4]]
    top = opp_intent.get('top', ordered[0][0] if ordered else '')
    return f"敌方意图(top={top}): " + ' | '.join(parts)


def annotate_series(worldstates, our_team='红', alpha=_DEFAULT_ALPHA):
    """对一局 worldstate 序列逐步推断意图。返回 list[opponent_intent]。"""
    tracker = GameIntentTracker()
    out = []
    prev = None
    for w in worldstates:
        oi = infer_opponent_intent(w, prev, our_team=our_team, tracker=tracker, alpha=alpha)
        out.append(oi)
        prev = w
    return out
