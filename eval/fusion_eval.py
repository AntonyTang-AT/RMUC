# -*- coding: utf-8 -*-
"""M2 fusion-level 评估(纯 numpy, 无 torch/transformers/sklearn, 单测离线可跑)。

为什么做: Track 2(轨2)要回答 "Plan B 物理 RL 源是否真学得动、且让融合后的宏观决策更贴合胜负"。

★ 承重事实(设计定锚): `labels.fusion.fuse` 把融合结果的 `global_mode` **锚定为 rule 头**的
  global_mode(line 52: `d['global_mode'] = src_rule.get(...)`), RL 源只覆盖每个单位类的
  `move/fire`。因此**不能用 global_mode_align_gap 区分 rule-only 与 rl 融合**(两者全球模式恒同)。
  RL 源的真实贡献在**单位级 move/fire**。故本模块的判别指标是 **RL 源自身的单位级质量**:
  攻击倾向(aggression)是否落在恰当胜势区间 + 该源是否合法/非退化。global_mode 只作为全局退化
  参考(见 fusion_overview.align_gap, 注明非判别项)。

三大口径:
  1. legality     —— RL 源/融合决策是否仍合法(sl.validate)。
  2. degeneracy   —— global_mode 是否退化为单模(Plan B 判据: 攻+均<0.10 或类别唯一);
                     及 RL 源 move 是否多样(常数策略 = 退化)。
  3. fav-alignment —— RL 源单位级攻击倾向 vs 红方有利度: Δ攻击度(RL−rule)在高 fav 时更高。
                     期望: physics RL 源 的 fav_align ≥ bandit RL 源(见 compare_rl_sources)。

★ 诚实上限: fav_align 是关联性/条件有利度,非因果(设计 §3);地基是 outcome_backtest.cv_auc_win_model
  (状态确实携带胜负)。仅 4 留出局,数值以方向与量级为准。
"""
import numpy as np

from collections import Counter

from eval import decision_gap as dgap
from eval.outcome_backtest import fav_from_worldstate

# global_mode 退化判据阈值: 攻+均占比低于此 → 视为被挤成"恒守/单模"
_AB_SHARE_FLOOR = 0.10
# 单位级 move 攻击倾向标度(取值 [-1,1]; 仅用于"相对高低"的排序/相关,非精确价值)。
# 注意: 单位级 move 词汇是 schema.layered.MOVES(推进/回防/游击/坚守/绕行/无), 与 global_mode 词汇(进攻/均衡/防守)不同。
_MOVE_AGGRESSION = {'推进': 1.0, '绕行': 0.4, '游击': 0.0, '无': 0.0, '坚守': -0.6, '回防': -1.0}


def _aggr(move):
    return _MOVE_AGGRESSION.get(move, 0.0)


def _unit_classes():
    from schema import layered as sl
    return list(sl.UNIT_CLASSES)


def _validate(d):
    from schema import layered as sl
    return sl.validate(d)[0]


def mode_distribution(decisions):
    """decisions: list[dict](决策) -> {global_mode: 归一占比}。空 -> {}。"""
    modes = [d.get('global_mode', '均衡') for d in decisions]
    if not modes:
        return {}
    c = Counter(modes)
    n = len(modes)
    return {k: round(v / n, 4) for k, v in sorted(c.items())}


def legal_rate(decisions):
    """合法分层决策占比。空 -> 0.0。"""
    if not decisions:
        return 0.0
    return sum(1 for d in decisions if _validate(d)) / len(decisions)


def attack_balance_share(mode_dist):
    """进攻+均衡 占比(Plan B 全局退化判据核心)。"""
    return round(mode_dist.get('进攻', 0.0) + mode_dist.get('均衡', 0.0), 4)


def is_degenerate(mode_dist):
    """global_mode 是否退化为单模/恒守。类别唯一 或 攻+均<0.10 -> True。"""
    if len(mode_dist) < 2:
        return True
    return attack_balance_share(mode_dist) < _AB_SHARE_FLOOR


def fusion_align_gap(decisions, worldstates, team='红'):
    """global_mode 对齐(参考项): 因 fuse 锚定 rule, 对 rule-only 与 rl 融合恒同 → 非判别项。"""
    modes = [d.get('global_mode', '均衡') for d in decisions]
    return dgap.global_mode_align_gap(modes, worldstates, team)


def rl_source_metrics(rl_decisions, rule_decisions, worldstates, team='红'):
    """RL 源自身单位级质量(判别口径)。rl/rule: list[dict] 同序、同点; worldstates: 对齐 list。

    返回 {'legal_rate', 'override_rate', 'move_diversity', 'fav_align'}.
      override_rate  : RL 与规则头在 (point,class) 上 move 不同的占比 —— 0 则 RL 是空转。
      move_diversity : 每单位类 RL 输出的不同 move 种类数 —— 全为 1 则常数策略(退化)。
      fav_align      : mean(Δ攻击度 | fav高) − mean(Δ攻击度 | fav低), Δ = RL−rule; fav 以中位数二分。
    """
    favs = [fav_from_worldstate(w, team) for w in worldstates]
    classes = _unit_classes()
    deltas = []
    override = total = 0
    distinct = {c: set() for c in classes}
    for i, (rd, dd) in enumerate(zip(rule_decisions, rl_decisions)):
        fav = favs[i] if i < len(favs) else 0.0
        for c in classes:
            ru = rd.get('units', {}).get(c)
            du = dd.get('units', {}).get(c)
            if not ru or not du:
                continue
            rm, dm = ru['move'], du['move']
            total += 1
            distinct[c].add(dm)
            if rm != dm:
                override += 1
            deltas.append((_aggr(dm) - _aggr(rm), fav))
    fav_align = None
    if deltas:
        arr = np.asarray(deltas, dtype=float)
        fav, d = arr[:, 1], arr[:, 0]
        med = float(np.median(fav))
        hi = d[fav > med] if (fav > med).any() else d
        lo = d[fav <= med] if (fav <= med).any() else d
        fav_align = round(float(hi.mean() - lo.mean()), 4)
    legal = legal_rate(rl_decisions)
    return {'legal_rate': round(legal, 4),
            'override_rate': round(override / total, 4) if total else 0.0,
            'move_diversity': {c: len(s) for c, s in distinct.items()},
            'fav_align': fav_align}


def fusion_overview(fused_result, worldstates, team='红'):
    """融合结果的全局面(参考口径): fuse() 输出 dict。返回合法性/退化/冲突 + 参考 align_gap。
    align_gap 因 fuse 锚定 rule 与 rule-only 恒同 → 列作参考, 非判别项。"""
    decs = fused_result['decisions']
    md = mode_distribution(decs)
    return {'n': len(decs),
            'legal_rate': round(legal_rate(decs), 4),
            'conflict_rate': round(fused_result.get('conflict_rate', 0.0), 4),
            'mode_dist': md,
            'attack_balance_share': attack_balance_share(md),
            'degenerate': is_degenerate(md),
            'align_gap': fusion_align_gap(decs, worldstates, team)}


def compare_rl_sources(physics_m, bandit_m):
    """physics vs bandit RL 源关键 delta(判别口径)。两者都有 'fav_align','legal_rate','override_rate'。
    返回 {'fav_align_delta_phys_minus_bandit', 'phys_ge_bandit_fav_align', 'both_legal', 'both_override'}."""
    fa_p, fa_b = physics_m.get('fav_align'), bandit_m.get('fav_align')
    def _bool(x):
        return bool(x) if x is not None else None
    return {'fav_align_delta_phys_minus_bandit':
                (round(fa_p - fa_b, 4) if (fa_p is not None and fa_b is not None) else None),
            'phys_ge_bandit_fav_align': _bool(fa_p >= fa_b) if (fa_p is not None and fa_b is not None) else None,
            'both_legal': (physics_m['legal_rate'] >= 0.9 and bandit_m['legal_rate'] >= 0.9),
            'both_override': (physics_m['override_rate'] > 0.0 and bandit_m['override_rate'] > 0.0),
            'physics_fav_align': fa_p, 'bandit_fav_align': fa_b}
