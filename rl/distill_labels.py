# -*- coding: utf-8 -*-
"""M2 RL 头蒸馏: 把 IPPO rollout 的 (move, fire) 转成合法 schema.layered 分层决策。
target/risk 由规则确定性填充; global_mode 默认由单位 move 聚合(from_units),
对照可切 game_theory(choose_global_mode)。"""
from schema import layered as sl
from worldstate import worldstate as ws
from tree import global_mode as gm
from game import game_theory as gt
from labels.fusion import mode_from_units


def override_by_actions(decision, per_cls_actions, w, mode):
    """把 per_cls {cls:(move,fire)} 覆盖到决策: 保留 target/risk(规则), 改 move/fire。"""
    for cls, (move, fire) in per_cls_actions.items():
        if cls not in decision['units']:
            continue
        unit = next((r for r in w['robots'].values() if r['team'] == '红' and r['type'] == cls), None)
        if unit is None:
            continue
        it = gm.derive_unit_intent(unit, w, mode)
        it['move'] = move
        it['fire'] = bool(fire)
        decision['units'][cls] = it
    return decision


def rollout_to_decisions(actions_per_dp, dps, bounds, team='红',
                         mode_source='from_units'):
    """actions_per_dp: list[dict{cls:(move,fire)}] (rl.train.IPPO.rollout_actions 输出)。
    mode_source:
      'from_units'(默认/Track2): global_mode = mode_from_units(决策), 让 fuse_mode 能分歧;
      'game_theory'(对照): 旧行为 choose_global_mode(w)。
    返回 list[dict] 合法分层决策。"""
    if mode_source not in ('from_units', 'game_theory'):
        raise ValueError(f'unknown mode_source={mode_source!r}')
    out = []
    for dp, acts in zip(dps, actions_per_dp):
        w = ws.build_worldstate(dp, bounds)
        # target/risk 填充用局势 mode; 最终写出的 global_mode 由 mode_source 决定
        provisional = gt.choose_global_mode(w)
        d = sl.build_empty_decision(dp['t'])
        for cls in sl.UNIT_CLASSES:
            move, fire = acts.get(cls, ('无', False))
            unit = next((r for r in w['robots'].values() if r['team'] == team and r['type'] == cls), None)
            if unit is None:
                continue  # 该类无单位 -> 保留 idle(合法)
            it = gm.derive_unit_intent(unit, w, provisional)
            it['move'] = move
            it['fire'] = bool(fire)
            d['units'][cls] = it
        if mode_source == 'from_units':
            d['global_mode'] = mode_from_units(d)
        else:
            d['global_mode'] = provisional
        out.append(d)
    return out


def validate_series(decisions):
    for d in decisions:
        ok, errs = sl.validate(d)
        if not ok:
            return False, errs
    return True, []
