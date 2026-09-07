# -*- coding: utf-8 -*-
"""M2 RL 头蒸馏: 把 IPPO rollout 的 (move, fire) 转成合法 schema.layered 分层决策。
target/risk 由规则确定性填充; global_mode 用博弈论 choose_global_mode(局势感知)。"""
from schema import layered as sl
from worldstate import worldstate as ws
from tree import global_mode as gm
from game import game_theory as gt
from rl.env import action_to_intent


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


def rollout_to_decisions(actions_per_dp, dps, bounds, team='红'):
    """actions_per_dp: list[dict{cls:(move,fire)}] (rl.train.IPPO.rollout_actions 输出)。
    返回 list[dict] 合法分层决策。"""
    out = []
    for dp, acts in zip(dps, actions_per_dp):
        w = ws.build_worldstate(dp, bounds)
        mode = gt.choose_global_mode(w)
        d = sl.build_empty_decision(dp['t'])
        d['global_mode'] = mode
        for cls in sl.UNIT_CLASSES:
            move, fire = acts.get(cls, ('无', False))
            unit = next((r for r in w['robots'].values() if r['team'] == team and r['type'] == cls), None)
            if unit is None:
                continue  # 该类无单位 -> 保留 idle(合法)
            it = gm.derive_unit_intent(unit, w, mode)
            it['move'] = move
            it['fire'] = bool(fire)
            d['units'][cls] = it
        out.append(d)
    return out


def validate_series(decisions):
    for d in decisions:
        ok, errs = sl.validate(d)
        if not ok:
            return False, errs
    return True, []
