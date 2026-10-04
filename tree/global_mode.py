from schema.layered import GLOBAL_MODES, TARGETS, MOVES, RISKS

SENTRY_STATES = ('def', 'hitbuilding', 'cruise')
STRATEGIC_INTENTS = (
    'StrategicDefendBase',
    'StrategicAttackOutpost',
    'StrategicPatrol',
)
_SENTRY_TO_STRATEGIC = {
    'def': 'StrategicDefendBase',
    'hitbuilding': 'StrategicAttackOutpost',
    'cruise': 'StrategicPatrol',
}
_STRATEGIC_TO_SENTRY = {v: k for k, v in _SENTRY_TO_STRATEGIC.items()}


def _mid_norm(x):
    return x  # norm_xy 已在 [0,1]; 中线 = 0.5


def _in_enemy_half(nx, team='红'):
    """红方敌半场 nx>0.5；蓝方敌半场 nx<0.5。"""
    if team == '红':
        return float(nx) > 0.5
    return float(nx) < 0.5


def _in_our_half(nx, team='红'):
    return not _in_enemy_half(nx, team=team)


def derive_global_mode(worldstate):
    robots = [r for r in worldstate['robots'].values() if r not in (None,)]
    if not robots:
        return '均衡'
    reds = [r for r in robots if r['team'] == '红']
    blues = [r for r in robots if r['team'] == '蓝']
    red_push = sum(1 for r in reds if r['norm_xy'][0] > 0.5)
    blue_push = sum(1 for r in blues if r['norm_xy'][0] < 0.5)
    red_hp = sum(r['hp_ratio'] for r in reds) / len(reds) if reds else 0.0
    blue_hp = sum(r['hp_ratio'] for r in blues) / len(blues) if blues else 0.0
    # 基地血量低 -> 防守倾向
    base_threat = any(b.get('hp_ratio', 1.0) < 0.5 for b in worldstate['buildings'].values() if b.get('team') == '红')
    push_adv = red_push - blue_push
    if push_adv >= 2 and blue_hp < 0.9 and not base_threat:
        return '进攻'
    if push_adv <= -2 or base_threat:
        return '防守'
    return '均衡'

def derive_unit_intent(unit, worldstate, global_mode, team='红'):
    hp = unit.get('hp_ratio', 1.0)
    vuln = unit.get('vuln', 0)
    dfire = unit.get('dfire', 0.0)
    x = unit['norm_xy'][0] if unit.get('norm_xy') else 0.5
    # 目标
    # ``norm_xy`` is rendered from the red-side coordinate system.  The
    # original rule was intentionally red-only; accepting ``team`` keeps its
    # old behaviour by default while making mirrored blue-side teachers and
    # runtime advice point toward the actual enemy half.
    if global_mode == '进攻' and _in_enemy_half(x, team=team):
        target = '敌方前哨站'
    elif global_mode == '防守':
        target = '敌方步兵'
    elif dfire > 0:
        target = '敌方英雄'
    else:
        target = '无'
    # 移动
    move = {'进攻': '推进', '防守': '回防', '均衡': '游击'}.get(global_mode, '无')
    if hp < 0.3 or vuln:
        move = '坚守'
    # 开火
    fire = bool(dfire > 0 or target.startswith('敌方'))
    # 风险
    risk = '高' if (hp < 0.3 or vuln) else ('中' if hp < 0.6 else '低')
    return {'target': target, 'move': move, 'fire': fire, 'risk': risk}


def sentry_to_strategic(state):
    """考核原图：雷达意图 ↔ 状态机三态。"""
    return _SENTRY_TO_STRATEGIC.get(state, 'StrategicPatrol')


def strategic_to_sentry(intent):
    return _STRATEGIC_TO_SENTRY.get(intent, 'cruise')


def derive_sentry_state(worldstate, team='红'):
    """瞬时几何规则派生哨兵三态 def/hitbuilding/cruise（诚实规则基线）。

    优先级与 strategy_label 窗口一致：敌进己半场 → def；否则哨兵越线+开火 → hitbuilding；
    否则 cruise。不读未来窗、不接触标注器真值。
    """
    robots = list((worldstate or {}).get('robots', {}).values())
    opp = '蓝' if team == '红' else '红'
    # 1) 敌方单位进入己方半场
    for r in robots:
        if r.get('team') != opp:
            continue
        nx = (r.get('norm_xy') or [0.5, 0.5])[0]
        if _in_our_half(nx, team=team):
            return 'def'
    # 2) 己方哨兵在敌半场且开火
    for r in robots:
        if r.get('team') != team or r.get('type') != '哨兵':
            continue
        nx = (r.get('norm_xy') or [0.5, 0.5])[0]
        if _in_enemy_half(nx, team=team) and float(r.get('dfire') or 0) > 0:
            return 'hitbuilding'
    return 'cruise'
