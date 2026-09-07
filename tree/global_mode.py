from schema.layered import GLOBAL_MODES, TARGETS, MOVES, RISKS

def _mid_norm(x):
    return x  # norm_xy 已在 [0,1]; 中线 = 0.5

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

def derive_unit_intent(unit, worldstate, global_mode):
    hp = unit.get('hp_ratio', 1.0)
    vuln = unit.get('vuln', 0)
    dfire = unit.get('dfire', 0.0)
    x = unit['norm_xy'][0] if unit.get('norm_xy') else 0.5
    # 目标
    if global_mode == '进攻' and x > 0.5:
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

_MODE_TO_SENTRY = {'防守': 'def', '均衡': 'cruise', '进攻': 'cruise'}

def derive_sentry_state(worldstate):
    """从世界状态派生一个**诚实**的哨兵三态(规则基线): map 全局模式, 不接触标注器真值。

    供 M1 --rule-baseline 作为不泄漏真值的下限; 也让 LLM-raw 得分有参照系。
    """
    return _MODE_TO_SENTRY.get(derive_global_mode(worldstate), 'cruise')
