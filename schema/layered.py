# schema/layered.py
UNIT_CLASSES = ('英雄', '工程', '步兵3', '步兵4', '空中', '哨兵')
GLOBAL_MODES = ('进攻', '均衡', '防守')
TARGETS = ('敌方哨兵', '敌方英雄', '敌方步兵', '敌方工程', '敌方前哨站', '敌方基地', '能量机关', '飞镖', '无')
MOVES = ('推进', '回防', '游击', '坚守', '绕行', '无')
RISKS = ('高', '中', '低')

def build_empty_decision(t):
    # 每类预置保守闲置决策, 保证"每台机器人"六类齐全的合法基线 (合规率 100%); 运行时按需覆盖
    idle = {'target': '无', 'move': '无', 'fire': False, 'risk': '低'}
    return {'t': t, 'global_mode': '均衡', 'units': {c: dict(idle) for c in UNIT_CLASSES}}

def _channels(entry):
    if not isinstance(entry, dict):
        return None
    return ('target', 'move', 'fire', 'risk')

def validate(decision):
    errs = []
    if not isinstance(decision, dict):
        return False, ['decision 不是 dict']
    if 't' not in decision:
        errs.append('missing t')
    if decision.get('global_mode') not in GLOBAL_MODES:
        errs.append('global_mode 非法')
    units = decision.get('units')
    if not isinstance(units, dict):
        errs.append('units 不是 dict')
        return False, errs
    for c in UNIT_CLASSES:
        if c not in units:
            errs.append(f'缺少单位类 {c}')
            continue
        e = units[c]
        if not isinstance(e, dict):
            errs.append(f'{c} 不是 dict')
            continue
        if e.get('target') not in TARGETS:
            errs.append(f'{c}.target 非法')
        if e.get('move') not in MOVES:
            errs.append(f'{c}.move 非法')
        if not isinstance(e.get('fire'), bool):
            errs.append(f'{c}.fire 非 bool')
        if e.get('risk') not in RISKS:
            errs.append(f'{c}.risk 非法')
    return (errs == []), errs

def conflict_rate(decisions):
    if not decisions or len(decisions) < 2:
        return 0.0
    conflicts = 0
    for a, b in zip(decisions, decisions[1:]):
        if a.get('global_mode') != b.get('global_mode'):
            conflicts += 1
        else:
            # compare unit intent flips (target) as a finer-grained conflict
            ua, ub = a.get('units') or {}, b.get('units') or {}
            if any((ua.get(c) or {}).get('target') != (ub.get(c) or {}).get('target')
                   for c in UNIT_CLASSES):
                conflicts += 1
    return conflicts / (len(decisions) - 1)
