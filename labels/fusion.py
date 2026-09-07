# -*- coding: utf-8 -*-
"""M2 三源置信融合 + 冲突标记: 规则头 / RL 头 / LLM 哨兵头 -> 一个合法分层决策。
每决策点、每单位类, 对 (move, fire) 做加权置信融合; 冲突时取最高置信源并标记 conflict。"""
from schema import layered as sl

DEFAULT_CONFS = {'rule': 0.5, 'rl': 0.3, 'llm': 0.2}


def fuse_channel(vals, confs):
    """vals: [(source_name, (move, fire))]。按 conf 加权; 返回 (top_value, conflict_bool)。"""
    weighted = {}
    for src, val in vals:
        if val is None:
            continue
        w = confs.get(src, 0.0)
        weighted[val] = weighted.get(val, 0.0) + w
    if not weighted:
        return None, False
    top = max(weighted, key=weighted.get)
    distinct = {v for _s, v in vals if v is not None}
    conflict = len(distinct) > 1  # 各源确有分歧 -> 标记冲突(取最高置信源)
    return top, conflict


def fuse(rule, rl, llm=None, confs=None):
    """rule/rl: list[dict] 分层决策(长度一致); llm: list[dict] 同一粒度(可 None)。
    返回 {'decisions': fused, 'conflict_rate': float, 'head_agreement': {...}}。"""
    confs = confs or DEFAULT_CONFS
    n = min(len(rule), len(rl)) if rl else len(rule)
    out = []
    overn = float(len(rule)) if rule else 0.0
    conflicts = 0
    agree_rl = agree_llm = 0
    for i in range(n):
        # 起底用规则头(若有), 否则 empty
        src_rule = rule[i] if i < len(rule) else None
        d = sl.build_empty_decision(src_rule['t'] if src_rule else (rl[i]['t'] if rl else 0.0))
        d['global_mode'] = src_rule.get('global_mode', '均衡') if src_rule else '均衡'
        pt_conf = False
        for cls in sl.UNIT_CLASSES:
            rv = src_rule['units'].get(cls) if src_rule else None
            lv = rl[i]['units'].get(cls) if (rl and i < len(rl)) else None
            xv = llm[i]['units'].get(cls) if (llm and i < len(llm)) else None
            mv_fire = []
            if rv: mv_fire.append(('rule', (rv['move'], rv['fire'])))
            if lv: mv_fire.append(('rl', (lv['move'], lv['fire'])))
            if xv: mv_fire.append(('llm', (xv['move'], xv['fire'])))
            (mv, fire), conflict = fuse_channel(mv_fire, confs)
            if mv is None:
                mv, fire = ('无', False)
            it = dict(rv) if rv else {'target': '无', 'move': mv, 'fire': fire, 'risk': '低'}
            it['move'] = mv
            it['fire'] = fire
            d['units'][cls] = it
            if rv and lv and rv['move'] == lv['move']:
                agree_rl += 1
            if rv and xv and rv['move'] == xv['move']:
                agree_llm += 1
            pt_conf = pt_conf or conflict
        out.append(d)
        if pt_conf:
            conflicts += 1
    return {'decisions': out,
            'conflict_rate': (conflicts / overn) if overn else 0.0,
            'head_agreement': {'rule_rl': (agree_rl / (n * len(sl.UNIT_CLASSES))) if n else 0.0,
                               'rule_llm': (agree_llm / (n * len(sl.UNIT_CLASSES))) if n else 0.0}}
