# -*- coding: utf-8 -*-
"""M2 置信融合 + 冲突标记: 规则头 / RL 头 / (可选)LLM 哨兵头 -> 合法分层决策。
每决策点、每单位类, 对 (move, fire) 做加权置信融合; 冲突时取最高置信源并标记 conflict。

交付主线: 规则保底 + 通道修复融合(CHANNEL_CONFS); LLM 默认不进融合(见 run_m2 --profile mainline)。
Track2 附录可用 mode_policy='fuse_mode' 让物理 RL 影响战略层。

旁路字段（敌方意图 probs / decision_meta tier）不在本模块写入；
由 run_m2.stage_fusion 与 finetune.make_dataset 在融合后挂载（见 labels.opp_intent / labels.router）。

global_mode:
  mode_policy='anchor_rule'(默认/对照): 锚定规则头 — 历史行为。
  mode_policy='fuse_mode'(Track2): 对 rule/rl/llm 的 mode 置信融合。
"""
from schema import layered as sl

# 三源置信。发现问题: 规则权重0.5>0.3>0.2且每通道在场 -> 规则恒胜, RL/LLM 发声为0;
# 且 LLM 与规则 96% 一致会"加强规则一票", 即使把 RL 权重抬高也会被 rule+llm 联盟平局压制。
DEFAULT_CONFS = {'rule': 0.5, 'rl': 0.3, 'llm': 0.2}

# 修复(Option A, 每通道建域专业化): 哨兵是 M1 慢环(规则+RL)专属通道 -> RL 在该通道有学习判断权;
# 且排空 LLM(llm=0, 它在此通道高度规则先导, 会与规则勾结压制 RL)。其余通道保留规则安全底座。
CHANNEL_CONFS = {
    '哨兵': {'rule': 0.4, 'rl': 0.6, 'llm': 0.0},
}

# global_mode 融合置信(与哨兵通道同向: 仿真附录允许 RL 影响战略层)
MODE_CONFS = {'rule': 0.4, 'rl': 0.6, 'llm': 0.0}


def _conf_selection(cls, confs):
    """显式传 confs(诊断/等权探针)则全局用; 否则走每通道默认。"""
    return confs if confs is not None else CHANNEL_CONFS.get(cls, DEFAULT_CONFS)


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


def mode_from_units(decision):
    """无显式 global_mode 时, 由单位 move 聚合启发式导出。

    轻量进攻偏向(消融 §7.12 / v3fix2): 推进门槛降至 1 且允许平票偏攻;
    防守需更强多数(≥3) — 缓解 Track2 free 轨近恒守。

    诚实 caveat: 这是防偏守/过硬门的启发式, **不得**单独当作「RL 战略优于规则」的证据;
    交付主线不依赖本函数过门结论。
    """
    units = (decision or {}).get('units') or {}
    moves = [u.get('move') for u in units.values() if u]
    n_push = sum(1 for m in moves if m == '推进')
    n_back = sum(1 for m in moves if m == '回防')
    if n_push >= 1 and n_push >= n_back:
        return '进攻'
    if n_back > n_push and n_back >= 3:
        return '防守'
    return '均衡'


def _source_mode(src):
    if not src:
        return None
    m = src.get('global_mode')
    if m in sl.GLOBAL_MODES:
        return m
    return mode_from_units(src)


def fuse_global_mode(src_rule, src_rl, src_llm=None, mode_confs=None):
    """对三源 global_mode 加权投票; 返回 (mode, conflict)。"""
    confs = mode_confs or MODE_CONFS
    vals = []
    for name, src in (('rule', src_rule), ('rl', src_rl), ('llm', src_llm)):
        m = _source_mode(src)
        if m is not None:
            vals.append((name, m))
    if not vals:
        return '均衡', False
    weighted = {}
    for src, m in vals:
        weighted[m] = weighted.get(m, 0.0) + confs.get(src, 0.0)
    top = max(weighted, key=weighted.get)
    conflict = len({m for _s, m in vals}) > 1
    return top, conflict


def fuse(rule, rl, llm=None, confs=None, mode_policy='anchor_rule', mode_confs=None):
    """rule/rl: list[dict] 分层决策(长度一致); llm: list[dict] 同一粒度(可 None)。
    mode_policy: 'anchor_rule' | 'fuse_mode'。
    返回 {'decisions': fused, 'conflict_rate': float, 'head_agreement': {...}}。
    注意: confs=None 必须透传给 _conf_selection 逐通道解析, 不能在此兜底成 DEFAULT_CONFS,
    否则 CHANNEL_CONFS 的逐通道覆盖永远不生效(rl 在哨兵上被 rule+LLM 联盟压制)。"""
    n = min(len(rule), len(rl)) if rl else len(rule)
    out = []
    overn = float(len(rule)) if rule else 0.0
    conflicts = 0
    agree_rl = agree_llm = 0
    for i in range(n):
        # 起底用规则头(若有), 否则 empty
        src_rule = rule[i] if i < len(rule) else None
        src_rl = rl[i] if (rl and i < len(rl)) else None
        src_llm = llm[i] if (llm and i < len(llm)) else None
        d = sl.build_empty_decision(src_rule['t'] if src_rule else (src_rl['t'] if src_rl else 0.0))
        if mode_policy == 'fuse_mode':
            d['global_mode'], _mc = fuse_global_mode(src_rule, src_rl, src_llm, mode_confs)
        else:
            # anchor_rule(默认对照): 战略层锁死规则头
            d['global_mode'] = src_rule.get('global_mode', '均衡') if src_rule else '均衡'
        pt_conf = False
        for cls in sl.UNIT_CLASSES:
            rv = src_rule['units'].get(cls) if src_rule else None
            lv = src_rl['units'].get(cls) if src_rl else None
            xv = src_llm['units'].get(cls) if src_llm else None
            mv_fire = []
            if rv: mv_fire.append(('rule', (rv['move'], rv['fire'])))
            if lv: mv_fire.append(('rl', (lv['move'], lv['fire'])))
            if xv: mv_fire.append(('llm', (xv['move'], xv['fire'])))
            (mv, fire), conflict = fuse_channel(mv_fire, _conf_selection(cls, confs))
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
                               'rule_llm': (agree_llm / (n * len(sl.UNIT_CLASSES))) if n else 0.0},
            'mode_policy': mode_policy}
