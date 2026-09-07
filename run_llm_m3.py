# -*- coding: utf-8 -*-
"""M3 分层决策的 LLM 对齐评估: 把 DeepSeek 慢环的哨兵三态映射成分层 4 通道决策,
与规则派生(局势感知)序列对齐度量。GPU-free; 仅 --llm 分支真正调 API。

用法:
  python run_llm_m3.py                    # 规则派生(不调 API)
  python run_llm_m3.py --llm --limit 20   # LLM 慢环 (DeepSeek, 读 .env)
"""
import os, json, argparse
import config
from data import db, features
from worldstate import worldstate as ws
from schema import layered as sl
from tree import global_mode as gm
from fastloop import behavior as bh
from eval import metrics

# 哨兵三态 -> 4 通道(合法值域见 schema.layered)
_INTENT = {
    'def':         {'target': '敌方步兵', 'move': '回防', 'fire': False, 'risk': '高'},
    'hitbuilding': {'target': '敌方前哨站', 'move': '推进', 'fire': True,  'risk': '中'},
    'cruise':      {'target': '无',        'move': '游击', 'fire': False, 'risk': '低'},
}


def llm_state_to_intent(state, sent=None):
    """把 LLM 哨兵状态映射为合法 4 通道; 未知状态回退 cruise。"""
    return dict(_INTENT.get(state, _INTENT['cruise']))


def _build_rule_decision(w):
    """规则派生: 用局势感知全局模式 + 单位意图生成一个合法分层决策(核心, 供兜底与参照)。"""
    mode = gm.derive_global_mode(w)
    d = sl.build_empty_decision(w['t'])
    d['global_mode'] = mode
    for rid, unit in w['robots'].items():
        if unit['team'] != '红' or unit['type'] not in d['units']:
            continue
        if unit['type'] == '哨兵' and gm.derive_unit_intent(unit, w, mode).get('fire') \
                and not bh.feasible('hitbuilding', {'robots': {7: unit}, 'bot_state': 'hitbuilding'}, bh.Rules()):
            d['units'][unit['type']] = {'target': '敌方步兵', 'move': '回防', 'fire': False, 'risk': '高'}
        else:
            d['units'][unit['type']] = gm.derive_unit_intent(unit, w, mode)
    return d


def _as_decision(w, llm_state):
    """LLM 哨兵状态 -> 合法分层决策(其余类用规则派生兜底以保 schema 完整)。"""
    d = _build_rule_decision(w)
    sent = next((u for u in w['robots'].values() if u['team'] == '红' and u['type'] == '哨兵'), None)
    if sent is not None:
        d['units']['哨兵'] = llm_state_to_intent(llm_state, sent)
    return d


def build_rule_decisions(dps, bounds):
    return [_build_rule_decision(ws.build_worldstate(dp, bounds)) for dp in dps]


def build_llm_decisions(dps, bounds, llm_state_fn, *, api_key=None, base_url=None, model=None):
    """对每个决策点: 先出规则决策, 再由 llm_state_fn 的哨兵状态覆盖哨兵通道。"""
    out = []
    for dp in dps:
        w = ws.build_worldstate(dp, bounds)
        state, _raw = llm_state_fn(w, api_key, base_url, model)
        out.append(_as_decision(w, state))
    return out


def assess(decisions_llm, decisions_rule):
    """LLM vs 规则 对齐评估: 哨兵通道合法性、与规则一致性、逐类、冲突率。"""
    n = len(decisions_llm)
    if n == 0:
        return {'n': 0}
    llm_sentry = [d['units']['哨兵']['move'] for d in decisions_llm]
    rule_sentry = [d['units']['哨兵']['move'] for d in decisions_rule]
    valid = sum(1 for d in decisions_llm if sl.validate(d)[0])
    return {
        'n': n,
        'llm_sentry_valid': valid / n,
        'compliance_rate': valid / n,
        'agreement': sum(1 for a, b in zip(llm_sentry, rule_sentry) if a == b) / n,
        'per_class': metrics.per_class(rule_sentry, llm_sentry,
                                       labels=sorted(set(rule_sentry) | set(llm_sentry))),
        'conflict_rate': sl.conflict_rate(decisions_llm),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--games', nargs='*', type=int, default=None)
    ap.add_argument('--llm', action='store_true', help='用 DeepSeek 慢环(默认仅规则派生)')
    ap.add_argument('--limit', type=int, default=20, help='LLM 分支跑多少个决策点(0=全部)')
    ap.add_argument('--out', default='out_m3')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    con = db.connect()
    gids = args.games or [config.GAME_REF]

    all_llm, all_rule = [], []
    for gid in gids:
        rows = db.timeseries_for(con, gid)
        dps = features.build_decision_points(rows, gid)
        b = features.field_bounds(dps)
        if args.limit > 0 and args.llm:
            dps = dps[:args.limit]
        all_rule += build_rule_decisions(dps, b)
        if args.llm:
            from run_llm_api import _load_env, llm_state
            cfg = _load_env()
            api_key = cfg['DEEPSEEK_API_KEY']
            base_url, model = cfg['DEEPSEEK_BASE_URL'], cfg['DEEPSEEK_MODEL']
            all_llm += build_llm_decisions(dps, b, llm_state, api_key=api_key, base_url=base_url, model=model)
        else:
            all_llm += build_rule_decisions(dps, b)  # 参照: 自洽, agreement 应=1.0
    con.close()

    report = assess(all_llm, all_rule)
    report['mode'] = 'llm_slowloop' if args.llm else 'rule_reference'
    with open(os.path.join(args.out, 'llm_m3_report.json'), 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
