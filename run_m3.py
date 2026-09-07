# -*- coding: utf-8 -*-
import os, argparse, json
import config
from data import db, features
from worldstate import worldstate as ws
from schema import layered as sl
from tree import global_mode as gm
from tree import behavior_prior as bp
from game import game_theory as gt
from fastloop import behavior as bh

def compute_decision_series(dps, bounds, opp_estimate=None):
    est = opp_estimate or [1/3, 1/3, 1/3]
    decisions, sigs = [], []
    for dp in dps:
        w = ws.build_worldstate(dp, bounds)
        mode = gt.choose_mode(gt.PAYOFF_3, est)      # 博弈论慢环: 选全局模式/意图
        d = sl.build_empty_decision(w['t'])
        d['global_mode'] = mode
        for rid, unit in w['robots'].items():
            if unit['team'] != '红' or unit['type'] not in d['units']:
                continue  # 只做己方(红)单位; 类型唯一, 不跨队冲突
            it = gm.derive_unit_intent(unit, w, mode)
            # 快环可行性校验 (spec §6.2): 哨兵血低/易伤 -> 禁进攻开火, 回退安全
            if unit['type'] == '哨兵' and it['fire'] and not bh.feasible(
                    'hitbuilding', {'robots': {7: unit}, 'bot_state': 'hitbuilding'}, bh.Rules()):
                it = {'target': it['target'], 'move': '回防', 'fire': False, 'risk': '高'}
            d['units'][unit['type']] = it
        # build_empty_decision 已预置六类闲置基线, 未出场类保持合法, 无需再补
        decisions.append(d)
        sigs.append(bp.state_sig(w, mode))
    # RULING 1 (singleton-wrap): bp.mine 期望 trajectories 为一系列序列(每个序列是一串 (sig,decision) 对),
    # 故包装成单条轨迹 [ [(sig, move), ...] ] —— 一局 = 一条 sig→move 流, 否则扁平 2 元组会解包报错。
    # 挖掘的是哨兵真实下游决策(单位 move), 而非把已在 sig 里的 mode 原样回显(避免 self-referential)。
    prior = bp.mine([[(sig, d['units']['哨兵']['move']) for sig, d in zip(sigs, decisions)]])
    return decisions, prior, sigs

def compliance_rate(decisions):
    if not decisions:
        return 0.0
    return sum(1 for d in decisions if sl.validate(d)[0]) / len(decisions)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--games', nargs='*', type=int, default=None)
    ap.add_argument('--out', default='out_m3')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    con = db.connect()
    gids = args.games or [config.GAME_REF]
    all_decisions, all_priors = [], []
    for gid in gids:
        rows = db.timeseries_for(con, gid)
        dps = features.build_decision_points(rows, gid)
        b = features.field_bounds(dps)
        decisions, prior, sigs = compute_decision_series(dps, b)
        all_decisions += decisions
        all_priors += prior
    con.close()
    mc = {}
    for d in all_decisions:
        mc[d['global_mode']] = mc.get(d['global_mode'], 0) + 1
    report = {
        'games': gids, 'n': len(all_decisions),
        'compliance_rate': compliance_rate(all_decisions),
        'conflict_rate': sl.conflict_rate(all_decisions),
        'mode_counts': mc,
        'prior': all_priors,
    }
    with open(os.path.join(args.out, 'report.json'), 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
