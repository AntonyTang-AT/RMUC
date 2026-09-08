# -*- coding: utf-8 -*-
"""M2 Track 2: Plan B RL 源重跑 -> 重算融合 -> fusion-level 评估(留出局)。

回答(轨2 设计 §1/§3): "Plan B 物理 RL 源是否真学得动、且让融合结果更好"。
对每个**留出局**(m2_report.json 的 eval_games)产三套决策做对照:
  rule-only               : fuse(rule, rule, None)   —— 确定性安全底座
  rl-physics(Plan B 对称) : fuse(rule, rl_physics)   —— IPPO 在 PhysicsEnv 上训 + rollout
  rl-bandit(旧模式参照)   : fuse(rule, rl_bandit)    —— IPPO 在 M2Env(replay-bandit)上训

判别口径(★ fuse 把 global_mode 锚定为 rule, 见 fusion_eval 定锚):
  RL 源自身单位级质量 rl_source_metrics(攻击倾向 fav-align / 多样性 / 合法性 / 与规则分歧),
  以及 physics-vs-bandit 的 compare_rl_sources delta。fusion_overview 仅列参考(全局退化/冲突)。
承重判据(设计 §3):  physics 的 fav_align ≥ bandit 的 fav_align(RL 真的学得动), 且都合法、非退化。

GPU-free(RL 头 CPU-torch);不训/不评 LLM(那是 Track 1)。只读 data.db;产物不 commit。

用法:
  python run_track2.py --rl-updates 80                                   # 全量(留出4局)
  python run_track2.py --limit 12 --rl-updates 40                        # 小样本冒烟(先看方向)
  python run_track2.py --out out_m2/track2_rl_fusion_report.json
"""
import os, sys, json, argparse

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)
os.chdir(REPO)

from data import db
import run_m2
from labels import fusion as fu
from worldstate import worldstate as ws
from eval import fusion_eval as fev

# 保留的最小合法率/退化底线(报告里标 legal_pass/degenerate, 不做硬门控)
_LEGAL_FLOOR = 0.9


def load_plan(m2_report):
    """读 m2_report.json -> {'eval_games', 'dataset_games', ...}。"""
    with open(m2_report, encoding='utf-8') as f:
        return json.load(f)


def build_train_ids(con, n):
    return run_m2.list_game_ids(con)[0:n]


def process_game(gid, train_ids, rl_updates, limit, rl_seed=0):
    """单留出局: 产 rule/rl_physics/rl_bandit 三头 + 融合 + 全指标。失败 -> None。"""
    dps, b = run_m2.load_decision_points(gid, limit)
    if not dps:
        print('game', gid, 'no decision points', flush=True)
        return None
    rule = run_m2.rule_head(dps, b)
    ws_list = [ws.build_worldstate(dp, b) for dp in dps]
    # RL 头: 与训练同窗(train_ids)一致, 只换 physics 模式; 策略按 cache 只训一次, 后续局复用
    rl_phys = run_m2.rl_head(dps, b, train_ids, rl_updates, rl_seed=rl_seed, physics=True)
    rl_band = run_m2.rl_head(dps, b, train_ids, rl_updates, rl_seed=rl_seed, physics=False)
    # 融合(global_mode 恒 rule): 看把 RL 源接进去后合法性/退化/冲突
    fused_rule = fu.fuse(rule, rule, None)
    fused_phys = fu.fuse(rule, rl_phys, None)
    # 判别: RL 源自身质量
    m_phys = fev.rl_source_metrics(rl_phys, rule, ws_list, '红')
    m_band = fev.rl_source_metrics(rl_band, rule, ws_list, '红')
    cmp_ = fev.compare_rl_sources(m_phys, m_band)
    ov_rule = fev.fusion_overview(fused_rule, ws_list, '红')
    ov_phys = fev.fusion_overview(fused_phys, ws_list, '红')
    return {'game_id': gid,
            'n': len(dps),
            'rl_source_physics': m_phys,
            'rl_source_bandit': m_band,
            'compare': cmp_,
            'fusion_rule_only': ov_rule,
            'fusion_rl_physics': ov_phys,
            'rl_physics_legal_pass': m_phys['legal_rate'] >= _LEGAL_FLOOR}


def aggregate(rows):
    """跨局聚合 key 指标: fav_align 口径 + 退化 + 合法 + 承重判据达标。"""
    if not rows:
        return {}
    def _avg_nones(values):
        got = [v for v in values if v is not None]
        return round(sum(got) / len(got), 4) if got else None

    ph = [r['rl_source_physics'] for r in rows]
    ba = [r['rl_source_bandit'] for r in rows]
    fav_align_p = _avg_nones([x['fav_align'] for x in ph])
    fav_align_b = _avg_nones([x['fav_align'] for x in ba])
    ge = [c['phys_ge_bandit_fav_align'] for c in
          (r['compare'] for r in rows) if c['phys_ge_bandit_fav_align'] is not None]
    all_legal = all(r['rl_physics_legal_pass'] for r in rows)
    all_non_degen = all(not r['fusion_rl_physics']['degenerate'] for r in rows)
    return {'n_games': len(rows),
            'avg_fav_align_physics': fav_align_p,
            'avg_fav_align_bandit': fav_align_b,
            'fav_align_delta_phys_minus_bandit':
                (round(fav_align_p - fav_align_b, 4)
                 if (fav_align_p is not None and fav_align_b is not None) else None),
            'n_phys_ge_bandit_fav_align': (sum(1 for x in ge if x) if ge else None),
            'n_phys_ge_bandit_fav_align_base': len(ge),
            'all_rl_physics_legal': all_legal,
            'all_fusion_non_degenerate': all_non_degen,
            # 承重判据: 至少半数局 physics 的 RL 源对齐更优 且 全局合法 + 非退化
            'loaded_criterion_met': bool(
                ge and (sum(1 for x in ge if x) >= max(1, len(ge) // 2))
                and all_legal and all_non_degen)}


def main():
    ap = argparse.ArgumentParser(description='Track 2: Plan B RL 源重跑 + 融合重算 + fusion-level 评估')
    ap.add_argument('--m2-report', default='out_m2/m2_report.json', help='含 eval_games/dataset_games')
    ap.add_argument('--out', default='out_m2/track2_rl_fusion_report.json')
    ap.add_argument('--train-games', type=int, default=3, help='RL 训练窗(与 run_m2 约定: 前 N 局)')
    ap.add_argument('--rl-updates', type=int, default=80, help='IPPO 每局更新轮数(长跑口径)')
    ap.add_argument('--limit', type=int, default=0, help='每局前 N 决策点(0=全部; 冒烟用小)')
    ap.add_argument('--rl-seed', type=int, default=0)
    ap.add_argument('--hardgate', action='store_true',
                    help='承重判据未达标则返回码 2(供脚本硬判; 默认只记不 halt)')
    args = ap.parse_args()

    plan = load_plan(args.m2_report)
    eval_games = plan.get('eval_games', [])
    if not eval_games:
        print('no eval_games in m2 report — abort')
        return 1

    con = db.connect()
    train_ids = build_train_ids(con, args.train_games)
    con.close()
    if not train_ids:
        print('no train games — abort')
        return 1
    print('train_window:', train_ids, '| eval_games:', eval_games, flush=True)

    rows = []
    for gid in eval_games:
        row = process_game(gid, train_ids, args.rl_updates, args.limit, args.rl_seed)
        if row:
            rows.append(row)
            print(json.dumps({k: v for k, v in row.items() if k != 'game_id'},
                             ensure_ascii=False), flush=True)

    agg = aggregate(rows)
    report = {'pipeline': 'track2_rl_fusion', 'train_window': train_ids,
              'eval_games': eval_games, 'rl_updates': args.rl_updates,
              'n_eval_games': len(rows), 'rows': rows, 'aggregate': agg,
              'caveat': ('global_mode 在融合里锚定为 rule(fuse 行52), RL 源只改单位级 move/fire; '
                         '故判别口径用 RL 源自身单位级 fav-align(攻击倾向随红方胜势), 非 global_mode。'
                         'fav_align 是关联性/条件有利度, 非因果; src 仅 4 留出局, 以方向与量级为准。')}
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps({'type': 'SAVE', 'report': args.out, 'aggregate': agg}, ensure_ascii=False), flush=True)
    if args.hardgate and not agg.get('loaded_criterion_met', True):
        print('HARDGATE: 承重判据未达标 (loaded_criterion_met=False)', flush=True)
        return 2
    return 0


if __name__ == '__main__':
    sys.exit(main())
