# -*- coding: utf-8 -*-
"""M2 Track 2(研究附录): Plan B 物理 RL -> 双轨融合评估(留出局)。

定位: **仿真/增值研究附录**, 非考核交付主线。交付主线见 run_m2 --profile mainline
(规则保底 + lookahead QLoRA)。hardgate 过关 ≠「RL 已优于规则」——须带符号物理 caveat。

仿真口径(v3fix2): PhysicsEnv MDP + 差分奖励 + fav对齐 bonus(0.02) + RL mode=from_units
(含进攻偏向启发式, 非优势证明)。
融合双轨: free(mode 可融合) + anchored(mode=rule, 对照)。
Bandit: 仅对照, 不进硬承重。

硬承重(v3fix2-rel-share): Physics 半数非常数 + legal + fav_align>0 半数
  + free.attack_balance_share>=rule(rule 自身退化局跳过; 否则绝对 degen)。

GPU-free(RL 头 CPU-torch);不训/不评 LLM。只读 data.db;产物不 commit。

用法:
  python run_track2.py --limit 12 --rl-updates 80 --out out_m2/track2_rl_fusion_smoke_v3fix2.json
  python run_track2.py --rl-updates 200 --hardgate --out out_m2/track2_rl_fusion_hardgate_v3fix2.json
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

_LEGAL_FLOOR = 0.9
_VIS_RADIUS = 8.0
_FAV_ALIGN_SCALE = 0.02  # 0.08 曾压回常数策略; 保持轻量对齐
_RL_MODE_SOURCE = 'from_units'


def load_plan(m2_report):
    with open(m2_report, encoding='utf-8') as f:
        return json.load(f)


def build_train_ids(con, n):
    return run_m2.list_game_ids(con)[0:n]


def _count_mode_diff(a_decs, b_decs):
    n = min(len(a_decs), len(b_decs))
    return sum(1 for i in range(n)
               if (a_decs[i] or {}).get('global_mode') != (b_decs[i] or {}).get('global_mode'))


def process_game(gid, train_ids, rl_updates, limit, rl_seed=0,
                 coherence_scale=0.25, ent_coef=0.02, reward_delta=True,
                 mode_policy='both', mode_source=_RL_MODE_SOURCE,
                 fav_align_scale=_FAV_ALIGN_SCALE):
    """单留出局: rule / rl_physics / rl_bandit + 双轨融合 + 全指标。"""
    dps, b = run_m2.load_decision_points(gid, limit)
    if not dps:
        print('game', gid, 'no decision points', flush=True)
        return None
    rule = run_m2.rule_head(dps, b)
    ws_list = [ws.build_worldstate(dp, b) for dp in dps]
    kw = dict(rl_seed=rl_seed, coherence_scale=coherence_scale, ent_coef=ent_coef,
              reward_delta=reward_delta, mode_source=mode_source,
              fav_align_scale=fav_align_scale)
    rl_phys = run_m2.rl_head(dps, b, train_ids, rl_updates, physics=True, **kw)
    rl_band = run_m2.rl_head(dps, b, train_ids, rl_updates, physics=False, **kw)

    fused_rule = fu.fuse(rule, rule, None, mode_policy='anchor_rule')
    do_anchor = mode_policy in ('both', 'anchor')
    do_free = mode_policy in ('both', 'free')
    fused_anchor = (fu.fuse(rule, rl_phys, None, mode_policy='anchor_rule')
                    if do_anchor else None)
    fused_free = (fu.fuse(rule, rl_phys, None, mode_policy='fuse_mode')
                  if do_free else None)

    m_phys = fev.rl_source_metrics(rl_phys, rule, ws_list, '红')
    m_band = fev.rl_source_metrics(rl_band, rule, ws_list, '红')
    cmp_ = fev.compare_rl_sources(m_phys, m_band)
    ov_rule = fev.fusion_overview(fused_rule, ws_list, '红')
    ov_anchor = (fev.fusion_overview(fused_anchor, ws_list, '红')
                 if fused_anchor is not None else None)
    ov_free = (fev.fusion_overview(fused_free, ws_list, '红')
               if fused_free is not None else None)
    ov_primary = ov_free if ov_free is not None else ov_anchor

    n_free_diff = 0
    if fused_free is not None:
        n_free_diff = _count_mode_diff(fused_free['decisions'], fused_rule['decisions'])

    return {'game_id': gid,
            'n': len(dps),
            'rl_source_physics': m_phys,
            'rl_source_bandit': m_band,
            'compare': cmp_,
            'fusion_rule_only': ov_rule,
            'fusion_rl_physics_anchored': ov_anchor,
            'fusion_rl_physics_free': ov_free,
            'fusion_rl_physics': ov_primary,
            'n_free_mode_diff_vs_rule': n_free_diff,
            'rl_physics_legal_pass': m_phys['legal_rate'] >= _LEGAL_FLOOR,
            'phys_fav_align_positive': (
                m_phys.get('fav_align') is not None and m_phys['fav_align'] > 0)}


def aggregate(rows):
    """跨局聚合。承重 v3-sim-rel-degen:
    Physics 半数非常数 + 合法 + fav_align>0 半数 + free 相对非退化
    (rule 自身退化的局跳过, 不拖垮硬门)。
    Bandit / phys>bandit 仅对照字段, 不进硬门。"""
    if not rows:
        return {}

    def _avg_nones(values):
        got = [v for v in values if v is not None]
        return round(sum(got) / len(got), 4) if got else None

    n = len(rows)
    half = max(1, n // 2)
    ph = [r['rl_source_physics'] for r in rows]
    ba = [r['rl_source_bandit'] for r in rows]
    fav_align_p = _avg_nones([x['fav_align'] for x in ph])
    fav_align_b = _avg_nones([x['fav_align'] for x in ba])
    comps = [r['compare'] for r in rows]
    ge = [c['phys_ge_bandit_fav_align'] for c in comps
          if c.get('phys_ge_bandit_fav_align') is not None]
    n_ge = sum(1 for x in ge if x) if ge else 0

    n_phys_const = sum(1 for x in ph if x.get('policy_constant'))
    n_band_const = sum(1 for x in ba if x.get('policy_constant'))
    n_phys_non_const = n - n_phys_const
    n_fav_pos = sum(1 for r in rows if r.get('phys_fav_align_positive'))
    all_legal = all(r['rl_physics_legal_pass'] for r in rows)

    rule_degen_ids = []
    for r in rows:
        ov = r.get('fusion_rule_only') or {}
        if ov.get('degenerate'):
            rule_degen_ids.append(r.get('game_id'))

    def _rel_non_degen(fuse_key):
        """相对非退化(v3fix2): 跳过 rule 退化局; free 攻+均份额不低于 rule 则过;
        否则回退到绝对 degenerate。无 fuse 轨 -> None。"""
        for r in rows:
            ov_f = r.get(fuse_key)
            if ov_f is None:
                return None
            ov_r = r.get('fusion_rule_only') or {}
            if ov_r.get('degenerate'):
                continue
            f_share = ov_f.get('attack_balance_share')
            r_share = ov_r.get('attack_balance_share')
            if (f_share is not None and r_share is not None
                    and f_share + 1e-9 >= r_share):
                continue  # 相对 rule 攻+均份额不下降
            if ov_f.get('degenerate'):
                return False
        # 全部 rule 都退化时: 无可检局, 视为通过(不拖垮)
        return True

    free_non_degen = _rel_non_degen('fusion_rl_physics_free')
    # 锚定轨仍报绝对非退化(对照)
    def _all_non_degen(key):
        vals = [r.get(key) for r in rows]
        if any(v is None for v in vals):
            return None
        return all(not v['degenerate'] for v in vals)

    anchor_non_degen = _all_non_degen('fusion_rl_physics_anchored')
    # 绝对 free(兼容旧字段观察)
    free_abs = _all_non_degen('fusion_rl_physics_free')
    primary_non_degen = (free_non_degen if free_non_degen is not None
                         else _rel_non_degen('fusion_rl_physics'))

    phys_non_const_pass = n_phys_non_const >= half
    fav_pos_pass = n_fav_pos >= half
    loaded = bool(phys_non_const_pass and all_legal and fav_pos_pass
                  and primary_non_degen is True)

    sim_gain = None
    if free_abs is not None and anchor_non_degen is not None:
        sim_gain = bool(free_abs and (not anchor_non_degen))

    n_free_diff = sum(int(r.get('n_free_mode_diff_vs_rule') or 0) for r in rows)

    return {
        'n_games': n,
        'avg_fav_align_physics': fav_align_p,
        'avg_fav_align_bandit': fav_align_b,
        'fav_align_delta_phys_minus_bandit':
            (round(fav_align_p - fav_align_b, 4)
             if (fav_align_p is not None and fav_align_b is not None) else None),
        'n_phys_ge_bandit_fav_align': (n_ge if ge else None),
        'n_phys_ge_bandit_fav_align_base': len(ge),
        'n_both_non_constant': sum(1 for c in comps if c.get('both_non_constant')),
        'n_physics_policy_constant': n_phys_const,
        'n_bandit_policy_constant': n_band_const,
        'n_physics_non_constant': n_phys_non_const,
        'n_phys_fav_align_positive': n_fav_pos,
        'avg_mean_move_diversity_physics': _avg_nones(
            [x.get('mean_move_diversity') for x in ph]),
        'avg_mean_move_diversity_bandit': _avg_nones(
            [x.get('mean_move_diversity') for x in ba]),
        'all_rl_physics_legal': all_legal,
        'all_fusion_non_degenerate': primary_non_degen,
        'all_fusion_non_degenerate_free': free_non_degen,  # 相对
        'all_fusion_non_degenerate_free_absolute': free_abs,
        'all_fusion_non_degenerate_anchored': anchor_non_degen,
        'sim_gain_free_fixes_anchor_degen': sim_gain,
        'n_free_mode_diff_vs_rule': n_free_diff,
        'rule_degenerate_games': rule_degen_ids,
        'loaded_criterion_met': loaded,
        'criterion_note': (
            'v3fix2-rel-share: phys non-constant majority + legal + fav_align>0 majority + '
            'free attack_balance_share >= rule (skip rule-degenerate games; else abs degen); '
            'bandit/phys>bandit are control-only'),
    }


def main():
    ap = argparse.ArgumentParser(description='Track 2 v3fix: Physics MDP + dual-fuse eval')
    ap.add_argument('--m2-report', default='out_m2/m2_report.json')
    ap.add_argument('--out', default='out_m2/track2_rl_fusion_report_v3fix.json')
    ap.add_argument('--train-games', type=int, default=3)
    ap.add_argument('--rl-updates', type=int, default=200)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--rl-seed', type=int, default=0)
    ap.add_argument('--mode-policy', default='both', choices=('both', 'anchor', 'free'),
                    help='融合轨: both=双轨(默认), free=主轨, anchor=对照')
    ap.add_argument('--mode-source', default=_RL_MODE_SOURCE,
                    choices=('from_units', 'game_theory'),
                    help='RL global_mode 来源(默认 from_units)')
    ap.add_argument('--fav-align-scale', type=float, default=_FAV_ALIGN_SCALE)
    ap.add_argument('--coherence-scale', type=float, default=0.25)
    ap.add_argument('--reward-delta', type=int, default=1, choices=(0, 1),
                    help='1=差分势能(默认), 0=绝对势能')
    ap.add_argument('--ent-coef', type=float, default=0.02)
    ap.add_argument('--hardgate', action='store_true')
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
    print('train_window:', train_ids, '| eval_games:', eval_games,
          '| mode_policy:', args.mode_policy, '| mode_source:', args.mode_source,
          '| coh:', args.coherence_scale, '| fav:', args.fav_align_scale,
          '| delta:', bool(args.reward_delta), flush=True)

    rows = []
    for gid in eval_games:
        row = process_game(
            gid, train_ids, args.rl_updates, args.limit, args.rl_seed,
            coherence_scale=args.coherence_scale, ent_coef=args.ent_coef,
            reward_delta=bool(args.reward_delta),
            mode_policy=args.mode_policy, mode_source=args.mode_source,
            fav_align_scale=args.fav_align_scale)
        if row:
            rows.append(row)
            print(json.dumps({k: v for k, v in row.items() if k != 'game_id'},
                             ensure_ascii=False), flush=True)

    agg = aggregate(rows)
    report = {
        'pipeline': 'track2_rl_fusion_v3fix',
        'sim_profile': 'physics_mdp',
        'primary_fusion': 'free',
        'vis_radius': _VIS_RADIUS,
        'rl_mode_source': args.mode_source,
        'fav_align_scale': args.fav_align_scale,
        'train_window': train_ids,
        'eval_games': eval_games,
        'rl_updates': args.rl_updates,
        'rl_seed': args.rl_seed,
        'coherence_scale': args.coherence_scale,
        'ent_coef': args.ent_coef,
        'reward_delta': bool(args.reward_delta),
        'mode_policy': args.mode_policy,
        'n_eval_games': len(rows),
        'rows': rows,
        'aggregate': agg,
        'control': {
            'bandit': 'compare-only; not in hard criterion',
            'anchored_fusion': 'control overview in rows[].fusion_rl_physics_anchored',
            'rule_degenerate_games': agg.get('rule_degenerate_games'),
        },
        'caveat': (
            'Track2=研究附录(非交付主线)。v3fix2: PhysicsEnv + fuse_mode(free) + '
            'mode=from_units(进攻偏向=防偏守启发式,非「优于rule」证明)。'
            'fav_align 为关联性非因果; 符号物理+模式级敌方≠完整裁判。'
            'bandit 为无因果对照; phys>bandit 不进硬门。'
            '退化硬门: free.attack_balance_share>=rule(rule 自身退化局跳过); 否则绝对 degen。'
            'seed 已钉死(np+torch); 4 留出局以方向与量级为准。'
            '交付主线: run_m2.py --profile mainline (lookahead QLoRA)。'
        ),
    }
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps({'type': 'SAVE', 'report': args.out, 'aggregate': agg},
                     ensure_ascii=False), flush=True)
    if args.hardgate and not agg.get('loaded_criterion_met', True):
        print('HARDGATE: 承重判据未达标 (loaded_criterion_met=False)', flush=True)
        return 2
    return 0


if __name__ == '__main__':
    sys.exit(main())
