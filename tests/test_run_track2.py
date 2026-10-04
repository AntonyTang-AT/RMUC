# -*- coding: utf-8 -*-
"""run_track2 纯函数单测: load_plan / aggregate(v3-sim-rel-degen)。"""
import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import run_track2 as r2


def test_load_plan(tmp_path):
    p = tmp_path / 'm2_report.json'
    p.write_text(json.dumps({'eval_games': [1, 2, 3], 'dataset_games': [4]}), encoding='utf-8')
    plan = r2.load_plan(str(p))
    assert plan['eval_games'] == [1, 2, 3] and plan['dataset_games'] == [4]


def _row(fav_align_p=0.2, fav_align_b=0.0, legal=True, degen_free=False,
         degen_anchor=False, degen_rule=False, const_p=False, const_b=False,
         game_id=1, n_free_diff=0):
    ph = {'fav_align': fav_align_p, 'legal_rate': 1.0 if legal else 0.5,
          'policy_constant': const_p, 'mean_move_diversity': 1.0 if const_p else 2.5}
    ba = {'fav_align': fav_align_b, 'legal_rate': 1.0,
          'policy_constant': const_b, 'mean_move_diversity': 1.0 if const_b else 2.0}
    both_nc = (not const_p) and (not const_b)
    phys_ge = None
    if both_nc and fav_align_p is not None and fav_align_b is not None:
        phys_ge = bool(fav_align_p > fav_align_b)
    return {
        'game_id': game_id,
        'rl_source_physics': ph, 'rl_source_bandit': ba,
        'compare': {'phys_ge_bandit_fav_align': phys_ge, 'both_non_constant': both_nc},
        'fusion_rule_only': {'degenerate': degen_rule},
        'fusion_rl_physics_free': {'degenerate': degen_free},
        'fusion_rl_physics_anchored': {'degenerate': degen_anchor},
        'fusion_rl_physics': {'degenerate': degen_free},
        'n_free_mode_diff_vs_rule': n_free_diff,
        'rl_physics_legal_pass': legal,
        'phys_fav_align_positive': (
            fav_align_p is not None and fav_align_p > 0),
    }


def test_aggregate_empty():
    assert r2.aggregate([]) == {}


def test_aggregate_v3sim_pass():
    """半数非常数 + fav>0 + free 相对非退化 + 合法 → 通过; bandit 差不影响。"""
    rows = [
        _row(0.2, 0.9, degen_free=False, degen_anchor=True, game_id=1),
        _row(0.1, 0.0, degen_free=False, degen_anchor=False, game_id=2),
        _row(-0.05, 0.0, degen_free=False, game_id=3),
    ]
    a = r2.aggregate(rows)
    assert a['n_physics_non_constant'] == 3
    assert a['n_phys_fav_align_positive'] == 2
    assert a['all_fusion_non_degenerate_free'] is True
    assert a['all_fusion_non_degenerate_anchored'] is False
    assert a['sim_gain_free_fixes_anchor_degen'] is True
    assert a['loaded_criterion_met'] is True


def test_aggregate_skips_rule_degenerate_for_free_gate():
    """rule 自身退化且 free 也退化 → 相对硬门仍可通过(跳过该局)。"""
    rows = [
        _row(0.2, 0.0, degen_free=True, degen_rule=True, game_id=1778661970735),
        _row(0.3, 0.0, degen_free=False, degen_rule=False, game_id=2),
    ]
    a = r2.aggregate(rows)
    assert a['rule_degenerate_games'] == [1778661970735]
    assert a['all_fusion_non_degenerate_free'] is True
    assert a['all_fusion_non_degenerate_free_absolute'] is False
    assert a['loaded_criterion_met'] is True
    assert 'rel-share' in a['criterion_note'] or 'rel-degen' in a['criterion_note']


def test_aggregate_fails_when_free_degenerate_and_rule_ok():
    """rule 非退化但 free 退化 → 硬门失败。"""
    rows = [_row(0.2, 0.0, degen_free=True, degen_rule=False),
            _row(0.3, 0.0, degen_free=True, degen_rule=False)]
    a = r2.aggregate(rows)
    assert a['all_fusion_non_degenerate_free'] is False
    assert a['loaded_criterion_met'] is False


def test_aggregate_fails_when_all_constant():
    rows = [_row(None, None, const_p=True, const_b=True),
            _row(None, None, const_p=True, const_b=True)]
    a = r2.aggregate(rows)
    assert a['n_physics_non_constant'] == 0
    assert a['loaded_criterion_met'] is False


def test_aggregate_fails_when_illegal():
    rows = [_row(0.2, 0.0, legal=False), _row(0.1, 0.0, legal=True)]
    a = r2.aggregate(rows)
    assert a['all_rl_physics_legal'] is False
    assert a['loaded_criterion_met'] is False


def test_aggregate_bandit_control_fields_present():
    rows = [_row(0.2, 0.0, n_free_diff=3), _row(0.3, 0.1, n_free_diff=5)]
    a = r2.aggregate(rows)
    assert 'n_phys_ge_bandit_fav_align' in a
    assert 'avg_fav_align_bandit' in a
    assert a['n_free_mode_diff_vs_rule'] == 8
    assert 'v3' in a['criterion_note']


def test_aggregate_free_share_ge_rule_passes_even_if_abs_flag():
    """free 绝对退化标记为真, 但攻+均份额不低于 rule → 相对硬门通过。"""
    rows = [
        _row(0.2, 0.0, degen_free=True, degen_rule=False, game_id=1),
        _row(0.3, 0.0, degen_free=False, degen_rule=False, game_id=2),
    ]
    rows[0]['fusion_rule_only'] = {
        'degenerate': False, 'attack_balance_share': 0.05}
    rows[0]['fusion_rl_physics_free'] = {
        'degenerate': True, 'attack_balance_share': 0.08}
    rows[0]['fusion_rl_physics'] = rows[0]['fusion_rl_physics_free']
    a = r2.aggregate(rows)
    assert a['all_fusion_non_degenerate_free'] is True
    assert a['loaded_criterion_met'] is True


def test_aggregate_free_share_worse_than_rule_fails():
    rows = [
        _row(0.2, 0.0, degen_free=True, degen_rule=False, game_id=1),
        _row(0.3, 0.0, degen_free=False, degen_rule=False, game_id=2),
    ]
    rows[0]['fusion_rule_only'] = {
        'degenerate': False, 'attack_balance_share': 0.40}
    rows[0]['fusion_rl_physics_free'] = {
        'degenerate': True, 'attack_balance_share': 0.02}
    rows[0]['fusion_rl_physics'] = rows[0]['fusion_rl_physics_free']
    a = r2.aggregate(rows)
    assert a['all_fusion_non_degenerate_free'] is False
    assert a['loaded_criterion_met'] is False
