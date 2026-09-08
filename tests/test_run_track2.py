# -*- coding: utf-8 -*-
"""run_track2 纯函数单测(无 torch/无 db 连接): load_plan / aggregate。
process_game 走 db+RL, 由真实冒烟跑覆盖, 不在此单测。"""
import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import run_track2 as r2


def test_load_plan(tmp_path):
    p = tmp_path / 'm2_report.json'
    p.write_text(json.dumps({'eval_games': [1, 2, 3], 'dataset_games': [4]}), encoding='utf-8')
    plan = r2.load_plan(str(p))
    assert plan['eval_games'] == [1, 2, 3] and plan['dataset_games'] == [4]


def _row(fav_align_p=None, fav_align_b=None, legal=True, degen=False):
    ph = {'fav_align': fav_align_p, 'legal_rate': 1.0 if legal else 0.5}
    ba = {'fav_align': fav_align_b, 'legal_rate': 1.0}
    return {'rl_source_physics': ph, 'rl_source_bandit': ba,
            'compare': {'phys_ge_bandit_fav_align': (fav_align_p >= fav_align_b)
                        if (fav_align_p is not None and fav_align_b is not None) else None},
            'fusion_rl_physics': {'degenerate': degen},
            'rl_physics_legal_pass': legal}


def test_aggregate_empty():
    assert r2.aggregate([]) == {}


def test_aggregate_loaded_criterion_met():
    rows = [_row(0.2, 0.0, legal=True, degen=False),
            _row(0.1, 0.3, legal=True, degen=False),   # 这局 physics<bandit, 但半数达标即可
            _row(0.3, 0.1, legal=True, degen=False)]
    a = r2.aggregate(rows)
    assert a['n_games'] == 3
    assert a['avg_fav_align_physics'] == 0.2
    assert a['avg_fav_align_bandit'] == round(0.4 / 3, 4)
    assert a['n_phys_ge_bandit_fav_align'] == 2 and a['n_phys_ge_bandit_fav_align_base'] == 3
    assert a['all_rl_physics_legal'] is True
    assert a['all_fusion_non_degenerate'] is True
    assert a['loaded_criterion_met'] is True


def test_aggregate_not_met_when_illegal():
    rows = [_row(0.2, 0.0, legal=False, degen=False), _row(0.1, 0.0, legal=True, degen=False)]
    a = r2.aggregate(rows)
    assert a['all_rl_physics_legal'] is False
    assert a['loaded_criterion_met'] is False


def test_aggregate_none_fav_align_skipped():
    rows = [_row(None, None, legal=True, degen=False),
            _row(0.2, 0.0, legal=True, degen=False)]
    a = r2.aggregate(rows)
    assert a['avg_fav_align_physics'] == 0.2      # None 侧被跳过
    assert a['n_phys_ge_bandit_fav_align_base'] == 1
    assert a['fav_align_delta_phys_minus_bandit'] == 0.2
