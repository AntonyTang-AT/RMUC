# -*- coding: utf-8 -*-
"""finetune.retrain_eval 纯函数单测(无 torch/transformers/sklearn, 离线可跑)。

只测不碰 GPU/db 的部分: load_records(jsonl 读取)、md.rebalance_records 被正确应用、
merge_delta(两行对照的承重判据 rollout_gap_ge_lookahead)。
"""
import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from finetune import retrain_eval as re


def _rec(mode):
    return {'instruction': f'局势: {mode}', 'completion': json.dumps({'global_mode': mode})}


def test_load_records(tmp_path):
    p = tmp_path / 'd.jsonl'
    rows = [_rec('进攻'), _rec('均衡'), _rec('防守')]
    with open(p, 'w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    got = re.load_records(str(p))
    assert len(got) == 3
    assert got[0]['completion'].startswith('{')  # completion 是决策 JSON 串


def test_merge_delta_rollout_ge_lookahead():
    la = {'method': 'lookahead', 'global_mode_align_gap': {'gap': -0.05}, 'eval': {'legal_rate': 0.95}}
    ro = {'method': 'rollout', 'global_mode_align_gap': {'gap': 0.1}, 'eval': {'legal_rate': 1.0}}
    d = re.merge_delta({'lookahead': la, 'rollout': ro})
    assert d['rollout_gap_ge_lookahead'] is True
    assert d['gap_delta_rollout_minus_lookahead'] == 0.15
    assert d['legal_rollout'] == 1.0 and d['legal_lookahead'] == 0.95


def test_merge_delta_none_gap():
    la = {'method': 'lookahead', 'global_mode_align_gap': {'gap': None}, 'eval': {'legal_rate': 0.9}}
    ro = {'method': 'rollout', 'global_mode_align_gap': {'gap': None}, 'eval': {'legal_rate': 0.9}}
    d = re.merge_delta({'lookahead': la, 'rollout': ro})
    assert d['gap_delta_rollout_minus_lookahead'] is None
    assert d['rollout_gap_ge_lookahead'] is None


def test_merge_delta_requires_both():
    d = re.merge_delta({'lookahead': {'method': 'lookahead'}})
    assert d == {}


def test_subsample_even_and_deterministic():
    pts = list(range(560))
    a = re.subsample(pts, 160)
    b = re.subsample(pts, 160)
    assert len(a) == 160
    assert a == b                        # 确定性
    assert a[0] == 0 and a[-1] == 559    # 均匀覆盖起点与终点
    # n<=0 或不足 n -> 原样
    assert re.subsample(pts, 0) == pts
    assert re.subsample(list(range(10)), 160) == list(range(10))
