# -*- coding: utf-8 -*-
"""llm_backtest 纯逻辑单测(无 GPU): 进攻度映射 + 决策-有利度对齐度。"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from eval import llm_backtest as lb


def test_aggression_maps():
    assert lb.aggression_of_move('推进') > 0
    assert lb.aggression_of_move('回防') < 0
    assert lb.aggression_of_move('坚守') < 0
    assert lb.aggression_of_move('无') == 0.0
    assert lb.aggression_of_move('未知词') == 0.0            # 未知 -> 中性
    assert lb.aggression_of_mode('进攻') > 0 and lb.aggression_of_mode('防守') < 0


def test_alignment_defensive_in_lower_fav():
    # 守(负进攻度)落在低有利度, 攻(正进攻度)落在高有利度 -> gap 为负(合理)
    pairs = [(-1.0, 0.2), (-0.5, 0.4), (1.0, 0.9), (1.0, 0.8)]
    r = lb.alignment(pairs)
    assert r['def_minus_aggr_gap'] < 0
    assert abs(r['def_mean_fav'] - 0.3) < 1e-9
    assert r['n_def'] == 2 and r['n_aggr'] == 2


def test_alignment_no_side_gives_none():
    assert lb.alignment([(0.0, 0.5), (0.0, 0.6)])['def_minus_aggr_gap'] is None
    assert lb.alignment([(-1.0, 0.5)])['aggr_mean_fav'] is None


def test_row_record_retains_raw_and_failed_decodes_without_future_fields():
    worldstate = {'t': 12.0, 'robots': {'r1': {'type': '英雄', 'hp_ratio': 0.5}}}
    record = lb.make_row_record(
        game_id=42, worldstate=worldstate, prompt='current-only prompt',
        raw_output='not json', decision=None, error='invalid JSON',
        latency_s=1.25, prompt_tokens=20, completion_tokens=3,
        adapter='adapter-a', run_metadata={'adapter_tree_sha256': 'abc'},
        visibility_policy='test-current-input')
    assert record['game_id'] == 42
    assert record['issued_at_s'] == 12.0
    assert record['parse_valid'] is False
    assert record['raw_output'] == 'not json'
    assert record['prompt_text'] == 'current-only prompt'
    assert record['visibility_policy'] == 'test-current-input'
    assert record['run_metadata']['adapter_tree_sha256'] == 'abc'
    assert record['issue_state'] == worldstate
    assert record['completion_tokens'] == 3
    assert 'future' not in record and 'outcome' not in record


def test_adapter_tree_fingerprint_is_stable_and_content_sensitive(tmp_path):
    adapter = tmp_path / 'adapter'
    adapter.mkdir()
    (adapter / 'adapter_config.json').write_text('{}', encoding='utf-8')
    weights = adapter / 'adapter_model.safetensors'
    weights.write_bytes(b'weights-a')
    first = lb._sha256_tree(adapter)
    assert first == lb._sha256_tree(adapter)
    weights.write_bytes(b'weights-b')
    assert first != lb._sha256_tree(adapter)
