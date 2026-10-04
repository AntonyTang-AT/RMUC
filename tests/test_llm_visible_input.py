import copy
import hashlib

import pytest

from data.radar_visibility_replay import PROVIDER_ID
from eval.llm_visible_input import (build_visible_llm_input,
                                    format_visible_training_instruction,
                                    load_visible_llm_input,
                                    mask_own_unit_slot,
                                    own_unit_slot_states,
                                    project_decision_to_observable_contract,
                                    validate_visible_llm_input)
from eval.llm_row_scorecard import score_rows
from eval.llm_visible_paired_run import _assert_frozen_prompts
from jev.api import candidate_prompts, compile_request


def _frame(t, enemy_x=19.0):
    return {
        'visibility_provider': PROVIDER_ID, 'self_team': '红', 'issued_at': t,
        'entities': {
            '红:1': {'entity_id': '红:1', 'team': '红', 'entity_type': '英雄',
                     'role': 'self', 'source': 'sensor', 'observability_state': 'current',
                     'observed_at': t, 'received_at': t, 'position': [8.0, 7.0],
                     'hp': 80, 'maxhp': 100, 'yaw': 40, 'small_heat': 2,
                     'future_result': 'hidden own outcome'},
            '蓝:2': {'entity_id': '蓝:2', 'team': '蓝', 'entity_type': '哨兵',
                     'role': 'opponent', 'source': 'radar', 'observability_state': 'current',
                     'observed_at': t, 'received_at': t, 'position': [enemy_x, 7.0],
                     'hp': 90, 'maxhp': 100, 'yaw': 270, 'ammo17': 100,
                     'secret_target': 'hidden opponent detail'},
        },
    }


def test_prompt_uses_only_selected_side_visible_history_and_current_frame():
    frames = [_frame(7), _frame(10), _frame(13, enemy_x=123.0)]
    row = build_visible_llm_input(frames, game_id=1, side='红', issued_at_s=10)
    assert validate_visible_llm_input(row)
    assert [frame['t'] for frame in row['visible_history']] == [7, 10]
    assert '123.0' not in row['prompt_text']
    for hidden in ('future_result', 'secret_target', 'hidden opponent detail', 'maxhp', 'yaw'):
        # Own maxhp/yaw may be present, but never on the opponent record.
        if hidden in ('maxhp', 'yaw'):
            assert hidden not in row['visible_history'][-1]['entities']['蓝:2']
        else:
            assert hidden not in row['prompt_text']
    assert row['issue_state']['robots']['红:1']['hp_ratio'] == 0.8
    assert 'hp_ratio' not in row['issue_state']['robots']['蓝:2']

    changed_future = copy.deepcopy(frames)
    changed_future[-1]['entities']['蓝:2']['position'] = [-999, -999]
    same = build_visible_llm_input(changed_future, game_id=1, side='红', issued_at_s=10)
    assert same['input_sha256'] == row['input_sha256']


def test_future_measurement_or_untrusted_provider_fails_closed():
    frame = _frame(10)
    frame['entities']['蓝:2']['observed_at'] = 11
    with pytest.raises(ValueError, match='future measurement'):
        build_visible_llm_input([frame], game_id=1, side='红', issued_at_s=10)
    frame = _frame(10)
    frame['visibility_provider'] = 'full_replay'
    with pytest.raises(ValueError, match='provider'):
        build_visible_llm_input([frame], game_id=1, side='红', issued_at_s=10)


def test_visible_scorecard_claim_requires_prompt_and_issue_state_integrity():
    row = build_visible_llm_input([_frame(10)], game_id=1, side='红', issued_at_s=10)
    row['decision'] = {'units': {'英雄': {'move': '坚守', 'fire': False}}}
    assert score_rows([row], [])['visibility_audit']['deployable_claim_allowed']

    forged = copy.deepcopy(row)
    forged['issue_state']['robots']['蓝:2']['hidden_future_hp'] = 0
    assert not score_rows([forged], [])['visibility_audit']['deployable_claim_allowed']
    forged = copy.deepcopy(row)
    forged['prompt_text'] += ' 未来结果'
    forged['input_sha256'] = hashlib.sha256(forged['prompt_text'].encode()).hexdigest()
    assert not validate_visible_llm_input(forged)
    forged = copy.deepcopy(row)
    forged['prompt_text'] = '未来结果 ' + forged['prompt_text']
    forged['input_sha256'] = hashlib.sha256(forged['prompt_text'].encode()).hexdigest()
    assert not validate_visible_llm_input(forged)


def test_future_and_stale_data_do_not_enter_measurement_fields():
    old = _frame(10)
    old['entities']['蓝:2']['observability_state'] = 'stale'
    row = build_visible_llm_input([old], game_id=1, side='红', issued_at_s=10)
    enemy = row['visible_history'][0]['entities']['蓝:2']
    assert 'position' not in enemy and 'hp' not in enemy
    assert validate_visible_llm_input(row)


def test_compact_training_instruction_uses_only_visible_projection_and_forecast_reference():
    visible = build_visible_llm_input([_frame(10)], game_id=4, side='红', issued_at_s=10)
    forecast = {
        'schema_version': 'forecast-context-v1',
        'predictions': [{
            'horizon_seconds': horizon,
            'global_mode': {'top': '均衡', 'confidence': 0.72,
                            'probabilities': {'均衡': .72, '进攻': .2, '防守': .08}},
            'opponent_trend': {'top': '推进', 'confidence': 0.61,
                               'probabilities': {'推进': .61, '稳定': .39}},
            'enemy_region': {'top': 'x3y1', 'confidence': 0.54,
                             'probabilities': {'x3y1': .54, 'x2y1': .31}},
            'model_comparison': {'enemy_region': {'model': 'x3y1', 'baseline': 'x2y1'}},
        } for horizon in (10, 30, 60, 120)],
    }
    instruction = format_visible_training_instruction(visible, forecast, max_chars=760)
    assert len(instruction) <= 760
    assert '预测参考(非事实)' in instruction
    assert 'M=x3y1/P=x2y1' in instruction
    assert 'hidden own outcome' not in instruction
    assert 'secret_target' not in instruction
    assert 'maxhp' not in instruction
    assert '己方固定槽位' in instruction
    assert '英雄=current' in instruction


def test_fixed_slots_and_counterfactual_mask_keep_unseen_unit_explicit():
    visible = build_visible_llm_input([_frame(10)], game_id=4, side='红', issued_at_s=10)
    masked = mask_own_unit_slot(visible, '英雄')
    assert validate_visible_llm_input(masked)
    assert own_unit_slot_states(masked)['英雄']['status'] == 'unseen'
    instruction = format_visible_training_instruction(masked, max_chars=760)
    assert '英雄=unseen' in instruction
    decision = {
        't': 10.0, 'global_mode': '均衡',
        'units': {name: {'target': '敌方哨兵', 'move': '推进', 'fire': True, 'risk': '低'}
                  for name in ('英雄', '工程', '步兵3', '步兵4', '空中', '哨兵')},
    }
    projected = project_decision_to_observable_contract(decision, masked)
    assert projected['units']['英雄'] == {
        'target': '无', 'move': '无', 'fire': False, 'risk': '高'}


def test_zero_hp_own_slot_is_destroyed_for_projection_contract():
    visible = build_visible_llm_input([_frame(10)], game_id=4, side='红', issued_at_s=10)
    visible['issue_state']['robots']['红:步兵3'] = {
        'team': '红', 'type': '步兵3', 'hp_ratio': 0.0,
        'observability_state': 'current',
    }
    assert own_unit_slot_states(visible)['步兵3']['status'] == 'destroyed'


def test_database_loader_queries_only_the_issue_time_prefix(monkeypatch):
    class Database:
        def execute(self, query, params):
            assert '"时刻秒"<=?' in query
            assert params == (4, 10.0)
            return [{'game_id': 4, '时刻秒': 10.0}]

    monkeypatch.setattr(
        'data.radar_visibility_replay.build_radar_visibility_snapshots',
        lambda rows, game_id: {'红': [_frame(10)], '蓝': []})
    row = load_visible_llm_input(Database(), game_id=4, side='红', issued_at_s=10)
    assert validate_visible_llm_input(row)


def test_paired_candidate_prompts_are_exactly_frozen():
    visible = build_visible_llm_input([_frame(10)], game_id=4, side='红', issued_at_s=10)
    questions = {'sentry_move': {'type': 'choice', 'instructions': '选择动作',
                 'criteria': {'推进': '前进', '坚守': '守区'}}}
    compiled = compile_request({'visible_prompt': visible['prompt_text']}, questions)[0]
    item = {'visible_input': visible, 'questions': questions,
            'candidate_keys': compiled['answer_keys'],
            'candidate_prompts': candidate_prompts(compiled)}
    _assert_frozen_prompts(item)
    item['candidate_prompts'][0] += ' future'
    with pytest.raises(ValueError, match='frozen candidate prompts'):
        _assert_frozen_prompts(item)
