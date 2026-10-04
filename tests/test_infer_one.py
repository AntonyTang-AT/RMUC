from finetune.infer_one import (extract_decision, extract_decision_with_error,
                                extract_decision_with_repair,
                                generate_instruction_results_with_bundle)
from schema import layered as sl


def _decision():
    decision = sl.build_empty_decision(3.0)
    decision['global_mode'] = '均衡'
    decision['units']['哨兵'] = {
        'target': '无', 'move': '游击', 'fire': False, 'risk': '中',
    }
    return decision


def test_extract_decision_ignores_explanation_and_markdown_fence():
    import json
    decision = _decision()
    text = '先说明原因。```json\n' + json.dumps(decision, ensure_ascii=False) + '\n```'
    assert extract_decision(text) == decision


def test_extract_decision_skips_invalid_json_and_accepts_later_valid_object():
    import json
    decision = _decision()
    text = '{"global_mode":"invalid"}\n' + json.dumps(decision, ensure_ascii=False)
    assert extract_decision(text) == decision


def test_extract_decision_normalizes_enum_whitespace():
    import json
    decision = _decision()
    decision['units']['哨兵']['move'] = ' 游击 '
    parsed = extract_decision(json.dumps(decision, ensure_ascii=False))
    assert parsed['units']['哨兵']['move'] == '游击'


def test_extract_decision_failure_categories_are_actionable():
    assert extract_decision_with_error('')[1] == 'empty_output'
    assert extract_decision_with_error('not json')[1] == 'no_json_object'
    assert extract_decision_with_error('{"global_mode":"进攻"')[1] == 'malformed_or_truncated_json'
    assert extract_decision_with_error('{"global_mode":"进攻"}')[1] == 'schema_invalid'


def test_runtime_contract_repair_maps_alias_and_fills_missing_units():
    import json
    decision = _decision()
    decision['units'].pop('英雄')
    decision['units']['工兵团'] = decision['units'].pop('工程')
    repaired, repairs, error = extract_decision_with_repair(
        json.dumps(decision, ensure_ascii=False))
    assert error == 'schema_repaired'
    assert set(repaired['units']) == set(sl.UNIT_CLASSES)
    assert 'unit_alias:工兵团->工程' in repairs
    assert 'missing_unit_idle:英雄' in repairs
    assert sl.validate(repaired)[0]


def test_runtime_contract_repair_normalizes_empty_risk_alias():
    import json
    decision = _decision()
    decision['units']['英雄'] = {
        'target': '无', 'move': '无', 'fire': False, 'risk': '无',
    }
    repaired, repairs, error = extract_decision_with_repair(
        json.dumps(decision, ensure_ascii=False))
    assert error == 'schema_repaired'
    assert repaired['units']['英雄']['risk'] == '低'
    assert repairs == ['enum_alias:英雄.risk:无->低']
    assert sl.validate(repaired)[0]


def test_generation_results_keep_strict_and_repair_metadata(monkeypatch):
    import json
    decision = _decision()
    decision['units'].pop('英雄')
    monkeypatch.setattr('finetune.infer_one.generate_instruction_texts_with_bundle',
                        lambda *args, **kwargs: [json.dumps(decision, ensure_ascii=False)])
    result = generate_instruction_results_with_bundle(None, None, ['visible input'])[0]
    assert result['strict_parse_valid'] is False
    assert result['decision']['units']['英雄']['move'] == '无'
    assert result['contract_repairs'] == ['missing_unit_idle:英雄']


def test_generation_results_retries_invalid_first_pass(monkeypatch):
    import json
    decision = _decision()
    invalid = dict(decision)
    invalid['units'] = dict(decision['units'])
    invalid['units'].pop('英雄')
    outputs = [json.dumps(invalid, ensure_ascii=False),
               json.dumps(decision, ensure_ascii=False)]

    def generate(*args, **kwargs):
        return [outputs.pop(0)]

    monkeypatch.setattr('finetune.infer_one.generate_instruction_texts_with_bundle', generate)
    result = generate_instruction_results_with_bundle(None, None, ['visible input'])[0]
    assert result['first_pass_strict_parse_valid'] is False
    assert 'retry_raw_output' in result
    assert result['retry_strict_parse_valid'] is True
    assert result['strict_parse_valid'] is True
    assert result['decision'] == decision


def test_batch_generation_restores_tokenizer_padding_side(monkeypatch):
    import json
    import torch
    from finetune import infer_one

    decision = _decision()

    class Encoded:
        input_ids = torch.tensor([[0, 1], [0, 1]])
        attention_mask = torch.tensor([[1, 1], [1, 1]])

    class Tokenizer:
        padding_side = 'right'
        eos_token_id = 2

        def __call__(self, prompts, **kwargs):
            assert len(prompts) == 2
            assert kwargs['padding'] is True
            return Encoded()

        def decode(self, row, skip_special_tokens=True):
            return json.dumps(decision, ensure_ascii=False)

    class Model:
        device = torch.device('cpu')

        def generate(self, input_ids, attention_mask, **kwargs):
            assert input_ids.shape[0] == 2
            return torch.tensor([[0, 1, 3], [0, 1, 3]])

    monkeypatch.setattr('finetune.lora_train.eval_prompt',
                        lambda worldstate, forecast_context=None: 'prompt')
    monkeypatch.setattr('worldstate.worldstate.build_worldstate',
                        lambda dp, bounds: {})
    result = infer_one.generate_batch_with_bundle(
        Tokenizer(), Model(), [({}, {}, None), ({}, {}, None)], max_new=8)
    assert result == [decision, decision]
