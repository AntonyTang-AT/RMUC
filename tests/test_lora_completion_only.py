def test_completion_only_masks_prompt_tokens(monkeypatch):
    from finetune import lora_train

    class Tokenizer:
        def __call__(self, text, **kwargs):
            # One token per character keeps the expected prefix boundary clear.
            return {"input_ids": list(range(len(text))),
                    "attention_mask": [1] * len(text)}

    record = {"instruction": "态势", "completion": '{"global_mode":"均衡"}'}
    text = lora_train._format(record)
    tok = Tokenizer()
    encoded = lora_train._encode_record(tok, record, max_len=len(text),
                                        completion_only=True)
    prefix = text.rsplit(record["completion"], 1)[0]
    assert encoded["labels"][:len(prefix)] == [-100] * len(prefix)
    assert encoded["labels"][len(prefix):] == list(range(len(prefix), len(text)))


def test_contract_repair_fills_missing_unit_fields_conservatively():
    from finetune.infer_one import extract_decision_with_repair

    raw = ('{"t":1,"global_mode":"均衡","units":{' +
           ','.join(
               f'"{name}":{{"target":"无","move":"无","fire":false}}'
               for name in ('英雄', '工程', '步兵3', '步兵4', '空中', '哨兵')) + '}}')
    decision, repairs, error = extract_decision_with_repair(raw)
    assert error == 'schema_repaired'
    assert decision['units']['英雄']['risk'] == '低'
    assert len(repairs) == 6
