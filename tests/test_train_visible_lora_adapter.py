import json


def test_train_visible_adapter_can_disable_class_balancing(monkeypatch, tmp_path):
    from eval import train_visible_lora_adapter as module
    from finetune import lora_train

    records = tmp_path / 'records.jsonl'
    records.write_text(json.dumps({'instruction': 'state', 'completion': '{}'}) + '\n', encoding='utf-8')
    captured = {}

    def fake_train(rows, **kwargs):
        captured['rows'] = rows
        captured.update(kwargs)
        return 'adapter', {'ok': True}

    monkeypatch.setattr(lora_train, 'train_lora', fake_train)
    result = module.main([
        '--records', str(records), '--out-dir', str(tmp_path / 'out'),
        '--no-class-balance',
    ])

    assert result == 0
    assert captured['class_balance'] is False
    assert len(captured['rows']) == 1
