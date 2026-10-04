import json

from eval.reconcile_llm_runtime_rows import reconcile
from schema import layered as sl


def _incomplete_decision():
    decision = sl.build_empty_decision(1.0)
    decision['units'].pop('英雄')
    return decision


def test_reconcile_preserves_strict_failure_and_delivers_narrow_repair():
    rows, summary = reconcile([{
        'raw_output': json.dumps(_incomplete_decision(), ensure_ascii=False),
        'rule_move': '游击',
    }])
    assert rows[0]['strict_parse_valid'] is False
    assert rows[0]['parse_valid'] is True
    assert rows[0]['contract_repairs'] == ['missing_unit_idle:英雄']
    assert summary['raw_parse_rate'] == 0.0
    assert summary['runtime_delivery_rate'] == 1.0
    assert summary['contract_repair_rate'] == 1.0
