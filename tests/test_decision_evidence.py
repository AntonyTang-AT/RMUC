from runtime.decision_evidence import (build_controller_record, build_issue_record,
                                       build_outcome_record, stable_sha256,
                                       validate_chain)


def _decision():
    return {
        't': 7.0, 'global_mode': '均衡',
        'units': {name: {'target': '无', 'move': '游击', 'fire': False, 'risk': '低'}
                  for name in ('英雄', '工程', '步兵3', '步兵4', '空中', '哨兵')},
    }


def test_hash_is_stable_across_dictionary_order():
    assert stable_sha256({'a': 1, 'b': 2}) == stable_sha256({'b': 2, 'a': 1})


def test_evidence_chain_requires_acknowledged_matching_command():
    issue = build_issue_record(match_id=7, side='红', decision_source='tier2_llm',
        visible_input={'schema_version': 1, 'entities': {}}, decision=_decision(),
        guard={'accepted': True, 'reasons': []}, decision_id='d1')
    controller = build_controller_record(decision_id='d1', command_id='c1', unit_id='红:1',
        issued_at_s=7, controller_received_at_s=7.1, controller_ack_at_s=7.2,
        ack_status='accepted', executed_action={'move': '游击'})
    outcome = build_outcome_record(decision_id='d1', unit_id='红:1', horizon_seconds=10,
        telemetry={'hp_ratio': 0.8})
    assert validate_chain(issue, controller, outcome) == (True, [])


def test_evidence_chain_never_treats_rejection_as_execution():
    issue = build_issue_record(match_id=7, side='红', decision_source='tier0_rule',
        visible_input={'entities': {}}, decision=_decision(), guard={'accepted': True}, decision_id='d1')
    controller = build_controller_record(decision_id='d1', command_id='c1', unit_id='红:1',
        issued_at_s=7, controller_received_at_s=7.1, controller_ack_at_s=7.2,
        ack_status='rejected', rejection_reason='safety')
    outcome = build_outcome_record(decision_id='d1', unit_id='红:1', horizon_seconds=10, telemetry={})
    assert validate_chain(issue, controller, outcome) == (False, ['command_not_accepted'])
