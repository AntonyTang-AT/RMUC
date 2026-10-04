from tree.sentry_state_machine import EventEnum, SentryStateMachine, StateEnum


def _worldstate(*, hp=1.0, vuln=0):
    return {
        'robots': {
            's': {'type': '哨兵', 'team': '红', 'hp_ratio': hp, 'vuln': vuln},
        },
        'buildings': {},
    }


def test_state_machine_has_named_three_state_contract():
    assert {state.value for state in StateEnum} == {'def', 'hitbuilding', 'cruise'}
    machine = SentryStateMachine()
    result = machine.transition(_worldstate(), proposed_state='hitbuilding')
    assert result.event == EventEnum.ATTACK_OPPORTUNITY.value
    assert result.state == 'hitbuilding'
    assert result.strategic_intent == 'StrategicAttackOutpost'


def test_state_machine_safety_overrides_attack_and_can_reset():
    machine = SentryStateMachine()
    result = machine.transition(_worldstate(hp=0.1), proposed_state='hitbuilding')
    assert result.event == EventEnum.SAFETY_OVERRIDE.value
    assert result.state == 'def'
    machine.reset()
    assert machine.current_state == StateEnum.CRUISE


def test_state_machine_base_threat_overrides_patrol():
    machine = SentryStateMachine()
    result = machine.transition(
        _worldstate(), proposed_state='cruise', semantic_situation={'base_threat': True})
    assert result.event == EventEnum.BASE_THREAT.value
    assert result.state == 'def'
