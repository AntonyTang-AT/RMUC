import pytest

from runtime.decision_request import to_decision_request
from runtime.single_executor import ExecutorRegistrationError, SingleExecutorArbiter
from schema.layered import build_empty_decision


def _request():
    decision = build_empty_decision(3.0)
    decision['decision_meta'] = {
        'sentry_state': 'cruise',
        'strategic_intent': 'StrategicPatrol',
    }
    return to_decision_request(decision)


def test_single_executor_rejects_second_registration():
    arbiter = SingleExecutorArbiter()
    arbiter.register('vehicle', lambda request: {'ack_status': 'accepted'})
    with pytest.raises(ExecutorRegistrationError):
        arbiter.register('shadow', lambda request: {'ack_status': 'accepted'})


def test_single_executor_dispatches_valid_request_to_registered_handler():
    observed = []
    arbiter = SingleExecutorArbiter()
    arbiter.register('vehicle', lambda request: observed.append(request) or {'ack_status': 'accepted'})
    receipt = arbiter.dispatch(_request())
    assert receipt['accepted_for_dispatch'] is True
    assert receipt['executor'] == 'vehicle'
    assert receipt['controller_response']['ack_status'] == 'accepted'
    assert len(observed) == 1
    assert arbiter.history[0]['dispatch_id'] == 1


def test_single_executor_refuses_dispatch_without_registered_handler():
    with pytest.raises(ExecutorRegistrationError):
        SingleExecutorArbiter().dispatch(_request())
