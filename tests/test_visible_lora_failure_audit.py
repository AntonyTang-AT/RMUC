from eval.audit_visible_lora_failures import audit


def _row(parsed, game=1, t=1.0, current=True):
    return {
        'parse_valid': parsed, 'game_id': game, 'issued_at_s': t,
        'prompt_text': 'visible prompt',
        'visible_history': [{'entities': {'x': {
            'observability_state': 'current' if current else 'stale'}}}],
        'forecast_context': {'predictions': [{'horizon_seconds': 10}]},
    }


def test_audit_groups_failures_without_treating_them_as_policy_labels():
    report = audit([_row(False, game=7, t=1), _row(False, game=7, t=4, current=False), _row(True)])
    assert report['parse_failures'] == 2
    assert report['failure_games']['7']['issued_at_s'] == [1.0, 4.0]
    assert report['failure_reasons']['unspecified'] == 2
    assert 'association only' in report['interpretation']
