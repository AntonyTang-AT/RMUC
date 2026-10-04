from eval.audit_llm_hdq_breakdown import _scenario


def test_scenario_uses_visible_forecast_flag():
    assert _scenario({'perspective': '红', 'run_metadata': {
        'forecast_context_enabled': True}}) == '红|forecast=true'
    assert _scenario({'perspective': '蓝', 'run_metadata': {}}) == '蓝|forecast=false'
