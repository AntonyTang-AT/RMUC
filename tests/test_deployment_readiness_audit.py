from eval.audit_deployment_readiness import _llm_gate, classify_head


def test_head_requires_accuracy_f1_and_probability_quality_improvement():
    result = classify_head(
        {'accuracy': .71, 'macro_f1': .60, 'brier': .22},
        {'accuracy': .70, 'macro_f1': .59, 'brier': .25})
    assert result['strictly_beats_persistence']


def test_head_rejects_accuracy_only_gain_when_macro_f1_falls():
    result = classify_head(
        {'accuracy': .71, 'macro_f1': .58, 'brier': .22},
        {'accuracy': .70, 'macro_f1': .59, 'brier': .25})
    assert not result['strictly_beats_persistence']


def test_head_marks_small_directional_gain_as_not_statistically_confirmed():
    paired_game_bootstrap = {'metrics': {
        'accuracy': {'ci_excludes_zero_in_favor_of_production': True},
        'macro_f1': {'ci_excludes_zero_in_favor_of_production': False},
        'brier': {'ci_excludes_zero_in_favor_of_production': True},
    }}
    result = classify_head(
        {'accuracy': .71, 'macro_f1': .60, 'brier': .22},
        {'accuracy': .70, 'macro_f1': .59, 'brier': .25},
        paired_game_bootstrap=paired_game_bootstrap)
    assert result['strictly_beats_persistence']
    assert result['confidence_status'] == 'directional_only'


def test_llm_gate_requires_both_sides_with_and_without_forecasts():
    rows = [{'side': '红', 'forecast_context_enabled': True, 'structural_gate': 'pass'}]
    gate = _llm_gate(rows)
    assert gate['structural_robustness_gate'] == 'fail'
    assert len(gate['missing_scenarios']) == 3


def test_llm_gate_passes_only_when_all_required_scenarios_are_structurally_valid():
    rows = [
        {'side': side, 'forecast_context_enabled': forecast, 'structural_gate': 'pass'}
        for side in ('红', '蓝') for forecast in (True, False)
    ]
    assert _llm_gate(rows)['structural_robustness_gate'] == 'pass'


def test_llm_gate_uses_candidate_acceptance_for_repair_and_semantic_gates():
    rows = [
        {'side': side, 'forecast_context_enabled': forecast, 'structural_gate': 'pass'}
        for side in ('红', '蓝') for forecast in (True, False)
    ]
    acceptance = {'gates': {
        'structural_reliability': 'fail',
        'strict_generation_reliability': 'fail',
        'semantic_sample_size': 'pass',
        'semantic_mode_coverage': 'pass',
        'semantic_hdq_by_global_mode_vs_legacy_rule': 'fail',
        'semantic_hdq_vs_legacy_rule': 'fail',
        'shadow_ready': False,
    }}

    gate = _llm_gate(rows, acceptance=acceptance)

    assert gate['source'] == 'llm_candidate_acceptance'
    assert gate['structural_robustness_gate'] == 'fail'
    assert gate['semantic_hdq_by_global_mode_gate'] == 'fail'
    assert gate['semantic_hdq_vs_legacy_rule_gate'] == 'fail'
