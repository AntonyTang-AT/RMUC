from eval.audit_llm_candidate_acceptance import audit


def _report(side, forecast, parsed=36, n=36):
    return {'result': {
        'side': side, 'forecast_context_enabled': forecast, 'parsed': parsed, 'n': n,
        'parse_rate': parsed / n, 'legal_rate': 1.0 if parsed == n else 0.0,
        'visible_rows_validated': n,
    }}


def _quality():
    scenarios = {}
    for side in ('红', '蓝'):
        for forecast in (True, False):
            scenarios[f'{side}|forecast={str(forecast).lower()}'] = {
                'end_to_end_hdq': {'n_games': 12, 'mean': .7, 'ci95': [.6, .8]},
                'end_to_end_hdq_delta_vs_legacy_rule': {'n_games': 12, 'mean': .1, 'ci95': [.01, .2]},
                'expected_global_mode_counts': {'进攻': 12, '防守': 12, '均衡': 12},
                'hdq_by_expected_global_mode': {
                    mode: {'n_rows': 12, 'end_to_end_hdq_delta_vs_legacy_rule': {
                        'n_games': 12, 'mean': .1, 'ci95': [.01, .2]}}
                    for mode in ('进攻', '防守', '均衡')
                },
                'orp': {},
            }
    return {'scenarios': scenarios}


def test_acceptance_requires_all_four_structural_scenarios():
    result = audit([_report('红', True)], _quality())
    assert result['gates']['structural_reliability'] == 'fail'
    assert len(result['missing_scenarios']) == 3


def test_acceptance_passes_shadow_gates_with_positive_hdq_and_coverage():
    reports = [_report(side, forecast) for side in ('红', '蓝') for forecast in (True, False)]
    result = audit(reports, _quality())
    assert result['gates']['structural_reliability'] == 'pass'
    assert result['gates']['semantic_hdq_vs_legacy_rule'] == 'pass'
    assert result['gates']['shadow_ready'] is True


def test_acceptance_allows_narrow_runtime_repair_but_reports_raw_gap():
    reports = [_report(side, forecast, parsed=12, n=12)
               for side in ('红', '蓝') for forecast in (True, False)]
    reports[0]['result'].update({
        'raw_parsed': 11,
        'raw_parse_rate': 11 / 12,
        'contract_repair_rate': 1 / 40,
        'runtime_delivery_rate': 1.0,
    })
    result = audit(reports, _quality())
    assert result['gates']['structural_reliability'] == 'pass'
    assert result['gates']['strict_generation_reliability'] == 'fail'


def test_acceptance_rejects_excessive_contract_repairs():
    reports = [_report(side, forecast, parsed=12, n=12)
               for side in ('红', '蓝') for forecast in (True, False)]
    reports[0]['result']['contract_repair_rate'] = 0.10
    result = audit(reports, _quality())
    assert result['gates']['structural_reliability'] == 'fail'


def test_acceptance_requires_all_global_modes_in_semantic_holdout():
    reports = [_report(side, forecast) for side in ('红', '蓝') for forecast in (True, False)]
    quality = _quality()
    quality['scenarios']['红|forecast=true']['expected_global_mode_counts']['均衡'] = 11

    result = audit(reports, quality)

    assert result['gates']['semantic_mode_coverage'] == 'insufficient'
    assert result['gates']['semantic_hdq_vs_legacy_rule'] == 'insufficient'
    assert result['gates']['shadow_ready'] is False


def test_acceptance_rejects_a_negative_mode_slice_despite_positive_aggregate():
    reports = [_report(side, forecast) for side in ('红', '蓝') for forecast in (True, False)]
    quality = _quality()
    quality['scenarios']['蓝|forecast=false']['hdq_by_expected_global_mode']['均衡'][
        'end_to_end_hdq_delta_vs_legacy_rule']['ci95'] = [-.05, .10]

    result = audit(reports, quality)

    assert result['gates']['semantic_hdq_by_global_mode_vs_legacy_rule'] == 'fail'
    assert result['gates']['semantic_hdq_vs_legacy_rule'] == 'fail'
    assert result['gates']['shadow_ready'] is False
