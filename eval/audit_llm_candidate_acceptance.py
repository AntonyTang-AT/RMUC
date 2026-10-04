"""Gate a visible-input LLM candidate on reliability and HDQ evidence."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

SIDES = ('红', '蓝')
MAX_CONTRACT_REPAIR_RATE = 0.05
MIN_MODE_ROWS = 12


def _scenario_key(side, forecast):
    return f'{side}|forecast={str(bool(forecast)).lower()}'


def audit(reports, quality, *, min_games=12, min_mode_rows=MIN_MODE_ROWS,
          min_mode_games=None):
    if min_mode_games is None:
        min_mode_games = int(min_games)
    by_scenario = {}
    for report in reports:
        result = report['result']
        key = _scenario_key(result['side'], result['forecast_context_enabled'])
        if key in by_scenario:
            raise ValueError(f'duplicate scenario report: {key}')
        parsed_all = int(result['parsed']) == int(result['n'])
        legal_all = parsed_all and float(result['legal_rate']) == 1.0
        visible_all = int(result['visible_rows_validated']) == int(result['n'])
        raw_parsed = int(result.get('raw_parsed', result['parsed']))
        raw_parse_rate = float(result.get('raw_parse_rate', result['parse_rate']))
        repair_rate = float(result.get('contract_repair_rate', 0.0))
        repair_budget_pass = repair_rate <= MAX_CONTRACT_REPAIR_RATE
        q = (quality.get('scenarios') or {}).get(key)
        expected_mode_counts = {
            mode: int(((q or {}).get('expected_global_mode_counts') or {}).get(mode, 0))
            for mode in ('进攻', '防守', '均衡')
        }
        mode_coverage_pass = q is not None and all(
            count >= int(min_mode_rows) for count in expected_mode_counts.values())
        mode_hdq = {}
        for mode in ('进攻', '防守', '均衡'):
            delta = (((q or {}).get('hdq_by_expected_global_mode') or {}).get(mode) or {}).get(
                'end_to_end_hdq_delta_vs_legacy_rule') or {}
            ci95 = delta.get('ci95')
            games = int(delta.get('n_games') or 0)
            mode_hdq[mode] = {
                'n_games': games,
                'ci95': ci95,
                'sample_size_pass': games >= int(min_mode_games),
                'positive_delta_pass': bool(ci95 is not None and float(ci95[0]) > 0.0),
            }
        mode_hdq_pass = all(value['sample_size_pass'] and value['positive_delta_pass']
                            for value in mode_hdq.values())
        by_scenario[key] = {
            'side': result['side'],
            'forecast_context_enabled': bool(result['forecast_context_enabled']),
            'n': int(result['n']),
            'parse_rate': float(result['parse_rate']),
            'runtime_delivery_rate': float(result.get('runtime_delivery_rate', result['parse_rate'])),
            'raw_parsed': raw_parsed,
            'raw_parse_rate': raw_parse_rate,
            'legal_rate_conditional_on_parse': float(result['legal_rate']),
            'contract_repair_rate': repair_rate,
            'unrecoverable_rows': int(result.get('unrecoverable_rows',
                                                  int(result['n']) - int(result['parsed']))),
            'visible_rows_validated': int(result['visible_rows_validated']),
            'expected_global_mode_counts': expected_mode_counts,
            'semantic_mode_coverage_pass': mode_coverage_pass,
            'hdq_by_expected_global_mode': mode_hdq,
            'semantic_mode_hdq_pass': mode_hdq_pass,
            'strict_generation_pass': bool(raw_parsed == int(result['n'])),
            'repair_budget_pass': repair_budget_pass,
            'structural_pass': bool(parsed_all and legal_all and visible_all and repair_budget_pass),
            'end_to_end_hdq': ((q or {}).get('end_to_end_hdq') or {}),
            'end_to_end_hdq_delta_vs_legacy_rule': (
                (q or {}).get('end_to_end_hdq_delta_vs_legacy_rule') or {}),
            'response_proxy': ((q or {}).get('orp') or {}),
            'quality_available': q is not None,
        }
    expected = {_scenario_key(side, forecast) for side in SIDES for forecast in (True, False)}
    missing = sorted(expected - set(by_scenario))
    structural_pass = not missing and all(row['structural_pass'] for row in by_scenario.values())
    semantic_rows = list(by_scenario.values())
    enough_games = all(int((row['end_to_end_hdq'] or {}).get('n_games') or 0) >= int(min_games)
                       for row in semantic_rows)
    enough_mode_coverage = all(row['semantic_mode_coverage_pass'] for row in semantic_rows)
    mode_hdq_sample_size = all(
        all(value['sample_size_pass'] for value in row['hdq_by_expected_global_mode'].values())
        for row in semantic_rows)
    mode_hdq_positive = all(
        all(value['positive_delta_pass'] for value in row['hdq_by_expected_global_mode'].values())
        for row in semantic_rows)
    enough_mode_hdq = mode_hdq_sample_size and mode_hdq_positive
    deltas = [row['end_to_end_hdq_delta_vs_legacy_rule'] for row in semantic_rows]
    semantic_positive = enough_games and enough_mode_coverage and enough_mode_hdq and all(
        value.get('ci95') is not None and float(value['ci95'][0]) > 0.0 for value in deltas)
    return {
        'schema_version': 'llm-candidate-acceptance-v1',
        'required_scenarios': sorted(expected),
        'scenarios': by_scenario,
        'missing_scenarios': missing,
        'gates': {
            'structural_reliability': 'pass' if structural_pass else 'fail',
            'strict_generation_reliability': (
                'pass' if not missing and all(row['strict_generation_pass']
                                               for row in by_scenario.values()) else 'fail'),
            'semantic_sample_size': ('pass' if enough_games else 'insufficient'),
            'semantic_mode_coverage': ('pass' if enough_mode_coverage else 'insufficient'),
            'semantic_hdq_by_global_mode_vs_legacy_rule': (
                'pass' if enough_mode_hdq else ('insufficient' if not mode_hdq_sample_size else 'fail')),
            'semantic_hdq_vs_legacy_rule': ('pass' if semantic_positive else
                                             ('insufficient' if not enough_games or not enough_mode_coverage
                                              or not mode_hdq_sample_size
                                              else 'fail')),
            'causal_policy_value': 'unverifiable_without_controller_execution_and_counterfactual_evidence',
            'shadow_ready': bool(structural_pass and semantic_positive),
        },
        'interpretation': [
            'A scenario must be fully parsed, schema-legal, and visibility-validated.',
            ('Runtime delivery may use only the narrow contract repair path; each scenario '
             f'must stay at or below {MAX_CONTRACT_REPAIR_RATE:.0%} repaired rows.'),
            'Strict raw-generation reliability is reported separately and never hidden by repair.',
            ('Each scenario must contain at least '
             f'{int(min_mode_rows)} look-ahead labels for every global mode before semantic promotion.'),
            ('Every global-mode slice must cover at least '
             f'{int(min_mode_games)} games and have a positive 95% CI lower bound for HDQ delta.'),
            'End-to-end HDQ assigns zero to invalid output before comparison with Tier 0.',
            'HDQ is a look-ahead operational-label metric, not a causal policy-value claim.',
            'Shadow readiness does not authorize active policy deployment.',
        ],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reports', nargs='+', required=True)
    parser.add_argument('--quality', required=True)
    parser.add_argument('--min-games', type=int, default=12)
    parser.add_argument('--min-mode-rows', type=int, default=MIN_MODE_ROWS)
    parser.add_argument('--min-mode-games', type=int, default=None)
    parser.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    reports = [json.loads(Path(path).read_text(encoding='utf-8')) for path in args.reports]
    quality = json.loads(Path(args.quality).read_text(encoding='utf-8'))
    result = audit(reports, quality, min_games=args.min_games,
                   min_mode_rows=args.min_mode_rows, min_mode_games=args.min_mode_games)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'output': str(output.resolve()), 'gates': result['gates']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
