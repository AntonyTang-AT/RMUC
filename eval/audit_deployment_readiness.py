"""Create a low-cost evidence matrix for forecast and LLM deployment readiness.

The project has several reports written at different times.  This auditor
never re-evaluates a model; it records which evidence is unified, which is a
newer standalone promotion artifact, and which older summaries are stale.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HORIZONS = ('10', '30', '60', '120')


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _mtime(path):
    value = Path(path).stat().st_mtime
    return datetime.fromtimestamp(value, tz=timezone.utc).isoformat()


def classify_head(production, persistence, paired_game_bootstrap=None):
    """Require directional improvement in accuracy, macro-F1, and Brier."""
    accuracy_delta = float(production['accuracy']) - float(persistence['accuracy'])
    f1_delta = float(production['macro_f1']) - float(persistence['macro_f1'])
    brier_delta = float(production['brier']) - float(persistence['brier'])
    result = {
        'accuracy_delta_pp': round(100.0 * accuracy_delta, 6),
        'macro_f1_delta_pp': round(100.0 * f1_delta, 6),
        'brier_delta': round(brier_delta, 9),
        'strictly_beats_persistence': bool(
            accuracy_delta > 0.0 and f1_delta > 0.0 and brier_delta < 0.0),
    }
    metrics = (paired_game_bootstrap or {}).get('metrics') or {}
    if metrics:
        result['confidence_status'] = ('confirmed' if all(
            bool((metrics.get(metric) or {}).get('ci_excludes_zero_in_favor_of_production'))
            for metric in ('accuracy', 'macro_f1', 'brier')) else 'directional_only')
        result['paired_game_bootstrap'] = paired_game_bootstrap
    else:
        result['confidence_status'] = 'not_available'
    return result


def _llm_status(report):
    result = report['result']
    parsed_all = int(result['parsed']) == int(result['n'])
    legal_all = parsed_all and float(result['legal_rate']) == 1.0
    return {
        'side': result['side'],
        'forecast_context_enabled': bool(result['forecast_context_enabled']),
        'n': int(result['n']),
        'parse_rate': float(result['parse_rate']),
        'legal_rate_conditional_on_parse': float(result['legal_rate']),
        'legacy_teacher_agreement': float(result['agreement_with_legacy_rule_teacher']),
        'conflict_rate': float(result['conflict_rate']),
        'visible_rows_validated': int(result['visible_rows_validated']),
        'structural_gate': 'pass' if legal_all else 'fail',
        'deployment_status': 'not_ready',
        'reason': ('No execution or causal policy-value ground truth is available; '
                   'legacy-rule agreement is comparison-only.'),
    }


def _llm_status_from_acceptance(acceptance):
    """Project the authoritative candidate acceptance into audit table rows."""
    rows = []
    for key, scenario in sorted((acceptance.get('scenarios') or {}).items()):
        rows.append({
            'side': scenario.get('side'),
            'forecast_context_enabled': bool(scenario.get('forecast_context_enabled')),
            'n': int(scenario.get('n') or 0),
            'parse_rate': float(scenario.get('parse_rate') or 0.0),
            'legal_rate_conditional_on_parse': float(
                scenario.get('legal_rate_conditional_on_parse') or 0.0),
            'legacy_teacher_agreement': None,
            'conflict_rate': None,
            'visible_rows_validated': int(scenario.get('visible_rows_validated') or 0),
            'structural_gate': 'pass' if scenario.get('structural_pass') else 'fail',
            'deployment_status': 'not_ready',
            'reason': ('Candidate acceptance records runtime delivery, strict generation, '
                       'repair budget, and HDQ gates.'),
        })
    return rows


def _llm_gate(llm_rows, acceptance=None):
    """Non-negotiable structural coverage, separate from policy usefulness.

    We deliberately do not turn legacy-rule agreement or conflict rate into a
    policy-quality pass/fail gate.  Neither is an execution or causal label.
    """
    expected = {(side, forecast) for side in ('红', '蓝') for forecast in (True, False)}
    observed = {(row['side'], row['forecast_context_enabled']) for row in llm_rows}
    missing = sorted(expected - observed, key=lambda value: (value[0], value[1]))
    structural_failures = [
        (row['side'], row['forecast_context_enabled'])
        for row in llm_rows if row['structural_gate'] != 'pass'
    ]
    gate = {
        'source': 'legacy_scenario_reports',
        'required_scenarios': [
            {'side': side, 'forecast_context_enabled': forecast}
            for side, forecast in sorted(expected, key=lambda value: (value[0], value[1]))
        ],
        'missing_scenarios': [
            {'side': side, 'forecast_context_enabled': forecast}
            for side, forecast in missing
        ],
        'structural_failures': [
            {'side': side, 'forecast_context_enabled': forecast}
            for side, forecast in structural_failures
        ],
        'structural_robustness_gate': ('pass' if not missing and not structural_failures
                                       else 'fail'),
        'policy_value_gate': 'unverifiable_without_execution_or_counterfactual_labels',
    }
    if acceptance is None:
        return gate

    gates = acceptance.get('gates') or {}
    gate.update({
        'source': 'llm_candidate_acceptance',
        'structural_robustness_gate': gates.get('structural_reliability', 'fail'),
        'strict_generation_gate': gates.get('strict_generation_reliability', 'fail'),
        'semantic_sample_size_gate': gates.get('semantic_sample_size', 'insufficient'),
        'semantic_mode_coverage_gate': gates.get('semantic_mode_coverage', 'insufficient'),
        'semantic_hdq_by_global_mode_gate': gates.get(
            'semantic_hdq_by_global_mode_vs_legacy_rule', 'fail'),
        'semantic_hdq_vs_legacy_rule_gate': gates.get('semantic_hdq_vs_legacy_rule', 'fail'),
        'shadow_ready': bool(gates.get('shadow_ready')),
    })
    return gate


def audit(report_path, summary_path, transition10_path, llm_reports=(),
          llm_acceptance_path=None):
    report = _read(report_path)
    summary = _read(summary_path)
    transition10 = _read(transition10_path)
    report_time = _mtime(report_path)
    summary_time = _mtime(summary_path)
    rows = []
    for task, horizons in report['global'].items():
        for horizon in HORIZONS:
            production = horizons[horizon]['production']
            persistence = horizons[horizon]['persistence']
            result = classify_head(production, persistence,
                                   horizons[horizon].get('paired_game_bootstrap'))
            rows.append({
                'kind': 'forecast_head', 'task': task, 'horizon_seconds': int(horizon),
                'evidence': 'unified_locked_test_report', 'evidence_time_utc': report_time,
                'status': 'pass' if result['strictly_beats_persistence'] else 'fail',
                **result,
            })

    # Keep the standalone sidecar visible until a newer unified report includes
    # it.  Once the report is newer, the sidecar is provenance only and must
    # not continue to block the unified forecast gate.
    evidence = transition10['promotion_evidence']
    candidate = evidence['candidate']['locked_test']
    persistence = evidence['persistence']['locked_test']
    transition_is_in_unified_report = Path(report_path).stat().st_mtime >= Path(transition10_path).stat().st_mtime
    rows.append({
        'kind': 'forecast_head', 'task': 'enemy_region', 'horizon_seconds': 10,
        'evidence': 'standalone_transition_promotion_artifact',
        'evidence_time_utc': _mtime(transition10_path),
        'status': ('superseded_by_unified_report' if transition_is_in_unified_report
                   else 'needs_unified_rerun'),
        'accuracy_delta_pp': round(100.0 * (candidate['accuracy'] - persistence['accuracy']), 6),
        'macro_f1_delta_pp': round(100.0 * (candidate['macro_f1'] - persistence['macro_f1']), 6),
        'strictly_beats_persistence': bool(
            candidate['accuracy'] > persistence['accuracy'] and
            candidate['macro_f1'] > persistence['macro_f1']),
        'reason': ('Included in the selected unified report.' if transition_is_in_unified_report
                   else 'Artifact is newer than the unified report; rerun the full runtime evaluator.'),
    })

    acceptance = None
    if llm_acceptance_path is not None and Path(llm_acceptance_path).exists():
        acceptance = _read(llm_acceptance_path)
    llm = (_llm_status_from_acceptance(acceptance) if acceptance is not None else
           [_llm_status(_read(path)) for path in llm_reports])
    llm_gate = _llm_gate(llm, acceptance=acceptance)
    stale_summary = Path(summary_path).stat().st_mtime < Path(report_path).stat().st_mtime
    unified_failures = [row for row in rows
                        if row['evidence'] == 'unified_locked_test_report' and row['status'] != 'pass']
    blocking_reasons = []
    if not transition_is_in_unified_report:
        blocking_reasons.append(
            'A newer 10-second enemy-region sidecar lacks a unified runtime re-evaluation.')
    if acceptance is None:
        blocking_reasons.append(
            'No current LLM candidate acceptance artifact is available; legacy scenario reports are diagnostic only.')
    else:
        if llm_gate['structural_robustness_gate'] != 'pass':
            blocking_reasons.append('LLM candidate fails the runtime structural reliability gate.')
        if llm_gate.get('semantic_hdq_vs_legacy_rule_gate') != 'pass':
            blocking_reasons.append('LLM candidate lacks a positive, mode-covered HDQ result versus the rule baseline.')
    blocking_reasons.append(
        'No controller execution or causal policy-value labels support a deployment claim.')
    return {
        'schema_version': 'deployment-readiness-audit-v1',
        'forecast': {
            'unified_report': str(Path(report_path)),
            'unified_report_time_utc': report_time,
            'older_summary': str(Path(summary_path)),
            'older_summary_time_utc': summary_time,
            'older_summary_is_stale': bool(stale_summary),
            'heads': rows,
            'unified_failures': [(row['task'], row['horizon_seconds']) for row in unified_failures],
        },
        'llm': llm,
        'llm_gate': llm_gate,
        'llm_acceptance': (str(Path(llm_acceptance_path)) if acceptance is not None else None),
        'overall_status': 'not_ready',
        'overall_reasons': blocking_reasons,
    }


def _markdown(result):
    lines = ['# Deployment Readiness Audit', '',
             f"Overall status: **{result['overall_status']}**", '',
             '## Forecast Heads', '',
             '| Task | Horizon | Evidence | Directional status | Confidence status | Accuracy delta (pp) | Macro-F1 delta (pp) |',
             '| --- | ---: | --- | --- | --- | ---: | ---: |']
    for row in result['forecast']['heads']:
        lines.append(
            f"| {row['task']} | {row['horizon_seconds']} | {row['evidence']} | {row['status']} | "
            f"{row.get('confidence_status', 'not_applicable')} | "
            f"{row.get('accuracy_delta_pp', float('nan')):.3f} | "
            f"{row.get('macro_f1_delta_pp', float('nan')):.3f} |")
    lines.extend(['', '## LLM Structural Evaluation', '',
                  '| Side | Forecast context | Parse rate | Conditional legal rate | Status |',
                  '| --- | --- | ---: | ---: | --- |'])
    for row in result['llm']:
        lines.append(
            f"| {row['side']} | {row['forecast_context_enabled']} | {row['parse_rate']:.3f} | "
            f"{row['legal_rate_conditional_on_parse']:.3f} | {row['deployment_status']} |")
    gate = result['llm_gate']
    lines.extend(['',
                  f"Structural robustness gate: **{gate['structural_robustness_gate']}**",
                  f"LLM evidence source: **{gate['source']}**",
                  f"Policy-value gate: **{gate['policy_value_gate']}**", ''])
    if gate['missing_scenarios']:
        lines.append('Missing scenarios: ' + ', '.join(
            f"{item['side']}/forecast={item['forecast_context_enabled']}"
            for item in gate['missing_scenarios']))
    if gate['structural_failures']:
        lines.append('Structural failures: ' + ', '.join(
            f"{item['side']}/forecast={item['forecast_context_enabled']}"
            for item in gate['structural_failures']))
    lines.extend(['', '## Blocking Conditions', ''])
    lines.extend(f'- {reason}' for reason in result['overall_reasons'])
    return '\n'.join(lines) + '\n'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', default='out_m2/production_ensemble_eval/report_current_transition10.json')
    parser.add_argument('--summary',
                        default='out_m2/production_ensemble_eval/production_summary_current_transition10.json')
    parser.add_argument('--transition10', default='out_m2/enemy_region_transition/production_manifest_10.json')
    parser.add_argument('--llm-reports', nargs='*', default=())
    parser.add_argument('--llm-acceptance',
                        default='out_m2/llm_v4_completion_only_run1_reliability/acceptance.json')
    parser.add_argument('--output', default='out_m2/deployment_readiness_audit_current.json')
    parser.add_argument('--markdown-output', default='out_m2/deployment_readiness_audit_current.md')
    args = parser.parse_args(argv)
    acceptance_path = ROOT / args.llm_acceptance if args.llm_acceptance else None
    result = audit(args.report, args.summary, args.transition10, args.llm_reports,
                   llm_acceptance_path=acceptance_path)
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    (ROOT / args.markdown_output).write_text(_markdown(result), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
