"""Create a compact acceptance summary from a production-ensemble report."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _head_summary(production, persistence, paired_game_bootstrap=None):
    accuracy_delta = float(production['accuracy']) - float(persistence['accuracy'])
    macro_f1_delta = float(production['macro_f1']) - float(persistence['macro_f1'])
    brier_delta = float(production['brier']) - float(persistence['brier'])
    result = {
        'production_accuracy': float(production['accuracy']),
        'baseline_accuracy': float(persistence['accuracy']),
        'accuracy_delta_pp': 100.0 * accuracy_delta,
        'production_macro_f1': float(production['macro_f1']),
        'baseline_macro_f1': float(persistence['macro_f1']),
        'macro_f1_delta_pp': 100.0 * macro_f1_delta,
        'production_brier': float(production['brier']),
        'baseline_brier': float(persistence['brier']),
        'double_gate': bool(accuracy_delta > 0.0 and macro_f1_delta > 0.0 and
                            brier_delta < 0.0),
    }
    paired_metrics = (paired_game_bootstrap or {}).get('metrics') or {}
    if paired_metrics:
        result['paired_game_bootstrap'] = paired_game_bootstrap
        result['confidence_confirmed_gate'] = bool(all(
            bool((paired_metrics.get(metric) or {}).get(
                'ci_excludes_zero_in_favor_of_production'))
            for metric in ('accuracy', 'macro_f1', 'brier')))
    else:
        result['confidence_confirmed_gate'] = None
    return result


def summarize(report):
    global_summary = {}
    for task, horizons in (report.get('global') or {}).items():
        global_summary[task] = {
            str(horizon): _head_summary(
                values['production'], values['persistence'],
                values.get('paired_game_bootstrap'))
            for horizon, values in horizons.items()
        }
    entity_summary = {}
    for horizon, values in (report.get('entity') or {}).items():
        entity_summary[str(horizon)] = {
            'presence_accuracy': values['presence']['accuracy'],
            'death_f1': values['death']['production']['f1'],
            'death_baseline_f1': values['death']['persistence']['f1'],
            'position_mae': values['position']['mae_normalized'],
            'position_baseline_mae': values['position']['baseline_mae_normalized'],
            'functional_exact': values['functional_zone']['exact_accuracy'],
            'functional_baseline': values['functional_zone']['baseline_exact_accuracy'],
            'canonical_exact': values['canonical_zone']['exact_accuracy'],
            'canonical_baseline': values['canonical_zone']['baseline_exact_accuracy'],
            'hp_mae': values['hp']['mae'],
            'hp_baseline_mae': values['hp']['persistence_mae'],
            'position_sidecar_enabled': values['position']['position_sidecar_enabled'],
        }
    region_marginals = {}
    for axis, horizons in (report.get('enemy_region_marginals') or {}).items():
        region_marginals[axis] = {}
        for horizon, values in horizons.items():
            production = values['production']
            persistence = values['persistence']
            paired = values.get('paired_game_bootstrap') or {}
            paired_metrics = paired.get('metrics') or {}
            region_marginals[axis][str(horizon)] = {
                'class_labels': values.get('class_labels'),
                'production_accuracy': production.get('accuracy'),
                'baseline_accuracy': persistence.get('accuracy'),
                'production_macro_f1': production.get('macro_f1'),
                'baseline_macro_f1': persistence.get('macro_f1'),
                'production_brier': production.get('brier'),
                'baseline_brier': persistence.get('brier'),
                'paired_game_bootstrap': paired,
                'confidence_confirmed_gate': bool(paired_metrics and all(
                    bool((paired_metrics.get(metric) or {}).get(
                        'ci_excludes_zero_in_favor_of_production'))
                    for metric in ('accuracy', 'macro_f1', 'brier'))),
            }
    return {
        'schema': 'production-ensemble-summary-v1',
        'source_report_schema': report.get('schema_version', report.get('schema')),
        'locked_test_games': int(report['locked_test_games']),
        'rows': int(report['rows']),
        'runtime_policy': report.get('runtime_policy'),
        'global': global_summary,
        'enemy_region_marginals': region_marginals,
        'entity': entity_summary,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    report = json.loads(Path(args.report).read_text(encoding='utf-8'))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summarize(report), ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'output': str(output.resolve())}, ensure_ascii=False))


if __name__ == '__main__':
    main()
