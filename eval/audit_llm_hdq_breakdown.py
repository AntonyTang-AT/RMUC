"""Break down delivered LLM HDQ errors by scenario, mode, unit, and field."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from schema import layered as sl


def _read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines()
            if line.strip()]


def _scenario(row):
    forecast = bool((row.get('run_metadata') or {}).get('forecast_context_enabled'))
    return f"{row.get('perspective')}|forecast={str(forecast).lower()}"


def _ratio(counter):
    return counter['correct'] / counter['scored'] if counter['scored'] else None


def _finalize_group(group):
    return {
        'rows': int(group['rows']),
        'runtime_delivered': int(group['runtime_delivered']),
        'strict_generated': int(group['strict_generated']),
        'repaired_rows': int(group['repaired_rows']),
        'mode_accuracy': _ratio(group['mode']),
        'mode_confusion_expected_rows_predicted_columns': [
            [int(group['mode_confusion'][(expected, predicted)])
             for predicted in sl.GLOBAL_MODES]
            for expected in sl.GLOBAL_MODES
        ],
        'unit_field_accuracy': {
            unit: {field: _ratio(values[field]) for field in ('target', 'move', 'fire', 'risk')}
            for unit, values in sorted(group['unit_fields'].items())
        },
        'candidate_minus_rule': {
            'mode_correct_rows': int(group['candidate_minus_rule']['mode_correct_rows']),
            'mode_wrong_rows': int(group['candidate_minus_rule']['mode_wrong_rows']),
            'unit_field_better': int(group['candidate_minus_rule']['unit_field_better']),
            'unit_field_worse': int(group['candidate_minus_rule']['unit_field_worse']),
        },
    }


def audit(rows, *, lookahead_delta=5):
    from eval import decision_quality_scorecard as quality

    contexts = quality._world_contexts(rows, lookahead_delta)
    groups = defaultdict(lambda: {
        'rows': 0,
        'runtime_delivered': 0,
        'strict_generated': 0,
        'repaired_rows': 0,
        'mode': Counter(),
        'mode_confusion': Counter(),
        'unit_fields': defaultdict(lambda: defaultdict(Counter)),
        'candidate_minus_rule': Counter(),
    })
    for row in rows:
        group = groups[_scenario(row)]
        group['rows'] += 1
        group['runtime_delivered'] += int(bool(row.get('parse_valid')))
        group['strict_generated'] += int(bool(row.get(
            'strict_parse_valid', row.get('parse_valid'))))
        group['repaired_rows'] += int(bool(row.get('contract_repairs')))
        if not bool(row.get('parse_valid')):
            continue
        game_id = int(row['game_id'])
        side = row.get('perspective')
        context = contexts.get(game_id, {}).get(round(float(row['issued_at_s']), 3))
        if context is None or side not in ('红', '蓝'):
            continue
        world, hindsight_modes, gm, _stability = context
        expected_mode = hindsight_modes[side]
        baseline_mode = gm.derive_global_mode(world)
        decision = row.get('decision') or {}
        predicted_mode = decision.get('global_mode')
        if predicted_mode not in sl.GLOBAL_MODES:
            continue
        group['mode']['scored'] += 1
        group['mode']['correct'] += int(predicted_mode == expected_mode)
        group['mode_confusion'][(expected_mode, predicted_mode)] += 1
        group['candidate_minus_rule']['mode_correct_rows'] += int(
            predicted_mode == expected_mode and baseline_mode != expected_mode)
        group['candidate_minus_rule']['mode_wrong_rows'] += int(
            predicted_mode != expected_mode and baseline_mode == expected_mode)
        expected = quality._decision_for_baseline(world, expected_mode, side)
        baseline = quality._decision_for_baseline(world, baseline_mode, side)
        for unit, values in quality._units_by_type(world, side).items():
            if len(values) != 1:
                continue
            actual = (decision.get('units') or {}).get(unit)
            if not isinstance(actual, dict):
                continue
            target = expected['units'][unit]
            rule = baseline['units'][unit]
            for field in ('target', 'move', 'fire', 'risk'):
                stats = group['unit_fields'][unit][field]
                stats['scored'] += 1
                stats['correct'] += int(actual.get(field) == target.get(field))
                group['candidate_minus_rule']['unit_field_better'] += int(
                    actual.get(field) == target.get(field) and rule.get(field) != target.get(field))
                group['candidate_minus_rule']['unit_field_worse'] += int(
                    actual.get(field) != target.get(field) and rule.get(field) == target.get(field))

    return {
        'schema_version': 'llm-hdq-breakdown-v1',
        'lookahead_delta': int(lookahead_delta),
        'source_contract': ('runtime-delivered decision; strict generation and contract repairs '
                            'are reported separately'),
        'scenarios': {key: _finalize_group(value) for key, value in sorted(groups.items())},
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rows', nargs='+', required=True)
    parser.add_argument('--lookahead-delta', type=int, default=5)
    parser.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    rows = []
    for path in args.rows:
        rows.extend(_read_jsonl(path))
    result = audit(rows, lookahead_delta=args.lookahead_delta)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'output': str(output.resolve()),
                      'scenarios': list(result['scenarios'])}, ensure_ascii=False))


if __name__ == '__main__':
    main()
