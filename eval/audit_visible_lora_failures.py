"""Summarize parse failures from visibility-safe LoRA evaluation rows.

This is diagnostics only: it identifies clusters and prompt-shape differences
without treating a parse failure as a tactical-policy correctness label.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _features(row):
    history = row.get('visible_history') or []
    latest = history[-1] if history else {}
    entities = list((latest.get('entities') or {}).values())
    current = [entity for entity in entities
               if entity.get('observability_state') == 'current']
    unknown = [entity for entity in entities
               if entity.get('observability_state') != 'current']
    forecasts = (row.get('forecast_context') or {}).get('predictions') or []
    return {
        'raw_output_chars': len(str(row.get('raw_output') or '')),
        'raw_output_has_json_start': int('{' in str(row.get('raw_output') or '')),
        'prompt_chars': len(str(row.get('prompt_text') or '')),
        'latest_entities': len(entities),
        'latest_current_entities': len(current),
        'latest_noncurrent_entities': len(unknown),
        'forecast_horizons': len(forecasts),
        'contract_valid': int(bool(row.get('contract_valid'))),
        'contract_repairs': len(row.get('contract_repairs') or []),
    }


def _mean(rows, key):
    return round(sum(float(row[key]) for row in rows) / len(rows), 4) if rows else None


def audit(rows):
    # New rows expose strict generation separately from runtime delivery.  Old
    # artifacts remain readable by treating ``parse_valid`` as both fields.
    failures = [row for row in rows if not bool(row.get(
        'strict_parse_valid', row.get('parse_valid')))]
    successes = [row for row in rows if bool(row.get(
        'strict_parse_valid', row.get('parse_valid')))]
    runtime_failures = [row for row in rows if not bool(row.get('parse_valid'))]
    repaired = [row for row in rows if bool(row.get('contract_repairs'))]
    fail_features = [_features(row) for row in failures]
    success_features = [_features(row) for row in successes]
    by_game = defaultdict(list)
    for row in failures:
        by_game[str(row.get('game_id'))].append(float(row.get('issued_at_s', 0.0)))
    feature_names = tuple(_features(rows[0]).keys()) if rows else ()
    return {
        'schema_version': 'visible-lora-failure-audit-v1',
        'rows': len(rows), 'parsed': len(successes), 'parse_failures': len(failures),
        'strict_parsed': len(successes), 'strict_parse_failures': len(failures),
        'runtime_delivery_failures': len(runtime_failures),
        'contract_repaired_rows': len(repaired),
        'failure_games': {
            game: {'count': len(times), 'issued_at_s': sorted(times)}
            for game, times in sorted(by_game.items())
        },
        'failure_reasons': dict(sorted(Counter(
            str(row.get('parse_error') or 'unspecified') for row in failures
        ).items())),
        'mean_features': {
            'failed': {key: _mean(fail_features, key) for key in feature_names},
            'parsed': {key: _mean(success_features, key) for key in feature_names},
        },
        'interpretation': (
            'Differences describe prompt-shape association only. Strict parsing and repaired '
            'runtime delivery are reported separately; this remains a structural diagnostic, '
            'not a tactical-policy correctness label.'),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rows', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    rows = [json.loads(line) for line in Path(args.rows).read_text(encoding='utf-8').splitlines()
            if line.strip()]
    result = audit(rows)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
