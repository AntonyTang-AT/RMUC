"""Re-score saved visible LLM rows under the current runtime parser.

This utility never regenerates a completion.  It preserves the original raw
text and records both the strict parser result and the narrow repaired runtime
result so historical candidates can be compared to a newer runtime contract.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from eval.evaluate_visible_lora_context import _score
from finetune.infer_one import extract_decision_with_error, extract_decision_with_repair


def _read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines()
            if line.strip()]


def reconcile(rows):
    """Return copied rows and a runtime-delivery structural summary."""
    updated, delivered, strict, rule_moves = [], [], [], []
    for row in rows:
        current = dict(row)
        raw_output = current.get('raw_output')
        strict_decision, parse_error = extract_decision_with_error(raw_output)
        contract_decision, repairs, contract_error = extract_decision_with_repair(raw_output)
        current.update({
            'schema_version': max(3, int(current.get('schema_version') or 0)),
            'decision': contract_decision,
            'strict_decision': strict_decision,
            'strict_parse_valid': strict_decision is not None,
            'parse_valid': contract_decision is not None,
            'parse_error': parse_error,
            'contract_decision': contract_decision,
            'contract_valid': contract_decision is not None,
            'contract_repairs': repairs,
            'contract_error': contract_error,
            'runtime_reconciliation': {
                'schema': 'runtime-contract-reconciliation-v1',
                'source': 'saved_raw_output_no_regeneration',
            },
        })
        updated.append(current)
        delivered.append(contract_decision)
        strict.append(strict_decision)
        rule_moves.append(current.get('rule_move'))

    runtime = _score(delivered, rule_moves)
    raw = _score(strict, rule_moves)
    runtime.update({
        'runtime_delivery_rate': runtime['parse_rate'],
        'raw_parsed': raw['parsed'],
        'raw_parse_rate': raw['parse_rate'],
        'raw_legal_rate': raw['legal_rate'],
        'contract_valid_rate': sum(bool(row['contract_valid']) for row in updated) /
        max(1, len(updated)),
        'contract_repair_rate': sum(bool(row['contract_repairs']) for row in updated) /
        max(1, len(updated)),
        'contract_repaired_rows': sum(bool(row['contract_repairs']) for row in updated),
        'unrecoverable_rows': sum(not bool(row['contract_valid']) for row in updated),
    })
    return updated, runtime


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rows', nargs='+', required=True)
    parser.add_argument('--reports', nargs='*', default=None,
                        help='Optional source reports, one per rows file, to refresh structurally.')
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args(argv)

    if args.reports is not None and len(args.reports) != len(args.rows):
        parser.error('--reports must contain one path for every --rows input')

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    manifest = {'schema_version': 'llm-runtime-row-reconciliation-v1',
                'source': 'saved raw completions; no model generation',
                'files': []}
    for index, source in enumerate(Path(path) for path in args.rows):
        rows, summary = reconcile(_read_jsonl(source))
        stem = source.name.replace('.rows.jsonl', '')
        row_path = output_dir / f'{stem}.rows.jsonl'
        report_path = output_dir / f'{stem}.runtime_delivery.json'
        row_path.write_text('\n'.join(
            json.dumps(row, ensure_ascii=False, separators=(',', ':')) for row in rows
        ) + ('\n' if rows else ''), encoding='utf-8')
        report_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
        item = {
            'source_rows': str(source.resolve()),
            'rows': str(row_path.resolve()),
            'summary': str(report_path.resolve()),
        }
        if args.reports is not None:
            source_report = Path(args.reports[index])
            report = json.loads(source_report.read_text(encoding='utf-8'))
            report['result'].update(summary)
            reconciled_report_path = output_dir / f'{stem}.report.json'
            reconciled_report_path.write_text(
                json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
            item['source_report'] = str(source_report.resolve())
            item['report'] = str(reconciled_report_path.resolve())
        manifest['files'].append(item)
    (output_dir / 'manifest.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'output_dir': str(output_dir.resolve()),
                      'files': len(manifest['files'])}, ensure_ascii=False))


if __name__ == '__main__':
    main()
