"""Build a compact index of authoritative and stale evaluation artifacts.

This is intentionally static: it reads existing manifests and reports without
rerunning training or model inference.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _latest(root: Path, pattern: str, fallback: Path) -> Path:
    candidates = [path for path in root.glob(pattern) if path.is_file()]
    return max(candidates, key=lambda path: path.stat().st_mtime) if candidates else fallback


def _file_entry(path: Path, *, role: str, status: str, reason: str) -> dict:
    path = path.resolve()
    stat = path.stat()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        'path': str(path),
        'role': role,
        'status': status,
        'reason': reason,
        'size_bytes': int(stat.st_size),
        'modified_at_utc': datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
        'sha256': digest,
    }


def build(root: Path = ROOT) -> dict:
    root = Path(root).resolve()
    report_dir = root / 'out_m2' / 'production_ensemble_eval'
    unified = _latest(report_dir, 'report*.json', report_dir / 'report.json')
    summary = _latest(report_dir, 'production_summary*.json',
                      report_dir / 'production_summary.json')
    transition10 = root / 'out_m2' / 'enemy_region_transition' / 'production_manifest_10.json'
    readiness = _latest(root / 'out_m2', 'deployment_readiness_audit*.json',
                         root / 'out_m2' / 'deployment_readiness_audit.json')
    metric_validity = root / 'out_m2' / 'decision_metric_validity_audit.json'
    training_dataset_audit = (root / 'out_m2' / 'temporal_visible_context_dataset_v4_run1' /
                              'training_dataset_audit_current.json')
    entries = []
    if unified.exists():
        entries.append(_file_entry(unified, role='authoritative_unified_forecast_report',
                                   status='authoritative',
                                   reason='Locked-test report for the exact evaluated runtime route.'))
    if summary.exists():
        status = 'stale' if unified.exists() and summary.stat().st_mtime < unified.stat().st_mtime else 'current'
        entries.append(_file_entry(
            summary, role='selected_production_summary', status=status,
            reason=('Summary aligned to the selected unified report.' if status == 'current' else
                    'Summary predates the selected unified report and is retained as stale provenance.')))
    if transition10.exists():
        included = unified.exists() and unified.stat().st_mtime >= transition10.stat().st_mtime
        entries.append(_file_entry(transition10, role='standalone_enemy_region_10s_sidecar',
                                   status=('included_in_unified_report' if included else 'needs_unified_rerun'),
                                   reason=('Provenance for the selected unified report.' if included else
                                           'Newer sidecar evidence is not yet folded into the unified report.')))
    if readiness.exists():
        entries.append(_file_entry(readiness, role='deployment_readiness_matrix',
                                   status='authoritative',
                                   reason='Static gate summary; it does not rerun models.'))
    if metric_validity.exists():
        entries.append(_file_entry(metric_validity, role='decision_metric_validity_audit',
                                   status='authoritative',
                                   reason='Holdout coverage and label-stability evidence for HDQ.'))
    if training_dataset_audit.exists():
        audit = json.loads(training_dataset_audit.read_text(encoding='utf-8'))
        passed = bool(audit.get('passed'))
        entries.append(_file_entry(
            training_dataset_audit, role='llm_training_dataset_audit',
            status=('authoritative' if passed else 'failed'),
            reason=('Visible-context completions, paired variants, and holdout isolation passed.'
                    if passed else
                    'Visible-context training-data audit has one or more failed checks.')))
    return {
        'schema_version': 'evidence-lineage-v1',
        'generated_at_utc': datetime.now(timezone.utc).isoformat(),
        'training_state': 'not_inferred_from_static_artifacts',
        'entries': entries,
        'selection_policy': {
            'forecast_metrics': 'authoritative_unified_forecast_report',
            'deployment_gate': 'deployment_readiness_matrix',
            'decision_metric_validity': 'decision_metric_validity_audit',
            'llm_training_data': 'llm_training_dataset_audit',
            'sidecar_promotion': ('included_in_unified_report' if unified.exists() and transition10.exists()
                                  and unified.stat().st_mtime >= transition10.stat().st_mtime
                                  else 'requires_unified_rerun'),
        },
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default=str(ROOT))
    parser.add_argument('--output', default='out_m2/evidence_lineage.json')
    args = parser.parse_args(argv)
    result = build(Path(args.root))
    output = Path(args.output)
    if not output.is_absolute():
        output = Path(args.root) / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'output': str(output.resolve()), 'entries': len(result['entries'])}, ensure_ascii=False))


if __name__ == '__main__':
    main()
