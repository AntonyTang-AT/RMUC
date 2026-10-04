import json
from pathlib import Path

from eval.build_evidence_lineage import build


def test_lineage_marks_older_summary_stale(tmp_path):
    root = tmp_path
    paths = [
        root / 'out_m2' / 'production_ensemble_eval' / 'report.json',
        root / 'out_m2' / 'production_ensemble_eval' / 'production_summary.json',
        root / 'out_m2' / 'enemy_region_transition' / 'production_manifest_10.json',
        root / 'out_m2' / 'deployment_readiness_audit.json',
        root / 'out_m2' / 'temporal_visible_context_dataset_v4_run1' /
        'training_dataset_audit_current.json',
    ]
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {'path': str(path)}
        if path.name == 'training_dataset_audit_current.json':
            payload['passed'] = True
        path.write_text(json.dumps(payload), encoding='utf-8')
    import os
    os.utime(paths[1], (paths[1].stat().st_atime, paths[1].stat().st_mtime - 10))
    result = build(root)
    statuses = {entry['role']: entry['status'] for entry in result['entries']}
    assert statuses['authoritative_unified_forecast_report'] == 'authoritative'
    assert statuses['selected_production_summary'] == 'stale'
    assert statuses['standalone_enemy_region_10s_sidecar'] == 'needs_unified_rerun'
    assert statuses['llm_training_dataset_audit'] == 'authoritative'
    assert result['training_state'] == 'not_inferred_from_static_artifacts'
