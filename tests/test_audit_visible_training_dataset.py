import json
import subprocess
import sys
from pathlib import Path

from eval.audit_visible_training_dataset import audit


def _row(game_id, t, side, variant):
    return {
        'instruction': 'visible',
        'completion': json.dumps({
            't': t, 'global_mode': '均衡',
            'units': {name: {'target': '无', 'move': '无', 'fire': False, 'risk': '低'}
                      for name in ('英雄', '工程', '步兵3', '步兵4', '空中', '哨兵')},
        }, ensure_ascii=False),
        'provenance': {
            'game_id': game_id, 'issued_at_s': t, 'perspective': side,
            'variant': variant, 'teacher_mode': '均衡',
            'forecast_context_sha256': 'abc' if variant == 'forecast' else None,
        },
    }


def test_audit_requires_each_side_time_pair_to_have_both_variants(tmp_path):
    dataset = tmp_path / 'records.jsonl'
    rows = [_row(10, 1.0, '红', 'forecast'), _row(10, 1.0, '红', 'no_forecast')]
    dataset.write_text('\n'.join(json.dumps(row, ensure_ascii=False) for row in rows), encoding='utf-8')
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps({'rows': 2}), encoding='utf-8')
    assert audit(dataset, manifest, heldout_games=())['passed']


def test_audit_rejects_unpaired_forecast_variant(tmp_path):
    dataset = tmp_path / 'records.jsonl'
    dataset.write_text(json.dumps(_row(10, 1.0, '蓝', 'forecast'), ensure_ascii=False), encoding='utf-8')
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps({'rows': 1}), encoding='utf-8')
    result = audit(dataset, manifest, heldout_games=())
    assert not result['passed']
    assert result['bad_variant_groups'] == 1


def test_audit_checks_additional_holdout_manifests(tmp_path):
    dataset = tmp_path / 'records.jsonl'
    rows = [_row(10, 1.0, '红', 'forecast'), _row(10, 1.0, '红', 'no_forecast')]
    dataset.write_text('\n'.join(json.dumps(row, ensure_ascii=False) for row in rows), encoding='utf-8')
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps({'rows': 2}), encoding='utf-8')
    heldout = tmp_path / 'reliability_holdout.json'
    heldout.write_text(json.dumps({'game_ids': [10, 11, 12]}), encoding='utf-8')

    result = audit(dataset, manifest, heldout_games=(), heldout_manifests=(heldout,))

    assert not result['passed']
    assert result['heldout_overlap'] == [10]
    assert result['heldout_games_checked'] == 3
    assert result['heldout_manifests'] == [{'path': str(heldout), 'games': 3}]


def test_v5_audit_rejects_nonconservative_completion_for_unseen_slot(tmp_path):
    dataset = tmp_path / 'records.jsonl'
    rows = [_row(10, 1.0, '红', 'forecast'), _row(10, 1.0, '红', 'no_forecast')]
    for row in rows:
        row['provenance']['slot_statuses'] = {
            '英雄': 'unseen', '工程': 'current', '步兵3': 'current',
            '步兵4': 'current', '空中': 'current', '哨兵': 'current',
        }
    dataset.write_text('\n'.join(json.dumps(row, ensure_ascii=False) for row in rows), encoding='utf-8')
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps({
        'rows': 2, 'schema_version': 'visible-temporal-context-dataset-v5-observable-slots'},
        ensure_ascii=False), encoding='utf-8')

    result = audit(dataset, manifest, heldout_games=())

    assert not result['passed']
    assert result['bad_observable_slot_contract'] == 2


def test_audit_script_runs_directly(tmp_path):
    dataset = tmp_path / 'records.jsonl'
    rows = [_row(10, 1.0, '红', 'forecast'), _row(10, 1.0, '红', 'no_forecast')]
    dataset.write_text('\n'.join(json.dumps(row, ensure_ascii=False) for row in rows), encoding='utf-8')
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps({'rows': 2}), encoding='utf-8')
    output = tmp_path / 'audit.json'
    root = Path(__file__).resolve().parents[1]

    completed = subprocess.run(
        [sys.executable, str(root / 'eval' / 'audit_visible_training_dataset.py'),
         '--dataset', str(dataset), '--manifest', str(manifest), '--output', str(output)],
        cwd=root, capture_output=True, text=True, check=False)

    assert completed.returncode == 0, completed.stderr
    assert json.loads(output.read_text(encoding='utf-8'))['passed']
