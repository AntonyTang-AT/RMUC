"""Build a local acceptance bundle without model weights or raw replay data."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIRS = (
    'data', 'eval', 'fastloop', 'finetune', 'forecast', 'game', 'labels',
    'model', 'rl', 'runtime', 'schema', 'scripts', 'tests', 'tree', 'worldstate',
)
DOCUMENT_DIRS = ('docs', '项目要求', '数据与资料')
ROOT_SUFFIXES = {'.py', '.md', '.txt'}
EVIDENCE_SUFFIXES = {'.json', '.md', '.html', '.png', '.csv'}
WEIGHT_SUFFIXES = {'.pt', '.pth', '.bin', '.safetensors', '.ckpt', '.npz', '.pkl'}
RAW_DATABASE_SUFFIXES = {'.db', '.sqlite', '.sqlite3'}
EXCLUDED_PARTS = {
    '.git', '__pycache__', '.venv_openjev', '.venv_openjev_cuda', '_work',
    'models_hf', 'third_party', 'ckpt', 'checkpoint-1600', 'checkpoint-1659',
}


def _excluded(path: Path) -> bool:
    parts = {part.lower() for part in path.parts}
    if parts.intersection(EXCLUDED_PARTS):
        return True
    if any(part.startswith('checkpoint-') for part in path.parts):
        return True
    if any(part.startswith('finetuned') for part in path.parts):
        return True
    return path.suffix.lower() in WEIGHT_SUFFIXES | RAW_DATABASE_SUFFIXES


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def collect_files(root: Path) -> list[Path]:
    selected: set[Path] = set()
    for directory in SOURCE_DIRS:
        source = root / directory
        if not source.exists():
            continue
        for path in source.rglob('*'):
            if path.is_file() and not _excluded(path.relative_to(root)):
                selected.add(path)
    for path in root.iterdir():
        if path.is_file() and path.name != '.env' and path.suffix.lower() in ROOT_SUFFIXES:
            selected.add(path)
    evidence = root / 'out_m2'
    if evidence.exists():
        for path in evidence.rglob('*'):
            relative = path.relative_to(root)
            if (path.is_file() and path.suffix.lower() in EVIDENCE_SUFFIXES and
                    not _excluded(relative)):
                selected.add(path)
    for directory in DOCUMENT_DIRS:
        docs = root / directory
        if not docs.exists():
            continue
        for path in docs.rglob('*'):
            if path.is_file() and path.suffix.lower() in EVIDENCE_SUFFIXES:
                selected.add(path)
    return sorted(selected, key=lambda item: str(item.relative_to(root)).lower())


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8'))


def _llm_readiness_failures(gates: dict) -> list[str]:
    failures = []
    if gates.get('structural_reliability') != 'pass':
        failures.append('LLM runtime structural reliability gate is not pass')
    if gates.get('semantic_mode_coverage') != 'pass':
        failures.append('LLM semantic mode-coverage gate is not pass')
    if gates.get('semantic_hdq_by_global_mode_vs_legacy_rule') != 'pass':
        failures.append('LLM semantic per-mode HDQ gate is not pass')
    if gates.get('semantic_hdq_vs_legacy_rule') != 'pass':
        failures.append('LLM semantic HDQ gate is not pass')
    return failures


def _metric_validity_failures(report: dict) -> list[str]:
    failures = []
    for side in ('红', '蓝'):
        summary = (report.get('by_side') or {}).get(side) or {}
        if not summary.get('mode_row_coverage_pass'):
            failures.append(f'{side} HDQ holdout lacks per-mode row coverage')
        if not summary.get('mode_game_coverage_pass'):
            failures.append(f'{side} HDQ holdout lacks per-mode game coverage')
    return failures


def _training_dataset_audit_failures(report: dict) -> list[str]:
    if not report.get('passed'):
        return ['LLM training dataset audit is not pass']
    return []


def verify_readiness(root: Path, llm_acceptance: Path | None) -> list[str]:
    failures = []
    deployment = root / 'out_m2' / 'deployment_readiness_audit_current.json'
    if not deployment.exists():
        failures.append(f'missing deployment audit: {deployment.relative_to(root)}')
    else:
        report = _read_json(deployment)
        if report.get('forecast', {}).get('unified_failures'):
            failures.append('forecast unified gate still has failures')
    metric_validity = root / 'out_m2' / 'decision_metric_validity_audit.json'
    if not metric_validity.exists():
        failures.append(f'missing decision-metric validity audit: {metric_validity.relative_to(root)}')
    else:
        failures.extend(_metric_validity_failures(_read_json(metric_validity)))
    training_dataset_audit = (root / 'out_m2' / 'temporal_visible_context_dataset_v4_run1' /
                              'training_dataset_audit_current.json')
    if not training_dataset_audit.exists():
        failures.append(f'missing LLM training dataset audit: '
                        f'{training_dataset_audit.relative_to(root)}')
    else:
        failures.extend(_training_dataset_audit_failures(_read_json(training_dataset_audit)))
    if llm_acceptance is None or not llm_acceptance.exists():
        failures.append('missing LLM acceptance artifact')
    else:
        gates = _read_json(llm_acceptance).get('gates') or {}
        failures.extend(_llm_readiness_failures(gates))
    return failures


def build(output: Path, *, llm_acceptance: Path | None, require_ready: bool) -> dict:
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f'refusing to overwrite existing package path: {output}')
    readiness_failures = verify_readiness(ROOT, llm_acceptance)
    if require_ready and readiness_failures:
        raise ValueError('readiness gate failed: ' + '; '.join(readiness_failures))

    files = collect_files(ROOT)
    package_root = output / 'rmuc_m2_acceptance'
    package_root.mkdir(parents=True)
    entries = []
    for source in files:
        relative = source.relative_to(ROOT)
        destination = package_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        entries.append({
            'path': str(relative).replace('\\', '/'),
            'bytes': source.stat().st_size,
            'sha256': _sha256(source),
        })
    manifest = {
        'schema_version': 'rmuc-m2-acceptance-package-v1',
        'purpose': 'local acceptance review bundle; no model weights or raw replay database copied',
        'readiness_failures_at_build': readiness_failures,
        'local_prerequisites': {
            'replay_database': '数据与资料/rmuc_2026_region_dataset.sqlite',
            'model_weights': 'remain in the original local out_m2 model directories',
            'python_requirements': 'requirements.txt',
        },
        'exclusions': {
            'weight_suffixes': sorted(WEIGHT_SUFFIXES),
            'raw_database_suffixes': sorted(RAW_DATABASE_SUFFIXES),
            'excluded_paths': sorted(EXCLUDED_PARTS),
            'raw_replay_database_copied': False,
            'environment_file_copied': False,
        },
        'files': entries,
    }
    manifest_path = package_root / 'ACCEPTANCE_MANIFEST.json'
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    readme = package_root / 'README_ACCEPTANCE.md'
    readme.write_text(
        '# RMUC M2 Acceptance Bundle\n\n'
        'This bundle contains source, tests, documentation, and generated evaluation evidence. '
        'Model weights and the raw replay SQLite database remain on the original local machine. '
        'See `ACCEPTANCE_MANIFEST.json` for checksums, exclusions, and local prerequisites.\n',
        encoding='utf-8')
    zip_path = output.with_suffix('.zip')
    with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for path in package_root.rglob('*'):
            if path.is_file():
                archive.write(path, path.relative_to(output))
    return {
        'package_root': str(package_root),
        'zip_path': str(zip_path),
        'file_count': len(entries),
        'readiness_failures_at_build': readiness_failures,
    }


def plan(*, llm_acceptance: Path | None, require_ready: bool) -> dict:
    readiness_failures = verify_readiness(ROOT, llm_acceptance)
    if require_ready and readiness_failures:
        raise ValueError('readiness gate failed: ' + '; '.join(readiness_failures))
    files = collect_files(ROOT)
    return {
        'file_count': len(files),
        'total_bytes': sum(path.stat().st_size for path in files),
        'readiness_failures_at_build': readiness_failures,
        'weight_files_selected': [str(path.relative_to(ROOT)) for path in files
                                  if path.suffix.lower() in WEIGHT_SUFFIXES],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--llm-acceptance', default=None)
    parser.add_argument('--allow-incomplete', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    kwargs = {
        'llm_acceptance': Path(args.llm_acceptance) if args.llm_acceptance else None,
        'require_ready': not args.allow_incomplete,
    }
    result = (plan(**kwargs) if args.dry_run else build(Path(args.output), **kwargs))
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
