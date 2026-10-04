"""Build a source-and-evidence acceptance archive without local model weights."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


SKIP_DIRS = {
    '.git', '.pytest_cache', '__pycache__', '.venv_openjev',
    '.venv_openjev_cuda', 'models_hf', 'adapter', 'ckpt',
}
SKIP_SUFFIXES = {
    '.safetensors', '.pt', '.pth', '.bin', '.npz', '.npy', '.sqlite', '.db',
    '.gz', '.tar', '.7z', '.onnx',
}
EVIDENCE_SUFFIXES = {'.json', '.md', '.html', '.txt', '.log'}


def _skip(path: Path, root: Path, destination: Path) -> tuple[bool, str | None]:
    rel = path.relative_to(root)
    parts = set(rel.parts)
    if path.resolve() == destination.resolve():
        return True, 'destination'
    if parts & SKIP_DIRS:
        return True, 'model_or_cache_directory'
    if path.name in {'.env', 'data.db', 'rm.db'}:
        return True, 'local_database_or_secret'
    if path.suffix.lower() in SKIP_SUFFIXES:
        return True, 'large_binary_or_dataset'
    # Generated out_m2 evidence is useful, but raw training/evaluation archives
    # are not part of a portable acceptance package.
    if rel.parts and rel.parts[0] == 'out_m2':
        if path.suffix.lower() not in EVIDENCE_SUFFIXES or path.stat().st_size > 20 * 1024 * 1024:
            return True, 'generated_cache'
    return False, None


def build_archive(root: Path, destination: Path) -> dict:
    root = root.resolve()
    destination = destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    included = []
    excluded = {}
    digest = hashlib.sha256()
    with ZipFile(destination, 'w', compression=ZIP_DEFLATED, compresslevel=6) as archive:
        for current, dirnames, filenames in os.walk(root):
            current_path = Path(current)
            dirnames[:] = sorted(name for name in dirnames if name not in SKIP_DIRS)
            for name in sorted(filenames):
                path = current_path / name
                skip, reason = _skip(path, root, destination)
                rel = path.relative_to(root).as_posix()
                if skip:
                    excluded.setdefault(reason or 'other', []).append(rel)
                    continue
                archive.write(path, rel)
                included.append(rel)
                digest.update(rel.encode('utf-8'))
                digest.update(hashlib.sha256(path.read_bytes()).digest())
        manifest = {
            'schema_version': 'rmuc-acceptance-package-v1',
            'root': str(root),
            'included_files': len(included),
            'included_sha256': digest.hexdigest(),
            'excluded_counts': {key: len(value) for key, value in excluded.items()},
            'excluded_policy': [
                'models_hf and all model weights are excluded',
                'all adapter/checkpoint directories are excluded',
                'large generated datasets, caches, local databases and secrets are excluded',
                'small JSON/Markdown/HTML evidence under out_m2 is retained',
            ],
            'files': included,
        }
        archive.writestr('PACKAGE_MANIFEST.json', json.dumps(manifest, ensure_ascii=False, indent=2))
    manifest['archive'] = str(destination)
    manifest['archive_bytes'] = destination.stat().st_size
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='.')
    parser.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    result = build_archive(Path(args.root), Path(args.output))
    print(json.dumps({key: result[key] for key in (
        'archive', 'archive_bytes', 'included_files', 'included_sha256', 'excluded_counts')},
        ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
