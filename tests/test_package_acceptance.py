import json
from pathlib import Path
from zipfile import ZipFile

from scripts.package_acceptance import build_archive


def test_acceptance_archive_excludes_weights_and_keeps_evidence(tmp_path):
    (tmp_path / 'src').mkdir()
    (tmp_path / 'models_hf').mkdir()
    (tmp_path / 'out_m2' / 'report').mkdir(parents=True)
    (tmp_path / 'src' / 'main.py').write_text('pass\n', encoding='utf-8')
    (tmp_path / 'models_hf' / 'model.safetensors').write_bytes(b'weight')
    (tmp_path / 'out_m2' / 'report' / 'acceptance.json').write_text('{}', encoding='utf-8')
    output = tmp_path / 'acceptance.zip'
    result = build_archive(tmp_path, output)
    assert result['included_files'] >= 2
    with ZipFile(output) as archive:
        names = set(archive.namelist())
        assert 'src/main.py' in names
        assert 'out_m2/report/acceptance.json' in names
        assert 'models_hf/model.safetensors' not in names
        manifest = json.loads(archive.read('PACKAGE_MANIFEST.json'))
        assert manifest['schema_version'] == 'rmuc-acceptance-package-v1'
