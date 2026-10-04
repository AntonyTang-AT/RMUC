import json

from eval.evaluate_llm_candidate_matrix import (_load_manifest_games,
                                                _write_failure_audit)


def test_matrix_loads_game_manifest(tmp_path):
    path = tmp_path / 'holdout.json'
    path.write_text(json.dumps({'game_ids': [7, 8]}), encoding='utf-8')
    assert _load_manifest_games(path) == [7, 8]


def test_matrix_writes_parse_failure_audit_for_each_rows_file(tmp_path):
    rows = tmp_path / 'candidate.rows.jsonl'
    rows.write_text(json.dumps({
        'parse_valid': False, 'parse_error': 'empty_output', 'game_id': 7,
        'issued_at_s': 2.0, 'visible_history': [],
    }, ensure_ascii=False) + '\n', encoding='utf-8')
    output = tmp_path / 'candidate.failure_audit.json'
    report = _write_failure_audit(rows, output)
    assert output.exists()
    assert report['failure_reasons'] == {'empty_output': 1}
