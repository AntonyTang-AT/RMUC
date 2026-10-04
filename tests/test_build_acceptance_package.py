from pathlib import Path

from scripts.build_acceptance_package import (_excluded, _llm_readiness_failures,
                                              _metric_validity_failures,
                                              _training_dataset_audit_failures,
                                              collect_files)


def test_package_excludes_model_weights_and_checkpoints():
    assert _excluded(Path('out_m2/finetuned_x/adapter/adapter_model.safetensors'))
    assert _excluded(Path('out_m2/finetuned_x/ckpt/checkpoint-100/optimizer.pt'))
    assert _excluded(Path('out_m2/temporal/model.pt'))


def test_package_excludes_raw_sqlite_databases():
    assert _excluded(Path('data/rm.db'))
    assert _excluded(Path('数据与资料/rmuc_2026_region_dataset.sqlite'))


def test_package_keeps_report_artifacts():
    assert not _excluded(Path('out_m2/production_ensemble_eval/report.json'))


def test_collected_project_files_do_not_include_weight_suffixes():
    root = Path(__file__).resolve().parents[1]
    assert all(path.suffix.lower() not in {'.pt', '.safetensors', '.npz', '.db', '.sqlite'}
               for path in collect_files(root))


def test_package_requires_semantic_mode_coverage_gate():
    failures = _llm_readiness_failures({
        'structural_reliability': 'pass',
        'semantic_mode_coverage': 'insufficient',
        'semantic_hdq_by_global_mode_vs_legacy_rule': 'pass',
        'semantic_hdq_vs_legacy_rule': 'pass',
    })
    assert failures == ['LLM semantic mode-coverage gate is not pass']


def test_package_requires_per_mode_hdq_gate():
    failures = _llm_readiness_failures({
        'structural_reliability': 'pass',
        'semantic_mode_coverage': 'pass',
        'semantic_hdq_by_global_mode_vs_legacy_rule': 'fail',
        'semantic_hdq_vs_legacy_rule': 'pass',
    })
    assert failures == ['LLM semantic per-mode HDQ gate is not pass']


def test_package_requires_hdq_metric_validity_for_both_sides():
    failures = _metric_validity_failures({
        'by_side': {
            '红': {'mode_row_coverage_pass': True, 'mode_game_coverage_pass': True},
            '蓝': {'mode_row_coverage_pass': False, 'mode_game_coverage_pass': True},
        },
    })
    assert failures == ['蓝 HDQ holdout lacks per-mode row coverage']


def test_package_requires_passing_llm_training_dataset_audit():
    assert _training_dataset_audit_failures({'passed': False}) == [
        'LLM training dataset audit is not pass']
    assert _training_dataset_audit_failures({'passed': True}) == []
