"""Repair/report-only pass for an already generated temporal ledger.

The historical evaluator stored model and persistence rows in one list and
did not retain persistence entity metrics.  Re-running the neural model is
expensive and materializes hundreds of thousands of JSON rows, so this pass
repairs the ledger accounting and computes the persistence entity baseline
directly from the compact causal test arrays.  New evaluations should use
``run_temporal_forecast.py``; this script is for old reports.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from forecast.entity import ENTITY_SLOTS
from forecast.temporal import HORIZONS
from worldstate.canonical_zone import CANONICAL_ZONE_LABELS, locate_zone
from worldstate.functional_zone import assign_functional_zone_normalized


def _baseline_metrics(npz_path):
    data = np.load(npz_path, allow_pickle=False)
    entity_y = data['entity_y']
    canonical_y = data.get('entity_canonical_zone_y')
    # Motion-schema compact rows have 16 slots × 20 values followed by a
    # small global tail.  Decode the issue-time positions in one vectorized
    # pass instead of rebuilding a Python dict for every sample.
    last = np.asarray(data['x'][:, -1, :320], dtype=np.float32).reshape(
        len(data['x']), len(ENTITY_SLOTS), 20)
    has_position = (last[:, :, 3] > 0.5) & (last[:, :, 9] > 0.5)
    positions = last[:, :, :2] * np.asarray([28.0, 15.0], dtype=np.float32)
    out = {}
    for h_idx, horizon in enumerate(HORIZONS):
        c = dict(presence_correct=0, presence_n=0, position_sum=0.0,
                 position_n=0, zone_correct=0, zone_n=0,
                 canonical_correct=0, canonical_n=0, legal=0, legal_n=0)
        for i in range(len(entity_y)):
            target = entity_y[i, h_idx]
            valid = target[:, 0] >= 0
            if not np.any(valid):
                continue
            for slot_index, (group, entity_type) in enumerate(ENTITY_SLOTS):
                if not valid[slot_index]:
                    continue
                present = bool(has_position[i, slot_index])
                truth_present = bool(target[slot_index, 0] > 0.5)
                c['presence_correct'] += int(present == truth_present)
                c['presence_n'] += 1
                if not present or not truth_present:
                    continue
                x, y = map(float, positions[i, slot_index])
                tx, ty = float(target[slot_index, 1]), float(target[slot_index, 2])
                c['position_sum'] += abs(x / 28.0 - tx) + abs(y / 15.0 - ty)
                c['position_n'] += 2
                current_zone = assign_functional_zone_normalized(x / 28.0, y / 15.0)
                target_zone = assign_functional_zone_normalized(tx, ty)
                c['zone_correct'] += int(current_zone == target_zone)
                c['zone_n'] += 1
                if canonical_y is not None:
                    label = int(canonical_y[i, h_idx, slot_index])
                    if label >= 0:
                        c['canonical_correct'] += int(
                            locate_zone(x, y) == CANONICAL_ZONE_LABELS[label])
                        c['canonical_n'] += 1
                # Staying in the current zone is always a legal route.
                c['legal'] += 1
                c['legal_n'] += 1
        out[f'{horizon}s'] = {
            'slot_count': len(ENTITY_SLOTS),
            'presence_accuracy': c['presence_correct'] / max(1, c['presence_n']),
            'position_mae_normalized': c['position_sum'] / max(1, c['position_n']),
            'raw_position_mae_normalized': c['position_sum'] / max(1, c['position_n']),
            'hp_mae': None,
            'functional_zone_accuracy': c['zone_correct'] / max(1, c['zone_n']),
            'raw_functional_zone_accuracy': c['zone_correct'] / max(1, c['zone_n']),
            'canonical_zone_accuracy': c['canonical_correct'] / max(1, c['canonical_n']),
            'raw_canonical_zone_accuracy': c['canonical_correct'] / max(1, c['canonical_n']),
            'transition_legal_rate': c['legal'] / max(1, c['legal_n']),
            'by_side': {}, 'by_slot': {},
        }
    return out


def repair(report_path, npz_path, out_path):
    report = json.loads(Path(report_path).read_text(encoding='utf-8'))
    models = report.get('models', {})
    scored_by_model = {name: int(value.get('scored_count', 0))
                       for name, value in models.items()}
    missing_by_model = {name: int(value.get('missing_count', 0))
                        for name, value in models.items()}
    baseline_unavailable = int(missing_by_model.get('persistence_v1', 0))
    if 'persistence_v1' in models:
        models['persistence_v1']['unavailable_count'] = baseline_unavailable
        models['persistence_v1']['missing_count'] = 0
        models['persistence_v1']['invalid_count'] = max(
            0, int(models['persistence_v1'].get('invalid_count', 0)) - baseline_unavailable)
        models['persistence_v1']['missing_rate'] = 0.0
    # Every model scores the same immutable key space; the model union is the
    # unique observation count.  For old reports the complete production model
    # count is a safe lower bound for that union.
    unique_scored = max(scored_by_model.values(), default=0)
    report['scored_count'] = unique_scored
    report['unique_scored_key_count'] = unique_scored
    report['scored_count_by_model'] = scored_by_model
    report['model_slot_count'] = sum(scored_by_model.values())
    report['missing_count_by_model'] = {
        name: int(value.get('missing_count', 0)) for name, value in models.items()
    }
    report['unavailable_count_by_model'] = {'persistence_v1': baseline_unavailable}
    report['unavailable_count'] = baseline_unavailable
    report['missing_count'] = sum(report['missing_count_by_model'].values())
    report['invalid_count'] = max(0, int(report.get('invalid_count', 0)) - baseline_unavailable)
    report['red_blue_complete_by_model'] = {
        name: bool(value.get('red_blue_complete')) for name, value in models.items()
    }
    report['accounting_note'] = (
        'scored_count is the unique target-key count; model_slot_count is the '
        'sum across model and persistence slots and must not be interpreted as '
        'independent observations.'
    )
    report['ledger_repair_note'] = (
        f'{baseline_unavailable} persistence rows with missing issue-time base labels '
        'are marked unavailable, not invalid or missing predictions.'
    )

    baseline = _baseline_metrics(npz_path)
    learned = report.get('entity_forecast_metrics', {})
    # The historical learned metrics were computed after the legal route
    # projection.  Preserve them under an explicit name; future reports also
    # include raw_* fields from the decoder.
    learned_projected = {}
    for horizon, value in learned.items():
        item = dict(value)
        item.setdefault('projected_position_mae_normalized', item.get('position_mae_normalized'))
        item.setdefault('projected_functional_zone_accuracy', item.get('functional_zone_accuracy'))
        item.setdefault('projected_canonical_zone_accuracy', item.get('canonical_zone_accuracy'))
        item['raw_metrics_available'] = bool('raw_position_mae_normalized' in value)
        learned_projected[horizon] = item
    report['entity_forecast_metrics'] = learned_projected
    report['entity_forecast_metrics_by_model'] = {
        next((name for name in models if name != 'persistence_v1'), 'learned_model'): learned_projected,
        'persistence_v1': baseline,
    }
    report['entity_baseline_note'] = (
        'Persistence entity metrics are computed from the issue-time compact '
        'state and are not copied from learned predictions. Historical learned '
        'position/zone values are projected metrics; use raw_* fields only in '
        'reports produced by the corrected evaluator.'
    )
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--report', required=True)
    ap.add_argument('--npz', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    report = repair(args.report, args.npz, args.out)
    print(json.dumps({
        'out': args.out,
        'scored_count': report['scored_count'],
        'model_slot_count': report['model_slot_count'],
        'entity_models': list(report['entity_forecast_metrics_by_model']),
    }, ensure_ascii=False))


if __name__ == '__main__':
    main()
