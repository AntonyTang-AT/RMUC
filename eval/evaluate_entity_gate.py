"""Validation-selected persistence gate for entity position/zone forecasts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from forecast.entity import current_positions_from_feature_row
from forecast.temporal import HORIZONS, load_compact
from eval.crossfit_validation_velocity_position import predict_frozen
from worldstate.functional_zone import FUNCTIONAL_ZONE_LABELS, assign_functional_zone_normalized

MAP = np.asarray([28.0, 15.0], dtype=np.float32)
THRESHOLDS_M = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0)


ZONE_INDEX = {name: index for index, name in enumerate(FUNCTIONAL_ZONE_LABELS)}


def _zone_array(points):
    """Compute polygon labels once; threshold candidates then use array ops."""
    values = np.asarray(points)
    output = np.full(values.shape[:2], -1, dtype=np.int16)
    for row, col in np.ndindex(values.shape[:2]):
        x, y = values[row, col]
        if np.isfinite(x) and np.isfinite(y):
            output[row, col] = ZONE_INDEX.get(
                assign_functional_zone_normalized(float(x), float(y)), -1)
    return output


def _load_current(data):
    current = np.full((len(data['x']), 16, 2), np.nan, dtype=np.float32)
    from forecast.entity import ENTITY_SLOTS
    for i in range(len(current)):
        values = current_positions_from_feature_row(data['x'][i, -1])
        for slot_index, (role, entity_type) in enumerate(ENTITY_SLOTS):
            point = values.get(f'{role}:{entity_type}')
            if point is not None:
                current[i, slot_index] = point
    return current / MAP[None, None, :]


def _score(points, current, target, horizon_index,
           predicted_zones=None, truth_zones=None):
    truth = target[:, horizon_index, :, 1:3]
    mask = ((target[:, horizon_index, :, 0] > 0.5) &
            np.isfinite(current).all(axis=-1))
    err = np.abs(points - truth)[mask]
    if predicted_zones is None:
        predicted_zones = _zone_array(points)
    if truth_zones is None:
        truth_zones = _zone_array(truth)
    zone_mask = mask & (predicted_zones >= 0) & (truth_zones >= 0)
    zone_accuracy = float(np.mean(
        predicted_zones[zone_mask] == truth_zones[zone_mask])) if np.any(zone_mask) else 0.0
    return {
        'position_mae_normalized': float(np.mean(err)) if err.size else None,
        'functional_zone_accuracy': zone_accuracy,
        'scored_slots': int(mask.sum()),
    }


def _predict(checkpoint, npz_path, device, batch_size):
    data = load_compact(npz_path)
    pred = predict_frozen(checkpoint, data['x'], device=device, batch_size=batch_size)
    return data, pred, _load_current(data)


def evaluate(checkpoint, validation_npz, test_npz, output, device='cpu', batch_size=1024):
    val, val_pred, val_current = _predict(checkpoint, validation_npz, device, batch_size)
    test, test_pred, test_current = _predict(checkpoint, test_npz, device, batch_size)
    val_target = np.asarray(val['entity_y'], dtype=np.float32)
    test_target = np.asarray(test['entity_y'], dtype=np.float32)
    selected = {}
    validation = {}
    test_report = {}
    for hi, horizon in enumerate(HORIZONS):
        val_truth = val_target[:, hi, :, 1:3]
        val_mask = ((val_target[:, hi, :, 0] > 0.5) &
                    np.isfinite(val_current).all(axis=-1))
        displacement_m = np.linalg.norm((val_pred[:, hi] - val_current) * MAP, axis=-1)
        val_current_zones = _zone_array(val_current)
        val_neural_zones = _zone_array(val_pred[:, hi])
        val_truth_zones = _zone_array(val_truth)
        candidates = []
        for threshold in THRESHOLDS_M:
            use_model = displacement_m >= float(threshold)
            gated = np.where(use_model[..., None], val_pred[:, hi], val_current)
            gated_zones = np.where(use_model, val_neural_zones, val_current_zones)
            score = _score(gated, val_current, val_target, hi,
                           predicted_zones=gated_zones,
                           truth_zones=val_truth_zones)
            # Select by zone first, then position error, with a slight
            # preference for fewer switches when scores are indistinguishable.
            objective = (0.55 * score['functional_zone_accuracy'] +
                         0.45 * (1.0 - score['position_mae_normalized']))
            candidates.append({'threshold_m': float(threshold),
                               'objective': float(objective), **score,
                               'switch_rate': float(np.mean(use_model[val_mask]))
                               if np.any(val_mask) else 0.0})
        best = max(candidates, key=lambda row: (row['objective'], -row['threshold_m']))
        selected[str(horizon)] = best
        validation[f'{horizon}s'] = {
            'persistence': _score(val_current, val_current, val_target, hi,
                                  predicted_zones=val_current_zones,
                                  truth_zones=val_truth_zones),
            'neural': _score(val_pred[:, hi], val_current, val_target, hi,
                             predicted_zones=val_neural_zones,
                             truth_zones=val_truth_zones),
            'candidates': candidates,
            'selected': best,
        }
        threshold = float(best['threshold_m'])
        test_displacement = np.linalg.norm((test_pred[:, hi] - test_current) * MAP, axis=-1)
        test_truth = test_target[:, hi, :, 1:3]
        test_current_zones = _zone_array(test_current)
        test_neural_zones = _zone_array(test_pred[:, hi])
        test_truth_zones = _zone_array(test_truth)
        test_gated = np.where((test_displacement >= threshold)[..., None],
                              test_pred[:, hi], test_current)
        test_gated_zones = np.where(test_displacement >= threshold,
                                    test_neural_zones, test_current_zones)
        test_report[f'{horizon}s'] = {
            'persistence': _score(test_current, test_current, test_target, hi,
                                  predicted_zones=test_current_zones,
                                  truth_zones=test_truth_zones),
            'neural': _score(test_pred[:, hi], test_current, test_target, hi,
                             predicted_zones=test_neural_zones,
                             truth_zones=test_truth_zones),
            'gated': _score(test_gated, test_current, test_target, hi,
                            predicted_zones=test_gated_zones,
                            truth_zones=test_truth_zones),
            'selected_threshold_m': threshold,
            'switch_rate': float(np.mean(test_displacement >= threshold)),
        }
    report = {
        'schema_version': 'entity-persistence-gate-v1',
        'checkpoint': str(checkpoint),
        'selection': 'validation-only objective; test used once for final audit',
        'thresholds_m': list(THRESHOLDS_M),
        'selected': selected,
        'validation': validation,
        'test': test_report,
    }
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--validation-npz', required=True)
    parser.add_argument('--test-npz', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--batch-size', type=int, default=1024)
    args = parser.parse_args()
    report = evaluate(args.checkpoint, args.validation_npz, args.test_npz,
                      args.output, args.device, args.batch_size)
    print(json.dumps(report['test'], ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
