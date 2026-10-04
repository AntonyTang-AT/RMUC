"""Evaluation for the sentinel three-state contract.

The default comparison is a causal current-frame rule prediction against the
short look-ahead state label (delta frames).  It reports red and blue views
separately and then averages the two views, so one side cannot hide the other.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from collections import Counter

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

STATES = ('def', 'hitbuilding', 'cruise')


def score_state_series(y_true, y_pred, labels=STATES):
    """Score a pair of sentinel state series with a fixed three-class schema."""
    y_true = list(y_true)
    y_pred = list(y_pred)
    if len(y_true) != len(y_pred):
        raise ValueError('y_true and y_pred must have the same length')
    labels = tuple(labels)
    unknown = [x for x in y_true + y_pred if x not in labels]
    if unknown:
        raise ValueError(f'unknown sentinel state: {unknown[0]}')

    confusion = {truth: {pred: 0 for pred in labels} for truth in labels}
    for truth, pred in zip(y_true, y_pred):
        confusion[truth][pred] += 1

    per_class = {}
    f1s = []
    for label in labels:
        tp = confusion[label][label]
        fp = sum(confusion[truth][label] for truth in labels if truth != label)
        fn = sum(confusion[label][pred] for pred in labels if pred != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1s.append(f1)
        per_class[label] = {
            'precision': round(precision, 6),
            'recall': round(recall, 6),
            'f1': round(f1, 6),
            'support': sum(confusion[label].values()),
        }

    transitions_true = sum(a != b for a, b in zip(y_true, y_true[1:]))
    transitions_pred = sum(a != b for a, b in zip(y_pred, y_pred[1:]))
    return {
        'n': len(y_true),
        'accuracy': round(sum(a == b for a, b in zip(y_true, y_pred)) / len(y_true), 6)
        if y_true else 0.0,
        'macro_f1': round(sum(f1s) / len(f1s), 6) if f1s else 0.0,
        'per_class': per_class,
        'confusion': confusion,
        'true_transition_count': transitions_true,
        'pred_transition_count': transitions_pred,
        'true_transition_rate': round(transitions_true / max(len(y_true) - 1, 1), 6),
        'pred_transition_rate': round(transitions_pred / max(len(y_pred) - 1, 1), 6),
    }


def _average_views(views):
    present = [views[side] for side in ('红', '蓝') if views.get(side)]
    if not present:
        return None
    out = {
        'n_sides': len(present),
        'accuracy': round(sum(x['accuracy'] for x in present) / len(present), 6),
        'macro_f1': round(sum(x['macro_f1'] for x in present) / len(present), 6),
        'true_transition_rate': round(sum(x['true_transition_rate'] for x in present) / len(present), 6),
        'pred_transition_rate': round(sum(x['pred_transition_rate'] for x in present) / len(present), 6),
        'per_class': {},
    }
    for state in STATES:
        out['per_class'][state] = {
            metric: round(sum(x['per_class'][state][metric] for x in present) / len(present), 6)
            for metric in ('precision', 'recall', 'f1')
        }
    return out


def evaluate_games(db_path, game_ids, delta=5):
    from data import db, features
    from labels import strategy_label
    from tree import global_mode
    from worldstate import worldstate

    views = {}
    for side in ('红', '蓝'):
        truth, pred = [], []
        per_game = []
        for gid in game_ids:
            con = sqlite3.connect(db_path)
            con.row_factory = sqlite3.Row
            rows = db.timeseries_for(con, gid)
            con.close()
            dps = features.build_decision_points(rows, gid)
            if not dps:
                continue
            bounds = features.field_bounds(dps)
            ws_list = [worldstate.build_worldstate(dp, bounds) for dp in dps]
            truth_game = strategy_label.sentry_state_series(ws_list, delta, team=side)
            pred_game = [global_mode.derive_sentry_state(w, team=side) for w in ws_list]
            truth.extend(truth_game)
            pred.extend(pred_game)
            per_game.append({
                'game_id': int(gid),
                **score_state_series(truth_game, pred_game),
            })
        report = score_state_series(truth, pred)
        report['per_game'] = per_game
        report['label_policy'] = f'future window [t, t+{delta}] with state priority def > hitbuilding > cruise'
        views[side] = report
    return {
        'schema_version': 'sentry-state-eval-v1',
        'states': list(STATES),
        'delta_frames': int(delta),
        'prediction': 'current-frame derive_sentry_state (no future input)',
        'truth': 'sentry_state_from_window (short look-ahead proxy label)',
        'red_view': views.get('红'),
        'blue_view': views.get('蓝'),
        'side_macro_average': _average_views(views),
        'caveat': 'look-ahead is a short-term behavior proxy, not an independent human gold label or causal win proof',
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--db', default=os.path.join(REPO, '数据与资料', 'rmuc_2026_region_dataset.sqlite'))
    parser.add_argument('--manifest', default=os.path.join(REPO, 'out_m2', 'eval_manifest.json'))
    parser.add_argument('--delta', type=int, default=5)
    parser.add_argument('--out', default=os.path.join(REPO, 'out_m2', 'sentry_state_eval_report.json'))
    args = parser.parse_args(argv)
    with open(args.manifest, encoding='utf-8') as f:
        manifest = json.load(f)
    report = evaluate_games(args.db, manifest['split']['eval_games'], delta=args.delta)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps({
        'out': args.out,
        'red': report['red_view']['macro_f1'],
        'blue': report['blue_view']['macro_f1'],
        'side_macro_average': report['side_macro_average']['macro_f1'],
    }, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
