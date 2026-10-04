# -*- coding: utf-8 -*-
"""统一的 M2 留出评估入口。

该脚本固定按 game_id 划分数据，避免同一局的时间点同时出现在训练和评估中。
当前不依赖 GPU 或人工标签，先报告：
  - schema 合法率
  - 决策序列冲突率
  - global_mode 相对 lookahead 代理标签的 accuracy / macro-F1
  - 每局结果与整体结果

外部模型可通过 --predictions 提供 JSONL。每行必须包含 game_id、side、index、完整 decision；
这里只评估红蓝双方决策序列。概率预测使用 eval.prediction_ledger 和
evaluate_forecast_ledger，不与单标签 legacy decision 接口混用。
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
import os
import sqlite3
import sys
from collections import Counter, defaultdict

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from data import db, features
from labels import strategy_label
from schema import layered as sl
from worldstate import worldstate
import run_m2
from eval.forecast_schema import validate_forecast
from eval.probabilistic_metrics import (
    aggregate_sides, calibration_bins, cluster_bootstrap_side_delta,
    multiclass_brier, multiclass_log_loss,
)


def _jsonable(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def _accuracy(y_true, y_pred):
    return float(sum(a == b for a, b in zip(y_true, y_pred)) / len(y_true)) if y_true else 0.0


def _macro_f1(y_true, y_pred, labels):
    scores = []
    for label in labels:
        tp = sum(a == label and b == label for a, b in zip(y_true, y_pred))
        fp = sum(a != label and b == label for a, b in zip(y_true, y_pred))
        fn = sum(a == label and b != label for a, b in zip(y_true, y_pred))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append(2 * precision * recall / (precision + recall)
                     if precision + recall else 0.0)
    return float(np.mean(scores)) if scores else 0.0


def _conflict_rate(decisions):
    return float(sl.conflict_rate(decisions)) if decisions else 0.0


def _digest(values):
    raw = json.dumps(values, ensure_ascii=False, separators=(',', ':')).encode('utf-8')
    return hashlib.sha256(raw).hexdigest()[:16]


FORECAST_KEY_FIELDS = ('game_id', 'side', 'issued_at', 'horizon', 'task', 'entity_or_region')


def forecast_key(row):
    """Build the immutable target identity used to join predictions to truth."""
    return tuple(row[field] for field in FORECAST_KEY_FIELDS)


def _target_is_finite_number(value):
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value))


def _quantile_pinball(observation, quantiles):
    """Mean pinball loss over the submitted quantile levels (lower is better)."""
    if not _target_is_finite_number(observation):
        raise ValueError('quantile target must be a finite number')
    losses = []
    for level, estimate in quantiles.items():
        tau = float(level)
        error = observation - estimate
        losses.append(max(tau * error, (tau - 1.0) * error))
    if not losses:
        raise ValueError('at least one quantile is required')
    return sum(losses) / len(losses)


def _side_metric_summary(scores_by_game, metric, n_bootstrap, seed):
    side_values = {side: {'weighted': 0.0, 'n': 0} for side in ('红', '蓝')}
    game_side = {}
    for game_id, sides in scores_by_game.items():
        game_side[game_id] = {}
        for side, values in sides.items():
            if not values:
                continue
            mean = sum(values) / len(values)
            game_side[game_id][side] = {'score': mean, 'n': len(values)}
            side_values[side]['weighted'] += sum(values)
            side_values[side]['n'] += len(values)
    sides = {
        side: ({'score': value['weighted'] / value['n'], 'n': value['n']}
               if value['n'] else None)
        for side, value in side_values.items()
    }
    paired = {game: values for game, values in game_side.items()
              if set(values) == {'红', '蓝'}}
    report = {'sides': sides, 'red_view': sides['红'], 'blue_view': sides['蓝'],
              'n_complete_games': len(paired)}
    if sides['红'] is not None and sides['蓝'] is not None:
        report.update(aggregate_sides(sides))
    else:
        report['side_macro_average'] = None
        report['pooled_micro'] = None
    if paired:
        bootstrap_rows = {
            game: [{'side': side, **summary} for side, summary in values.items()]
            for game, values in paired.items()
        }
        report['red_minus_blue_ci'] = cluster_bootstrap_side_delta(
            bootstrap_rows, metric='score', n_bootstrap=n_bootstrap, seed=seed,
        )
        report['red_minus_blue_ci']['population'] = 'complete games with both side scores'
    else:
        report['red_minus_blue_ci'] = None
    return report


def evaluate_forecast_ledger(predictions, truth_by_key, expected_keys, now,
                             n_bootstrap=2000, seed=0, model_versions=None,
                             unavailable_keys_by_model=None):
    """Score matured archived forecasts against already-built exact-time targets.

    ``truth_by_key`` contains only uncensored outcomes keyed by ``forecast_key``.
    ``expected_keys`` enumerates both-side prediction opportunities. Predictions
    are inputs only; future truth is never copied into feature snapshots here.
    """
    if not _target_is_finite_number(now):
        raise ValueError('now must be a finite game timestamp')
    predictions = list(predictions)
    expected = set()
    for raw_key in expected_keys:
        try:
            key = tuple(raw_key)
        except TypeError as exc:
            raise ValueError('expected keys must be iterable tuples') from exc
        if len(key) != len(FORECAST_KEY_FIELDS):
            raise ValueError('expected key must contain six fields')
        try:
            hash(key)
        except TypeError as exc:
            raise ValueError('expected key fields must be hashable') from exc
        expected.add(key)
    if not expected:
        raise ValueError('expected_keys must not be empty')
    if not hasattr(truth_by_key, 'get'):
        raise ValueError('truth_by_key must map forecast keys to scalar targets')
    for key in expected:
        if (len(key) != len(FORECAST_KEY_FIELDS) or key[1] not in ('红', '蓝')
                or not _target_is_finite_number(key[2])
                or not _target_is_finite_number(key[3]) or key[3] <= 0
                or not isinstance(key[0], (int, str))
                or not isinstance(key[4], str)
                or not (isinstance(key[5], str) and key[5]
                        or isinstance(key[5], int) and not isinstance(key[5], bool))):
            raise ValueError('expected key must contain game, red/blue side, time, horizon, task, and target')

    active_expected = {key for key in expected if key[2] + key[3] <= now}
    immature_expected = expected - active_expected
    active_groups = defaultdict(set)
    for key in active_expected:
        active_groups[(key[0], key[2], key[3], key[4])].add(key[1])
    scorable_expected = {
        key for key in active_expected
        if key in truth_by_key and truth_by_key[key] is not None
    }
    censored = active_expected - scorable_expected

    records_by_model_key = defaultdict(list)
    seen_by_model_key = set()
    invalid_by_model = Counter()
    immature_by_model = Counter()
    unexpected_by_model = Counter()
    duplicate_by_model = Counter()
    missing_by_model = Counter()
    unavailable_by_model = Counter()
    invalid_records = []
    for row in predictions:
        if not isinstance(row, dict) or '_ledger_error' in row:
            invalid_records.append({'errors': [row.get('_ledger_error', 'record is not an object')]
                                    if isinstance(row, dict) else ['record is not an object']})
            continue
        valid, errors = validate_forecast(row)
        if not valid:
            invalid_records.append({'prediction_id': row.get('prediction_id'), 'errors': errors})
            model = row.get('model_version')
            if isinstance(model, str) and model:
                invalid_by_model[model] += 1
                if all(field in row for field in FORECAST_KEY_FIELDS):
                    try:
                        key = forecast_key(row)
                        hash(key)
                    except (KeyError, TypeError):
                        key = None
                    if key is not None and key in expected:
                        seen_by_model_key.add((model, key))
            continue
        model = row['model_version']
        key = forecast_key(row)
        if key not in expected:
            unexpected_by_model[model] += 1
            continue
        seen_by_model_key.add((model, key))
        if row['valid_at'] > now:
            immature_by_model[model] += 1
            continue
        records_by_model_key[model, key].append(row)

    models = set(model_versions or ())
    models.update(row['model_version'] for row in predictions
                  if isinstance(row, dict) and '_ledger_error' not in row
                  and isinstance(row.get('model_version'), str) and row['model_version'])
    models.update(invalid_by_model)
    if not models:
        raise ValueError('model_versions is required when no forecast identifies a model')

    per_model = defaultdict(lambda: defaultdict(lambda: defaultdict(
        lambda: defaultdict(lambda: {'红': [], '蓝': []}))))
    calibration = defaultdict(lambda: {'confidence': [], 'correct': []})
    model_reports = {}
    model_counts = {}
    scored_by_model_key = defaultdict(set)
    for model in sorted(models):
        scored = 0
        unmatched_truth = len(censored)
        unavailable = set((unavailable_keys_by_model or {}).get(model, ()))
        for key in scorable_expected:
            if key in unavailable:
                unavailable_by_model[model] += 1
                continue
            candidates = records_by_model_key.get((model, key), [])
            if not candidates:
                if (model, key) not in seen_by_model_key:
                    missing_by_model[model] += 1
                continue
            if len(candidates) != 1:
                duplicate_by_model[model] += len(candidates)
                continue
            row = candidates[0]
            actual = truth_by_key[key]
            bucket = f"{row['task']}@{row['horizon']}s"
            if row.get('probabilities', row.get('probability_distribution')) is not None:
                probabilities = row.get('probabilities', row.get('probability_distribution'))
                labels = row['labels']
                brier = multiclass_brier(actual, probabilities, labels)
                log_loss = multiclass_log_loss(actual, probabilities, labels)
                for metric, value in (('brier', brier), ('log_loss', log_loss)):
                    per_model[model][bucket][metric][key[0]][key[1]].append(value)
                predicted = max(labels, key=lambda label: probabilities[label])
                calibration[model]['confidence'].append(probabilities[predicted])
                calibration[model]['correct'].append(int(predicted == actual))
            else:
                value = _quantile_pinball(actual, row['quantiles'])
                per_model[model][bucket]['mean_pinball'][key[0]][key[1]].append(value)
            scored += 1
            scored_by_model_key[model].add(key)
        model_counts[model] = {
            'expected_count': len(scorable_expected),
            'scored_count': scored,
            'missing_count': missing_by_model[model],
            'unavailable_count': unavailable_by_model[model],
            'duplicate_count': duplicate_by_model[model],
            'unexpected_count': unexpected_by_model[model],
            'invalid_count': (invalid_by_model[model] + duplicate_by_model[model]
                              + unexpected_by_model[model] + missing_by_model[model]),
            'immature_count': immature_by_model[model],
            'target_unavailable_count': unmatched_truth,
            'missing_rate': (missing_by_model[model] /
                             max(1, len(scorable_expected) - unavailable_by_model[model])
                             if scorable_expected else 0.0),
        }
        expected_groups = defaultdict(lambda: {'红': [], '蓝': []})
        for key in active_expected:
            if key in unavailable:
                continue
            expected_groups[(key[0], key[2], key[3], key[4])][key[1]].append(key)
        complete_groups = [sides for sides in expected_groups.values()
                           if sides['红'] and sides['蓝']]
        model_counts[model]['red_blue_group_count'] = len(complete_groups)
        model_counts[model]['active_group_count'] = len(expected_groups)
        model_counts[model]['red_blue_pair_coverage_rate'] = (
            len(complete_groups) / len(expected_groups) if expected_groups else 0.0)
        model_counts[model]['red_blue_complete'] = bool(complete_groups) and all(
            all(key in scored_by_model_key[model]
                for side in ('红', '蓝') for key in sides[side] if key in scorable_expected)
            and all(any(key in scorable_expected for key in sides[side]) for side in ('红', '蓝'))
            for sides in complete_groups
        ) and all(set(sides) == {'红', '蓝'} for sides in active_groups.values())

    for model in sorted(models):
        buckets = per_model[model]
        model_reports[model] = {'by_task_horizon': {}, **model_counts[model]}
        for bucket, metric_rows in buckets.items():
            model_reports[model]['by_task_horizon'][bucket] = {
                metric: _side_metric_summary(values, metric, n_bootstrap, seed)
                for metric, values in metric_rows.items()
            }
        reliability = calibration.get(model, {'confidence': [], 'correct': []})
        model_reports[model]['top_label_calibration'] = (
            calibration_bins(reliability['confidence'], reliability['correct'])
            if reliability['confidence'] else []
        )

    total_model_slots = (len(scorable_expected) * len(models) -
                         sum(unavailable_by_model.values()))
    # A key is one game/side/time/horizon/task target.  The same key is
    # evaluated once per model, so summing model scored counts double-counts
    # the observations whenever a baseline is included alongside a model.
    unique_scored_keys = set().union(*(scored_by_model_key[m] for m in models)) if models else set()
    total_missing = sum(missing_by_model.values())
    total_invalid = (len(invalid_records) + sum(duplicate_by_model.values())
                     + sum(unexpected_by_model.values()) + total_missing)
    return {
        'schema_version': 1,
        'evaluation_mode': 'matured_forecast_ledger',
        'as_of': now,
        'expected_count': len(active_expected),
        'scorable_expected_count': len(scorable_expected),
        'immature_expected_count': len(immature_expected),
        'scored_count': len(unique_scored_keys),
        'scored_count_by_model': {
            model: counts['scored_count'] for model, counts in model_counts.items()
        },
        'model_slot_count': sum(counts['scored_count'] for counts in model_counts.values()),
        'unique_scored_key_count': len(unique_scored_keys),
        'invalid_count': total_invalid,
        'invalid_records': invalid_records,
        'duplicate_count': sum(duplicate_by_model.values()),
        'unexpected_count': sum(unexpected_by_model.values()),
        'missing_count': total_missing,
        'missing_rate': total_missing / total_model_slots if total_model_slots else 0.0,
        'unavailable_count': sum(unavailable_by_model.values()),
        'unavailable_count_by_model': dict(unavailable_by_model),
        'immature_count': sum(immature_by_model.values()),
        'target_unavailable_count': len(censored),
        'red_blue_complete': bool(model_counts) and all(
            counts['red_blue_complete'] for counts in model_counts.values()
        ),
        'red_blue_group_count_by_model': {
            model: counts.get('red_blue_group_count', 0)
            for model, counts in model_counts.items()
        },
        'active_group_count_by_model': {
            model: counts.get('active_group_count', 0)
            for model, counts in model_counts.items()
        },
        'red_blue_pair_coverage_by_model': {
            model: counts.get('red_blue_pair_coverage_rate', 0.0)
            for model, counts in model_counts.items()
        },
        'models': model_reports,
    }


def _load_report(path):
    with open(path, encoding='utf-8') as f:
        report = json.load(f)
    eval_games = [int(x) for x in report.get('eval_games', [])]
    dataset_games = [int(x) for x in report.get('dataset_games', [])]
    if not eval_games:
        raise ValueError(f'no eval_games in {path}')
    return dataset_games, eval_games


def make_manifest(report_path, db_path, out_path):
    dataset_games, eval_games = _load_report(report_path)
    con = sqlite3.connect(db_path)
    counts = {}
    for gid in dataset_games + eval_games:
        row = con.execute('SELECT COUNT(*) FROM timeseries WHERE game_id=?', (gid,)).fetchone()
        counts[str(gid)] = int(row[0]) if row else 0
    con.close()
    manifest = {
        'version': 1,
        'source_report': os.path.abspath(report_path),
        'db': os.path.abspath(db_path),
        'split': {
            'train_or_dataset_games': dataset_games,
            'eval_games': eval_games,
        },
        'timeseries_rows': counts,
        'eval_games_sha256_prefix': _digest(eval_games),
        'policy': 'group_by_game_id; no frame-level random split',
    }
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return manifest


def _load_predictions(path):
    if not path:
        return {}
    out = {}
    with open(path, encoding='utf-8') as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if 'game_id' not in row or 'side' not in row or 'index' not in row:
                raise ValueError(f'{path}:{line_no}: need game_id, side, and index')
            if row['side'] not in ('红', '蓝'):
                raise ValueError(f'{path}:{line_no}: side must be 红 or 蓝')
            key = (int(row['game_id']), row['side'], int(row['index']))
            if key in out:
                raise ValueError(f'{path}:{line_no}: duplicate prediction key {key!r}')
            out[key] = row
    return out


def _perspective_view(dps, bounds, side):
    """Map the blue view into the existing red-oriented rule head coordinates."""
    if side not in ('红', '蓝'):
        raise ValueError('side must be 红 or 蓝')
    if side == '红':
        return dps
    view = deepcopy(dps)
    xmin, xmax = bounds['xmin'], bounds['xmax']
    for dp in view:
        for state in dp['robots'].values():
            state['team'] = '蓝' if state['team'] == '红' else '红'
            state['x'] = xmin + xmax - state['x']
            state['dx'] = -state.get('dx', 0.0)
            state['yaw'] = (180.0 - state.get('yaw', 0.0)) % 360.0
        dp['team'] = {
            ('蓝' if team == '红' else '红'): values
            for team, values in dp['team'].items()
        }
    return view


def _evaluate_game(gid, db_path, mode='rule', predictions=None, delta=5, side='红'):
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    rows = db.timeseries_for(con, gid)
    con.close()
    dps = features.build_decision_points(rows, gid)
    if not dps:
        return {'game_id': gid, 'side': side, 'n': 0, 'expected_n': 0, 'error': 'no decision points'}
    bounds = features.field_bounds(dps)
    ws_list = [worldstate.build_worldstate(dp, bounds) for dp in dps]
    proxy = strategy_label.strategy_series(ws_list, delta, team=side)
    perspective_dps = _perspective_view(dps, bounds, side)
    if mode == 'rule':
        decisions = run_m2.rule_head(perspective_dps, bounds)
    elif mode == 'lookahead':
        decisions = run_m2.rule_head(perspective_dps, bounds)
        decisions, _ = strategy_label.relabel_decision_series(ws_list, decisions, delta, team=side)
    elif mode == 'predictions':
        decisions, scored_proxy, decision_legal = [], [], []
        missing_predictions = 0
        invalid_predictions = 0
        for i, dp in enumerate(dps):
            row = (predictions or {}).get((gid, side, i))
            if row is None:
                missing_predictions += 1
                continue
            decision = row.get('decision')
            if not isinstance(decision, dict):
                invalid_predictions += 1
                continue
            is_legal = bool(sl.validate(decision)[0])
            decision_legal.append(is_legal)
            if not is_legal:
                invalid_predictions += 1
                continue
            decisions.append(decision)
            scored_proxy.append(proxy[i])
    else:
        raise ValueError(f'unknown mode: {mode}')

    legal = [sl.validate(d)[0] for d in decisions]
    modes = [d.get('global_mode') for d in decisions]
    comparison_proxy = scored_proxy if mode == 'predictions' else proxy
    return {
        'game_id': gid,
        'side': side,
        'n': len(decisions),
        'expected_n': len(dps),
        'legal_rate': (float(sum(decision_legal) / len(decision_legal))
                       if mode == 'predictions' and decision_legal
                       else float(sum(legal) / len(legal)) if legal else None),
        'legal_n': len(decision_legal) if mode == 'predictions' else len(legal),
        'conflict_rate': _conflict_rate(decisions),
        'mode_accuracy_vs_lookahead': _accuracy(comparison_proxy, modes),
        'mode_macro_f1_vs_lookahead': _macro_f1(comparison_proxy, modes, sl.GLOBAL_MODES),
        'mode_counts': dict(Counter(modes)),
        'proxy_counts': dict(Counter(comparison_proxy)),
        'missing_predictions': missing_predictions if mode == 'predictions' else 0,
        'invalid_predictions': invalid_predictions if mode == 'predictions' else 0,
    }


def run_benchmark(report_path, db_path, manifest_path, modes, predictions_path=None, delta=5):
    if manifest_path and os.path.exists(manifest_path):
        with open(manifest_path, encoding='utf-8') as f:
            manifest = json.load(f)
        eval_games = [int(x) for x in manifest['split']['eval_games']]
    else:
        _, eval_games = _load_report(report_path)
        manifest = make_manifest(report_path, db_path, manifest_path or os.path.join(REPO, 'out_m2', 'eval_manifest.json'))
    predictions = _load_predictions(predictions_path)
    results = {}
    for mode in modes:
        rows = [_evaluate_game(gid, db_path, mode, predictions, delta, side)
                for gid in eval_games for side in ('红', '蓝')]
        metric_names = ('legal_rate', 'conflict_rate', 'mode_accuracy_vs_lookahead',
                        'mode_macro_f1_vs_lookahead')
        metric_reports = {}
        for metric in metric_names:
            per_side = {}
            for side in ('红', '蓝'):
                side_rows = [row for row in rows if row.get('side') == side]
                count_key = 'legal_n' if metric == 'legal_rate' else 'n'
                valid_rows = [row for row in side_rows if row.get(metric) is not None]
                n = sum(row.get(count_key, 0) for row in valid_rows)
                per_side[side] = {
                    'score': (sum(row[metric] * row.get(count_key, 0)
                                  for row in valid_rows) / n if n else None),
                    'n': n,
                }
            if per_side['红']['n'] and per_side['蓝']['n']:
                summary = aggregate_sides(per_side)
            else:
                summary = {
                    'red_view': per_side['红'] if per_side['红']['n'] else None,
                    'blue_view': per_side['蓝'] if per_side['蓝']['n'] else None,
                    'side_macro_average': None,
                    'pooled_micro': None,
                }
            paired = {}
            for gid in eval_games:
                game_rows = [row for row in rows if row['game_id'] == gid]
                views = {row['side']: row for row in game_rows}
                count_key = 'legal_n' if metric == 'legal_rate' else 'n'
                if all(side in views and views[side].get(count_key, 0)
                       and views[side].get(metric) is not None for side in ('红', '蓝')):
                    paired[gid] = [
                        {'side': side, 'score': views[side].get(metric, 0.0),
                         'n': views[side][count_key]}
                        for side in ('红', '蓝')
                    ]
            summary['red_minus_blue_ci'] = (
                cluster_bootstrap_side_delta(paired, metric='score', seed=0)
                if paired else None
            )
            if summary['red_minus_blue_ci'] is not None:
                summary['red_minus_blue_ci']['population'] = 'complete games with both side scores'
            metric_reports[metric] = summary
        total_n = sum(row.get('n', 0) for row in rows)
        expected_n = sum(row.get('expected_n', 0) for row in rows)
        missing_n = sum(row.get('missing_predictions', 0) for row in rows)
        invalid_n = sum(row.get('invalid_predictions', 0) for row in rows)
        side_macro = {
            metric: report['side_macro_average']['score']
            if report['side_macro_average'] else None
            for metric, report in metric_reports.items()
        }
        pooled_micro = {
            metric: report['pooled_micro']['score'] if report['pooled_micro'] else None
            for metric, report in metric_reports.items()
        }
        per_side_metrics = {
            side: {metric: metric_reports[metric]['red_view' if side == '红' else 'blue_view']['score']
                   if metric_reports[metric]['red_view' if side == '红' else 'blue_view'] else None
                   for metric in metric_names}
            for side in ('红', '蓝')
        }
        results[mode] = {
            'eval_games': eval_games,
            'n_points': total_n,
            'n_expected_points': expected_n,
            'missing_predictions': missing_n,
            'missing_rate': missing_n / expected_n if expected_n else 0.0,
            'invalid_predictions': invalid_n,
            'invalid_rate': invalid_n / expected_n if expected_n else 0.0,
            'red_blue_complete': missing_n == 0 and invalid_n == 0 and all(
                sum(row.get('expected_n', 0) for row in rows
                    if row['game_id'] == gid and row['side'] == side) > 0
                for gid in eval_games for side in ('红', '蓝')
            ),
            'red_view': per_side_metrics['红'],
            'blue_view': per_side_metrics['蓝'],
            'side_macro_average': side_macro,
            'pooled_micro': pooled_micro,
            'metrics': metric_reports,
            'per_game': rows,
        }
    return {
        'version': 1,
        'manifest': manifest,
        'delta': delta,
        'proxy_label': 'lookahead; not human gold and not causal outcome proof',
        'models': results,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--report', default=os.path.join(REPO, 'out_m2', 'm2_report.json'))
    ap.add_argument('--db', default=None)
    ap.add_argument('--manifest', default=os.path.join(REPO, 'out_m2', 'eval_manifest.json'))
    ap.add_argument('--predictions', default=None, help='JSONL: game_id,side,index,decision')
    ap.add_argument('--modes', nargs='+', default=['rule', 'lookahead'])
    ap.add_argument('--delta', type=int, default=5)
    ap.add_argument('--out', default=os.path.join(REPO, 'out_m2', 'benchmark_report.json'))
    ap.add_argument('--make-manifest-only', action='store_true')
    args = ap.parse_args()
    db_path = args.db or os.path.join(REPO, '数据与资料', 'rmuc_2026_region_dataset.sqlite')
    manifest = make_manifest(args.report, db_path, args.manifest)
    if args.make_manifest_only:
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return 0
    report = run_benchmark(args.report, db_path, args.manifest, args.modes,
                           args.predictions, args.delta)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps({k: {m: v for m, v in value.items() if m != 'per_game'}
                      for k, value in report['models'].items()}, ensure_ascii=False, indent=2))
    print('SAVE', args.out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
