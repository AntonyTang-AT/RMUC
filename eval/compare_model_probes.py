"""Compare Qwen and Open-Jev probe artifacts on an identical causal sample.

The script never mixes model-specific benchmark numbers with the RM split.  It
requires the same ``(game_id, side, issued_at)`` keys and reports the overlap,
side balance, and paired metric differences.  A small overlap is explicitly
marked diagnostic rather than accepted as a model ranking.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _load(path):
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    rows = {(int(row['game_id']), row['side'], round(float(row['issued_at']), 3)): row
            for row in data.get('rows', [])}
    return data, rows


def _paired_metrics(left_rows, right_rows, horizons=(10, 30, 60, 120)):
    result = {}
    common = sorted(set(left_rows) & set(right_rows))
    for horizon in horizons:
        a, b = [], []
        for key in common:
            lrow, rrow = left_rows[key], right_rows[key]
            label = lrow.get('label', {}).get(str(horizon))
            if label is None or rrow.get('label', {}).get(str(horizon)) != label:
                continue
            labels = ['进攻', '均衡', '防守']
            if label not in labels:
                continue
            truth = labels.index(label)
            lp = np.asarray(lrow['probabilities'][str(horizon)], dtype=float)
            rp = np.asarray(rrow['probabilities'][str(horizon)], dtype=float)
            a.append((lp, truth)); b.append((rp, truth))
        if not a:
            result[f'global_mode@{horizon}s'] = {'n': 0}
            continue
        def stats(rows):
            probs = np.asarray([x[0] for x in rows])
            truth = np.asarray([x[1] for x in rows])
            one_hot = np.eye(3)[truth]
            return {
                'accuracy': float(np.mean(probs.argmax(axis=1) == truth)),
                'brier': float(np.mean(np.sum((probs - one_hot) ** 2, axis=1))),
                'log_loss': float(-np.mean(np.log(np.clip(probs[np.arange(len(truth)), truth], 1e-12, 1.0)))),
            }
        ls, rs = stats(a), stats(b)
        result[f'global_mode@{horizon}s'] = {
            'n': len(a), 'qwen': ls, 'open_jev': rs,
            'open_jev_minus_qwen': {metric: rs[metric] - ls[metric] for metric in ls},
        }
    return result, common


def compare(qwen_path, jev_path):
    qwen, qrows = _load(qwen_path)
    jev, jrows = _load(jev_path)
    paired, common = _paired_metrics(qrows, jrows)
    sides = {side: sum(key[1] == side for key in common) for side in ('红', '蓝')}
    return {
        'schema': 'paired-model-probe-comparison-v1',
        'qwen': {'path': str(qwen_path), 'model': qwen.get('model'), 'n_reported': len(qrows)},
        'open_jev': {'path': str(jev_path), 'model': jev.get('model'), 'n_reported': len(jrows)},
        'common_n': len(common), 'common_by_side': sides,
        'paired_metrics': paired,
        'acceptance': {
            'same_samples': len(common) == min(len(qrows), len(jrows)),
            'side_balanced': sides.get('红', 0) == sides.get('蓝', 0),
            'n_sufficient_for_ranking': len(common) >= 128,
            'interpretation': ('ranking_ready' if len(common) >= 128 else
                               'diagnostic_only_expand_to_at_least_128_paired_rows'),
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--qwen', default='out_m2/temporal_forecast/qwen_probe.json')
    ap.add_argument('--open-jev', default='out_m2/temporal_forecast/open_jev_probe.json')
    ap.add_argument('--output', default='out_m2/temporal_forecast/qwen_jev_paired_comparison.json')
    args = ap.parse_args()
    result = compare(Path(args.qwen), Path(args.open_jev))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()

