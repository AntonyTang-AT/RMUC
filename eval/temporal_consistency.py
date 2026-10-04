"""Rolling forecast consistency checks for the causal replay artifact.

The requested comparison is anchored at issue time ``t``:

* the long forecast issued at ``t`` for ``t+30``;
* the short forecast issued as close as the decision grid permits to
  ``t+20`` for ``t+30`` (the current data are sampled every 3 seconds, so this
  is normally ``t+21`` and the quantisation is reported);
* the matured truth attached to the 30-second target at ``t``.

No future values are used to build either forecast.  The module only scores
already materialised replay frames and is therefore safe for acceptance
reports as well as deployment telemetry.
"""

from __future__ import annotations

import math
from statistics import median


def _distribution(forecast, key, actual=None):
    probs = (forecast or {}).get(f"{key}_probabilities")
    if isinstance(probs, dict) and probs:
        values = {str(label): max(0.0, float(value or 0.0))
                  for label, value in probs.items()}
        total = sum(values.values())
        if total > 0:
            return {label: value / total for label, value in values.items()}
    label = (forecast or {}).get(key)
    if label is None:
        label = actual
    return {str(label): 1.0} if label is not None else {}


def _brier(probs, truth):
    if truth is None or not probs:
        return None
    labels = set(probs) | {str(truth)}
    return sum((float(probs.get(label, 0.0)) -
                (1.0 if label == str(truth) else 0.0)) ** 2
               for label in labels)


def _js(left, right):
    labels = set(left) | set(right)
    if not labels:
        return None
    def kl(a, b):
        value = 0.0
        for label in labels:
            pa, pb = float(a.get(label, 0.0)), float(b.get(label, 0.0))
            if pa > 0:
                value += pa * math.log(pa / max(pb, 1e-12), 2)
        return value
    mid = {label: (left.get(label, 0.0) + right.get(label, 0.0)) / 2.0
           for label in labels}
    return 0.5 * (kl(left, mid) + kl(right, mid))


def _nearest_frame(frames, target, tolerance):
    candidates = [(abs(float(frame.get('t', 0.0)) - target), frame)
                  for frame in frames]
    candidates = [item for item in candidates if item[0] <= tolerance]
    return min(candidates, key=lambda item: item[0])[1] if candidates else None


def evaluate_replay_frames(payload, *, requested_offset_seconds=20.0,
                           long_horizon=30, short_horizon=10,
                           tolerance_seconds=None):
    """Return long/short/actual consistency metrics from replay frames."""
    frames = list((payload or {}).get('frames') or [])
    # A replay can contain one-second observation frames while decisions are
    # issued every three seconds.  Score each forecast issue only once; held
    # frames are presentation copies and must not multiply-count a forecast.
    decision_frames = [frame for frame in frames
                       if frame.get('decision_updated', True)]
    if decision_frames:
        frames = decision_frames
    step = float((payload or {}).get('decision_step_seconds') or 3.0)
    tolerance = (float(tolerance_seconds) if tolerance_seconds is not None
                 else max(step / 2.0, 1e-6))
    per_task = {key: [] for key in ('global_mode', 'opponent_trend')}
    route_agreements = []
    used_offsets = []
    pairs = 0
    for frame in frames:
        issued_t = float(frame.get('t', 0.0))
        short_frame = _nearest_frame(frames, issued_t + float(requested_offset_seconds), tolerance)
        if short_frame is None:
            continue
        used_t = float(short_frame.get('t', 0.0))
        if used_t <= issued_t:
            continue
        for side in ('红', '蓝'):
            long_view = ((frame.get('views') or {}).get(side) or {})
            short_view = ((short_frame.get('views') or {}).get(side) or {})
            long_head = ((long_view.get('horizons') or {}).get(str(long_horizon)) or {})
            short_head = ((short_view.get('horizons') or {}).get(str(short_horizon)) or {})
            long_pred = long_head.get('forecast') or {}
            short_pred = short_head.get('forecast') or {}
            actual = long_head.get('actual') or {}
            for task, key in (('global_mode', 'mode'), ('opponent_trend', 'trend')):
                truth = actual.get(key)
                long_probs = _distribution(long_pred, key, truth)
                short_probs = _distribution(short_pred, key, truth)
                if truth is None or not long_probs or not short_probs:
                    continue
                per_task[task].append({
                    'long_correct': long_pred.get(key) == truth,
                    'short_correct': short_pred.get(key) == truth,
                    'top_agreement': long_pred.get(key) == short_pred.get(key),
                    'long_brier': _brier(long_probs, truth),
                    'short_brier': _brier(short_probs, truth),
                    'js_divergence': _js(long_probs, short_probs),
                })
            long_entities = long_head.get('entities') or {}
            short_entities = short_head.get('entities') or {}
            for slot, item in long_entities.items():
                other = short_entities.get(slot)
                if not other:
                    continue
                route_agreements.append(float(item.get('to_zone') == other.get('to_zone')))
        pairs += 1
        used_offsets.append(used_t - issued_t)

    def aggregate(rows):
        if not rows:
            return {'n': 0, 'long_accuracy': None, 'short_accuracy': None,
                    'top_agreement': None, 'long_brier': None,
                    'short_brier': None, 'short_brier_improvement': None,
                    'js_divergence': None}
        long_brier = sum(row['long_brier'] for row in rows) / len(rows)
        short_brier = sum(row['short_brier'] for row in rows) / len(rows)
        return {
            'n': len(rows),
            'long_accuracy': sum(row['long_correct'] for row in rows) / len(rows),
            'short_accuracy': sum(row['short_correct'] for row in rows) / len(rows),
            'top_agreement': sum(row['top_agreement'] for row in rows) / len(rows),
            'long_brier': long_brier,
            'short_brier': short_brier,
            'short_brier_improvement': long_brier - short_brier,
            'js_divergence': sum(row['js_divergence'] for row in rows) / len(rows),
        }

    return {
        'schema': 'temporal-consistency-v1',
        'requested_offset_seconds': float(requested_offset_seconds),
        'decision_step_seconds': step,
        'matching_tolerance_seconds': tolerance,
        'pairs': pairs,
        'offset_used_seconds': (median(used_offsets) if used_offsets else None),
        'offset_quantization_error_seconds': (median(used_offsets) - float(requested_offset_seconds)
                                              if used_offsets else None),
        'by_task': {task: aggregate(rows) for task, rows in per_task.items()},
        'entity_route_agreement': (sum(route_agreements) / len(route_agreements)
                                   if route_agreements else None),
        'entity_route_comparisons': len(route_agreements),
        'interpretation': 'short forecast at t+20 (nearest decision tick) versus long forecast at t, both scored against actual t+30 truth',
    }
