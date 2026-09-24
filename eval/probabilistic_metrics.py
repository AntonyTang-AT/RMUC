"""Proper forecast scores and paired red/blue game-level summaries.

Lower scores are better. Score aggregation weights observations inside each
side; the main side macro score then gives red and blue equal weight.
"""

import math
import random
from collections.abc import Mapping


_SIDES = ('红', '蓝')


def _cases(y_true, probabilities, labels):
    labels = tuple(labels)
    if not labels or len(set(labels)) != len(labels):
        raise ValueError('labels must be nonempty and unique')
    if isinstance(probabilities, Mapping):
        cases = [(y_true, probabilities)]
    else:
        if isinstance(y_true, (str, bytes)):
            raise ValueError('batch probabilities require batch labels')
        cases = list(zip(y_true, probabilities, strict=True))
        if not cases:
            raise ValueError('at least one forecast is required')
    for observed, distribution in cases:
        if observed not in labels:
            raise ValueError(f'unknown observed label: {observed!r}')
        if not isinstance(distribution, Mapping) or set(distribution) != set(labels):
            raise ValueError('probabilities must contain exactly the declared labels')
        values = [distribution[label] for label in labels]
        if any(isinstance(p, bool) or not isinstance(p, (int, float)) or
               not math.isfinite(p) or p < 0 or p > 1 for p in values):
            raise ValueError('probabilities must be finite values in [0, 1]')
        if not math.isclose(sum(values), 1.0, rel_tol=0, abs_tol=1e-6):
            raise ValueError('probability sum must equal 1')
    return cases, labels


def multiclass_brier(y_true, probabilities, labels):
    """Mean summed squared error across all declared classes."""
    cases, labels = _cases(y_true, probabilities, labels)
    return sum(sum((dist[label] - (label == observed)) ** 2 for label in labels)
               for observed, dist in cases) / len(cases)


def multiclass_log_loss(y_true, probabilities, labels, epsilon=1e-15):
    """Mean negative log probability of the observed class."""
    if not 0 < epsilon < 1 or not math.isfinite(epsilon):
        raise ValueError('epsilon must be finite and within (0, 1)')
    cases, _ = _cases(y_true, probabilities, labels)
    return -sum(math.log(max(dist[observed], epsilon)) for observed, dist in cases) / len(cases)


def _finite_vector(values):
    vector = tuple(values)
    if not vector or any(isinstance(x, bool) or not isinstance(x, (int, float)) or
                         not math.isfinite(x) for x in vector):
        raise ValueError('vector must contain finite numeric values')
    return vector


def _ensemble_score(observation, samples):
    samples = tuple(samples)
    if not samples:
        raise ValueError('at least one sample is required')
    dimension = len(observation)
    if any(len(sample) != dimension for sample in samples):
        raise ValueError('sample and observation dimensions differ')
    distance = lambda a, b: math.dist(a, b)
    first = sum(distance(sample, observation) for sample in samples) / len(samples)
    second = sum(distance(a, b) for a in samples for b in samples) / (2 * len(samples) ** 2)
    return first - second


def crps_ensemble(observation, samples):
    """Empirical CRPS with ordered sample pairs, including self pairs."""
    if (isinstance(observation, bool) or not isinstance(observation, (int, float))
            or not math.isfinite(observation)):
        raise ValueError('observation must be finite')
    return _ensemble_score((observation,), [_finite_vector((value,)) for value in samples])


def energy_score(observation_vector, sample_vectors):
    """Empirical Euclidean energy score for a joint vector forecast."""
    observation = _finite_vector(observation_vector)
    return _ensemble_score(observation, [_finite_vector(sample) for sample in sample_vectors])


def _validate_score(part, metric='score'):
    score = part.get(metric)
    n = part.get('n')
    if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
        raise ValueError(f'{metric} must be finite')
    if isinstance(n, bool) or not isinstance(n, int) or n <= 0:
        raise ValueError('n must be a positive integer')
    return score, n


def aggregate_sides(per_side):
    """Aggregate two already summarized views with equal-side macro weight."""
    if set(per_side) != set(_SIDES):
        raise ValueError('both red and blue side summaries are required')
    red, nr = _validate_score(per_side['红'])
    blue, nb = _validate_score(per_side['蓝'])
    return {
        'red_view': dict(per_side['红']),
        'blue_view': dict(per_side['蓝']),
        'side_macro_average': {'score': (red + blue) / 2, 'n_sides': 2},
        'pooled_micro': {'score': (red * nr + blue * nb) / (nr + nb), 'n_points': nr + nb},
    }


def aggregate_game_scores(per_game_side):
    """Collapse per-game scores by observations, then balance the two sides."""
    if not per_game_side:
        raise ValueError('at least one game is required')
    totals = {side: {'weighted': 0.0, 'n': 0} for side in _SIDES}
    for game, views in per_game_side.items():
        if set(views) != set(_SIDES):
            raise ValueError(f'game {game!r} must contain both sides')
        for side in _SIDES:
            score, n = _validate_score(views[side])
            totals[side]['weighted'] += score * n
            totals[side]['n'] += n
    sides = {side: {'score': totals[side]['weighted'] / totals[side]['n'],
                    'n': totals[side]['n']} for side in _SIDES}
    return {'sides': sides, **aggregate_sides(sides), 'n_games': len(per_game_side)}


def _valid_clusters(rows_by_game):
    if not rows_by_game:
        raise ValueError('at least one game cluster is required')
    clusters = []
    for game, rows in rows_by_game.items():
        group = list(rows)
        if {row.get('side') for row in group} != set(_SIDES):
            raise ValueError(f'game {game!r} must contain red and blue rows')
        clusters.append(group)
    return clusters


def bootstrap_game_clusters(rows_by_game, n_bootstrap=2000, seed=0):
    """Draw games with replacement, retaining all rows of each selected game."""
    if isinstance(n_bootstrap, bool) or not isinstance(n_bootstrap, int) or n_bootstrap <= 0:
        raise ValueError('n_bootstrap must be a positive integer')
    clusters = _valid_clusters(rows_by_game)
    rng = random.Random(seed)
    return [[list(clusters[rng.randrange(len(clusters))]) for _ in clusters]
            for _ in range(n_bootstrap)]


def _paired_delta(clusters, metric):
    totals = {side: [0.0, 0] for side in _SIDES}
    for cluster in clusters:
        for row in cluster:
            side = row['side']
            if side not in totals:
                raise ValueError(f'unknown side: {side!r}')
            score, n = _validate_score(row, metric)
            totals[side][0] += score * n
            totals[side][1] += n
    if any(n == 0 for _, n in totals.values()):
        raise ValueError('both side views are required')
    return totals['红'][0] / totals['红'][1] - totals['蓝'][0] / totals['蓝'][1]


def _quantile(sorted_values, q):
    pos = (len(sorted_values) - 1) * q
    low = math.floor(pos)
    high = math.ceil(pos)
    return sorted_values[low] + (sorted_values[high] - sorted_values[low]) * (pos - low)


def cluster_bootstrap_side_delta(rows_by_game, metric='score', n_bootstrap=2000, seed=0):
    """Estimate red minus blue score and its paired 95% percentile interval.

    Each row has ``side``, ``n``, and the named mean score. Whole games are
    sampled so both views and all of their prediction rows stay paired.
    """
    clusters = _valid_clusters(rows_by_game)
    if not isinstance(metric, str) or not metric:
        raise ValueError('metric must name a row score field')
    point = _paired_delta(clusters, metric)
    draws = bootstrap_game_clusters(rows_by_game, n_bootstrap=n_bootstrap, seed=seed)
    estimates = sorted(_paired_delta(draw, metric) for draw in draws)
    return {'point_estimate': point, 'ci_low': _quantile(estimates, .025),
            'ci_high': _quantile(estimates, .975), 'n_games': len(clusters),
            'n_bootstrap': n_bootstrap, 'metric': metric, 'delta': '红-蓝'}


def calibration_bins(probabilities, outcomes, n_bins=10):
    """Return equal-width reliability bins with counts and observed rates."""
    if isinstance(n_bins, bool) or not isinstance(n_bins, int) or n_bins <= 0:
        raise ValueError('n_bins must be a positive integer')
    pairs = list(zip(probabilities, outcomes, strict=True))
    bins = [{'lower': i / n_bins, 'upper': (i + 1) / n_bins,
             'count': 0, 'mean_confidence': None, 'observed_frequency': None}
            for i in range(n_bins)]
    sums = [[0.0, 0.0] for _ in bins]
    for p, y in pairs:
        if isinstance(p, bool) or not isinstance(p, (int, float)) or not math.isfinite(p) or not 0 <= p <= 1:
            raise ValueError('probability must be finite and in [0, 1]')
        if y not in (0, 1) or isinstance(y, bool):
            raise ValueError('outcome must be 0 or 1')
        i = min(int(p * n_bins), n_bins - 1)
        bins[i]['count'] += 1
        sums[i][0] += p
        sums[i][1] += y
    for row, (sum_p, sum_y) in zip(bins, sums):
        if row['count']:
            row['mean_confidence'] = sum_p / row['count']
            row['observed_frequency'] = sum_y / row['count']
    return bins
