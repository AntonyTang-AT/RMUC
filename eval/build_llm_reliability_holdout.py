"""Select a deterministic, training-disjoint temporal holdout for LLM checks."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def evenly_spaced(values, count):
    values = list(values)
    if count <= 0 or count > len(values):
        raise ValueError('count must be in 1..len(values)')
    if count == 1:
        return [values[len(values) // 2]]
    indexes = sorted({round(i * (len(values) - 1) / (count - 1)) for i in range(count)})
    if len(indexes) != count:
        raise ValueError('sampling produced duplicate indices')
    return [values[index] for index in indexes]


def build(training_manifest, current_holdout, all_game_ids, count, *, excluded_manifests=None):
    trained = set(int(value) for value in training_manifest['game_ids'])
    smoke = set(int(value) for value in current_holdout)
    excluded_manifests = excluded_manifests or {}
    prior_holdouts = {
        str(name): set(int(value) for value in manifest['game_ids'])
        for name, manifest in excluded_manifests.items()
    }
    prior_union = set().union(*prior_holdouts.values()) if prior_holdouts else set()
    candidates = [int(value) for value in all_game_ids
                  if int(value) not in trained and int(value) not in smoke
                  and int(value) not in prior_union]
    selected = evenly_spaced(candidates, int(count))
    return {
        'schema_version': 'llm-reliability-holdout-v1',
        'selection': ('chronological uniform spacing over games disjoint from v4 training, '
                      'smoke holdout, and prior holdout manifests'),
        'trained_game_count': len(trained),
        'smoke_holdout_game_count': len(smoke),
        'prior_holdout_game_count': len(prior_union),
        'prior_holdouts': {
            name: {
                'game_count': len(ids),
                'overlap': sorted(ids.intersection(selected)),
            }
            for name, ids in prior_holdouts.items()
        },
        'candidate_game_count': len(candidates),
        'game_ids': selected,
        'training_overlap': sorted(trained.intersection(selected)),
        'smoke_overlap': sorted(smoke.intersection(selected)),
        'prior_holdout_overlap': sorted(prior_union.intersection(selected)),
        'required_scenarios': [
            {'side': side, 'forecast_context_enabled': forecast}
            for side in ('红', '蓝') for forecast in (True, False)
        ],
        'evaluation_contract': {
            'sampling': 'uniform decision-time sampling within each game',
            'points_per_game': 12,
            'visibility_policy': 'rmuc_2026_referee_radar_protocol_v1',
            'semantic_metric': 'end_to_end_hindsight_decision_quality',
        },
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-manifest',
                        default='out_m2/temporal_visible_context_dataset_v4_run1/manifest.json')
    parser.add_argument('--current-holdout', nargs='*', type=int, default=(
        1778661259223, 1778661970735, 1778663517697, 1778664469370))
    parser.add_argument('--exclude-manifests', nargs='*', default=(),
                        help='prior holdout manifests whose games must remain excluded')
    parser.add_argument('--count', type=int, default=20)
    parser.add_argument('--output', default='out_m2/llm_reliability_holdout.json')
    args = parser.parse_args(argv)
    import run_m2
    from data import db
    with db.connect() as con:
        all_game_ids = run_m2.list_game_ids(con)
    manifest = json.loads((ROOT / args.training_manifest).read_text(encoding='utf-8'))
    excluded = {
        name: json.loads((ROOT / name).read_text(encoding='utf-8'))
        for name in args.exclude_manifests
    }
    result = build(manifest, args.current_holdout, all_game_ids, args.count,
                   excluded_manifests=excluded)
    if (result['training_overlap'] or result['smoke_overlap'] or
            result['prior_holdout_overlap']):
        raise ValueError('holdout overlap detected')
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'output': str(output), 'games': len(result['game_ids']),
                      'game_ids': result['game_ids']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
