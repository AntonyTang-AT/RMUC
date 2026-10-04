"""Audit holdout coverage and label stability for the HDQ decision metric."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SIDES = ('红', '蓝')
MODES = ('进攻', '防守', '均衡')


def summarize(rows, *, min_mode_rows=12, min_mode_games=12):
    """Summarize precomputed issue rows with hindsight mode and stability."""
    by_side = {}
    for side in SIDES:
        selected = [row for row in rows if row.get('side') == side]
        counts = Counter(row.get('expected_global_mode') for row in selected)
        games = defaultdict(set)
        for row in selected:
            mode = row.get('expected_global_mode')
            if mode in MODES:
                games[mode].add(int(row['game_id']))
        stable = [row for row in selected if row.get('label_stable_next_5') is not None]
        stable_count = sum(row.get('label_stable_next_5') is True for row in stable)
        mode_rows_pass = all(counts[mode] >= int(min_mode_rows) for mode in MODES)
        mode_games_pass = all(len(games[mode]) >= int(min_mode_games) for mode in MODES)
        by_side[side] = {
            'issue_rows': len(selected),
            'mode_rows': {mode: int(counts[mode]) for mode in MODES},
            'mode_complete_games': {mode: len(games[mode]) for mode in MODES},
            'label_stability_next_5': {
                'eligible_rows': len(stable),
                'stable_rows': stable_count,
                'coverage': stable_count / max(1, len(stable)),
            },
            'mode_row_coverage_pass': mode_rows_pass,
            'mode_game_coverage_pass': mode_games_pass,
        }
    return by_side


def audit(game_ids, *, points_per_game=12, lookahead_delta=5,
          min_mode_rows=12, min_mode_games=12):
    import run_m2
    from eval import decision_quality_scorecard as quality
    from eval.evaluate_visible_lora_context import _sample_decision_points

    issue_rows = []
    for game_id in game_ids:
        points, _bounds = run_m2.load_decision_points(int(game_id))
        for point in _sample_decision_points(points, int(points_per_game), sampling='uniform'):
            for side in SIDES:
                issue_rows.append({
                    'game_id': int(game_id),
                    'issued_at_s': float(point['t']),
                    'perspective': side,
                })
    contexts = quality._world_contexts(issue_rows, int(lookahead_delta))
    rows = []
    for issue in issue_rows:
        context = contexts[int(issue['game_id'])][round(float(issue['issued_at_s']), 3)]
        _world, modes, _gm, stability = context
        side = issue['perspective']
        rows.append({
            'game_id': int(issue['game_id']),
            'side': side,
            'expected_global_mode': modes[side],
            'label_stable_next_5': stability[side],
        })
    return {
        'schema_version': 'decision-metric-validity-audit-v1',
        'contract': {
            'metric': 'end_to_end_hindsight_decision_quality',
            'sampling': 'uniform decision-time sampling within each holdout game',
            'lookahead_delta_frames': int(lookahead_delta),
            'stable_label_definition': 'hindsight mode remains unchanged for the next five decision frames',
            'min_mode_rows': int(min_mode_rows),
            'min_mode_games': int(min_mode_games),
        },
        'games': [int(value) for value in game_ids],
        'points_per_game': int(points_per_game),
        'by_side': summarize(rows, min_mode_rows=min_mode_rows,
                             min_mode_games=min_mode_games),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--games-manifest', default='out_m2/llm_reliability_holdout.json')
    parser.add_argument('--points-per-game', type=int, default=12)
    parser.add_argument('--lookahead-delta', type=int, default=5)
    parser.add_argument('--min-mode-rows', type=int, default=12)
    parser.add_argument('--min-mode-games', type=int, default=12)
    parser.add_argument('--output', default='out_m2/decision_metric_validity_audit.json')
    args = parser.parse_args(argv)
    manifest = json.loads((ROOT / args.games_manifest).read_text(encoding='utf-8'))
    result = audit(
        manifest['game_ids'], points_per_game=args.points_per_game,
        lookahead_delta=args.lookahead_delta, min_mode_rows=args.min_mode_rows,
        min_mode_games=args.min_mode_games)
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'output': str(output.resolve()), 'games': len(result['games']),
                      'by_side': result['by_side']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
