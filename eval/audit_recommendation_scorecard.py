"""All-history recommendation response scorecard (observational proxies only).

Run from the project root:
    python eval/audit_recommendation_scorecard.py

Produces action × unit type × viewing side coverage and 10s/30s response
classes.  These classes describe later observable behavior, never controller
execution, recommendation accuracy, or causal utility.
"""
from __future__ import annotations

import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.db import connect, events_for, timeseries_for
from data.radar_visibility_replay import build_radar_visibility_snapshots
from eval.build_battle_replay_visual import _compact_current
from worldstate.functional_zone import (
    FIELD_HEIGHT_M, FIELD_WIDTH_M, ZONE_POLYGONS,
    assign_functional_zone, zone_centroid,
)

SIDES = ('红', '蓝')
HORIZONS = (10, 30)
MOBILE = {'英雄', '工程', '步兵3', '步兵4', '空中', '哨兵'}
BOOTSTRAPS = 1000
SEED = 20260927
EPSILON_MAP_UNITS = 0.05


def _row_zone(row):
    if row is None or row['x'] is None or row['y'] is None:
        return None
    x, y = float(row['x']), float(row['y'])
    return None if (x == 0 and y == 0) else assign_functional_zone(x, y)


def _distance_to_zone_center(row, goal):
    center = zone_centroid(goal) if goal else None
    if row is None or center is None or _row_zone(row) is None:
        return None
    return math.hypot(float(row['x']) - center[0], float(row['y']) - center[1])


def _point_in_polygon(x, y, polygon):
    inside = False
    previous = polygon[-1]
    for current in polygon:
        x1, y1 = previous
        x2, y2 = current
        if (y1 > y) != (y2 > y):
            cross_x = (x2 - x1) * (y - y1) / ((y2 - y1) or 1e-12) + x1
            if x < cross_x:
                inside = not inside
        previous = current
    return inside


def _distance_to_zone_polygon(row, goal):
    """Shortest point-to-edge distance in metres; zero inside the planning polygon."""
    if row is None or goal not in ZONE_POLYGONS or row['x'] is None or row['y'] is None:
        return None
    x, y = float(row['x']), float(row['y'])
    if not (0.0 <= x <= FIELD_WIDTH_M and 0.0 <= y <= FIELD_HEIGHT_M):
        return None
    polygon = ZONE_POLYGONS[goal]
    if _point_in_polygon(x, y, polygon):
        return 0.0
    distances = []
    for (x1, y1), (x2, y2) in zip(polygon, polygon[1:] + polygon[:1]):
        dx, dy = x2 - x1, y2 - y1
        length_sq = dx * dx + dy * dy
        if length_sq <= 1e-12:
            distances.append(math.hypot(x - x1, y - y1))
            continue
        ratio = max(0.0, min(1.0, ((x - x1) * dx + (y - y1) * dy) / length_sq))
        nearest_x, nearest_y = x1 + ratio * dx, y1 + ratio * dy
        distances.append(math.hypot(x - nearest_x, y - nearest_y))
    return min(distances) if distances else None


def _response_class(guidance, current, future, shots, horizon):
    """Return an action-aligned response class and traceable metric record."""
    label = guidance.get('action_label')
    if label == 'continue_attack_visible_target':
        if shots is None:
            return None, {'metric': 'logged_fire_event', 'reason': 'shot_event_join_missing'}
        fired = any(shots)
        return ('aligned' if fired else 'neutral'), {
            'metric': 'logged_fire_event_within_horizon',
            'fire_event_count': sum(shots),
            'reason': 'fire only shows a shot occurred; target linkage and causal relation to advice remain unknown',
        }
    zone0 = current.get('zone')
    zone1 = _row_zone(future)
    if zone1 is None:
        return None, {'metric': None, 'reason': 'future_position_or_zone_missing'}

    goal = guidance.get('movement_goal_zone')
    if goal is None and label in ('retreat_low_hp', 'return_resource_to_supply',
                                  'continue_resource_route'):
        goal = guidance.get('target_zone')
    if goal:
        row0 = current.get('_truth_row')
        polygon_d0 = _distance_to_zone_polygon(row0, goal)
        polygon_d1 = _distance_to_zone_polygon(future, goal)
        center_d0 = _distance_to_zone_center(row0, goal)
        center_d1 = _distance_to_zone_center(future, goal)
        if polygon_d0 is None or polygon_d1 is None:
            return None, {'metric': 'distance_to_recommended_zone_polygon',
                          'goal_zone': goal, 'reason': 'current_or_future_position_out_of_field_or_polygon_missing'}
        delta = polygon_d0 - polygon_d1
        center_delta = (center_d0 - center_d1
                        if center_d0 is not None and center_d1 is not None else None)
        category = ('aligned' if delta > EPSILON_MAP_UNITS else
                    'reverse' if delta < -EPSILON_MAP_UNITS else 'neutral')
        return category, {
            'metric': 'distance_reduction_to_recommended_zone_polygon',
            'goal_zone': goal,
            'distance_reduction_map_units': round(delta, 4),
            'inside_target_polygon': {'at_issue': polygon_d0 == 0.0,
                                      'at_horizon': polygon_d1 == 0.0},
            'centroid_distance_reduction_diagnostic': (round(center_delta, 4)
                                                        if center_delta is not None else None),
            'epsilon_map_units': EPSILON_MAP_UNITS,
            'reason': 'primary metric is shortest distance to the planning polygon (inside=0); centroid change is diagnostic only; neither proves instruction execution',
        }

    if label in ('cover_and_confirm_target', 'guard_defense_zone'):
        if zone0 is None:
            return None, {'metric': 'same_functional_zone', 'reason': 'current_zone_missing'}
        retained = zone1 == zone0
        return ('aligned' if retained else 'reverse'), {
            'metric': 'same_functional_zone_at_horizon',
            'current_zone': zone0,
            'future_zone': zone1,
            'reason': 'zone retention is consistent with hold advice; it is not proof of target confirmation or command execution',
        }

    return None, {'metric': None, 'reason': 'no_action_specific_observable_response_defined'}


def _cluster_ci(games, categories, reps=BOOTSTRAPS, seed=SEED):
    """Whole-game bootstrap intervals for the three response proportions."""
    game_ids = [gid for gid, count in games.items() if count['n']]
    if not game_ids:
        return {'n_games': 0, 'ci95': None}
    rng = random.Random(seed)
    samples = {label: [] for label in categories}
    for _ in range(reps):
        drawn = [rng.choice(game_ids) for __ in game_ids]
        denom = sum(games[gid]['n'] for gid in drawn)
        totals = {label: sum(games[gid][label] for gid in drawn) for label in samples}
        for label in samples:
            samples[label].append(totals[label] / denom if denom else 0.0)
    out = {}
    for label, values in samples.items():
        values.sort()
        out[label] = [round(values[int(.025 * reps)], 4),
                      round(values[min(reps - 1, int(.975 * reps))], 4)]
    return {'n_games': len(game_ids), 'ci95': out}


def main():
    # group -> per-horizon recommendation counts and game-cluster response data
    groups = defaultdict(lambda: {
        'issued': 0,
        'metrics': {h: {'scorable': 0, 'missing': Counter(),
                        'categories': Counter(), 'games': defaultdict(Counter)}
                    for h in HORIZONS},
    })
    games_used = []
    all_counts = Counter()

    with connect() as con:
        game_ids = [int(row[0]) for row in con.execute(
            'SELECT DISTINCT game_id FROM timeseries ORDER BY game_id')]
        for game_id in game_ids:
            rows = sorted(timeseries_for(con, game_id),
                          key=lambda row: (float(row['时刻秒']), row['阵营'], str(row['robot_id'])))
            if not rows:
                continue
            games_used.append(game_id)
            truth = defaultdict(dict)
            for row in rows:
                truth[float(row['时刻秒'])][f"{row['阵营']}:{row['robot_id']}"] = row
            first_time = min(truth)
            snaps = build_radar_visibility_snapshots(rows, game_id=game_id)
            shots = defaultdict(list)
            for event in events_for(con, game_id):
                if event['事件类型'] != '发弹':
                    continue
                shot_team = event['阵营']
                shot_robot = str(event['robot_id'])
                shots[(shot_team, shot_robot)].append(float(event['时刻秒']))

            for side in SIDES:
                for snap in snaps.get(side, []):
                    t = float(snap['issued_at'])
                    if abs((t - first_time) / 3 - round((t - first_time) / 3)) > 1e-6:
                        continue
                    entities = list(snap['entities'].values())
                    by_entity_id = snap['entities']
                    for entity in entities:
                        if entity.get('team') != side or entity.get('entity_type') not in MOBILE:
                            continue
                        current = _compact_current(entity, entities, side)
                        if current.get('life_state') == 'destroyed':
                            continue
                        guidance = current.get('special_state', {}).get('decision_guidance') or {}
                        label = guidance.get('action_label')
                        if not guidance.get('applicable') or not guidance.get('action') or not label:
                            continue
                        unit = current.get('type') or entity.get('entity_type') or '未知'
                        key = (label, side, unit)
                        group = groups[key]
                        group['issued'] += 1
                        all_counts['issued_recommendations'] += 1
                        entity_id = str(entity.get('entity_id'))
                        truth_row0 = truth.get(t, {}).get(entity_id)
                        current['_truth_row'] = truth_row0
                        robot_id = str(entity.get('robot_id', entity_id.split(':')[-1]))
                        event_times = shots.get((side, robot_id), [])

                        for horizon in HORIZONS:
                            record = group['metrics'][horizon]
                            future = truth.get(t + horizon, {}).get(entity_id)
                            if future is None:
                                record['missing']['exact_future_row_missing'] += 1
                                continue
                            all_counts[f'exact_{horizon}s_rows'] += 1
                            window_shots = [shot_t for shot_t in event_times
                                            if t < shot_t <= t + horizon]
                            category, detail = _response_class(
                                guidance, current, future, window_shots, horizon)
                            if category is None:
                                record['missing'][detail.get('reason', 'not_scorable')] += 1
                                continue
                            record['scorable'] += 1
                            record['categories'][category] += 1
                            game_count = record['games'][game_id]
                            game_count['n'] += 1
                            game_count[category] += 1
                            # Persist one compact example in group output, not every frame.
                            record.setdefault('example', {
                                'unit_type': unit,
                                'advice': guidance.get('action'),
                                'reason': guidance.get('action_reason') or guidance.get('short_text'),
                                'metric_detail': detail,
                            })

    results = []
    for (action, side, unit), group in sorted(groups.items()):
        entry = {'recommendation_type': action, 'side': side, 'unit_type': unit,
                 'recommendations_issued': group['issued'], 'horizons': {}}
        for horizon in HORIZONS:
            raw = group['metrics'][horizon]
            denominator = raw['scorable']
            counts = raw['categories']
            game_data = raw['games']
            rates = {label: (counts[label] / denominator if denominator else None)
                     for label in ('aligned', 'neutral', 'reverse')}
            defined_categories = (('aligned', 'neutral') if action == 'continue_attack_visible_target'
                                  else ('aligned', 'neutral', 'reverse'))
            ci = _cluster_ci(game_data, defined_categories,
                             seed=SEED + sum(map(ord, action + side + unit + str(horizon))))
            missing_total = sum(raw['missing'].values())
            entry['horizons'][str(horizon)] = {
                'observed_outcome_n': denominator + missing_total,
                'scorable_n': denominator,
                'coverage_of_issued_advice': round(denominator / group['issued'], 4)
                    if group['issued'] else None,
                'response_denominator': denominator,
                'defined_response_categories': list(defined_categories),
                'response_counts': {label: (counts[label] if label in defined_categories else None)
                                    for label in ('aligned', 'neutral', 'reverse')},
                'response_rates': {label: (round(counts[label] / denominator, 4)
                                           if denominator and label in defined_categories else None)
                                   for label in ('aligned', 'neutral', 'reverse')},
                'game_cluster_bootstrap_95ci': ci,
                'execution_status': {'status': 'unobservable',
                                     'reason': 'no controller command or acknowledgement log; response class is not execution'},
                'unscorable_reasons': dict(raw['missing']),
                'response_metric_example': raw.get('example'),
            }
        results.append(entry)

    output = {
        'scope': {
            'match_count': len(games_used),
            'match_ids': games_used,
            'side_groups': list(SIDES),
            'decision_anchor_seconds': 3,
            'horizons_seconds': list(HORIZONS),
            'source': 'all game_id groups in timeseries table; causal-order rows only; recommendation generated from current radar snapshot; no future row passed to recommendation function',
        },
        'total_counts': dict(all_counts),
        'definitions': {
            'aligned': '后续可观测行为朝建议目标/保持目标区域/出现建议相关开火事件的方向变化；不是执行确认。',
            'neutral': '观测到足够数据，但未出现明显朝目标进展；对攻击建议表示窗口内未观测到新发弹事件。',
            'reverse': '移动建议的目标区中心距离明显增加，或保持区域建议后观测位置离开原区。',
            'movement_metric': f'主要指标：到建议目标区规划多边形边界的最短距离变化（区内距离为0）；减少超过 {EPSILON_MAP_UNITS} 地图单位记 aligned，增加超过阈值记 reverse，其余 neutral。到质心的距离变化只作诊断。多边形来自规划级功能区图，并非毫米级CAD边界。',
            'hold_metric': 'cover_and_confirm_target / guard_defense_zone：10s/30s仍在当前功能区记 aligned，离开记 reverse。',
            'attack_metric': 'continue_attack_visible_target：时间窗内有“发弹”事件记 aligned，无发弹记 neutral；无可观测反向行为定义。发弹不能证明目标匹配、指令执行或建议导致该事件。',
            'confidence_interval': '2000次整场比赛 cluster bootstrap，按建议记录数加权；不是把帧独立重采样。CI 是响应比例的区间。',
        },
        'limits': [
            'aligned / neutral / reverse are observable response categories, not recommendation accuracy, command execution, or causal effect.',
            'Recommendation groups are generated for the currently observed friendly unit states; no issued controller action/acknowledgement log exists.',
            'Movement response uses shortest distance to the planning-level functional-zone polygon (inside=0); centroid distance is retained only as a diagnostic. These polygons are approximate and may not match millimetre-level field edges or intra-zone obstacles.',
            'Hold response only measures zone occupancy; it does not prove target confirmation, successful defense, or tactical value.',
            'Attack response uses a unit-level logged fire event; the event target may be absent or unrelated to the nearby candidate.',
            'Exact 10s/30s rows may be absent near end of a game or during telemetry gaps. Those rows are explicitly excluded from response denominators and reported as missing.',
            'Sparse recommendation × unit × side strata can have unstable bootstrap intervals; always inspect recommendations_issued, response_denominator, game count in the CI, and missing reasons.',
        ],
        'results': results,
    }
    path = ROOT / 'out_m2' / 'recommendation_response_scorecard_all_matches.json'
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding='utf-8')
    brief = {
        'output': str(path),
        'matches': len(games_used),
        'total_recommendations': all_counts['issued_recommendations'],
        'strata': len(results),
        'action_totals': {action: sum(row['recommendations_issued'] for row in results
                                      if row['recommendation_type'] == action)
                          for action in sorted({row['recommendation_type'] for row in results})},
    }
    print(json.dumps(brief, ensure_ascii=False))


if __name__ == '__main__':
    main()
