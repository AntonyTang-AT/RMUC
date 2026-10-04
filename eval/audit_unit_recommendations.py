"""Audit per-entity advice against later *observed* behavior, not commands.

Run from the project root:
    python eval/audit_unit_recommendations.py

The first eight chronological full match groups are used as a fixed exploratory
sample so this report is comparable with docs/analysis/2026-09-27-decision-metric-audit.md.
No future values are passed to the recommendation builder. Bootstrap units are
whole games, keeping the two sides and repeated frames from one game together.
"""
from __future__ import annotations

import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.db import connect, timeseries_for
from data.radar_visibility_replay import build_radar_visibility_snapshots
from eval.build_battle_replay_visual import _compact_current
from worldstate.functional_zone import assign_functional_zone, zone_centroid
from worldstate.unit_state import candidate_moves

GAME_IDS = (1778631047459, 1778631715625, 1778632575621,
            1778633483984, 1778634274546,
            1778637960624, 1778638792994, 1778639477645)
MOBILE = {'英雄', '工程', '步兵3', '步兵4', '空中', '哨兵'}
SIDES = ('红', '蓝')
SEED = 20260927
BOOTSTRAPS = 4000


def zone(row):
    if row is None or row['x'] is None or row['y'] is None:
        return None
    x, y = float(row['x']), float(row['y'])
    return None if (x == 0 and y == 0) else assign_functional_zone(x, y)


def distance_to_zone(row, name):
    center = zone_centroid(name)
    if row is None or center is None or zone(row) is None:
        return None
    return ((float(row['x'])-center[0])**2 + (float(row['y'])-center[1])**2)**.5


def legal_route_hops(unit_type, side, start, goal):
    """Shortest legal zone-hop distance; -1 means no route is known."""
    if not start or not goal:
        return -1
    if start == goal:
        return 0
    queue = [(start, 0)]
    seen = {start}
    while queue:
        current, hops = queue.pop(0)
        for nxt in candidate_moves(unit_type, side, current):
            if nxt == goal:
                return hops + 1
            if nxt not in seen:
                seen.add(nxt)
                queue.append((nxt, hops + 1))
    return -1


def rate(a, n):
    return a/n if n else None


def bootstrap_paired(game_rows, key_a, key_b, seed=SEED):
    """Cluster bootstrap mean game-level (advice - hold) rate difference."""
    vals = []
    for row in game_rows:
        n_a = row.get(key_a + '_n', 0)
        n_b = row.get(key_b + '_n', 0)
        if n_a and n_b:
            vals.append(row.get(key_a + '_hits', 0)/n_a -
                        row.get(key_b + '_hits', 0)/n_b)
    if not vals:
        return {'games': 0, 'mean_delta_pp': None, 'ci95_pp': None}
    rng = random.Random(seed)
    samples = []
    for _ in range(BOOTSTRAPS):
        picked = [rng.choice(vals) for __ in vals]
        samples.append(sum(picked)/len(picked))
    samples.sort()
    mean = sum(vals)/len(vals)
    return {'games': len(vals), 'mean_delta_pp': round(100*mean, 3),
            'ci95_pp': [round(100*samples[int(.025*BOOTSTRAPS)], 3),
                        round(100*samples[min(BOOTSTRAPS-1, int(.975*BOOTSTRAPS))], 3)]}


def bootstrap_game_mean(game_rows, sum_key, n_key, seed=SEED):
    """Game-cluster CI for a continuous observed-response proxy."""
    vals = [row[sum_key] / row[n_key] for row in game_rows if row.get(n_key, 0)]
    if not vals:
        return {'games': 0, 'mean': None, 'ci95': None}
    rng = random.Random(seed)
    samples = sorted(sum(rng.choice(vals) for _ in vals) / len(vals)
                     for _ in range(BOOTSTRAPS))
    return {'games': len(vals), 'mean': round(sum(vals) / len(vals), 4),
            'ci95': [round(samples[int(.025 * BOOTSTRAPS)], 4),
                     round(samples[min(BOOTSTRAPS-1, int(.975 * BOOTSTRAPS))], 4)]}


def main():
    results = {side: {'advice_match_3s_hits': 0, 'hold_match_3s_hits': 0,
                      'checkable_3s': 0, 'legal_first_steps': 0,
                      'first_step_checkable': 0,
                      'known_provenance': Counter(), 'actions': Counter(),
                      'action_location_pairs': Counter(), 'entity_types': defaultdict(Counter),
                      'nonhold': Counter(), 'nonhold_by_type': defaultdict(Counter),
                      't10': Counter(),
                      'games': {}} for side in SIDES}
    game_sides = defaultdict(lambda: {side: Counter() for side in SIDES})
    used_games = []

    with connect() as con:
        for game_id in GAME_IDS:
            rows = sorted(timeseries_for(con, game_id),
                          key=lambda r: (float(r['时刻秒']), r['阵营'], str(r['robot_id'])))
            if not rows:
                continue
            used_games.append(game_id)
            truth = defaultdict(dict)
            for row in rows:
                t = float(row['时刻秒'])
                truth[t][f"{row['阵营']}:{row['robot_id']}"] = row
            snapshots = build_radar_visibility_snapshots(rows, game_id=game_id)
            first = min(truth)
            for side in SIDES:
                for snap in snapshots[side]:
                    t = float(snap['issued_at'])
                    if abs((t-first)/3 - round((t-first)/3)) > 1e-6:
                        continue
                    later = truth.get(t+3)
                    if later is None:
                        continue
                    all_entities = list(snap['entities'].values())
                    for entity in all_entities:
                        if entity.get('team') != side or entity.get('entity_type') not in MOBILE:
                            continue
                        current = _compact_current(entity, all_entities, side)
                        if current.get('life_state') == 'destroyed':
                            continue
                        advice = current.get('special_state', {}).get('decision_guidance') or {}
                        if not advice.get('applicable') or not advice.get('action'):
                            continue
                        target = advice.get('first_step_zone')
                        start = current.get('zone')
                        actual = later.get(entity['entity_id'])
                        actual_zone = zone(actual)
                        if not start or not target:
                            continue
                        legal = set(candidate_moves(current['type'], side, start)) | {start}
                        rs = results[side]
                        rs['actions'][advice['action']] += 1
                        rs['action_location_pairs'][(advice['action'], start, target)] += 1
                        rs['entity_types'][current['type']][advice['action']] += 1
                        rs['first_step_checkable'] += 1
                        if target in legal:
                            rs['legal_first_steps'] += 1
                        # A second fixed horizon separates immediate movement
                        # response from completion/progress. These remain
                        # observed-behavior proxies, not command execution.
                        later10 = truth.get(t+10)
                        actual10 = later10.get(entity['entity_id']) if later10 else None
                        actual_zone10 = zone(actual10)
                        route = advice.get('path_zones') or []
                        route_goal = (advice.get('target_zone')
                                      if len(route) > 1 and route[0] == start
                                      else target)
                        if actual_zone10 is not None:
                            rs['t10']['checkable'] += 1
                            g10 = game_sides[game_id][side]
                            g10['advice_t10_n'] += 1
                            g10['advice_t10_hits'] += int(actual_zone10 == target)
                            if target == start:
                                rs['t10']['hold_n'] += 1
                                rs['t10']['hold_hits'] += int(actual_zone10 == start)
                            else:
                                rs['t10']['move_n'] += 1
                                rs['t10']['move_arrival_hits'] += int(actual_zone10 == route_goal)
                                g10['move_n'] += 1
                                g10['move_arrival_n'] += 1
                                g10['move_arrival_hits'] += int(actual_zone10 == route_goal)
                                g10['move_stay_n'] += 1
                                g10['move_stay_hits'] += int(actual_zone10 == start)
                                rs['t10']['move_stay_hits'] += int(actual_zone10 == start)
                                d0 = distance_to_zone(truth.get(t, {}).get(entity['entity_id']), route_goal)
                                d10 = distance_to_zone(actual10, route_goal)
                                if d0 is not None and d10 is not None:
                                    rs['t10']['move_distance_n'] += 1
                                    rs['t10']['move_progress_hits'] += int(d10 < d0 - 1e-6)
                                    rs['t10']['move_distance_reduction_sum'] += d0 - d10
                                    g10['move_distance_n'] += 1
                                    g10['move_progress_hits'] += int(d10 < d0 - 1e-6)
                                    g10['move_distance_reduction_sum'] += d0 - d10
                                    if d0 > 1e-6:
                                        progress = max(-1.0, min(1.0, (d0-d10)/d0))
                                        rs['t10']['progress_n'] += 1
                                        rs['t10']['progress_sum'] += progress
                                        rs['t10']['progress_positive'] += int(progress > 1e-6)
                                        g10['progress_n'] += 1
                                        g10['progress_sum'] += progress
                                hops0 = legal_route_hops(current['type'], side, start, route_goal)
                                hops10 = legal_route_hops(current['type'], side, actual_zone10, route_goal)
                                if hops0 >= 0 and hops10 >= 0:
                                    rs['t10']['route_hops_n'] += 1
                                    rs['t10']['route_progress_hits'] += int(hops10 < hops0)
                                    rs['t10']['route_hops_reduced_sum'] += hops0 - hops10
                        if actual_zone is None:
                            continue
                        rs['checkable_3s'] += 1
                        rs['advice_match_3s_hits'] += int(actual_zone == target)
                        rs['hold_match_3s_hits'] += int(actual_zone == start)
                        g = game_sides[game_id][side]
                        g['advice_match_3s_n'] += 1
                        g['advice_match_3s_hits'] += int(actual_zone == target)
                        g['hold_match_3s_n'] += 1
                        g['hold_match_3s_hits'] += int(actual_zone == start)
                        rs['known_provenance']['functional_zone_derived_from_xy'] += 1
                        for ev in advice.get('evidence', []):
                            source = ev.get('source', 'unspecified')
                            label = ('observed' if source in ('current_observation', 'current_position',
                                                                'direct_sensor_field', 'self_sensor')
                                     else 'map_derived' if source == 'functional_zone_graph'
                                     else 'inferred' if source in ('visible_geometry_inference',
                                                                  'functional_zone_prior')
                                     else 'unknown_or_unspecified')
                            rs['known_provenance'][label] += 1
                        if target != start:
                            nr = rs['nonhold']
                            nr['n'] += 1
                            nr['match_3s'] += int(actual_zone == target)
                            nr['hold_match_3s'] += int(actual_zone == start)
                            ds, dt = distance_to_zone(truth.get(t, {}).get(entity['entity_id']), target), distance_to_zone(actual, target)
                            if ds is not None and dt is not None:
                                nr['distance_n'] += 1
                                nr['moved_closer_3s'] += int(dt < ds-1e-6)
                                nr['distance_reduction_sum'] += ds-dt
                                if ds > 1e-6:
                                    progress = max(-1.0, min(1.0, (ds-dt)/ds))
                                    nr['progress_3s_n'] += 1
                                    nr['progress_3s_sum'] += progress
                                    nr['progress_3s_positive'] += int(progress > 1e-6)
                                    g = game_sides[game_id][side]
                                    g['progress_3s_n'] += 1
                                    g['progress_3s_sum'] += progress
                            typ = current['type']
                            b = rs['nonhold_by_type'][typ]
                            b['n'] += 1
                            b['match_3s'] += int(actual_zone == target)
                            b['hold_match_3s'] += int(actual_zone == start)

    out = {'title': 'Per-entity advice observability with t+3/t+10 location proxies',
           'sample_game_ids': used_games, 'sample_games': len(used_games),
           'decision_cadence_seconds': 3,
           'scope': '8 chronological complete games; both sides; observed replay only',
           'results': {}, 'limitations': [
               'No issued movement/target/attack command is stored; observed position is not command agreement.',
               'Position response is not evidence that advice caused movement or improved a match outcome.',
               'Repeated entities/frames are correlated; confidence interval resamples entire games.',
               'Action applicability and evidence provenance describe this rule implementation, not human-validated correctness.'
           ]}
    for side in SIDES:
        r = results[side]
        games = []
        for gid in used_games:
            g = game_sides[gid][side]
            games.append({'game_id': gid, **dict(g)})
        d = bootstrap_paired(games, 'advice_match_3s', 'hold_match_3s')
        den = r['checkable_3s']
        action_json = [{'action': a, 'n': n} for a, n in r['actions'].most_common()]
        pair_json = [{'action': a, 'from_zone': f, 'to_zone': t, 'n': n}
                     for (a, f, t), n in r['action_location_pairs'].most_common(20)]
        nonhold = dict(r['nonhold'])
        if nonhold.get('n'):
            nonhold['advice_zone_match_rate'] = rate(nonhold['match_3s'], nonhold['n'])
            nonhold['stay_put_match_rate'] = rate(nonhold['hold_match_3s'], nonhold['n'])
            nonhold['moved_closer_rate'] = rate(nonhold.get('moved_closer_3s', 0), nonhold.get('distance_n', 0))
            nonhold['mean_distance_reduction_map_units'] = rate(nonhold.get('distance_reduction_sum', 0), nonhold.get('distance_n', 0))
            nonhold['directional_progress_proxy_t_plus_3'] = {
                'metric': 'clipped_normalized_goal_distance_reduction',
                'definition': 'clip((distance_at_issue - distance_at_t+3) / distance_at_issue, -1, 1)',
                'n': nonhold.get('progress_3s_n', 0),
                'mean': rate(nonhold.get('progress_3s_sum', 0), nonhold.get('progress_3s_n', 0)),
                'positive_rate': rate(nonhold.get('progress_3s_positive', 0), nonhold.get('progress_3s_n', 0)),
                'game_cluster_bootstrap': bootstrap_game_mean(
                    [game_sides[gid][side] for gid in used_games], 'progress_3s_sum', 'progress_3s_n'),
                'interpretation': 'positive means the observed robot moved geometrically closer to the recommended first-step zone; not proof the command was issued or followed',
            }
        t10 = dict(r['t10'])
        t10['hold_zone_occupancy_rate'] = rate(t10.get('hold_hits', 0), t10.get('hold_n', 0))
        t10['move_goal_arrival_rate'] = rate(t10.get('move_arrival_hits', 0), t10.get('move_n', 0))
        t10['move_stay_put_rate'] = rate(t10.get('move_stay_hits', 0), t10.get('move_n', 0))
        t10['move_goal_vs_hold_same_rows_game_cluster_delta'] = bootstrap_paired(
            games, 'move_arrival', 'move_stay')
        t10['move_toward_goal_rate'] = rate(t10.get('move_progress_hits', 0), t10.get('move_distance_n', 0))
        t10['mean_goal_distance_reduction_map_units'] = rate(
            t10.get('move_distance_reduction_sum', 0), t10.get('move_distance_n', 0))
        t10['legal_route_progress_rate'] = rate(t10.get('route_progress_hits', 0), t10.get('route_hops_n', 0))
        t10['mean_legal_route_hops_reduced'] = rate(
            t10.get('route_hops_reduced_sum', 0), t10.get('route_hops_n', 0))
        t10['directional_progress_proxy'] = {
            'metric': 'clipped_normalized_goal_distance_reduction',
            'definition': 'clip((distance_at_issue - distance_at_t+10) / distance_at_issue, -1, 1)',
            'n': t10.get('progress_n', 0),
            'mean': rate(t10.get('progress_sum', 0), t10.get('progress_n', 0)),
            'positive_rate': rate(t10.get('progress_positive', 0), t10.get('progress_n', 0)),
            'game_cluster_bootstrap': bootstrap_game_mean(games, 'progress_sum', 'progress_n'),
            'interpretation': 'positive means the observed robot moved geometrically closer to the recommended route goal; not proof the command was issued or followed',
        }
        out['results'][side] = {
            'checkable_recommendations': den,
            'legal_first_step_checkable': r['first_step_checkable'],
            'legal_first_step_rate': rate(r['legal_first_steps'], r['first_step_checkable']),
            'advice_zone_match_t_plus_3_rate': rate(r['advice_match_3s_hits'], den),
            'stay_put_baseline_zone_match_t_plus_3_rate': rate(r['hold_match_3s_hits'], den),
            'game_cluster_bootstrap_advice_minus_stay_put_pp': d,
            'nonhold_advice': nonhold,
            't_plus_10s_observed_response': t10,
            'actions': action_json,
            'most_common_action_routes': pair_json,
            'observability_provenance_counts_not_unique_units': dict(r['known_provenance']),
            'per_type_actions': {k: dict(v) for k, v in r['entity_types'].items()},
            'per_game': games,
        }
        out['results'][side]['metric_layers'] = {
            'feasibility': {
                'name': 'legal_first_step_rate',
                'value': out['results'][side]['legal_first_step_rate'],
                'n': r['first_step_checkable'],
                'definition': '首步区域是否为当前位置或区域图允许的相邻区域；不包含真实行驶时间和安全性',
            },
            'behavior_response': {
                't_plus_3': {
                    'advised_zone_rate': out['results'][side]['advice_zone_match_t_plus_3_rate'],
                    'stay_put_baseline': out['results'][side]['stay_put_baseline_zone_match_t_plus_3_rate'],
                    'paired_game_cluster_delta_pp': d,
                    'nonhold_directional_progress': nonhold.get('directional_progress_proxy_t_plus_3'),
                },
                't_plus_10': {
                    'move_goal_arrival_rate': t10['move_goal_arrival_rate'],
                    'same_rows_stay_put_baseline': t10['move_stay_put_rate'],
                    'goal_vs_stay_paired_game_cluster_delta_pp': t10['move_goal_vs_hold_same_rows_game_cluster_delta'],
                    'directional_progress': t10['directional_progress_proxy'],
                },
                'interpretation': '观测到的位置变化，不是执行/遵从率',
            },
            'outcome_proxy': {
                'status': 'not_scored_in_this_unit_location_audit',
                'available_candidates': ['机器人HP变化', '设施HP变化', '资源阶段变化', '区域控制变化'],
                'interpretation': '须按建议动作和目标逐条匹配；结果变化本身不代表建议造成变化',
            },
            'causal_effect': {
                'status': 'not_identifiable',
                'reason': '缺少控制指令/ACK日志且没有反事实干预；不能把行为响应或结果代理解释成因果收益',
            },
        }
    out_path = ROOT / 'out_m2' / 'unit_recommendation_quality_audit.json'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'output': str(out_path), 'sample_games': used_games,
                      'results': {s: {k: out['results'][s][k] for k in (
                          'checkable_recommendations', 'legal_first_step_rate',
                          'advice_zone_match_t_plus_3_rate',
                          'stay_put_baseline_zone_match_t_plus_3_rate',
                          'game_cluster_bootstrap_advice_minus_stay_put_pp', 'nonhold_advice',
                          't_plus_10s_observed_response')}
                                  for s in SIDES}}, ensure_ascii=False))


if __name__ == '__main__':
    main()
