"""Full-history, action-specific decision response scorecard.

This is a replay-observation audit, not command execution or causal utility.
It scores only fields that can be joined to exact t+3s/t+10s telemetry and
keeps route legality/reachability, observable response, outcome proxies, and
coverage/unknown separate. Results are stratified by action × unit × side.

Run from repository root: python eval/audit_all_history_action_response.py
"""
from __future__ import annotations

import json
import random
import sys
from collections import Counter, defaultdict, deque
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.db import connect, events_for, timeseries_for
from data.radar_visibility_replay import build_radar_visibility_snapshots
from eval.build_battle_replay_visual import _compact_current
from worldstate.functional_zone import (ZONE_POLYGONS, assign_functional_zone,
                                        candidate_moves)

SIDES = ('红', '蓝')
MOBILE = {'英雄', '工程', '步兵3', '步兵4', '空中', '哨兵'}
HORIZONS = (3, 10)
BOOTSTRAPS = 1000
SEED = 20260927


def zone(row):
    if row is None or row['x'] is None or row['y'] is None:
        return None
    x, y = float(row['x']), float(row['y'])
    return None if (x == 0 and y == 0) else assign_functional_zone(x, y)


def hp_ratio(row):
    if row is None:
        return None
    try:
        hp, maximum = float(row['当前血量']), float(row['最大血量'])
    except (TypeError, ValueError):
        return None
    return hp / maximum if maximum > 0 else None


def distance_to_zone_polygon(row, target):
    """Map-metre distance to planning polygon; zero while inside it."""
    if row is None or row['x'] is None or row['y'] is None or target not in ZONE_POLYGONS:
        return None
    x, y = float(row['x']), float(row['y'])
    polygon = ZONE_POLYGONS[target]
    inside = False
    previous = polygon[-1]
    for current in polygon:
        x1, y1 = previous
        x2, y2 = current
        if (y1 > y) != (y2 > y):
            cross_x = (x2-x1)*(y-y1)/((y2-y1) or 1e-12)+x1
            if x < cross_x:
                inside = not inside
        previous = current
    if inside:
        return 0.0
    distances = []
    for (x1, y1), (x2, y2) in zip(polygon, polygon[1:]+polygon[:1]):
        dx, dy = x2-x1, y2-y1
        length2 = dx*dx+dy*dy
        ratio = max(0.0, min(1.0, ((x-x1)*dx+(y-y1)*dy)/length2)) if length2 else 0.0
        distances.append(((x-(x1+ratio*dx))**2+(y-(y1+ratio*dy))**2)**.5)
    return min(distances) if distances else None


def route_hops(unit, side, start, goal):
    if not start or not goal:
        return None
    if start == goal:
        return 0
    todo, seen = deque([(start, 0)]), {start}
    while todo:
        current, hops = todo.popleft()
        for nxt in candidate_moves(unit, side, current):
            if nxt == goal:
                return hops + 1
            if nxt not in seen:
                seen.add(nxt)
                todo.append((nxt, hops + 1))
    return None


def check_route(unit, side, start, first, goal, path):
    """Return tri-state first-step/full-path legality and goal reachability."""
    first_legal = None
    if start and first:
        first_legal = first == start or first in candidate_moves(unit, side, start)
    full_legal = None
    if path:
        path = list(path)
        if start and goal and path[0] == start and path[-1] == goal:
            full_legal = all(b in candidate_moves(unit, side, a)
                             for a, b in zip(path, path[1:]))
        else:
            full_legal = False
    hops = route_hops(unit, side, start, goal)
    return first_legal, full_legal, (hops is not None if goal and start else None), hops


def confirmed_target_id_for_scoring(advice):
    """Return a target ID only when the issue-time sensor explicitly linked it.

    Nearest-visible geometry is useful context for a recommendation, but it
    cannot identify which robot a projectile is aimed at. Do not score that
    candidate's HP/survival as the outcome of a target-specific advice.
    """
    if not isinstance(advice, dict):
        return None
    evaluation = advice.get('evaluation') or {}
    if evaluation.get('target_linkage') != 'explicit_sensor_target':
        return None
    return advice.get('target_unit_id')


def add_metric(group, game, metric, hit=None, value=None):
    group[metric + '_n'] += 1
    game[metric + '_n'] += 1
    if hit is not None:
        group[metric + '_hits'] += int(bool(hit))
        game[metric + '_hits'] += int(bool(hit))
    if value is not None:
        group[metric + '_sum'] += float(value)
        game[metric + '_sum'] += float(value)


def cluster_ci(game_cells, metric, *, seed):
    cells = [(g, c) for g, c in game_cells.items() if c[metric + '_n']]
    if not cells:
        return {'n_games': 0, 'ci95': None}
    rng = random.Random(seed)
    rates = []
    for _ in range(BOOTSTRAPS):
        sample = [rng.choice(cells)[1] for __ in cells]
        den = sum(c[metric + '_n'] for c in sample)
        num = sum(c[metric + '_hits'] for c in sample)
        rates.append(num / den if den else 0.0)
    rates.sort()
    return {'n_games': len(cells), 'ci95': [round(rates[int(.025*BOOTSTRAPS)], 4),
            round(rates[min(BOOTSTRAPS-1, int(.975*BOOTSTRAPS))], 4)]}


def new_group():
    return {'issued': 0, 'unknown': Counter(), 'route': Counter(),
            'horizon': {h: Counter() for h in HORIZONS},
            'games': {h: defaultdict(Counter) for h in HORIZONS}}


def main():
    groups = defaultdict(new_group)
    games_used = []
    total = Counter()
    with connect() as con:
        game_ids = [int(r[0]) for r in con.execute(
            'SELECT DISTINCT game_id FROM timeseries ORDER BY game_id')]
        for gid in game_ids:
            rows = sorted(timeseries_for(con, gid),
                          key=lambda r: (float(r['时刻秒']), r['阵营'], str(r['robot_id'])))
            if not rows:
                continue
            games_used.append(gid)
            truth = defaultdict(dict)
            for row in rows:
                truth[float(row['时刻秒'])][f"{row['阵营']}:{row['robot_id']}"] = row
            first_time = min(truth)
            snaps = build_radar_visibility_snapshots(rows, game_id=gid)
            side_snaps = {s: {float(x['issued_at']): x for x in snaps.get(s, [])}
                          for s in SIDES}
            fire_events = defaultdict(list)
            for event in events_for(con, gid):
                if event['事件类型'] == '发弹':
                    fire_events[(event['阵营'], str(event['robot_id']))].append(
                        float(event['时刻秒']))

            for side in SIDES:
                for snap in snaps.get(side, []):
                    t = float(snap['issued_at'])
                    if abs((t-first_time)/3-round((t-first_time)/3)) > 1e-6:
                        continue
                    entities = list(snap['entities'].values())
                    for entity in entities:
                        if entity.get('team') != side or entity.get('entity_type') not in MOBILE:
                            continue
                        current = _compact_current(entity, entities, side)
                        if current.get('life_state') == 'destroyed':
                            continue
                        advice = (current.get('special_state') or {}).get('decision_guidance') or {}
                        action = advice.get('action_label')
                        if not advice.get('applicable') or not advice.get('action') or not action:
                            continue
                        unit = current.get('type') or entity.get('entity_type') or '未知'
                        key = (action, unit, side)
                        group = groups[key]
                        group['issued'] += 1
                        total['issued'] += 1
                        start = current.get('zone')
                        goal = advice.get('movement_goal_zone')
                        if goal is None and action in ('retreat_low_hp',
                                'return_resource_to_supply', 'continue_resource_route'):
                            goal = advice.get('target_zone')
                        first = advice.get('first_step_zone')
                        path = advice.get('path_zones') or []
                        moving = bool(goal and goal != start)
                        if moving:
                            group['route']['movement_issued'] += 1
                            first_ok, path_ok, reachable, hops0 = check_route(
                                unit, side, start, first, goal, path)
                            for metric, result in (('first_step_legal', first_ok),
                                                   ('full_path_legal', path_ok),
                                                   ('goal_reachable', reachable)):
                                if result is None:
                                    group['route'][metric + '_unknown'] += 1
                                else:
                                    group['route'][metric + '_n'] += 1
                                    group['route'][metric + '_yes'] += int(result)
                            if hops0 is None:
                                group['route']['shortest_hops_unknown'] += 1
                            else:
                                group['route']['shortest_hops_n'] += 1
                                group['route']['shortest_hops_sum'] += hops0
                        else:
                            group['route']['non_movement_or_no_goal'] += 1

                        entity_id = str(entity.get('entity_id'))
                        robot_id = str(entity.get('robot_id', entity_id.split(':')[-1]))
                        evaluation = advice.get('evaluation') or {}
                        explicit_target = (
                            action == 'continue_attack_visible_target'
                            and evaluation.get('target_linkage') == 'explicit_sensor_target')
                        target_id = (confirmed_target_id_for_scoring(advice)
                                     if explicit_target else None)
                        for h in HORIZONS:
                            cell = group['horizon'][h]
                            game_cell = group['games'][h][gid]
                            future = truth.get(t+h, {}).get(entity_id)
                            if future is None:
                                cell['exact_future_row_missing'] += 1
                                group['unknown'][f'exact_{h}s_future_row_missing'] += 1
                                continue
                            cell['future_row_n'] += 1
                            game_cell['future_row_n'] += 1
                            z1 = zone(future)
                            if z1 is None:
                                cell['future_zone_missing'] += 1
                            else:
                                cell['future_zone_n'] += 1
                                game_cell['future_zone_n'] += 1
                                if moving:
                                    h1 = route_hops(unit, side, z1, goal)
                                    if hops0 is not None and h1 is not None:
                                        # positive = fewer legal graph hops remain
                                        add_metric(cell, game_cell, 'route_hop_progress',
                                                  h1 < hops0, hops0-h1)
                                    else:
                                        cell['route_hop_progress_unknown'] += 1
                                elif action in ('cover_and_confirm_target', 'guard_defense_zone'):
                                    add_metric(cell, game_cell, 'hold_zone_response', z1 == start)
                            if moving:
                                d0 = distance_to_zone_polygon(truth.get(t, {}).get(entity_id), goal)
                                d1 = distance_to_zone_polygon(future, goal)
                                if d0 is not None and d1 is not None:
                                    # positive = closer to target zone in continuous map metres
                                    add_metric(cell, game_cell, 'goal_distance_reduction_m', value=d0-d1)
                                else:
                                    cell['goal_distance_unknown'] += 1

                            hp0, hp1 = hp_ratio(truth.get(t, {}).get(entity_id)), hp_ratio(future)
                            if hp0 is None or hp1 is None:
                                cell['own_hp_unknown'] += 1
                            else:
                                add_metric(cell, game_cell, 'own_hp_delta', value=hp1-hp0)
                                add_metric(cell, game_cell, 'own_survival', hp1 > 0)
                            if action not in ('continue_attack_visible_target',
                                               'cover_and_confirm_target'):
                                cell['target_state_not_applicable'] += 1
                            elif not explicit_target:
                                cell['target_identity_unconfirmed'] += 1
                            elif target_id is None:
                                cell['target_id_unknown'] += 1
                            else:
                                target0 = truth.get(t, {}).get(str(target_id))
                                target1 = truth.get(t+h, {}).get(str(target_id))
                                th0, th1 = hp_ratio(target0), hp_ratio(target1)
                                if th0 is None or th1 is None:
                                    cell['target_state_unknown'] += 1
                                else:
                                    add_metric(cell, game_cell, 'target_hp_delta', value=th1-th0)
                                    add_metric(cell, game_cell, 'target_survival', th1 > 0)
                            if action in ('continue_attack_visible_target', 'cover_and_confirm_target'):
                                shots = fire_events.get((side, robot_id), [])
                                fired = any(t < shot <= t+h for shot in shots)
                                add_metric(cell, game_cell, 'observed_firing_event', fired)

                            # Execution/command logging is absent in this database.
                            cell['execution_unknown'] += 1

    results = []
    for (action, unit, side), group in sorted(groups.items()):
        row = {'action_label': action, 'unit_type': unit, 'side': side,
               'issued_advice_n': group['issued'], 'route': {}, 'horizons': {}}
        route = group['route']
        row['route'] = {'movement_advice_n': route['movement_issued'],
                        'non_movement_or_no_goal_n': route['non_movement_or_no_goal']}
        for metric in ('first_step_legal', 'full_path_legal', 'goal_reachable'):
            n, yes = route[metric+'_n'], route[metric+'_yes']
            row['route'][metric] = {'yes_n': yes, 'scorable_n': n,
                'unknown_n': route[metric+'_unknown'],
                'rate': round(yes/n, 4) if n else None,
                'coverage_of_movement_advice': round(n/max(1,route['movement_issued']), 4)}
        row['route']['mean_shortest_legal_hops_to_goal'] = (
            round(route['shortest_hops_sum']/route['shortest_hops_n'], 3)
            if route['shortest_hops_n'] else None)
        for h in HORIZONS:
            c = group['horizon'][h]
            games = group['games'][h]
            den = c['future_row_n']
            horizon = {'issued_advice_n': group['issued'], 'exact_future_row_n': den,
                'exact_future_row_coverage': round(den/group['issued'], 4) if group['issued'] else None,
                'future_zone_n': c['future_zone_n'], 'unscorable_or_unknown': {},
                'metrics': {}, 'execution': {'status': 'unknown', 'n': 0,
                    'coverage': 0.0, 'reason': 'no controller command/ack telemetry'}}
            for k, v in c.items():
                if k.endswith('_missing') or k.endswith('_unknown') or k in (
                        'target_id_unknown', 'target_state_unknown',
                        'target_identity_unconfirmed', 'execution_unknown'):
                    horizon['unscorable_or_unknown'][k] = v
            for metric in ('route_hop_progress', 'hold_zone_response', 'own_survival',
                           'target_survival', 'observed_firing_event'):
                n, hits = c[metric+'_n'], c[metric+'_hits']
                if n or metric in ('route_hop_progress', 'hold_zone_response',
                                   'target_survival', 'observed_firing_event'):
                    horizon['metrics'][metric] = {'hits_n': hits, 'scorable_n': n,
                        'coverage_of_issued_advice': round(n/group['issued'], 4)
                            if group['issued'] else None,
                        'rate': round(hits/n, 4) if n else None,
                        'game_cluster_bootstrap_95ci': cluster_ci(
                            games, metric, seed=SEED+sum(map(ord, action+unit+side+metric+str(h))))}
            for metric in ('own_hp_delta', 'target_hp_delta', 'goal_distance_reduction_m'):
                n = c[metric+'_n']
                value_key = ('mean_distance_reduction_m' if metric == 'goal_distance_reduction_m'
                             else 'mean_fractional_hp_change')
                horizon['metrics'][metric] = {value_key:
                    round(c[metric+'_sum']/n, 4) if n else None, 'scorable_n': n,
                    'coverage_of_issued_advice': round(n/group['issued'], 4)
                        if group['issued'] else None,
                    'interpretation': ('planning-polygon distance reduction in metres; movement response proxy'
                        if metric == 'goal_distance_reduction_m' else
                        'observed outcome proxy; confounded by the whole match')}
            row['horizons'][str(h)] = horizon
        results.append(row)

    output = {'schema_version': 1,
        'scope': {'games_n': len(games_used), 'game_ids': games_used,
            'sides': list(SIDES), 'decision_issue_step_s': 3,
            'response_horizons_s': list(HORIZONS), 'issued_advice_n': total['issued'],
            'join': 'same game + same entity + exact timestamp t+3 or t+10; future telemetry is scoring-only'},
        'definitions': {
            'route_hop_progress': 'shortest legal adjacency hops remaining decreased by t+H; descriptive movement consistency only',
            'hold_zone_response': 'same functional zone at t+H for hold/guard action labels',
            'observed_firing_event': 'own logged fire event in (t,t+H]; does not identify target hit or advice-caused firing',
            'target_hp/survival_scoring': 'only sensor-confirmed explicit target IDs are linked; nearest-visible geometric candidates remain unconfirmed and are excluded',
            'own_hp_delta/target_hp_delta': 'change in normalized HP; confounded observational outcome proxy',
            'unknown': 'not scorable due to missing row, location, HP, target ID/state, or absent controller logs; never imputed as failure',
            'route': 'functional-zone planning graph only, not physical collision-free path or ETA'},
        'limitations': [
            'Behavior response does not prove the controller received or executed the recommendation.',
            'HP, survival, target state, and fire events are influenced by all participants; no causal effect is identified.',
            'A logged fire event is not a target-specific hit. The target may be missing or unrelated.',
            'Route legality/reachability uses an approximate functional-zone graph; it is not CAD-level physical feasibility.',
            'Exact t+3/t+10 absent from a match-end tail is censored and counted as missing, not a negative result.',
            'Strata with fewer than five contributing games should be treated as unstable even when row counts are large.' ],
        'results': results}
    path = ROOT/'out_m2'/'all_history_action_response_scorecard.json'
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding='utf-8')
    brief = {'output': str(path), 'games_n': len(games_used),
             'issued_advice_n': total['issued'], 'strata_n': len(results),
             'action_totals': {a: sum(r['issued_advice_n'] for r in results
                                      if r['action_label'] == a)
                               for a in sorted({r['action_label'] for r in results})}}
    print(json.dumps(brief, ensure_ascii=False))


if __name__ == '__main__':
    main()
