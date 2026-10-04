"""Action-label by unit-type audit of observable responses and constraints.

Run from the repository root: python eval/audit_advice_action_type_response.py
This is an observational response audit, not action execution or causal utility.
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
from eval.audit_unit_recommendations import GAME_IDS, MOBILE
from eval.build_battle_replay_visual import _compact_current
from worldstate.functional_zone import assign_functional_zone, candidate_moves, zone_centroid

SIDES = ('红', '蓝')
BOOTSTRAPS = 4000
SEED = 20260927


def zone(row):
    if row is None or row['x'] is None or row['y'] is None:
        return None
    x, y = float(row['x']), float(row['y'])
    return None if (x == 0 and y == 0) else assign_functional_zone(x, y)


def hp(row):
    try:
        return float(row['当前血量']) if row and row['当前血量'] is not None else None
    except (TypeError, ValueError):
        return None


def dist_to_zone(row, target):
    c = zone_centroid(target) if target else None
    if c is None or zone(row) is None:
        return None
    return ((float(row['x'])-c[0])**2 + (float(row['y'])-c[1])**2)**.5


def route_is_legal(unit, side, start, first, path):
    if not start or not first:
        return None
    if first == start:
        return True
    route = list(path or [])
    if len(route) < 2 or route[0] != start or route[1] != first:
        return False
    return all(b in set(candidate_moves(unit, side, a)) for a, b in zip(route, route[1:]))


def bootstrap_rate(group_game_counts, hit_key, n_key, *, seed=SEED):
    games = [g for g, r in group_game_counts.items() if r[n_key] > 0]
    if not games:
        return {'n_games': 0, 'ci95': None}
    rng = random.Random(seed)
    values = []
    for _ in range(BOOTSTRAPS):
        chosen = [rng.choice(games) for __ in games]
        den = sum(group_game_counts[g][n_key] for g in chosen)
        values.append(sum(group_game_counts[g][hit_key] for g in chosen)/den if den else 0.0)
    values.sort()
    return {'n_games': len(games),
            'ci95': [round(values[int(.025*BOOTSTRAPS)], 4),
                     round(values[min(BOOTSTRAPS-1, int(.975*BOOTSTRAPS))], 4)]}


def add_rate(group, game, name, hit):
    group[name + '_n'] += 1
    group[name + '_hits'] += int(bool(hit))
    game[name + '_n'] += 1
    game[name + '_hits'] += int(bool(hit))


def main():
    aggregate = defaultdict(Counter)
    by_game = defaultdict(lambda: defaultdict(Counter))
    scope_counts = Counter()
    used = []
    with connect() as con:
        for gid in GAME_IDS:
            rows = sorted(timeseries_for(con, gid),
                          key=lambda r: (float(r['时刻秒']), r['阵营'], str(r['robot_id'])))
            if not rows:
                continue
            used.append(gid)
            truth = defaultdict(dict)
            for row in rows:
                truth[float(row['时刻秒'])][f"{row['阵营']}:{row['robot_id']}"] = row
            snapshots = {side: list(items) for side, items in
                         build_radar_visibility_snapshots(rows, game_id=gid).items()}
            snap_by_time = {side: {float(s['issued_at']): s for s in items}
                            for side, items in snapshots.items()}
            first = min(truth)
            for side in SIDES:
                for snap in snapshots[side]:
                    t = float(snap['issued_at'])
                    if abs((t-first)/3 - round((t-first)/3)) > 1e-6:
                        continue
                    later_snaps = {h: snap_by_time[side].get(t+h) for h in (3, 10)}
                    all_entities = list(snap['entities'].values())
                    for entity in all_entities:
                        if entity.get('team') != side or entity.get('entity_type') not in MOBILE:
                            continue
                        current = _compact_current(entity, all_entities, side)
                        if current.get('life_state') == 'destroyed':
                            continue
                        guidance = current.get('special_state', {}).get('decision_guidance') or {}
                        label = guidance.get('action_label')
                        if not guidance.get('applicable') or not label or not guidance.get('action'):
                            continue
                        unit = current['type']
                        key = (label, unit, side)
                        a, g = aggregate[key], by_game[key][gid]
                        a['advice_n'] += 1
                        g['advice_n'] += 1
                        scope_counts[(side, unit, label)] += 1
                        start = current.get('zone')
                        first_step = guidance.get('first_step_zone')
                        path = guidance.get('path_zones') or []
                        if first_step is not None and start is not None:
                            add_rate(a, g, 'first_step_legal',
                                     route_is_legal(unit, side, start, first_step, path))
                        if first_step and start and first_step != start:
                            a['movement_advice_n'] += 1
                            g['movement_advice_n'] += 1
                        else:
                            a['hold_advice_n'] += 1
                            g['hold_advice_n'] += 1

                        entity_id = entity['entity_id']
                        row0 = truth.get(t, {}).get(entity_id)
                        for h in (3, 10):
                            future = truth.get(t+h, {}).get(entity_id)
                            if future is None:
                                continue
                            a[f'observed_{h}s_n'] += 1
                            g[f'observed_{h}s_n'] += 1
                            future_zone = zone(future)
                            hp0, hp1 = hp(row0), hp(future)
                            if hp0 is not None and hp1 is not None:
                                a[f'hp_delta_{h}s_sum'] += hp1-hp0
                                a[f'hp_delta_{h}s_n'] += 1
                                g[f'hp_delta_{h}s_sum'] += hp1-hp0
                                g[f'hp_delta_{h}s_n'] += 1
                                add_rate(a, g, f'survived_{h}s', hp1 > 0)
                            if start and future_zone:
                                add_rate(a, g, f'zone_retained_{h}s', future_zone == start)
                                if first_step:
                                    add_rate(a, g, f'first_step_reached_{h}s', future_zone == first_step)
                                if first_step and first_step != start:
                                    d0, d1 = dist_to_zone(row0, first_step), dist_to_zone(future, first_step)
                                    if d0 is not None and d1 is not None:
                                        add_rate(a, g, f'moved_closer_{h}s', d1 < d0-1e-6)
                                        a[f'distance_reduction_{h}s_sum'] += d0-d1
                                        a[f'distance_reduction_{h}s_n'] += 1
                                    if len(path) > 1:
                                        if future_zone in path:
                                            progressed = path.index(future_zone) > path.index(start) if start in path else False
                                            add_rate(a, g, f'path_progress_{h}s', progressed)
                                        else:
                                            add_rate(a, g, f'path_zone_observed_{h}s', False)
                        # Cover/confirm advice has one directly observable short-term
                        # signal: a new own firing/ammo delta. This remains response,
                        # not confirmation that this advice caused the fire.
                        if label == 'cover_and_confirm_target':
                            for h in (3, 10):
                                future_snap = later_snaps[h]
                                if future_snap is None:
                                    continue
                                later_entity = future_snap['entities'].get(entity_id)
                                if later_entity is None:
                                    continue
                                future_current = _compact_current(
                                    later_entity, list(future_snap['entities'].values()), side)
                                status = future_current.get('special_state', {}).get('attack_status')
                                fire = status == 'attacking'
                                add_rate(a, g, f'fire_state_by_{h}s', fire)
                                target_id = guidance.get('target_unit_id')
                                if target_id is not None:
                                    target_visible = str(target_id) in future_snap['entities']
                                    add_rate(a, g, f'same_candidate_visible_{h}s', target_visible)

    results = []
    for (label, unit, side), counts in sorted(aggregate.items()):
        game_counts = by_game[(label, unit, side)]
        row = {'action_label': label, 'unit_type': unit, 'view_side': side,
               'advice_n': counts['advice_n'], 'movement_advice_n': counts['movement_advice_n'],
               'hold_advice_n': counts['hold_advice_n'],
               'first_step_legal_rate': counts['first_step_legal_hits']/counts['first_step_legal_n']
                   if counts['first_step_legal_n'] else None,
               'first_step_legal_n': counts['first_step_legal_n'], 'metrics': {}}
        for metric, denom_key in (
            ('zone_retained_3s', 'zone_retained_3s_n'),
            ('zone_retained_10s', 'zone_retained_10s_n'),
            ('first_step_reached_3s', 'first_step_reached_3s_n'),
            ('first_step_reached_10s', 'first_step_reached_10s_n'),
            ('moved_closer_3s', 'moved_closer_3s_n'),
            ('moved_closer_10s', 'moved_closer_10s_n'),
            ('path_progress_3s', 'path_progress_3s_n'),
            ('path_progress_10s', 'path_progress_10s_n'),
            ('fire_state_by_3s', 'fire_state_by_3s_n'),
            ('fire_state_by_10s', 'fire_state_by_10s_n'),
            ('same_candidate_visible_3s', 'same_candidate_visible_3s_n'),
            ('same_candidate_visible_10s', 'same_candidate_visible_10s_n'),
        ):
            n = counts[denom_key]
            hit_key = metric + '_hits'
            if not n:
                continue
            ci = bootstrap_rate(game_counts, hit_key, denom_key,
                                seed=SEED + sum(map(ord, label+unit+side+metric)))
            row['metrics'][metric] = {
                'hits': counts[hit_key], 'n': n, 'rate': round(counts[hit_key]/n, 4),
                'game_cluster_bootstrap_95ci': ci,
            }
        for h in (3, 10):
            n = counts[f'hp_delta_{h}s_n']
            if n:
                row['metrics'][f'mean_own_hp_delta_{h}s'] = {
                    'mean': round(counts[f'hp_delta_{h}s_sum']/n, 3), 'n': n,
                    'interpretation': 'observed outcome association; no action attribution',
                }
            n = counts[f'distance_reduction_{h}s_n']
            if n:
                row['metrics'][f'mean_distance_reduction_to_first_step_{h}s'] = {
                    'map_units': round(counts[f'distance_reduction_{h}s_sum']/n, 4), 'n': n,
                }
        results.append(row)

    output = {
        'sample_game_ids': used, 'sample_games': len(used), 'sides': list(SIDES),
        'decision_step_seconds': 3,
        'method': 'action_label × unit_type × view_side; exact t+3/t+10 replay rows; 4,000 bootstrap draws resampling whole games',
        'results': results,
        'limits': [
            'Action response means a subsequent observed state/event aligned with advice; it is not proof the command was issued or executed.',
            'The DB has no unit-level controller command/ack log. HP/zone changes are confounded by other units and opponent actions.',
            'Candidate visibility uses the radar snapshot and does not prove firing at that candidate.',
            'Small strata are unstable; report each numerator/denominator and game-cluster CI, and suppress conclusions when fewer than 5 games contribute.',
            'Stay/hold advice naturally has high zone retention; compare it as a response measure, not policy benefit.'
        ],
    }
    path = ROOT / 'out_m2' / 'advice_action_type_response.json'
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding='utf-8')
    # Print only actionable strata for a quick, legible run summary.
    print(json.dumps({'output': str(path), 'sample_games': len(used),
                      'n_strata': len(results),
                      'top': [{k: x.get(k) for k in ('action_label','unit_type','view_side','advice_n','movement_advice_n','hold_advice_n','first_step_legal_rate','metrics')}
                              for x in results if x['advice_n'] >= 20]}, ensure_ascii=False))


if __name__ == '__main__':
    main()
