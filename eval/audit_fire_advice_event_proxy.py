"""Check combat advice against logged firing events (behavior proxy only).

Usage from project root: python eval/audit_fire_advice_event_proxy.py

This test uses the first eight chronological games, both sensor sides, the
existing 3-second decision cadence, and a (t, t+3] window. It never presents a
shot event as proof that an advice caused or instructed a shot.
"""
from __future__ import annotations

import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.db import connect, events_for, timeseries_for
from data.radar_visibility_replay import build_radar_visibility_snapshots
from eval.audit_unit_recommendations import GAME_IDS, MOBILE
from eval.build_battle_replay_visual import _compact_current

SIDES = ('红', '蓝')
TARGET_LABEL = 'continue_attack_visible_target'
SEED = 20260927
BOOTSTRAPS = 4000


def bootstrap_delta(game_rows):
    # Paired game-level delta: firing rate with this advice minus the eligible
    # non-target-advice reference rate, computed within each complete game.
    deltas = []
    for row in game_rows:
        if row['target_n'] and row['reference_n']:
            deltas.append(row['target_fired']/row['target_n'] -
                          row['reference_fired']/row['reference_n'])
    if not deltas:
        return {'games': 0, 'delta_pp': None, 'ci95_pp': None}
    rng = random.Random(SEED)
    samples = sorted(sum(rng.choice(deltas) for _ in deltas)/len(deltas)
                     for __ in range(BOOTSTRAPS))
    delta = sum(deltas)/len(deltas)
    return {'games': len(deltas), 'delta_pp': round(delta*100, 3),
            'ci95_pp': [round(samples[int(.025*BOOTSTRAPS)]*100, 3),
                        round(samples[min(BOOTSTRAPS-1, int(.975*BOOTSTRAPS))]*100, 3)]}


def main():
    data = {side: {'target': Counter(), 'reference': Counter(),
                   'by_type': defaultdict(lambda: {'target': Counter(), 'reference': Counter()}),
                   'by_game': defaultdict(lambda: {'target': Counter(), 'reference': Counter()}),
                   'by_game_type': defaultdict(lambda: defaultdict(
                       lambda: {'target': Counter(), 'reference': Counter()})),
                   'target_ids': Counter(), 'evidence': Counter()}
            for side in SIDES}
    used = []
    with connect() as con:
        for game_id in GAME_IDS:
            rows = sorted(timeseries_for(con, game_id),
                          key=lambda r: (float(r['时刻秒']), r['阵营'], str(r['robot_id'])))
            if not rows:
                continue
            used.append(game_id)
            events = events_for(con, game_id)
            shots = defaultdict(set)
            for event in events:
                if event['事件类型'] == '发弹':
                    shots[(event['阵营'], str(event['robot_id']))].add(float(event['时刻秒']))
            snapshots = build_radar_visibility_snapshots(rows, game_id=game_id)
            first = min(float(row['时刻秒']) for row in rows)
            for side in SIDES:
                for snapshot in snapshots[side]:
                    t = float(snapshot['issued_at'])
                    if abs((t-first)/3 - round((t-first)/3)) > 1e-6:
                        continue
                    entities = list(snapshot['entities'].values())
                    for entity in entities:
                        if entity.get('team') != side or entity.get('entity_type') not in MOBILE:
                            continue
                        current = _compact_current(entity, entities, side)
                        if current.get('life_state') == 'destroyed':
                            continue
                        guidance = current.get('special_state', {}).get('decision_guidance') or {}
                        label = guidance.get('action_label')
                        if not guidance.get('applicable') or not label:
                            continue
                        # Mark presence of one or more observed shots, not command execution.
                        robot_id = str(entity.get('robot_id', entity.get('entity_id', '')).split(':')[-1])
                        fired = int(any(t < shot_t <= t+3 for shot_t in shots[(side, robot_id)]))
                        fired10 = int(any(t < shot_t <= t+10 for shot_t in shots[(side, robot_id)]))
                        group = 'target' if label == TARGET_LABEL else 'reference'
                        row = data[side]
                        row[group]['n'] += 1
                        row[group]['fired'] += fired
                        row[group]['fired10'] += fired10
                        row['by_type'][current['type']][group]['n'] += 1
                        row['by_type'][current['type']][group]['fired'] += fired
                        row['by_type'][current['type']][group]['fired10'] += fired10
                        row['by_game'][game_id][group]['n'] += 1
                        row['by_game'][game_id][group]['fired'] += fired
                        row['by_game_type'][game_id][current['type']][group]['n'] += 1
                        row['by_game_type'][game_id][current['type']][group]['fired'] += fired
                        if group == 'target':
                            row['target_ids']['target_id_known'] += int(guidance.get('target_unit_id') is not None)
                            row['evidence'][str(current.get('special_state', {}).get('evidence', 'unknown'))] += 1

    result = {'sample_games': used, 'sample_count': len(used),
              'question': 'Among active friendly units, does continue_attack_visible_target advice precede a logged shot within 3 seconds more often than other advice?',
              'per_side': {}, 'interpretation': [
                  'A firing event confirms that the robot fired; it does not confirm a command was issued, that the target matched, or that advice caused the shot.',
                  'The reference comprises different actions and states, so this is a descriptive selection/corroboration proxy, not a causal comparison or accuracy.',
                  'The events table has no attacker-to-target link for firing. Target-specific damage cannot be attributed to the advised unit from this join.',
                  'Decision windows are repeated observations; uncertainty resamples whole games, not individual robot-frames.'
              ]}
    for side, d in data.items():
        target, ref = d['target'], d['reference']
        games = []
        for gid in used:
            g = d['by_game'][gid]
            games.append({'game_id': gid, 'target_n': g['target']['n'],
                          'target_fired': g['target']['fired'],
                          'reference_n': g['reference']['n'],
                          'reference_fired': g['reference']['fired']})
        type_balanced_game_deltas = []
        type_pooled = {'target_n': 0, 'target_fired': 0, 'reference_weighted_fired': 0}
        for gid in used:
            strata = d['by_game_type'][gid]
            weighted_delta, weight = 0.0, 0
            for unit, groups in strata.items():
                tn, tf = groups['target']['n'], groups['target']['fired']
                rn, rf = groups['reference']['n'], groups['reference']['fired']
                if not tn or not rn:
                    continue
                w = tn
                weighted_delta += w*(tf/tn-rf/rn)
                weight += w
                type_pooled['target_n'] += tn
                type_pooled['target_fired'] += tf
                type_pooled['reference_weighted_fired'] += w*(rf/rn)
            if weight:
                type_balanced_game_deltas.append(weighted_delta/weight)
        if type_balanced_game_deltas:
            rng = random.Random(SEED)
            samples = sorted(sum(rng.choice(type_balanced_game_deltas)
                                 for _ in type_balanced_game_deltas)/len(type_balanced_game_deltas)
                             for __ in range(BOOTSTRAPS))
            type_bootstrap = {
                'games': len(type_balanced_game_deltas),
                'delta_pp': round(100*sum(type_balanced_game_deltas)/len(type_balanced_game_deltas), 3),
                'ci95_pp': [round(100*samples[int(.025*BOOTSTRAPS)], 3),
                            round(100*samples[min(BOOTSTRAPS-1, int(.975*BOOTSTRAPS))], 3)],
                'method': 'within each game, compare recommendation and non-target-action reference within each same unit type; weight strata by target-advice count; resample whole games',
            }
        else:
            type_bootstrap = {'games': 0, 'delta_pp': None, 'ci95_pp': None}
        result['per_side'][side] = {
            'target_advice': {'n': target['n'], 'shot_event_within_3s': target['fired'],
                              'shot_event_rate': target['fired']/target['n'] if target['n'] else None,
                              'shot_event_within_10s': target['fired10'],
                              'shot_event_rate_10s': target['fired10']/target['n'] if target['n'] else None,
                              'target_entity_id_known_count': d['target_ids']['target_id_known'],
                              'evidence_source_counts': dict(d['evidence'])},
            'other_action_advice_reference': {'n': ref['n'], 'shot_event_within_3s': ref['fired'],
                                               'shot_event_rate': ref['fired']/ref['n'] if ref['n'] else None,
                                               'shot_event_within_10s': ref['fired10'],
                                               'shot_event_rate_10s': ref['fired10']/ref['n'] if ref['n'] else None},
            'descriptive_rate_delta_vs_reference': (target['fired']/target['n']-ref['fired']/ref['n']) if target['n'] and ref['n'] else None,
            'game_cluster_bootstrap_delta': bootstrap_delta(games),
            'same_unit_type_weighted_comparison': {
                **type_bootstrap,
                'pooled_target_n': type_pooled['target_n'],
                'pooled_target_fire_rate': (type_pooled['target_fired']/type_pooled['target_n']
                                            if type_pooled['target_n'] else None),
                'reference_fire_rate_weighted_by_target_counts': (type_pooled['reference_weighted_fired']/type_pooled['target_n']
                                                                 if type_pooled['target_n'] else None),
                'warning': 'Descriptive; non-target-action units differ in state and targeting. Same-type stratification does not remove confounding.',
            },
            'by_type': {unit: {
                'target_advice': {'n': values['target']['n'], 'shots': values['target']['fired'],
                                  'rate': values['target']['fired']/values['target']['n'] if values['target']['n'] else None,
                                  'shots_10s': values['target']['fired10'],
                                  'rate_10s': values['target']['fired10']/values['target']['n'] if values['target']['n'] else None},
                'reference': {'n': values['reference']['n'], 'shots': values['reference']['fired'],
                              'rate': values['reference']['fired']/values['reference']['n'] if values['reference']['n'] else None,
                              'shots_10s': values['reference']['fired10'],
                              'rate_10s': values['reference']['fired10']/values['reference']['n'] if values['reference']['n'] else None},
            } for unit, values in d['by_type'].items()},
            'per_game': games,
        }
    output = ROOT / 'out_m2' / 'fire_advice_event_proxy.json'
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'output': str(output), 'per_side': {
        side: {k: result['per_side'][side][k] for k in (
            'target_advice', 'other_action_advice_reference',
            'descriptive_rate_delta_vs_reference', 'game_cluster_bootstrap_delta')}
        for side in SIDES}}, ensure_ascii=False))


if __name__ == '__main__':
    main()
