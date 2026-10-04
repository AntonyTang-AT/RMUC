"""Verify hero 42mm event coverage and applicability of fire-continuation advice."""
from __future__ import annotations

import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data.db import connect, events_for, timeseries_for
from data.radar_visibility_replay import build_radar_visibility_snapshots
from eval.audit_unit_recommendations import GAME_IDS, SIDES
from eval.build_battle_replay_visual import _compact_current


def main():
    out = {'sample_games': [], 'event_coverage': {}, 'rule': {}, 'limits': [
        'Official events do not map a firing event to a target entity.',
        'A 3-second no-shot window can be correct for a robot whose target leaves line of sight, whose weapon is cooling, or whose tactical action changes.',
        'Event and ammo counters verify occurrence/coverage only; they do not prove recommendation execution or utility.'
    ]}
    with connect() as con:
        full_coverage = con.execute("""
            WITH ammo_deltas AS (
              SELECT game_id, 阵营, robot_id,
                     SUM(CASE WHEN previous_ammo IS NOT NULL
                               THEN MAX(0.0, 累计42mm发弹 - previous_ammo)
                               ELSE 0.0 END) AS ammo42_delta
              FROM (
                SELECT game_id, 阵营, robot_id, 累计42mm发弹,
                       LAG(累计42mm发弹) OVER (
                         PARTITION BY game_id, 阵营, robot_id ORDER BY 时刻秒
                       ) AS previous_ammo
                FROM timeseries
                WHERE 机器人类型='英雄' AND 累计42mm发弹 IS NOT NULL
              )
              GROUP BY game_id, 阵营, robot_id
            ), fire_events AS (
              SELECT game_id, 阵营, robot_id, COUNT(*) AS event_count
              FROM events
              WHERE 事件类型='发弹' AND 类别='42mm' AND 机器人类型='英雄'
              GROUP BY game_id, 阵营, robot_id
            ), joined AS (
              SELECT ammo_deltas.game_id, ammo_deltas.阵营, ammo_deltas.robot_id,
                     ammo_deltas.ammo42_delta, COALESCE(fire_events.event_count,0) AS event_count
              FROM ammo_deltas LEFT JOIN fire_events
                USING(game_id, 阵营, robot_id)
            )
            SELECT COUNT(*) AS hero_robot_games,
                   SUM(CASE WHEN ammo42_delta>0 THEN 1 ELSE 0 END) AS with_ammo_delta,
                   SUM(CASE WHEN event_count>0 THEN 1 ELSE 0 END) AS with_42mm_event,
                   SUM(CASE WHEN ammo42_delta>0 AND event_count>0 THEN 1 ELSE 0 END) AS both_positive,
                   SUM(CASE WHEN ammo42_delta>0 AND event_count=0 THEN 1 ELSE 0 END) AS ammo_only,
                   SUM(CASE WHEN ammo42_delta=0 AND event_count>0 THEN 1 ELSE 0 END) AS event_only,
                   SUM(ammo42_delta) AS ammo42_positive_delta_sum,
                   SUM(event_count) AS event42_total,
                   COUNT(DISTINCT game_id) AS games
            FROM joined
        """).fetchone()
        mismatch_rows = con.execute("""
            WITH ammo_deltas AS (
              SELECT game_id, 阵营, robot_id,
                     SUM(CASE WHEN previous_ammo IS NOT NULL
                               THEN MAX(0.0, 累计42mm发弹 - previous_ammo)
                               ELSE 0.0 END) AS ammo42_delta
              FROM (
                SELECT game_id, 阵营, robot_id, 累计42mm发弹,
                       LAG(累计42mm发弹) OVER (
                         PARTITION BY game_id, 阵营, robot_id ORDER BY 时刻秒
                       ) AS previous_ammo
                FROM timeseries
                WHERE 机器人类型='英雄' AND 累计42mm发弹 IS NOT NULL
              )
              GROUP BY game_id, 阵营, robot_id
            ), fire_events AS (
              SELECT game_id, 阵营, robot_id, COUNT(*) AS event_count
              FROM events
              WHERE 事件类型='发弹' AND 类别='42mm' AND 机器人类型='英雄'
              GROUP BY game_id, 阵营, robot_id
            )
            SELECT a.game_id, a.阵营 AS side, a.robot_id,
                   a.ammo42_delta, COALESCE(e.event_count,0) AS event_count,
                   a.ammo42_delta - COALESCE(e.event_count,0) AS delta
            FROM ammo_deltas a LEFT JOIN fire_events e
              USING(game_id, 阵营, robot_id)
            WHERE ABS(a.ammo42_delta - COALESCE(e.event_count,0)) > 1e-6
            ORDER BY ABS(a.ammo42_delta - COALESCE(e.event_count,0)) DESC,
                     a.game_id, side, a.robot_id
        """).fetchall()
        interval_rows = con.execute("""
            SELECT robot_type, gap_s FROM (
              SELECT 机器人类型 AS robot_type,
                     时刻秒 - LAG(时刻秒) OVER (
                       PARTITION BY game_id, 阵营, robot_id ORDER BY 时刻秒
                     ) AS gap_s
              FROM events WHERE 事件类型='发弹'
            ) WHERE gap_s IS NOT NULL AND gap_s >= 0
        """).fetchall()
        by_type_gaps = defaultdict(list)
        for row in interval_rows:
            by_type_gaps[row['robot_type']].append(float(row['gap_s']))
        samples = []
        for gid in GAME_IDS:
            rows = sorted(timeseries_for(con, gid),
                          key=lambda r: (float(r['时刻秒']), r['阵营'], str(r['robot_id'])))
            if not rows:
                continue
            events = events_for(con, gid)
            out['sample_games'].append(gid)
            by_robot = defaultdict(lambda: {'first': None, 'last': None,
                                             'positive_42mm_delta': 0.0, 'delta_seconds': 0})
            for row in rows:
                if row['机器人类型'] != '英雄' or row['累计42mm发弹'] is None:
                    continue
                key = (row['阵营'], str(row['robot_id']))
                t = float(row['时刻秒'])
                cum = float(row['累计42mm发弹'])
                v = by_robot[key]
                if v['last'] is not None:
                    old_t, old_cum = v['last']
                    # Sum positive increments only; reset/rounding anomalies are not counted as firing.
                    v['positive_42mm_delta'] += max(0.0, cum-old_cum)
                    v['delta_seconds'] += 1
                else:
                    v['first'] = (t, cum)
                v['last'] = (t, cum)
            ev_shots = Counter()
            ev_by_sec = defaultdict(set)
            for event in events:
                if event['事件类型'] == '发弹' and event['类别'] == '42mm':
                    key = (event['阵营'], str(event['robot_id']))
                    ev_shots[key] += 1
                    ev_by_sec[key].add(float(event['时刻秒']))
            # Coverage compare is intentionally at robot/game total level:
            # ammo counter may count projectiles while one event can mean a burst.
            total = Counter()
            for key, value in by_robot.items():
                if value['delta_seconds'] <= 0:
                    continue
                total['robots_with_counter_delta'] += 1
                total['counter_positive_delta'] += int(value['positive_42mm_delta'] > 0)
                total['ammo42_delta_sum'] += value['positive_42mm_delta']
                total['42mm_event_count'] += ev_shots[key]
                total['robots_with_both'] += int(value['positive_42mm_delta'] > 0 and ev_shots[key] > 0)
                total['robots_counter_only'] += int(value['positive_42mm_delta'] > 0 and ev_shots[key] == 0)
                total['robots_event_only'] += int(value['positive_42mm_delta'] == 0 and ev_shots[key] > 0)
            samples.append(dict(total))

        sums = Counter()
        for s in samples:
            sums.update(s)
        out['event_coverage'] = {
            'comparison': 'per hero robot over each sampled match: positive increase in cumulative 42mm ammo versus count of 42mm “发弹” events',
            'games': len(out['sample_games']),
            'robots_with_counter_delta': sums['robots_with_counter_delta'],
            'robots_with_both_counter_delta_and_event': sums['robots_with_both'],
            'robots_counter_delta_only': sums['robots_counter_only'],
            'robots_event_only': sums['robots_event_only'],
            'ammo42_positive_delta_sum': sums['ammo42_delta_sum'],
            'event42_count': sums['42mm_event_count'],
            'total_counts_equal': round(sums['ammo42_delta_sum']) == sums['42mm_event_count'],
        }
        out['event_coverage_all_games'] = {
            'comparison': 'all database games; hero-game-side robot aggregates; positive cumulative 42mm ammo increments versus 42mm 发弹 event counts',
            **{key: full_coverage[key] for key in full_coverage.keys()},
            'counts_equal': round(full_coverage['ammo42_positive_delta_sum'] or 0) == (full_coverage['event42_total'] or 0),
        }
        out['event_mismatch_diagnostics'] = {
            'mismatch_group_count': len(mismatch_rows),
            'mismatch_abs_delta_sum': float(sum(abs(float(row['delta'])) for row in mismatch_rows)),
            'mismatch_groups': [
                {
                    'game_id': int(row['game_id']),
                    'side': str(row['side']),
                    'robot_id': str(row['robot_id']),
                    'ammo42_delta': float(row['ammo42_delta'] or 0.0),
                    'event_count': int(row['event_count'] or 0),
                    'delta': float(row['delta'] or 0.0),
                }
                for row in mismatch_rows
            ],
            'interpretation': 'A mismatch is a counter/event accounting difference at one hero-game-side robot. It is not evidence that a shot was missed by the sensor; inspect burst semantics, reset/rounding, and event duplication before using it for recommendation labels.',
        }
        interval_summary = {}
        for unit, gaps in by_type_gaps.items():
            sorted_gaps = sorted(gaps)
            def quantile(q):
                if not sorted_gaps:
                    return None
                return sorted_gaps[min(len(sorted_gaps)-1, max(0, math.ceil(q*len(sorted_gaps))-1))]
            interval_summary[unit] = {
                'event_gap_n': len(gaps),
                'median_gap_seconds': quantile(.5),
                'p90_gap_seconds': quantile(.9),
                'fraction_next_shot_within_3s': (sum(gap <= 3 for gap in gaps)/len(gaps) if gaps else None),
            }
        out['intershot_intervals_all_games'] = interval_summary

        rule_counts = Counter()
        types = defaultdict(Counter)
        for gid in out['sample_games']:
            rows = sorted(timeseries_for(con, gid),
                          key=lambda r: (float(r['时刻秒']), r['阵营'], str(r['robot_id'])))
            truth_first = min(float(row['时刻秒']) for row in rows)
            snapshots = build_radar_visibility_snapshots(rows, game_id=gid)
            for side in SIDES:
                for snap in snapshots[side]:
                    t = float(snap['issued_at'])
                    if abs((t-truth_first)/3-round((t-truth_first)/3)) > 1e-6:
                        continue
                    entities = list(snap['entities'].values())
                    for entity in entities:
                        if entity.get('team') != side or entity.get('entity_type') not in {'英雄', '步兵3', '步兵4', '哨兵', '空中'}:
                            continue
                        current = _compact_current(entity, entities, side)
                        if current.get('life_state') == 'destroyed':
                            continue
                        state = current.get('special_state') or {}
                        advice = state.get('decision_guidance') or {}
                        typ = current.get('type')
                        label = advice.get('action_label')
                        attack = state.get('attack_status')
                        firing_source = state.get('evidence')
                        selected_target = state.get('attacking_target') or state.get('possible_target') or {}
                        selected_target_reason = selected_target.get('reason', 'unknown')
                        if attack == 'attacking':
                            types[typ]['attacking_state'] += 1
                            types[typ]['attacking_evidence_' + str(firing_source)] += 1
                        if label == 'continue_attack_visible_target':
                            types[typ]['continue_advice'] += 1
                            types[typ]['target_id_known'] += int(advice.get('target_unit_id') is not None)
                            types[typ]['target_explicit_sensor'] += int(selected_target_reason == 'explicit_target')
                            types[typ]['target_nearest_visible_candidate'] += int(selected_target_reason == 'nearest_visible_enemy')
                            types[typ]['target_source_unknown'] += int(selected_target_reason not in ('explicit_target', 'nearest_visible_enemy'))
                            types[typ]['direct_sensor_evidence'] += int(firing_source == 'direct_sensor_field')
                            rule_counts['continue_advice_total'] += 1
                            rule_counts['hero_advice'] += int(typ == '英雄')
        out['rule'] = {
            'trigger_chain': 'data.radar_visibility_replay adds ammo_delta from positive increments in cumulative 17mm/42mm counters; firing_inferred=(ammo_delta>0). worldstate._attack_signal accepts firing_inferred. If attack_status==attacking and explicit/visible nearby target exists, derive_unit_guidance selects continue_attack_visible_target.',
            'from_code': ['data/radar_visibility_replay.py::build_radar_visibility_snapshots', 'worldstate/unit_state.py::_attack_signal', 'worldstate/unit_state.py::_combat_state', 'worldstate/unit_state.py::derive_unit_guidance'],
            'caveat': 'Thus hero advice is not a generic firing action: it is triggered after a positive observed ammo delta plus an explicit or geometry-inferred visible target. The 3-second follow-up event proxy tests continued firing persistence. It cannot test target identity/action intent.',
            'counts': {'all_continue_advice': rule_counts['continue_advice_total'],
                       'hero_continue_advice': rule_counts['hero_advice'],
                       'by_unit_type': {u: dict(c) for u, c in types.items()}},
        }

    path = ROOT / 'out_m2' / 'hero_fire_event_coverage.json'
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'output': str(path), 'event_coverage': out['event_coverage'],
                      'event_coverage_all_games': out['event_coverage_all_games'],
                      'event_mismatch_diagnostics': out['event_mismatch_diagnostics'],
                      'intershot_intervals_all_games': out['intershot_intervals_all_games'],
                      'rule': out['rule']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
