"""Score row-wise LLM decisions against exact future replay telemetry.

This is an observational response scorecard, not execution or causal utility.
It deliberately does not load a model or scan a database at import time.  The
issue row is immutable; future rows are joined by exact (game, entity, second).
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from worldstate.functional_zone import assign_functional_zone, candidate_moves
from eval.llm_visible_input import validate_visible_llm_input

MOBILE = {'英雄', '工程', '步兵3', '步兵4', '空中', '哨兵'}
MOVE_MOVES = {'推进', '游击', '绕行', '回防'}
HOLD_MOVES = {'坚守', '无'}


def _entity_id(row):
    if row.get('entity_id') is not None:
        return str(row['entity_id'])
    if row.get('阵营') is not None and row.get('robot_id') is not None:
        return f"{row['阵营']}:{row['robot_id']}"
    if row.get('team') is not None and row.get('robot_id') is not None:
        return f"{row['team']}:{row['robot_id']}"
    return None


def _zone(row):
    if row is None:
        return None
    x, y = row.get('x'), row.get('y')
    if x is None or y is None:
        pos = row.get('position')
        if isinstance(pos, (list, tuple)) and len(pos) >= 2:
            x, y = pos[:2]
    try:
        x, y = float(x), float(y)
    except (TypeError, ValueError):
        return None
    if x == 0 and y == 0:
        return None
    return assign_functional_zone(x, y)


def _hp_ratio(row):
    if not row:
        return None
    if row.get('hp_ratio') is not None:
        try:
            value = float(row['hp_ratio'])
            return value if 0 <= value <= 1 else None
        except (TypeError, ValueError):
            return None
    try:
        hp = float(row.get('当前血量', row.get('hp')))
        maxhp = float(row.get('最大血量', row.get('maxhp')))
    except (TypeError, ValueError):
        return None
    return hp / maxhp if maxhp > 0 else None


def _issue_units(issue_state):
    robots = issue_state.get('robots') or {}
    result = {}
    for key, value in robots.items():
        row = dict(value)
        typ, team = row.get('type', row.get('entity_type')), row.get('team')
        if typ not in MOBILE or team not in ('红', '蓝'):
            continue
        eid = str(key) if ':' in str(key) else f'{team}:{row.get("robot_id", key)}'
        result.setdefault((team, typ), []).append((eid, row))
    return result


def _future_index(future_rows):
    index = {}
    for row in future_rows:
        eid = _entity_id(row)
        if eid is None:
            continue
        try:
            key = (int(row.get('game_id')), eid, float(row.get('时刻秒', row.get('t'))))
        except (TypeError, ValueError):
            continue
        # Duplicate exact rows are unsafe to resolve arbitrarily.
        index[key] = None if key in index else row
    return index


def score_rows(issue_records, future_rows, *, event_rows=(), horizons=(3, 10)):
    """Return per-action/unit/side observable proxy rates and explicit unknowns.

    The LLM schema only names unit *classes*, not unit IDs or destination
    zones. A class is scored only where exactly one own mobile entity of that
    class exists at issue time. Target HP is scored only with an explicit
    ``target_unit_id`` plus ``target_linkage='explicit_sensor_target'``.
    """
    future = _future_index(future_rows)
    fire_by_entity = defaultdict(list)
    for event in event_rows:
        if event.get('事件类型') != '发弹':
            continue
        eid = _entity_id(event)
        if eid is None:
            continue
        try:
            fire_by_entity[(int(event['game_id']), eid)].append(float(event['时刻秒']))
        except (KeyError, TypeError, ValueError):
            continue
    groups = defaultdict(lambda: {'issued': 0, 'unknown': Counter(), 'route': Counter(),
                                   'h': {int(x): Counter() for x in horizons}})
    for record in issue_records:
        try:
            game = int(record['game_id'])
            t = float(record.get('issued_at_s', record['issue_state']['t']))
        except (KeyError, TypeError, ValueError):
            continue
        side = record.get('perspective')
        if side not in ('红', '蓝'):
            side = (record.get('run_metadata') or {}).get('decision_side')
        issue_units = _issue_units(record.get('issue_state') or {})
        decision = record.get('decision') or {}
        units = decision.get('units') or {}
        for typ, advice in units.items():
            if typ not in MOBILE or not isinstance(advice, dict):
                continue
            # Preserve a cell even when perspective/entity attribution is unknown.
            group_key = (str(advice.get('move', 'unknown')), typ,
                         side if side in ('红', '蓝') else 'unknown')
            group = groups[group_key]
            group['issued'] += 1
            if side not in ('红', '蓝'):
                group['unknown']['decision_perspective_unknown'] += 1
                continue
            candidates = issue_units.get((side, typ), [])
            if len(candidates) != 1:
                group['unknown']['unit_identity_ambiguous_or_missing'] += 1
                continue
            eid, issue_unit = candidates[0]
            start_zone = issue_unit.get('functional_zone', issue_unit.get('zone')) or _zone(issue_unit)
            move = advice.get('move')
            goal = advice.get('movement_goal_zone')
            if goal:
                first = advice.get('first_step_zone')
                if start_zone and first:
                    legal = first == start_zone or first in candidate_moves(typ, side, start_zone)
                    group['route']['first_step_scorable'] += 1
                    group['route']['first_step_legal_yes'] += int(legal)
                else:
                    group['route']['first_step_unknown'] += 1
            else:
                group['route']['goal_missing'] += 1
            target_id = advice.get('target_unit_id')
            target_linkage = advice.get('target_linkage')
            for h, cell in group['h'].items():
                cell['issued'] += 1
                actual = future.get((game, eid, t + h))
                if actual is None:
                    cell['exact_future_row_missing'] += 1
                    continue
                cell['exact_future_row_n'] += 1
                z1 = _zone(actual)
                if start_zone and z1:
                    changed = (z1 != start_zone)
                    if move in MOVE_MOVES:
                        cell['movement_response_n'] += 1
                        cell['movement_response_yes'] += int(changed)
                    elif move in HOLD_MOVES:
                        cell['hold_zone_response_n'] += 1
                        cell['hold_zone_response_yes'] += int(not changed)
                else:
                    cell['zone_response_unknown'] += 1
                if advice.get('fire') is True:
                    cell['fire_advice_n'] += 1
                    fired = any(t < shot <= t + h
                                for shot in fire_by_entity.get((game, eid), ()))
                    cell['observed_firing_event_n'] += 1
                    cell['observed_firing_event_yes'] += int(fired)
                hp0, hp1 = _hp_ratio(issue_unit), _hp_ratio(actual)
                if hp0 is None or hp1 is None:
                    cell['own_hp_unknown'] += 1
                else:
                    cell['own_hp_delta_n'] += 1
                    cell['own_hp_delta_sum'] += hp1 - hp0
                    cell['own_survival_n'] += 1
                    cell['own_survival_yes'] += int(hp1 > 0)
                # There is no controller command/ack table in this dataset.
                cell['execution_unknown'] += 1
                if target_id is None or target_linkage != 'explicit_sensor_target':
                    cell['target_id_unknown'] += 1
                else:
                    # A target outcome requires exact target telemetry too.
                    target = future.get((game, str(target_id), t + h))
                    if target is None or _hp_ratio(target) is None:
                        cell['target_state_unknown'] += 1
                    else:
                        hp_target_issue = None
                        for rows in issue_units.values():
                            hp_target_issue = next((r for i, r in rows if i == str(target_id)), None)
                            if hp_target_issue is not None:
                                break
                        hp0t, hp1t = _hp_ratio(hp_target_issue), _hp_ratio(target)
                        if hp0t is None:
                            cell['target_state_unknown'] += 1
                        else:
                            cell['target_hp_delta_n'] += 1
                            cell['target_hp_delta_sum'] += hp1t - hp0t
    output = []
    for (action, typ, side), group in sorted(groups.items()):
        result = {'action': action, 'unit_type': typ, 'side': side,
                  'issued_n': group['issued'], 'unknown': dict(group['unknown']),
                  'route': {
                      'first_step_scorable_n': group['route']['first_step_scorable'],
                      'first_step_legal_rate': (
                          group['route']['first_step_legal_yes'] /
                          group['route']['first_step_scorable']
                          if group['route']['first_step_scorable'] else None),
                      'first_step_unknown_n': group['route']['first_step_unknown'],
                      'goal_missing_n': group['route']['goal_missing'],
                  }, 'horizons': {}}
        for h, c in group['h'].items():
            n = c['exact_future_row_n']
            metrics = {}
            for name, den, num in (
                ('movement_response', c['movement_response_n'], c['movement_response_yes']),
                ('hold_zone_response', c['hold_zone_response_n'], c['hold_zone_response_yes']),
                ('observed_firing_event', c['observed_firing_event_n'], c['observed_firing_event_yes']),
                ('own_survival', c['own_survival_n'], c['own_survival_yes'])):
                metrics[name] = {'scorable_n': den, 'rate': num / den if den else None,
                                 'coverage_of_issued': den / group['issued'] if group['issued'] else 0.0}
            metrics['own_hp_delta'] = {'scorable_n': c['own_hp_delta_n'],
                'mean_fractional_hp_change': c['own_hp_delta_sum'] / c['own_hp_delta_n']
                if c['own_hp_delta_n'] else None,
                'coverage_of_issued': c['own_hp_delta_n'] / group['issued'] if group['issued'] else 0.0}
            metrics['target_hp_delta'] = {'scorable_n': c['target_hp_delta_n'],
                'mean_fractional_hp_change': c['target_hp_delta_sum'] / c['target_hp_delta_n']
                if c['target_hp_delta_n'] else None,
                'coverage_of_issued': c['target_hp_delta_n'] / group['issued'] if group['issued'] else 0.0}
            result['horizons'][str(h)] = {
                'exact_future_row_n': n,
                'exact_future_row_coverage': n / c['issued'] if c['issued'] else 0.0,
                'metrics': metrics,
                'unknown': {k: v for k, v in c.items()
                            if k.endswith('unknown') or k.endswith('missing')},
                'execution': {'status': 'unknown', 'coverage': 0.0,
                              'reason': 'no controller command/ack telemetry'},
            }
        output.append(result)
    return {
        'schema_version': 1,
        'scope': {'issue_rows_n': len(issue_records), 'response_horizons_s': list(horizons),
                  'join': 'exact game_id + entity_id + issue second + horizon'},
        'visibility_audit': {
            'legacy_omniscient_rows_n': sum(
                'legacy_complete_timeseries' in str(r.get('visibility_policy', ''))
                for r in issue_records),
            'verified_visible_rows_n': sum(validate_visible_llm_input(r)
                                           for r in issue_records),
            'deployable_claim_allowed': bool(issue_records) and all(
                validate_visible_llm_input(r) for r in issue_records),
        },
        'limitations': [
            'Behavior and HP are observational response/outcome proxies, not command execution or causal effects.',
            'The legacy unit-class schema has no movement goal, so route goal progress is unknown unless enriched.',
            'Target HP is unscored unless target ID and explicit sensor target linkage are both present.',
            'Missing/duplicate exact future joins remain unknown, never a failure.',
            'A visibility-policy string alone is insufficient; every issue row needs a verified causal sensor/radar prompt and matching issue state.',
            'Any legacy complete-timeseries issue input invalidates deployment-quality claims.'],
        'results': output,
    }


def _read_jsonl(path):
    rows = []
    with open(path, encoding='utf-8') as stream:
        for line in stream:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _load_exact_outcomes(db_path, records, horizons=(3, 10)):
    """Read only requested future seconds for these issue rows/games."""
    wanted = defaultdict(set)
    for record in records:
        try:
            game = int(record['game_id'])
            t = float(record.get('issued_at_s', record['issue_state']['t']))
        except (KeyError, TypeError, ValueError):
            continue
        for horizon in horizons:
            wanted[game].add(t + float(horizon))
    rows, events = [], []
    with sqlite3.connect(db_path) as con:
        con.row_factory = sqlite3.Row
        for game, times in wanted.items():
            values = sorted(times)
            # Stay below SQLite's conservative variable limit.
            for offset in range(0, len(values), 400):
                batch = values[offset:offset + 400]
                slots = ','.join('?' for _ in batch)
                rows.extend(dict(row) for row in con.execute(
                    f'SELECT * FROM timeseries WHERE game_id=? AND "时刻秒" IN ({slots})',
                    [game, *batch]))
            if values:
                events.extend(dict(row) for row in con.execute(
                    'SELECT * FROM events WHERE game_id=? AND "事件类型"=? AND "时刻秒">? AND "时刻秒"<=?',
                    (game, '发弹', min(values) - max(horizons), max(values))))
    return rows, events


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--rows', required=True, help='LLM issue-time JSONL only')
    ap.add_argument('--db', default=None, help='replay SQLite; defaults to config.DB_PATH')
    ap.add_argument('--out', required=True)
    args = ap.parse_args(argv)
    records = _read_jsonl(args.rows)
    if args.db:
        db_path = args.db
    else:
        from config import DB_PATH
        db_path = DB_PATH
    games = sorted({int(r['game_id']) for r in records})
    future_rows, event_rows = _load_exact_outcomes(db_path, records)
    report = score_rows(records, future_rows, event_rows=event_rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'output': str(Path(args.out).resolve()),
                      'games_n': len(games), 'issue_rows_n': len(records),
                      'deployable_claim_allowed': report['visibility_audit']['deployable_claim_allowed']},
                     ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
