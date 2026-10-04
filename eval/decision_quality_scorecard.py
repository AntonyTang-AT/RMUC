"""Score visible LLM decisions against replay-derived decision labels.

The primary metric is HDQ (Hindsight Decision Quality): agreement with the
same side-specific look-ahead label contract used to build the visible v4
dataset. It is a reproducible operational-label metric, not a claim of causal
policy value. ORP (Observable Response Proxy) is a secondary fixed-horizon
diagnostic that checks whether movement/fire observations are compatible with
the emitted unit actions.

All aggregate intervals resample complete games and average red/blue within a
game first. Missing telemetry is reported as unknown rather than failure.
"""
from __future__ import annotations

import argparse
import json
import random
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from schema import layered as sl
from eval.llm_visible_input import (conservative_slot_action, own_unit_slot_states,
                                    project_decision_to_observable_contract)

ROOT = Path(__file__).resolve().parents[1]
SIDES = ('红', '蓝')
MOBILE = set(sl.UNIT_CLASSES)
MOVE_FORWARD = '推进'
MOVE_BACK = '回防'
MOVE_HOLD = {'坚守', '无'}
MOVE_FREE = {'游击', '绕行'}
POSITION_EPSILON_M = 0.25
BOOTSTRAPS = 2000
GLOBAL_MODE_WEIGHTS = (0.2, 0.4, 0.6, 0.8)


def _read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines()
            if line.strip()]


def _entity_id(team, robot_id):
    return f'{team}:{robot_id}'


def _row_position(row):
    position = row.get('position')
    if isinstance(position, (list, tuple)) and len(position) >= 2:
        try:
            return float(position[0]), float(position[1])
        except (TypeError, ValueError):
            return None
    try:
        return float(row['x']), float(row['y'])
    except (KeyError, TypeError, ValueError):
        return None


def _oriented_dx(team, current, future):
    p0, p1 = _row_position(current), _row_position(future)
    if p0 is None or p1 is None:
        return None
    dx = p1[0] - p0[0]
    return dx if team == '红' else -dx


def _response_for_move(move, team, current, future):
    """Return [0, 1] movement compatibility, or None when unscorable."""
    dx = _oriented_dx(team, current, future)
    if dx is None:
        return None
    if move == MOVE_FORWARD:
        return 1.0 if dx > POSITION_EPSILON_M else (0.5 if dx > 0 else 0.0)
    if move == MOVE_BACK:
        return 1.0 if dx < -POSITION_EPSILON_M else (0.5 if dx < 0 else 0.0)
    if move in MOVE_HOLD:
        return 1.0 if abs(dx) <= POSITION_EPSILON_M else 0.0
    if move in MOVE_FREE:
        return 1.0 if abs(dx) > POSITION_EPSILON_M else 0.5
    return None


def _bootstrap(values, seed=20260929):
    values = list(values)
    if not values:
        return {'n_games': 0, 'mean': None, 'ci95': None}
    rng = random.Random(seed)
    samples = sorted(
        sum(rng.choice(values) for _ in values) / len(values)
        for _ in range(BOOTSTRAPS))
    return {
        'n_games': len(values),
        'mean': sum(values) / len(values),
        'ci95': [samples[int(0.025 * BOOTSTRAPS)],
                 samples[min(BOOTSTRAPS - 1, int(0.975 * BOOTSTRAPS))]],
    }


def _balanced_game_metric(rows, key):
    """Equal-weight observations within each game, then equal-weight games."""
    by_game_side = defaultdict(dict)
    for row in rows:
        value = row.get(key)
        if value is None or row.get('side') not in SIDES:
            continue
        by_game_side[int(row['game_id'])].setdefault(row['side'], []).append(float(value))
    game_values = []
    for sides in by_game_side.values():
        side_means = [sum(values) / len(values) for values in sides.values() if values]
        if side_means:
            game_values.append(sum(side_means) / len(side_means))
    return _bootstrap(game_values)


def _paired_game_delta(rows, candidate_key, baseline_key):
    """Clustered candidate-minus-baseline delta on identical issue rows."""
    by_game_side = defaultdict(lambda: defaultdict(list))
    for row in rows:
        candidate, baseline = row.get(candidate_key), row.get(baseline_key)
        if candidate is None or baseline is None or row.get('side') not in SIDES:
            continue
        by_game_side[int(row['game_id'])][row['side']].append(
            float(candidate) - float(baseline))
    game_values = []
    for sides in by_game_side.values():
        side_means = [sum(values) / len(values) for values in sides.values() if values]
        if side_means:
            game_values.append(sum(side_means) / len(side_means))
    return _bootstrap(game_values)


def _weighted_hdq(mode, unit_field, global_mode_weight):
    """Combine strategic and per-unit agreement for one valid decision."""
    if mode is None or unit_field is None:
        return None
    return (float(global_mode_weight) * float(mode) +
            (1.0 - float(global_mode_weight)) * float(unit_field))


def _label_stable_next(labels, index, frames=5):
    """Return whether a hindsight label survives the next fixed frame window."""
    if index < 0 or index + int(frames) >= len(labels):
        return None
    return all(value == labels[index] for value in labels[index + 1:index + int(frames) + 1])


def _expected_mode_counts(rows):
    """Keep rare strategic modes visible in every scenario report."""
    return {
        mode: sum(row.get('expected_global_mode') == mode for row in rows)
        for mode in sl.GLOBAL_MODES
    }


def _hdq_by_expected_mode(rows):
    """Score each strategic mode separately so aggregate HDQ cannot mask a regression."""
    output = {}
    for mode in sl.GLOBAL_MODES:
        selected = [row for row in rows if row.get('expected_global_mode') == mode]
        output[mode] = {
            'n_rows': len(selected),
            'end_to_end_hdq': _balanced_game_metric(selected, 'end_to_end_hdq'),
            'end_to_end_hdq_delta_vs_legacy_rule': _paired_game_delta(
                selected, 'end_to_end_hdq', 'baseline_hdq'),
        }
    return output


def _world_contexts(records, lookahead_delta):
    import run_m2
    from labels import strategy_label
    from tree import global_mode as gm
    from worldstate import worldstate as ws

    contexts = {}
    for game_id in sorted({int(row['game_id']) for row in records}):
        dps, bounds = run_m2.load_decision_points(game_id)
        worlds = [ws.build_worldstate(dp, bounds) for dp in dps]
        modes = {
            side: strategy_label.strategy_series(
                worlds, int(lookahead_delta), team=side,
                margin=strategy_label.DEF_MARGIN)
            for side in SIDES
        }
        stable = {
            side: [_label_stable_next(modes[side], index) for index in range(len(worlds))]
            for side in SIDES
        }
        by_time = {}
        for index, world in enumerate(worlds):
            by_time[round(float(world['t']), 3)] = (
                world, {side: modes[side][index] for side in SIDES}, gm,
                {side: stable[side][index] for side in SIDES})
        contexts[game_id] = by_time
    return contexts


def _telemetry(db_path, records):
    wanted = defaultdict(set)
    games = sorted({int(row['game_id']) for row in records})
    for row in records:
        game, t = int(row['game_id']), float(row['issued_at_s'])
        wanted[game].add(round(t + 10.0, 3))
    future, fire = {}, defaultdict(set)
    with sqlite3.connect(db_path) as con:
        con.row_factory = sqlite3.Row
        for game in games:
            rows = con.execute('SELECT * FROM timeseries WHERE game_id=?', (game,)).fetchall()
            for item in rows:
                value = dict(item)
                t = round(float(value['时刻秒']), 3)
                if t in wanted[game]:
                    eid = _entity_id(value['阵营'], value['robot_id'])
                    future[(game, eid, t)] = value
            events = con.execute(
                'SELECT * FROM events WHERE game_id=? AND "事件类型"=?',
                (game, '发弹')).fetchall()
            for item in events:
                value = dict(item)
                eid = _entity_id(value['阵营'], value['robot_id'])
                fire[(game, eid)].add(float(value['时刻秒']))
    return future, fire


def _units_by_type(world, side):
    output = {}
    for key, unit in (world.get('robots') or {}).items():
        if unit.get('team') == side and unit.get('type') in MOBILE:
            entity_id = str(unit.get('entity_id') or key)
            if ':' not in entity_id:
                entity_id = _entity_id(side, entity_id)
            output.setdefault(unit['type'], []).append((entity_id, unit))
    return output


def _decision_for_baseline(world, issue_state, mode, side):
    """Build the legacy reference using only slots observable at issue time."""
    from tree import global_mode as gm
    decision = sl.build_empty_decision(float(world['t']))
    decision['global_mode'] = mode
    record = {'perspective': side, 'issue_state': issue_state}
    slots = own_unit_slot_states(record)
    by_id = {str(unit.get('entity_id') or key): unit
             for key, unit in (world.get('robots') or {}).items()}
    for unit_type, slot in slots.items():
        if slot['status'] != 'current':
            decision['units'][unit_type] = conservative_slot_action(slot['status'])
            continue
        observed = slot['entity'] or {}
        full = by_id.get(str(slot['entity_id'])) or {}
        unit = dict(full)
        unit['hp_ratio'] = observed.get('hp_ratio', unit.get('hp_ratio', 1.0))
        unit['dfire'] = 1.0 if observed.get('firing_inferred') is True else 0.0
        unit['vuln'] = 0
        decision['units'][unit_type] = gm.derive_unit_intent(unit, world, mode, team=side)
    return project_decision_to_observable_contract(decision, record)


def _quality(decision, world, issue_state, side, hindsight_mode, baseline_mode):
    """Return label agreement fields for one parsed decision."""
    record = {'perspective': side, 'issue_state': issue_state}
    baseline = _decision_for_baseline(world, issue_state, baseline_mode, side)
    expected = _decision_for_baseline(world, issue_state, hindsight_mode, side)
    slots = own_unit_slot_states(record)
    baseline_unit_rows, baseline_field_rows = [], []
    for unit_type, slot in slots.items():
        if slot['status'] != 'current':
            continue
        baseline_unit = baseline['units'][unit_type]
        expected_unit = expected['units'][unit_type]
        baseline_unit_rows.append(float(baseline_unit == expected_unit))
        baseline_field_rows.extend(float(baseline_unit.get(field) == expected_unit.get(field))
                                   for field in ('target', 'move', 'fire', 'risk'))
    baseline_unit_field = (sum(baseline_field_rows) / len(baseline_field_rows)
                           if baseline_field_rows else None)
    baseline_hdq = (0.4 * float(baseline_mode == hindsight_mode) +
                    0.6 * baseline_unit_field) if baseline_unit_field is not None else None
    hidden_slots = [unit_type for unit_type, slot in slots.items()
                    if slot['status'] != 'current']
    if not isinstance(decision, dict) or not sl.validate(decision)[0]:
        return {'mode': None, 'unit_exact': None, 'unit_field': None,
                'hdq': None, 'baseline_mode': float(baseline_mode == hindsight_mode),
                'baseline_unit_field': baseline_unit_field, 'baseline_hdq': baseline_hdq,
                'hidden_slot_safety': 0.0 if hidden_slots else None,
                'observable_unit_slots': len(sl.UNIT_CLASSES) - len(hidden_slots)}
    unit_rows, field_rows = [], []
    predicted_units = decision.get('units') or {}
    for unit_type, slot in slots.items():
        if slot['status'] != 'current' or unit_type not in predicted_units:
            continue
        expected_unit, actual_unit = expected['units'][unit_type], predicted_units[unit_type]
        unit_rows.append(float(actual_unit == expected_unit))
        field_rows.extend(float(actual_unit.get(field) == expected_unit.get(field))
                          for field in ('target', 'move', 'fire', 'risk'))
    unit_exact = sum(unit_rows) / len(unit_rows) if unit_rows else None
    unit_field = sum(field_rows) / len(field_rows) if field_rows else None
    hidden_safety = [
        float(predicted_units.get(unit_type) == conservative_slot_action(slots[unit_type]['status']))
        for unit_type in hidden_slots
    ]
    return {
        'mode': float(decision.get('global_mode') == hindsight_mode),
        'unit_exact': unit_exact,
        'unit_field': unit_field,
        'hdq': (0.4 * float(decision.get('global_mode') == hindsight_mode) +
                0.6 * unit_field) if unit_field is not None else None,
        'baseline_mode': float(baseline_mode == hindsight_mode),
        'baseline_unit_field': baseline_unit_field,
        'baseline_hdq': baseline_hdq,
        'hidden_slot_safety': (sum(hidden_safety) / len(hidden_safety)
                               if hidden_safety else None),
        'observable_unit_slots': len(sl.UNIT_CLASSES) - len(hidden_slots),
    }


def _response(decision, issue_state, world, side, game_id, t, future, fire):
    if not isinstance(decision, dict) or not sl.validate(decision)[0]:
        return {'score': None, 'movement': None, 'fire': None, 'coverage': 0.0}
    by_type = _units_by_type(world, side)
    movement, firing = [], []
    future_t = round(t + 10.0, 3)
    for unit_type, values in by_type.items():
        if len(values) != 1:
            continue
        eid, _world_unit = values[0]
        current = (issue_state.get('robots') or {}).get(eid)
        next_row = future.get((game_id, eid, future_t))
        action = (decision.get('units') or {}).get(unit_type) or {}
        if current is not None and next_row is not None:
            value = _response_for_move(action.get('move'), side, current, next_row)
            if value is not None:
                movement.append(value)
        if action.get('fire') is True and next_row is not None:
            fired = any(t < event_t <= t + 10.0 for event_t in fire.get((game_id, eid), ()))
            firing.append(float(fired))
    components = movement + firing
    possible_components = len(by_type) + sum(
        bool((decision.get('units') or {}).get(unit_type, {}).get('fire') is True)
        for unit_type in by_type)
    return {
        'score': sum(components) / len(components) if components else None,
        'movement': sum(movement) / len(movement) if movement else None,
        'fire': sum(firing) / len(firing) if firing else None,
        'coverage': len(components) / max(1, possible_components),
    }


def score(records, db_path, *, lookahead_delta=5):
    contexts = _world_contexts(records, lookahead_delta)
    future, fire = _telemetry(db_path, records)
    scored = []
    for record in records:
        game_id, t, side = int(record['game_id']), float(record['issued_at_s']), record.get('perspective')
        context = contexts.get(game_id, {}).get(round(t, 3))
        if context is None or side not in SIDES:
            continue
        world, hindsight_modes, gm, stability = context
        hindsight_mode = hindsight_modes[side]
        baseline_mode = gm.derive_global_mode(world)
        issue_state = record.get('issue_state') or {}
        quality = _quality(record.get('decision'), world, issue_state, side,
                           hindsight_mode, baseline_mode)
        response = _response(record.get('decision'), record.get('issue_state') or {}, world,
                             side, game_id, t, future, fire)
        row = {'game_id': game_id, 'side': side, 'issued_at_s': t,
               'forecast_context_enabled': bool((record.get('run_metadata') or {}).get(
                   'forecast_context_enabled')),
               'expected_global_mode': hindsight_mode,
               'label_stable_next_5': stability[side],
               'parse_valid': bool(record.get('parse_valid')) and quality['mode'] is not None,
               **quality, 'response_score': response['score'],
               'response_movement': response['movement'],
               'response_fire': response['fire'],
               'response_coverage': response['coverage']}
        scored.append(row)

    scenarios = {}
    for scenario in sorted({
        (row['side'], row['forecast_context_enabled']) for row in scored
    }):
        selected = [row for row in scored if
                    (row['side'], row['forecast_context_enabled']) == scenario]
        parsed = [row for row in selected if row['parse_valid']]
        end_to_end = [{**row, 'end_to_end_hdq': (row['hdq'] if row['parse_valid'] else 0.0)}
                      for row in selected]
        stable_rows = [row for row in end_to_end if row.get('label_stable_next_5') is True]
        weight_sensitivity = {}
        for global_weight in GLOBAL_MODE_WEIGHTS:
            candidate_key = f'candidate_hdq_global_weight_{global_weight:.1f}'
            baseline_key = f'baseline_hdq_global_weight_{global_weight:.1f}'
            weighted_rows = []
            for row in selected:
                candidate = (_weighted_hdq(row.get('mode'), row.get('unit_field'), global_weight)
                             if row['parse_valid'] else 0.0)
                baseline = _weighted_hdq(row.get('baseline_mode'),
                                         row.get('baseline_unit_field'), global_weight)
                weighted_rows.append({**row, candidate_key: candidate, baseline_key: baseline})
            weight_sensitivity[f'global_mode_weight={global_weight:.1f}'] = {
                'candidate_end_to_end_hdq': _balanced_game_metric(weighted_rows, candidate_key),
                'delta_vs_legacy_rule': _paired_game_delta(weighted_rows, candidate_key, baseline_key),
            }
        scenarios[f'{scenario[0]}|forecast={str(scenario[1]).lower()}'] = {
            'side': scenario[0], 'forecast_context_enabled': scenario[1],
            'n': len(selected), 'parsed': len(parsed),
            'parse_rate': len(parsed) / max(1, len(selected)),
            'expected_global_mode_counts': _expected_mode_counts(selected),
            'label_stability': {
                'definition': 'hindsight label remains unchanged for the next five decision frames',
                'coverage': _balanced_game_metric(end_to_end, 'label_stable_next_5'),
                'stable_rows': len(stable_rows),
                'end_to_end_hdq': _balanced_game_metric(stable_rows, 'end_to_end_hdq'),
                'end_to_end_hdq_delta_vs_legacy_rule': _paired_game_delta(
                    stable_rows, 'end_to_end_hdq', 'baseline_hdq'),
            },
            'hdq': {key: _balanced_game_metric(parsed, key)
                    for key in ('hdq', 'mode', 'unit_exact', 'unit_field')},
            'end_to_end_hdq': _balanced_game_metric(end_to_end, 'end_to_end_hdq'),
            'end_to_end_hdq_delta_vs_legacy_rule': _paired_game_delta(
                end_to_end, 'end_to_end_hdq', 'baseline_hdq'),
            'hdq_by_expected_global_mode': _hdq_by_expected_mode(end_to_end),
            'legacy_rule_baseline': {
                'mode_accuracy': _balanced_game_metric(parsed, 'baseline_mode'),
                'hdq': _balanced_game_metric(selected, 'baseline_hdq'),
            },
            'hdq_weight_sensitivity': weight_sensitivity,
            'orp': {key: _balanced_game_metric(parsed, key)
                    for key in ('response_score', 'response_movement', 'response_fire')},
            'response_coverage': _balanced_game_metric(selected, 'response_coverage'),
        }
    return {
        'schema_version': 'decision-quality-scorecard-v1',
        'metric_contract': {
            'primary': ('observable end-to-end HDQ = 0 for any unparseable/schema-invalid output; otherwise '
                        '0.4 * hindsight global-mode accuracy + 0.6 * mean per-field unit-intent accuracy '
                        'over own slots currently observable at issue time'),
            'conditional_semantic': 'HDQ on parsed rows only, reported to diagnose label fidelity separately from reliability',
            'secondary': 'ORP = mean compatible movement response and observed firing response within 10 seconds',
            'aggregation': 'complete-game bootstrap; red/blue averaged within each game before game averaging',
            'hindsight_label': f'side-specific strategy_label.forward_strategy with delta={int(lookahead_delta)} frames',
            'stability_diagnostic': ('stable-label HDQ retains rows whose hindsight mode remains '
                                     'unchanged for the next five decision frames; it is diagnostic only.'),
            'unknown_policy': ('unseen, destroyed, and ambiguous own slots are excluded from the unit-action '
                               'denominator and reported separately through hidden-slot safety'),
            'weight_sensitivity': ('The primary HDQ uses global-mode weight 0.4. '
                                   'The report also shows 0.2, 0.4, 0.6, and 0.8 as a '
                                   'robustness diagnostic; promotion still uses the primary weight.'),
        },
        'limitations': [
            'HDQ measures agreement with an observable operational look-ahead label, not final match outcome or causal policy value.',
            'Stable-label HDQ excludes transition-adjacent rows and therefore is never used in place of full end-to-end HDQ.',
            'ORP is an observational compatibility proxy; no controller receipt or acknowledgement exists.',
            'The movement response proxy uses oriented x displacement and a 0.25 m threshold; it is not a path-execution proof.',
            'A future fire event does not prove that the emitted decision caused the shot or that the target was hit.',
            'Counterfactual policy value requires controlled execution or matched/randomized interventions.',
        ],
        'rows': len(scored),
        'scenarios': scenarios,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rows', nargs='+', required=True)
    parser.add_argument('--db', default=None)
    parser.add_argument('--lookahead-delta', type=int, default=5)
    parser.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    if args.db is None:
        from config import DB_PATH
        args.db = DB_PATH
    records = []
    for path in args.rows:
        records.extend(_read_jsonl(path))
    result = score(records, args.db, lookahead_delta=args.lookahead_delta)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'output': str(output.resolve()), 'rows': result['rows'],
                      'scenarios': list(result['scenarios'])}, ensure_ascii=False))


if __name__ == '__main__':
    main()
