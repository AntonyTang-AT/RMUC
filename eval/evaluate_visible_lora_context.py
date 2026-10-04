"""Evaluate a LoRA Tier2 adapter on auditable radar/sensor-only inputs.

The regular forecast-context ablation predates the strict visible prompt
contract and reconstructs a world state from complete replay decision points.
This evaluator deliberately never passes such a state to the LLM.  It stores
the exact validated visible input used for each generation, making its JSONL
compatible with ``eval.llm_row_scorecard``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from schema import layered as sl


def _sample_decision_points(points, limit, sampling='uniform'):
    """Keep chronological points while avoiding a first-frames-only sample."""
    if limit is None or int(limit) <= 0 or len(points) <= int(limit):
        return list(points)
    if sampling == 'head':
        return list(points[:int(limit)])
    if sampling != 'uniform':
        raise ValueError(f'unsupported sampling policy: {sampling}')
    count = int(limit)
    indexes = sorted({round(index * (len(points) - 1) / (count - 1))
                      for index in range(count)}) if count > 1 else [len(points) // 2]
    return [points[index] for index in indexes]


def _load_games(game_ids, limit, sampling='uniform'):
    import run_m2
    output = []
    for game_id in game_ids:
        dps, bounds = run_m2.load_decision_points(int(game_id))
        dps = _sample_decision_points(dps, limit, sampling=sampling)
        if dps:
            output.append((int(game_id), dps, bounds))
    return output


def _score(decisions, rule_moves):
    pairs = [(decision, rule_move) for decision, rule_move in zip(decisions, rule_moves)
             if decision is not None]
    valid = [decision for decision, _rule_move in pairs]
    return {
        'n': len(decisions),
        'parsed': len(valid),
        'parse_rate': len(valid) / max(1, len(decisions)),
        'legal_rate': sum(bool(sl.validate(decision)[0]) for decision in valid) / max(1, len(valid)),
        'agreement_with_legacy_rule_teacher': sum(
            decision.get('units', {}).get('哨兵', {}).get('move', '无') == rule_move
            for decision, rule_move in pairs) / max(1, len(valid)),
        'conflict_rate': sl.conflict_rate(valid) if len(valid) > 1 else 0.0,
    }


def _visible_requests(game_id, dps, bounds, *, side, checkpoint, device,
                      use_forecast, include_region10, instruction_max_chars):
    """Build all prompts from chronological visibility snapshots only."""
    import run_m2
    from data.radar_visibility_replay import PROVIDER_ID
    from eval.llm_visible_input import (build_visible_llm_input,
                                        format_visible_training_instruction,
                                        validate_visible_llm_input)
    from runtime.temporal_forecast import resolve_temporal_forecaster
    from tree import global_mode as gm
    from worldstate import worldstate as ws
    from worldstate.forecast_context import build_forecast_context

    snapshots = run_m2.load_observable_snapshots(game_id, team=side)
    contexts = [None] * len(dps)
    if use_forecast:
        options = {}
        if not include_region10:
            options['enemy_region_transition_path'] = str(
                Path('out_m2/enemy_region_transition/production_manifest.json').resolve())
        forecaster = resolve_temporal_forecaster(
            checkpoint, device=device, team=side, context=12,
            adaptive_router=None, auto_mature_feedback=False, **options)
        # Feed all one-second snapshots before selecting the decision clock.
        predictions = run_m2.precompute_temporal_forecasts(dps, snapshots, forecaster)
        contexts = [build_forecast_context(prediction) if prediction else None
                    for prediction in predictions]

    requests = []
    for dp, context in zip(dps, contexts):
        issued_at = float(dp.get('t', 0.0))
        visible = build_visible_llm_input(
            snapshots, game_id=game_id, side=side, issued_at_s=issued_at,
            history_frames=4)
        if not validate_visible_llm_input(visible):
            raise ValueError(f'visible input audit failed at game={game_id} t={issued_at}')
        instruction = format_visible_training_instruction(
            visible, forecast_context=context if use_forecast else None,
            max_chars=instruction_max_chars)
        # This is a historical teacher comparison only.  It is not a visible
        # policy oracle and is deliberately named as such in the output.
        legacy_move = {'def': '回防', 'hitbuilding': '推进', 'cruise': '游击'}.get(
            gm.derive_sentry_state(ws.build_worldstate(dp, bounds), team=side), '游击')
        requests.append({
            'instruction': instruction,
            'visible': visible,
            'forecast_context': context if use_forecast else None,
            'legacy_rule_move': legacy_move,
            'forecast_provider': (PROVIDER_ID if use_forecast else None),
        })
    return requests


def evaluate(adapter, games, *, side='红', device='cuda', checkpoint='auto',
             use_forecast=True, include_region10=True, max_new=200,
             prompt_max_length=896, batch_size=4, instruction_max_chars=640,
             rows_output=None, contract_retry=True):
    from finetune.infer_one import (extract_decision_with_error,
                                    extract_decision_with_repair,
                                    generate_instruction_texts_with_bundle,
                                    load_lora_bundle)

    tokenizer, model = load_lora_bundle(adapter, device=device)
    delivered_decisions, strict_decisions, first_pass_decisions, rule_moves, rows = [], [], [], [], []
    for game_id, dps, bounds in games:
        requests = _visible_requests(
            game_id, dps, bounds, side=side, checkpoint=checkpoint,
            device='cpu', use_forecast=use_forecast,
            include_region10=include_region10,
            instruction_max_chars=instruction_max_chars)
        for start in range(0, len(requests), max(1, int(batch_size))):
            block = requests[start:start + max(1, int(batch_size))]
            block_texts = generate_instruction_texts_with_bundle(
                tokenizer, model, [item['instruction'] for item in block],
                max_new=max_new, prompt_max_length=prompt_max_length)
            for item, raw_output in zip(block, block_texts):
                strict_decision, parse_error = extract_decision_with_error(raw_output)
                contract_decision, contract_repairs, contract_error = (
                    extract_decision_with_repair(raw_output))
                visible = dict(item['visible'])
                visible.update({
                    'schema_version': 3,
                    'adapter': os.path.abspath(adapter),
                    # ``decision`` mirrors the object that a deployable Tier2
                    # backend returns to the runtime.  Keep the strict result
                    # separately so repairs cannot be mistaken for a raw-model
                    # structural success.
                    'decision': contract_decision,
                    'strict_decision': strict_decision,
                    'raw_output': raw_output,
                    'parse_error': parse_error,
                    'parse_valid': contract_decision is not None,
                    'strict_parse_valid': strict_decision is not None,
                    'contract_decision': contract_decision,
                    'contract_repairs': contract_repairs,
                    'contract_error': contract_error,
                    'contract_valid': contract_decision is not None,
                    'instruction_sha256': hashlib.sha256(
                        item['instruction'].encode('utf-8')).hexdigest(),
                    'run_metadata': {
                        'input_contract': 'visible-radar-sensor-plus-causal-forecast-v1',
                        'forecast_context_enabled': bool(use_forecast),
                        'region10_transition_enabled': bool(include_region10),
                        'instruction_max_chars': int(instruction_max_chars),
                        'prompt_max_length': int(prompt_max_length),
                        'legacy_rule_teacher': 'complete-replay current-state policy; comparison only',
                    },
                    'rule_move': item['legacy_rule_move'],
                })
                if use_forecast:
                    visible['forecast_context'] = item['forecast_context']
                rows.append(visible)
                delivered_decisions.append(contract_decision)
                strict_decisions.append(strict_decision)
                first_pass_decisions.append(strict_decision)
                rule_moves.append(item['legacy_rule_move'])
        # A single targeted regeneration is allowed for malformed/schema-invalid
        # first attempts. Keep both generations so retry reliability cannot be
        # confused with first-pass reliability or deterministic contract repair.
        game_start = len(rows) - len(requests)
        retry_indexes = [i for i in range(len(requests))
                         if not rows[game_start + i]['strict_parse_valid']]
        retry_outputs = {}
        if contract_retry and retry_indexes:
            contract_prefix = (
                '契约纠错重试：只输出一个可解析JSON对象，不要解释、代码围栏或多余字段。'
                'global_mode只能是进攻/防守/均衡；units必须含英雄、工程、步兵3、步兵4、空中、哨兵；'
                '每个单位必须含target、move、fire、risk四项，fire为布尔值，risk只能是高/中/低，禁止risk=无。\n')
            retry_instructions = [
                contract_prefix + requests[i]['instruction'] for i in retry_indexes]
            generated = generate_instruction_texts_with_bundle(
                tokenizer, model, retry_instructions, max_new=max_new,
                prompt_max_length=prompt_max_length)
            retry_outputs = dict(zip(retry_indexes, generated))
        for local_index, retry_raw in retry_outputs.items():
            row_index = game_start + local_index
            retry_decision, retry_error = extract_decision_with_error(retry_raw)
            row = rows[row_index]
            row['first_pass_raw_output'] = row['raw_output']
            row['first_pass_strict_parse_valid'] = row['strict_parse_valid']
            row['retry_raw_output'] = retry_raw
            row['retry_parse_error'] = retry_error
            row['retry_strict_parse_valid'] = retry_decision is not None
            row['raw_output'] = retry_raw
            row['strict_decision'] = retry_decision
            row['strict_parse_valid'] = retry_decision is not None
            row['parse_error'] = retry_error
            if retry_decision is not None:
                row['decision'] = retry_decision
                row['contract_decision'] = retry_decision
                row['contract_repairs'] = []
                row['contract_error'] = None
                row['contract_valid'] = True
                delivered_decisions[row_index] = retry_decision
                strict_decisions[row_index] = retry_decision
            else:
                retry_contract, retry_repairs, retry_contract_error = (
                    extract_decision_with_repair(retry_raw))
                if retry_contract is not None:
                    row['decision'] = retry_contract
                    row['contract_decision'] = retry_contract
                    row['contract_repairs'] = retry_repairs
                    row['contract_error'] = retry_contract_error
                    row['contract_valid'] = True
                    delivered_decisions[row_index] = retry_contract
                else:
                    row['decision'] = None
                    row['contract_decision'] = None
                    row['contract_repairs'] = []
                    row['contract_error'] = retry_contract_error
                    row['contract_valid'] = False
                    delivered_decisions[row_index] = None
        print(f'[game {game_id}] evaluated {len(requests)} visible decision points', flush=True)
    result = _score(delivered_decisions, rule_moves)
    strict_result = _score(first_pass_decisions, rule_moves)
    final_strict_result = _score(strict_decisions, rule_moves)
    result.update({
        'runtime_delivery_rate': result['parse_rate'],
        'raw_parsed': strict_result['parsed'],
        'raw_parse_rate': strict_result['parse_rate'],
        'raw_legal_rate': strict_result['legal_rate'],
        'contract_valid_rate': sum(bool(row.get('contract_valid')) for row in rows) / max(1, len(rows)),
        'contract_repair_rate': sum(bool(row.get('contract_repairs')) for row in rows) / max(1, len(rows)),
        'contract_repaired_rows': sum(bool(row.get('contract_repairs')) for row in rows),
        'unrecoverable_rows': sum(not bool(row.get('contract_valid')) for row in rows),
        'first_pass_raw_parsed': strict_result['parsed'],
        'first_pass_raw_parse_rate': strict_result['parse_rate'],
        'post_retry_raw_parsed': final_strict_result['parsed'],
        'post_retry_raw_parse_rate': final_strict_result['parse_rate'],
        'contract_retry_rows': sum('retry_raw_output' in row for row in rows),
        'contract_retry_successes': sum(bool(row.get('retry_strict_parse_valid')) for row in rows),
    })
    result.update({
        'adapter': os.path.abspath(adapter), 'side': side,
        'forecast_context_enabled': bool(use_forecast),
        'region10_transition_enabled': bool(include_region10),
        'visible_rows_validated': sum(
            bool(row.get('visibility_policy') == 'rmuc_2026_referee_radar_protocol_v1')
            for row in rows),
        'rows_output': os.path.abspath(rows_output) if rows_output else None,
        'interpretation': ('Structural output evaluation plus observational row artifacts only; '
                           'legacy rule agreement is not a deployable-policy or causal utility claim.'),
    })
    if rows_output:
        destination = Path(rows_output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text('\n'.join(
            json.dumps(row, ensure_ascii=False, separators=(',', ':')) for row in rows
        ) + ('\n' if rows else ''), encoding='utf-8')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--adapter', required=True)
    parser.add_argument('--games', nargs='+', required=True, type=int)
    parser.add_argument('--limit', type=int, default=10,
                        help='points per game; 0 keeps all points')
    parser.add_argument('--sampling', choices=('uniform', 'head'), default='uniform')
    parser.add_argument('--side', default='红', choices=('红', '蓝'))
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--temporal-checkpoint', default='auto')
    parser.add_argument('--without-forecast-context', action='store_true')
    parser.add_argument('--disable-region10-transition', action='store_true')
    parser.add_argument('--max-new', type=int, default=200)
    parser.add_argument('--prompt-max-length', type=int, default=896)
    parser.add_argument('--instruction-max-chars', type=int, default=640)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--rows-output', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args(argv)

    games = _load_games(args.games, args.limit, sampling=args.sampling)
    report = evaluate(
        args.adapter, games, side=args.side, device=args.device,
        checkpoint=args.temporal_checkpoint,
        use_forecast=not args.without_forecast_context,
        include_region10=not args.disable_region10_transition,
        max_new=args.max_new, prompt_max_length=args.prompt_max_length,
        instruction_max_chars=args.instruction_max_chars,
        batch_size=args.batch_size, rows_output=args.rows_output)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({
        'schema_version': 'visible-lora-forecast-context-eval-v1',
        'games': [game_id for game_id, _dps, _bounds in games],
        'limit_per_game': int(args.limit), 'result': report,
        'sampling_policy': args.sampling,
        'sampled_issued_at_s': {
            str(game_id): [float(dp.get('t', 0.0)) for dp in dps]
            for game_id, dps, _bounds in games
        },
    }, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
