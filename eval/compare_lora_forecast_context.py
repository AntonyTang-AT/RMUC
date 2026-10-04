"""Ablation for the forecast-aware Qwen LoRA training path.

The comparison keeps games, decision points, base model and decoding fixed:

* old adapter + legacy prompt;
* old adapter + forecast context;
* new adapter trained with forecast context.

It reports schema validity, rule agreement and decision conflict.  The
forecast context is generated causally from the visibility-masked snapshots
at the issue time; no future row is passed to the prompt.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from schema import layered as sl


def _load_games(game_ids, limit):
    import run_m2
    out = []
    for gid in game_ids:
        dps, bounds = run_m2.load_decision_points(int(gid), limit)
        if dps:
            out.append((int(gid), dps, bounds))
    return out


def _contexts_for_game(gid, dps, checkpoint, device='cpu', include_region10=True):
    import run_m2
    from runtime.temporal_forecast import resolve_temporal_forecaster
    from worldstate.forecast_context import build_forecast_context

    snapshots = run_m2.align_observable_snapshots(
        dps, run_m2.load_observable_snapshots(gid, team='红'))
    options = {}
    if not include_region10:
        # An absolute explicit path selects the existing 30s manifest without
        # triggering the resolver's default auto-load of the 10s sidecar.
        options['enemy_region_transition_path'] = str(
            Path('out_m2/enemy_region_transition/production_manifest.json').resolve())
    forecaster = resolve_temporal_forecaster(
        checkpoint, device=device, team='红', context=12,
        adaptive_router=None, auto_mature_feedback=False, **options)
    contexts = []
    for snapshot in snapshots:
        if snapshot is None:
            contexts.append(None)
        else:
            contexts.append(build_forecast_context(forecaster.predict(snapshot)))
    return contexts


def _score_decisions(decisions, rule_moves):
    pairs = [(decision, rule_move) for decision, rule_move in zip(decisions, rule_moves)
             if decision is not None]
    valid = [decision for decision, _rule_move in pairs]
    legal = sum(bool(sl.validate(d)[0]) for d in valid) / max(1, len(valid))
    agreement = sum(
        decision.get('units', {}).get('哨兵', {}).get('move', '无') == rule_move
        for decision, rule_move in pairs
    ) / max(1, len(valid))
    conflict = sl.conflict_rate(valid) if len(valid) > 1 else 0.0
    return {
        'n': len(decisions),
        'parsed': len(valid),
        'parse_rate': len(valid) / max(1, len(decisions)),
        'legal_rate': legal,
        'agreement_with_rule': agreement,
        'conflict_rate': conflict,
    }


def evaluate_adapter(adapter, games, *, base=None, device='cuda', use_context=False,
                     checkpoint='auto', max_new=200, prompt_max_length=896,
                     batch_size=1, rows_output=None, variant='unknown', resume=True,
                     include_region10=True):
    from finetune.infer_one import generate_batch_with_bundle, load_lora_bundle
    from tree import global_mode as gm
    from worldstate import worldstate as ws

    tok, model = load_lora_bundle(adapter, base=base, device=device)
    decisions = []
    rule_moves = []
    completed = set()
    if rows_output and resume and os.path.exists(rows_output):
        with open(rows_output, encoding='utf-8') as stream:
            for line in stream:
                if not line.strip():
                    continue
                row = json.loads(line)
                if str(row.get('variant')) != str(variant):
                    continue
                completed.add((str(row.get('variant')), int(row['game_id']),
                               float(row['issued_at'])))
                decisions.append(row.get('decision'))
                rule_moves.append(row.get('rule_move', '游击'))
    rows_stream = None
    if rows_output:
        Path(rows_output).parent.mkdir(parents=True, exist_ok=True)
        rows_stream = open(rows_output, 'a', encoding='utf-8')
    context_count = 0
    for gid, dps, bounds in games:
        contexts = (_contexts_for_game(gid, dps, checkpoint, device='cpu',
                                       include_region10=include_region10)
                    if use_context else [None] * len(dps))
        pending = []
        pending_meta = []
        for dp, context in zip(dps, contexts):
            issued_at = float(dp.get('t', 0.0))
            key = (str(variant), int(gid), issued_at)
            if key in completed:
                context_count += int(context is not None)
                continue
            rule_move = {'def': '回防', 'hitbuilding': '推进',
                         'cruise': '游击'}.get(
                             gm.derive_sentry_state(ws.build_worldstate(dp, bounds)),
                             '游击')
            pending.append((dp, bounds, context))
            pending_meta.append((gid, issued_at, rule_move, context))
        for start in range(0, len(pending), max(1, int(batch_size))):
            batch = pending[start:start + max(1, int(batch_size))]
            meta = pending_meta[start:start + max(1, int(batch_size))]
            batch_decisions = generate_batch_with_bundle(
                tok, model, batch, max_new=max_new,
                prompt_max_length=prompt_max_length)
            for decision, (row_gid, issued_at, rule_move, context) in zip(batch_decisions, meta):
                decisions.append(decision)
                rule_moves.append(rule_move)
                context_count += int(context is not None)
                if rows_stream:
                    row = {'schema_version': 1, 'variant': variant,
                           'game_id': int(row_gid), 'issued_at': issued_at,
                           'rule_move': rule_move, 'decision': decision,
                           'parse_valid': decision is not None}
                    rows_stream.write(json.dumps(row, ensure_ascii=False) + '\n')
                    rows_stream.flush()
                    completed.add((str(variant), int(row_gid), issued_at))
    if rows_stream:
        rows_stream.close()
    result = _score_decisions(decisions, rule_moves)
    result['adapter'] = os.path.abspath(adapter)
    result['forecast_context_enabled'] = bool(use_context)
    result['forecast_context_attached'] = int(context_count)
    result['batch_size'] = int(batch_size)
    result['max_new_tokens'] = int(max_new)
    if rows_output:
        result['rows_output'] = os.path.abspath(rows_output)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--old-adapter', required=True)
    parser.add_argument('--new-adapter', required=True)
    parser.add_argument('--games', nargs='+', required=True, type=int)
    parser.add_argument('--limit', type=int, default=20)
    parser.add_argument('--base', default=None)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--temporal-checkpoint', default='auto')
    parser.add_argument('--max-new', type=int, default=200)
    parser.add_argument('--prompt-max-length', type=int, default=896)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--rows-output', default='',
                        help='optional JSONL incremental rows; existing rows are resumed')
    parser.add_argument('--only', choices=['all', 'old_legacy', 'old_forecast', 'new_forecast'],
                        default='all')
    parser.add_argument('--disable-region10-transition', action='store_true',
                        help='use the existing 30s sidecar without the 10s transition expert')
    parser.add_argument('--no-resume', action='store_true')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    games = _load_games(args.games, args.limit)
    variants = {
        'old_adapter_legacy_prompt': (args.old_adapter, False),
        'old_adapter_forecast_prompt': (args.old_adapter, True),
        'new_adapter_forecast_prompt': (args.new_adapter, True),
    }
    selected = list(variants)
    if args.only != 'all':
        selected = [{'old_legacy': 'old_adapter_legacy_prompt',
                     'old_forecast': 'old_adapter_forecast_prompt',
                     'new_forecast': 'new_adapter_forecast_prompt'}[args.only]]
    results = {}
    for variant in selected:
        adapter, use_context = variants[variant]
        results[variant] = evaluate_adapter(
            adapter, games, base=args.base, device=args.device,
            use_context=use_context, checkpoint=args.temporal_checkpoint,
            max_new=args.max_new, prompt_max_length=args.prompt_max_length,
            batch_size=args.batch_size,
            rows_output=args.rows_output or None, variant=variant,
            resume=not args.no_resume,
            include_region10=not args.disable_region10_transition)
    report = {
        'schema_version': 'llm-forecast-context-ablation-v1',
        'games': [gid for gid, _, _ in games],
        'limit_per_game': int(args.limit),
        'prompt_max_length': int(args.prompt_max_length),
        'results': results,
        'causal_context_contract': 'visibility-masked issue-time snapshots only; no future rows',
        'region10_transition_enabled': not args.disable_region10_transition,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(results, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
