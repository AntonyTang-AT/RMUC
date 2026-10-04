"""Build side-correct, visibility-safe LoRA training records.

This is intentionally separate from the earlier v1 builder.  That builder
could safely replace a red prompt with a blue visible view, but retained the
red completion and red forecast sidecar.  Such a pairing is not valid
supervision for blue-side inference.  Here each side gets its own causal
forecast stream and its own short-window historical teacher completion.

Every source decision produces two records: one with the forecast reference
and one without it.  The second variant teaches the adapter to retain the
strict JSON contract if the optional forecaster is unavailable at runtime.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SIDES = ('红', '蓝')
VARIANTS = ('forecast', 'no_forecast')
from schema import layered as sl


def _completion(decision):
    return json.dumps(decision, ensure_ascii=False, separators=(',', ':'))


def build_side_teacher_decisions(dps, bounds, *, side, lookahead_delta=5,
                                 margin=None):
    """Produce one historical label per point for ``side``.

    These are supervised labels only.  Future frames participate exclusively
    in the offline look-ahead label construction; prompt construction remains
    strictly causal and visibility-masked.
    """
    if side not in SIDES:
        raise ValueError(f'unsupported side: {side}')
    from labels import strategy_label as labels
    from schema import layered as sl
    from tree import global_mode as gm
    from worldstate import worldstate as ws

    ws_list = [ws.build_worldstate(dp, bounds) for dp in dps]
    labels_out = labels.strategy_series(
        ws_list, int(lookahead_delta), team=side,
        margin=labels.DEF_MARGIN if margin is None else float(margin))
    decisions = []
    for world, mode in zip(ws_list, labels_out):
        decision = sl.build_empty_decision(world['t'])
        decision['global_mode'] = mode
        for cls in sl.UNIT_CLASSES:
            unit = next((row for row in world.get('robots', {}).values()
                         if row.get('team') == side and row.get('type') == cls), None)
            if unit is not None:
                decision['units'][cls] = gm.derive_unit_intent(
                    unit, world, mode, team=side)
        valid, errors = sl.validate(decision)
        if not valid:
            raise ValueError(f'invalid {side} teacher decision: {errors}')
        decisions.append(decision)
    return decisions, labels_out


def _forecast_contexts(game_id, dps, *, side, checkpoint, device):
    import run_m2
    from runtime.temporal_forecast import resolve_temporal_forecaster
    from worldstate.forecast_context import build_forecast_context

    snapshots = run_m2.load_observable_snapshots(int(game_id), team=side)
    forecaster = resolve_temporal_forecaster(
        checkpoint, device=device, team=side, context=12,
        adaptive_router=None, auto_mature_feedback=False)
    if forecaster is None:
        raise ValueError('a temporal checkpoint is required for forecast variants')
    predictions = run_m2.precompute_temporal_forecasts(dps, snapshots, forecaster)
    contexts = [build_forecast_context(item) if item else None for item in predictions]
    if any(item is None for item in contexts):
        missing = sum(item is None for item in contexts)
        raise ValueError(f'game {game_id}/{side}: {missing} missing causal forecasts')
    return snapshots, contexts


def build_game_records(game_id, *, sides=SIDES, checkpoint='auto', device='cpu',
                       max_chars=640, lookahead_delta=5, margin=None,
                       blue_hero_unseen_every=5):
    """Build matched forecast/no-forecast records for one historical game."""
    import run_m2
    from eval.llm_visible_input import (build_visible_llm_input,
                                        format_visible_training_instruction,
                                        mask_own_unit_slot,
                                        own_unit_slot_states,
                                        project_decision_to_observable_contract,
                                        validate_visible_llm_input)

    dps, bounds = run_m2.load_decision_points(int(game_id))
    if not dps:
        raise ValueError(f'game {game_id} has no decision points')
    output = []
    for side in sides:
        teachers, modes = build_side_teacher_decisions(
            dps, bounds, side=side, lookahead_delta=lookahead_delta, margin=margin)
        snapshots, contexts = _forecast_contexts(
            game_id, dps, side=side, checkpoint=checkpoint, device=device)
        if len(teachers) != len(contexts):
            raise ValueError(f'game {game_id}/{side}: teacher/context length mismatch')
        for index, (dp, teacher, context, mode) in enumerate(
                zip(dps, teachers, contexts, modes)):
            issued_at = float(dp.get('t', 0.0))
            visible = build_visible_llm_input(
                snapshots, game_id=int(game_id), side=side,
                issued_at_s=issued_at, history_frames=4)
            if not validate_visible_llm_input(visible):
                raise ValueError(f'game {game_id}/{side}/{index}: visibility audit failed')
            views = [('observed', visible)]
            slots = own_unit_slot_states(visible)
            if (side == '蓝' and int(blue_hero_unseen_every) > 0
                    and index % int(blue_hero_unseen_every) == 0
                    and slots['英雄']['status'] == 'current'):
                views.append(('blue_hero_unseen', mask_own_unit_slot(visible, '英雄')))
            for augmentation, visible_view in views:
                projected_teacher = project_decision_to_observable_contract(teacher, visible_view)
                if not sl.validate(projected_teacher)[0]:
                    raise ValueError(f'game {game_id}/{side}/{index}: observable teacher invalid')
                adjusted_slots = [
                    unit_type for unit_type in sl.UNIT_CLASSES
                    if projected_teacher['units'][unit_type] != teacher['units'][unit_type]
                ]
                view_slots = own_unit_slot_states(visible_view)
                for variant, forecast_context in (
                        ('forecast', context), ('no_forecast', None)):
                    instruction = format_visible_training_instruction(
                        visible_view, forecast_context=forecast_context, max_chars=max_chars)
                    if variant == 'forecast' and '预测参考(非事实)' not in instruction:
                        raise ValueError(f'game {game_id}/{side}/{index}: forecast prompt lost reference')
                    if variant == 'no_forecast' and '预测参考' in instruction:
                        raise ValueError(f'game {game_id}/{side}/{index}: no-forecast prompt contaminated')
                    output.append({
                        'instruction': instruction,
                        'completion': _completion(projected_teacher),
                        'provenance': {
                            'game_id': int(game_id), 'issued_at_s': issued_at,
                            'perspective': side, 'variant': variant,
                            'teacher': 'side-specific-lookahead-observable-v2',
                            'teacher_mode': mode,
                            'lookahead_delta': int(lookahead_delta),
                            'visibility_policy': visible_view['visibility_policy'],
                            'visible_input_sha256': visible_view['input_sha256'],
                            'slot_statuses': {name: value['status']
                                              for name, value in view_slots.items()},
                            'visibility_adjusted_slots': adjusted_slots,
                            'augmentation': augmentation,
                            'forecast_context_sha256': (
                                hashlib.sha256(json.dumps(
                                    forecast_context, ensure_ascii=False, sort_keys=True,
                                    separators=(',', ':')).encode('utf-8')).hexdigest()
                                if forecast_context else None),
                        },
                    })
    return output


def build_dataset(game_ids, *, checkpoint='auto', device='cpu', max_chars=640,
                  lookahead_delta=5, margin=None, seed=42,
                  blue_hero_unseen_every=5):
    records = []
    for game_id in game_ids:
        rows = build_game_records(
            int(game_id), checkpoint=checkpoint, device=device,
            max_chars=max_chars, lookahead_delta=lookahead_delta, margin=margin,
            blue_hero_unseen_every=blue_hero_unseen_every)
        records.extend(rows)
        print(json.dumps({'game_id': int(game_id), 'records': len(rows)},
                         ensure_ascii=False), flush=True)
    random.Random(int(seed)).shuffle(records)
    return records


def _read_game_ids(path):
    payload = json.loads(Path(path).read_text(encoding='utf-8'))
    return [int(value) for value in payload['game_ids']]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--game-manifest',
                        default='out_m2/temporal_visible_context_dataset_v3/manifest.json')
    parser.add_argument('--out-dir', default='out_m2/temporal_visible_context_dataset_v4')
    parser.add_argument('--temporal-checkpoint', default='auto')
    parser.add_argument('--forecast-device', default='cpu')
    parser.add_argument('--max-chars', type=int, default=640)
    parser.add_argument('--lookahead-delta', type=int, default=5)
    parser.add_argument('--margin', type=float, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--blue-hero-unseen-every', type=int, default=5,
                        help='add one blue-hero-unseen view every N issue points; 0 disables')
    args = parser.parse_args(argv)

    game_ids = _read_game_ids(ROOT / args.game_manifest)
    records = build_dataset(
        game_ids, checkpoint=args.temporal_checkpoint, device=args.forecast_device,
        max_chars=args.max_chars, lookahead_delta=args.lookahead_delta,
        margin=args.margin, seed=args.seed,
        blue_hero_unseen_every=args.blue_hero_unseen_every)
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = out_dir / 'm2_dataset.jsonl'
    dataset_path.write_text(
        ''.join(json.dumps(row, ensure_ascii=False, separators=(',', ':')) + '\n'
                for row in records), encoding='utf-8')
    variant_counts = Counter(row['provenance']['variant'] for row in records)
    side_counts = Counter(row['provenance']['perspective'] for row in records)
    mode_counts = Counter(row['provenance']['teacher_mode'] for row in records)
    augmentation_counts = Counter(row['provenance'].get('augmentation', 'observed')
                                  for row in records)
    unavailable_slot_counts = Counter(
        unit_type for row in records
        for unit_type, status in (row['provenance'].get('slot_statuses') or {}).items()
        if status != 'current')
    visible_hashes = [row['provenance']['visible_input_sha256'] for row in records]
    manifest = {
        'schema_version': 'visible-temporal-context-dataset-v5-observable-slots',
        'rows': len(records), 'game_ids': game_ids,
        'perspectives': list(SIDES), 'variants': list(VARIANTS),
        'variant_counts': dict(variant_counts), 'side_counts': dict(side_counts),
        'teacher_mode_counts': dict(mode_counts),
        'visibility_policy': 'rmuc_2026_referee_radar_protocol_v1',
        'all_visible_inputs_validated': True,
        'teacher_contract': 'side-specific offline look-ahead labels; never sent in prompts',
        'unit_label_contract': ('fixed six output slots; unseen or destroyed own slots receive '
                                'a deterministic conservative action; unobserved enemy robot '
                                'targets are suppressed'),
        'augmentation_counts': dict(augmentation_counts),
        'unavailable_slot_counts': dict(unavailable_slot_counts),
        'blue_hero_unseen_every': int(args.blue_hero_unseen_every),
        'forecast_contract': 'side-specific causal issue-time forecast-context-v1; reference_only',
        'prompt_max_chars': int(args.max_chars),
        'lookahead_delta': int(args.lookahead_delta), 'margin': args.margin,
        'visible_input_set_sha256': hashlib.sha256(
            '\n'.join(visible_hashes).encode('utf-8')).hexdigest(),
    }
    (out_dir / 'manifest.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'dataset': str(dataset_path), 'manifest': str(out_dir / 'manifest.json'),
                      'rows': len(records), 'games': len(game_ids),
                      'variant_counts': dict(variant_counts),
                      'side_counts': dict(side_counts)}, ensure_ascii=False), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
