"""Prepare identical visible-input candidate prompts for Qwen/Open-JEV smoke.

This preparation step reads no future telemetry.  Model runners must record
their own candidate scoring settings; a two-action smoke is not a full policy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from data import db
from eval.llm_visible_input import load_visible_llm_input
from jev.api import candidate_prompts, compile_request


def prepare(rows, output):
    prepared = []
    con = db.connect()
    try:
        for game_id, side, t in rows:
            visible = load_visible_llm_input(
                con, game_id=game_id, side=side, issued_at_s=t,
                history_frames=4)
            questions = {'sentry_move': {
                'type': 'choice',
                'instructions': '仅根据当前及过去可见记录，选择我方哨兵未来3秒更适合的动作。',
                'criteria': {
                    '推进': '向前压进或提升前线压力',
                    '坚守': '留在当前区域并保持掩护',
                },
            }}
            records = compile_request({'visible_prompt': visible['prompt_text']}, questions)
            prompts = candidate_prompts(records[0])
            prepared.append({
                'visible_input': visible, 'questions': questions,
                'candidate_keys': records[0]['answer_keys'],
                'candidate_prompts': prompts,
                'candidate_prompt_sha256': [hashlib.sha256(p.encode()).hexdigest()
                                            for p in prompts],
            })
    finally:
        con.close()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', encoding='utf-8') as stream:
        for item in prepared:
            stream.write(json.dumps(item, ensure_ascii=False, allow_nan=False) + '\n')
    return prepared


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default='out_m2/llm_visible_smoke/paired_inputs.jsonl')
    args = parser.parse_args()
    selection = [(gid, side, t)
                 for gid in (1778668127515, 1778668980465)
                 for side in ('红', '蓝') for t in (60.0, 180.0)]
    rows = prepare(selection, args.out)
    print(json.dumps({'output': str(Path(args.out).resolve()), 'n': len(rows),
                      'red_n': 4, 'blue_n': 4,
                      'action_candidates': ['推进', '坚守']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
