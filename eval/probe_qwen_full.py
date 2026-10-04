"""Efficient full RM holdout probe for the Qwen LoRA checkpoint.

The original diagnostic scored one sample at a time.  This runner batches all
candidate rows belonging to several samples, keeping the exact same prompt and
candidate-token scoring contract while making a 128/512-row test practical on
one GPU.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from forecast.temporal import HORIZONS, MODES, target_labels  # noqa: E402
from eval.probe_open_jev import _choose, _groups, _state  # noqa: E402


def _score_batch(model, tokenizer, requests, device):
    """Score prefixes in one forward each and read the three label tokens.

    The RM labels are single tokenizer tokens.  Therefore this is exactly the
    same candidate-token likelihood as the original probe, but it avoids
    running the whole transformer three times for every question.
    """
    rows, metadata = [], []
    candidate_ids = [tokenizer(candidate, add_special_tokens=False)['input_ids']
                     for candidate in MODES]
    for sample_key, horizon, prefix in requests:
        prefix_ids = tokenizer(prefix, add_special_tokens=False)['input_ids']
        if any(len(value) != 1 for value in candidate_ids):
            raise ValueError('RM mode labels must remain single-token candidates')
        rows.append(prefix_ids)
        metadata.append((sample_key, horizon, len(prefix_ids)))
    max_len = max(len(row) for row in rows)
    input_ids = torch.full((len(rows), max_len), tokenizer.pad_token_id,
                           dtype=torch.long, device=device)
    attention = torch.zeros_like(input_ids)
    for index, row in enumerate(rows):
        offset = max_len - len(row)
        input_ids[index, offset:] = torch.tensor(row, dtype=torch.long, device=device)
        attention[index, offset:] = 1
    with torch.inference_mode():
        logits = model(input_ids=input_ids, attention_mask=attention,
                       use_cache=False, logits_to_keep=1).logits.float()
    scores = {}
    label_tokens = torch.tensor([value[0] for value in candidate_ids],
                                dtype=torch.long, device=device)
    for index, (sample_key, horizon, prefix_len) in enumerate(metadata):
        token_logp = torch.log_softmax(logits[index, -1], dim=-1)
        values = token_logp.index_select(0, label_tokens).detach().cpu().tolist()
        scores.setdefault(sample_key, {})[str(horizon)] = values
    probabilities = {}
    for sample_key, horizon_scores in scores.items():
        probabilities[sample_key] = {}
        for horizon, values in horizon_scores.items():
            values = np.asarray(values, dtype=float)
            values -= values.max()
            probs = np.exp(values)
            probabilities[sample_key][horizon] = (probs / probs.sum()).tolist()
    return probabilities


def _metrics(rows):
    result = {}
    for horizon in HORIZONS:
        probs = np.asarray([row['probabilities'][str(horizon)] for row in rows], dtype=float)
        labels = np.asarray([MODES.index(row['label'][str(horizon)]) for row in rows], dtype=int)
        one_hot = np.eye(len(MODES), dtype=float)[labels]
        result[f'global_mode@{horizon}s'] = {
            'n': int(len(labels)),
            'accuracy': float(np.mean(probs.argmax(axis=1) == labels)),
            'brier': float(np.mean(np.sum((probs - one_hot) ** 2, axis=1))),
            'log_loss': float(-np.mean(np.log(np.clip(probs[np.arange(len(labels)), labels], 1e-12, 1.0)))),
        }
    result['side_macro_accuracy'] = float(np.mean([
        np.mean([int(np.argmax(row['probabilities']['30']) == MODES.index(row['label']['30']))
                 for row in rows if row['side'] == side])
        for side in ('红', '蓝') if any(row['side'] == side for row in rows)
    ])) if rows else None
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='out_m2/forecast_dataset/test.jsonl.gz')
    parser.add_argument('--adapter', default='out_m2/finetuned_lookahead/adapter')
    parser.add_argument('--output', default='out_m2/temporal_forecast/qwen_probe_full.json')
    parser.add_argument('--max-examples', type=int, default=128)
    parser.add_argument('--context', type=int, default=6)
    parser.add_argument('--batch-examples', type=int, default=8)
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel

    meta = Path(args.adapter).parent / 'meta.json'
    base = json.loads(meta.read_text(encoding='utf-8'))['base'] if meta.is_file() else str(REPO / 'models_hf' / 'Qwen2.5-3B-Instruct')
    tokenizer = AutoTokenizer.from_pretrained(base)
    # Left padding keeps the final position at the end of each real prefix;
    # this is required when logits_to_keep=1 is used below.
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                               bnb_4bit_quant_type='nf4', bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(base, quantization_config=quant,
                                                 device_map={'': 0}).eval()
    model = PeftModel.from_pretrained(model, args.adapter).eval()
    device = next(model.parameters()).device

    selected = _choose(_groups(Path(args.input)), args.max_examples)
    rows = []
    for start in range(0, len(selected), max(1, args.batch_examples)):
        block = selected[start:start + max(1, args.batch_examples)]
        requests = []
        for gid, group_rows, index in block:
            current = group_rows[index]
            state = _state(group_rows, index, args.context)
            context = json.dumps(state, ensure_ascii=False, separators=(',', ':'))
            sample_key = f'{gid}:{current["side"]}:{current["issued_at"]}'
            for horizon in HORIZONS:
                prefix = ('You are a robotic competition tactical analyst.\n'
                          f'Predict the team\'s global tactical mode {horizon} seconds after the current time.\n'
                          'Use the causal observable history below.\nContext:\n' + context +
                          '\nAnswer with one label: 进攻, 均衡, or 防守. Answer: ')
                requests.append((sample_key, horizon, prefix))
        scored = _score_batch(model, tokenizer, requests, device)
        for gid, group_rows, index in block:
            current = group_rows[index]
            sample_key = f'{gid}:{current["side"]}:{current["issued_at"]}'
            rows.append({
                'game_id': gid, 'side': current['side'], 'issued_at': current['issued_at'],
                'probabilities': scored[sample_key],
                'label': {str(h): target_labels(current, h)['global_mode'] for h in HORIZONS},
            })
        print(f'probed {min(start + len(block), len(selected))}/{len(selected)}', flush=True)
    report = {
        'schema_version': 1,
        'model': 'Qwen2.5-3B-Instruct + finetuned_lookahead LoRA',
        'base': base, 'adapter': args.adapter,
        'input_contract': 'observable-only causal feature history',
        'selection': 'same deterministic red/blue balanced RM test selector as Open-Jev probe',
        'n_examples': len(rows), 'red_n': sum(row['side'] == '红' for row in rows),
        'blue_n': sum(row['side'] == '蓝' for row in rows),
        'metrics': _metrics(rows), 'rows': rows,
    }
    path = Path(args.output); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'output': str(path), 'n_examples': len(rows), 'metrics': report['metrics']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
