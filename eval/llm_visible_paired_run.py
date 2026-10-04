"""Small frozen-input binary sentry-action smoke for Qwen and Open-JEV.

Both models receive exactly the candidate prompt strings saved by
``llm_visible_paired_smoke`` and select the larger candidate score.  Their
heads differ (next-token Yes/No versus Open-JEV learned scalar head), so this
checks the input/evaluation pipeline, not model superiority.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

from jev.api import candidate_prompts, compile_request


def _read(path):
    with Path(path).open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _write(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')


def _assert_frozen_prompts(item):
    record = compile_request(
        {'visible_prompt': item['visible_input']['prompt_text']},
        item['questions'])[0]
    if candidate_prompts(record) != item['candidate_prompts']:
        raise ValueError('frozen candidate prompts differ from current Open-JEV compiler')
    if record['answer_keys'] != item['candidate_keys']:
        raise ValueError('candidate ordering changed')


def _finalize(item, model, probabilities, latency, settings):
    keys = item['candidate_keys']
    if len(keys) != len(probabilities) or not all(math.isfinite(v) and v >= 0 for v in probabilities):
        raise ValueError('candidate probability vector invalid')
    total = sum(probabilities)
    if total <= 0:
        raise ValueError('candidate probability vector sums to zero')
    probabilities = [float(v / total) for v in probabilities]
    choice = keys[max(range(len(keys)), key=lambda i: probabilities[i])]
    row = dict(item['visible_input'])
    row.update({
        'model': model, 'decision': {'units': {'哨兵': {
            'move': choice, 'fire': False, 'target': '无', 'risk': '中'}}},
        'parse_valid': True, 'latency_s': float(latency),
        'candidate_keys': keys, 'candidate_probabilities': probabilities,
        'candidate_prompt_sha256': item['candidate_prompt_sha256'],
        'run_metadata': settings,
    })
    return row


def run_jev(items, *, endpoint):
    from jev.client import Client
    client = Client(endpoint=endpoint, timeout=300)
    output = []
    for item in items:
        _assert_frozen_prompts(item)
        start = time.perf_counter()
        response = client.ask(
            state={'visible_prompt': item['visible_input']['prompt_text']},
            questions=item['questions'])
        latency = time.perf_counter() - start
        prob_map = response['answers']['sentry_move']['probabilities']
        probs = [float(prob_map[key]) for key in item['candidate_keys']]
        output.append(_finalize(item, 'Open-JEV-2B', probs, latency, {
            'checkpoint': 'out_m2/openjev_local_checkpoint',
            'selection': 'binary sentry candidate scorer; deterministic argmax',
            'endpoint': endpoint, 'candidate_prompt_contract': 'jev.api.candidate_prompts',
        }))
        print(f'Open-JEV {len(output)}/{len(items)} {latency:.1f}s', flush=True)
    return output


def run_qwen(items, *, adapter, base, max_length):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True)
    yes = tokenizer(' Yes', add_special_tokens=False)['input_ids']
    no = tokenizer(' No', add_special_tokens=False)['input_ids']
    if len(yes) != 1 or len(no) != 1:
        raise ValueError('Qwen Yes/No must be one token for this scorer')
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                               bnb_4bit_quant_type='nf4', bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(
        base, quantization_config=quant, device_map={'': 0},
        local_files_only=True).eval()
    model = PeftModel.from_pretrained(model, adapter, local_files_only=True).eval()
    output = []
    device = next(model.parameters()).device
    for item in items:
        _assert_frozen_prompts(item)
        start = time.perf_counter()
        correct_probs = []
        token_lengths = []
        for candidate in item['candidate_prompts']:
            encoded = tokenizer(candidate, return_tensors='pt', truncation=False)
            length = int(encoded.input_ids.shape[1])
            if length > max_length:
                raise ValueError(f'candidate prompt has {length} tokens > {max_length}; refusing truncation')
            token_lengths.append(length)
            with torch.inference_mode():
                logits = model(input_ids=encoded.input_ids.to(device),
                               attention_mask=encoded.attention_mask.to(device),
                               use_cache=False).logits[0, -1].float()
            correct_probs.append(float(torch.sigmoid(logits[yes[0]] - logits[no[0]]).cpu()))
        latency = time.perf_counter() - start
        output.append(_finalize(item, 'Qwen2.5-3B-LoRA', correct_probs, latency, {
            'base': str(Path(base).resolve()), 'adapter': str(Path(adapter).resolve()),
            'selection': 'binary sentry candidate scorer; deterministic argmax',
            'score_method': 'softmax Yes/No next-token logits, then normalize candidate Yes probabilities',
            'max_prompt_tokens': max_length, 'observed_candidate_token_lengths': token_lengths,
            'candidate_prompt_contract': 'jev.api.candidate_prompts',
        }))
        print(f'Qwen {len(output)}/{len(items)} {latency:.1f}s', flush=True)
    return output


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', default='out_m2/llm_visible_smoke/paired_inputs.jsonl')
    ap.add_argument('--model', choices=['qwen', 'jev'], required=True)
    ap.add_argument('--out', default='')
    ap.add_argument('--endpoint', default='http://127.0.0.1:8791/v1/systemone')
    ap.add_argument('--qwen-base', default='models_hf/Qwen2.5-3B-Instruct')
    ap.add_argument('--qwen-adapter', default='out_m2/finetuned_lookahead/adapter')
    ap.add_argument('--max-length', type=int, default=4096)
    args = ap.parse_args()
    items = _read(args.input)
    rows = (run_jev(items, endpoint=args.endpoint) if args.model == 'jev' else
            run_qwen(items, adapter=args.qwen_adapter,
                     base=args.qwen_base, max_length=args.max_length))
    output = args.out or f'out_m2/llm_visible_smoke/{args.model}_rows.jsonl'
    _write(output, rows)
    print(json.dumps({'output': str(Path(output).resolve()), 'rows': len(rows),
                      'model': args.model}, ensure_ascii=False))


if __name__ == '__main__':
    main()
