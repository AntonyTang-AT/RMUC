# -*- coding: utf-8 -*-
"""单点 LoRA 分层决策推理。torch/transformers/peft 惰性 import；模块级零 torch。

供 runtime Tier2 与 eval 复用。无 GPU 时不要调用 load_lora_bundle。
"""
from __future__ import annotations

import json
import os
from copy import deepcopy

from schema import layered as sl


_UNIT_ALIASES = {
    '工兵团': '工程',
    '工程兵': '工程',
}

# ``风险=无`` appears in otherwise complete generations when a unit has no
# target.  The closed DecisionRequest contract intentionally represents that
# state as the conservative lowest risk, not as a fourth risk enum.  Keep this
# conversion narrow and auditable; strict parsing still reports the raw output
# as schema-invalid.
_RISK_ALIASES = {
    '无': '低',
}


def repair_decision_contract(decision):
    """Apply only deterministic contract repairs; return (decision, repairs)."""
    if not isinstance(decision, dict):
        return None, []
    repaired = deepcopy(decision)
    units = repaired.get('units')
    if not isinstance(units, dict):
        return None, []
    repairs = []
    normalized = {}
    for raw_name, action in units.items():
        name = _UNIT_ALIASES.get(str(raw_name), str(raw_name))
        if name != raw_name:
            repairs.append(f'unit_alias:{raw_name}->{name}')
        if name not in sl.UNIT_CLASSES:
            return None, []
        if name in normalized:
            return None, []
        if not isinstance(action, dict):
            return None, []
        normalized_action = dict(action)
        defaults = {
            'target': '无',
            'move': '无',
            'fire': False,
            'risk': '低',
        }
        for field, default in defaults.items():
            if field not in normalized_action:
                normalized_action[field] = default
                repairs.append(f'missing_field_default:{name}.{field}')
        raw_risk = normalized_action.get('risk')
        if raw_risk in _RISK_ALIASES:
            normalized_action['risk'] = _RISK_ALIASES[raw_risk]
            repairs.append(f'enum_alias:{name}.risk:{raw_risk}->{normalized_action["risk"]}')
        normalized[name] = normalized_action
    idle = {'target': '无', 'move': '无', 'fire': False, 'risk': '低'}
    for name in sl.UNIT_CLASSES:
        if name not in normalized:
            normalized[name] = dict(idle)
            repairs.append(f'missing_unit_idle:{name}')
    repaired['units'] = normalized
    if not sl.validate(repaired)[0]:
        return None, []
    return repaired, repairs


def extract_decision_with_error(text):
    """Extract one schema-valid decision and retain a failure category otherwise."""
    if not text or not str(text).strip():
        return None, 'empty_output'
    decoder = json.JSONDecoder()
    # Do not greedily capture from the first ``{`` to the last ``}``.  Model
    # explanations, markdown fences, and a second example can otherwise make
    # an otherwise valid decision unparsable.  Each JSON object is decoded
    # structurally and accepted only after the closed decision schema passes.
    saw_json_start = False
    saw_decodable_object = False
    for start, char in enumerate(str(text)):
        if char != '{':
            continue
        saw_json_start = True
        try:
            decision, _end = decoder.raw_decode(str(text)[start:])
        except (json.JSONDecodeError, TypeError, ValueError):
            continue
        if not isinstance(decision, dict):
            continue
        saw_decodable_object = True
        # Causal generation frequently inserts a leading/trailing space inside
        # enum strings (for example ``"move": " 游击"``).  This is a formatting
        # artifact, not a semantic disagreement.  Normalize only the closed
        # schema fields before validation; never alter free-form text because
        # the decision schema intentionally contains no free-form action fields.
        for key in ('global_mode',):
            if isinstance(decision.get(key), str):
                decision[key] = decision[key].strip()
        units = decision.get('units')
        if isinstance(units, dict):
            for entry in units.values():
                if not isinstance(entry, dict):
                    continue
                for key in ('target', 'move', 'risk'):
                    if isinstance(entry.get(key), str):
                        entry[key] = entry[key].strip()
        if sl.validate(decision)[0]:
            return decision, None
    if not saw_json_start:
        return None, 'no_json_object'
    if saw_decodable_object:
        return None, 'schema_invalid'
    return None, 'malformed_or_truncated_json'


def extract_decision(text):
    """从生成文本抽合法分层决策；失败返回 None。"""
    return extract_decision_with_error(text)[0]


def extract_decision_with_repair(text):
    """Parse a decision and apply the narrow runtime contract repair policy."""
    decision, error = extract_decision_with_error(text)
    if decision is not None:
        return decision, [], None
    if not text or not str(text).strip():
        return None, [], 'empty_output'
    decoder = json.JSONDecoder()
    for start, char in enumerate(str(text)):
        if char != '{':
            continue
        try:
            candidate, _end = decoder.raw_decode(str(text)[start:])
        except (json.JSONDecodeError, TypeError, ValueError):
            continue
        if not isinstance(candidate, dict):
            continue
        if isinstance(candidate.get('global_mode'), str):
            candidate['global_mode'] = candidate['global_mode'].strip()
        if isinstance(candidate.get('units'), dict):
            for action in candidate['units'].values():
                if isinstance(action, dict):
                    for key in ('target', 'move', 'risk'):
                        if isinstance(action.get(key), str):
                            action[key] = action[key].strip()
        repaired, repairs = repair_decision_contract(candidate)
        if repaired is not None and repairs:
            return repaired, repairs, 'schema_repaired'
    return None, [], error or 'schema_invalid'


def load_lora_bundle(adapter_path, base=None, device='cuda'):
    """加载 tokenizer + Peft 模型。仅在有卡路径调用。"""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel
    import torch
    from finetune.lora_train import LORA_BASE

    meta_path = os.path.join(os.path.dirname(adapter_path), 'meta.json')
    base = base or LORA_BASE
    if os.path.exists(meta_path):
        with open(meta_path, encoding='utf-8') as f:
            meta = json.load(f)
        base = meta.get('base', base)

    # The adapter directory was exported by an older transformers release
    # whose tokenizer_config stores extra_special_tokens as a list. Newer
    # releases expect a mapping and fail before model loading. The adapter
    # does not change the tokenizer vocabulary, so load the matching base
    # tokenizer instead; this also keeps inference reproducible after a
    # transformers upgrade.
    tok = AutoTokenizer.from_pretrained(base, local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if str(device).startswith('cuda'):
        _idx = int(device.split(':')[1]) if ':' in str(device) else torch.cuda.current_device()
        device_map = {'': _idx}
    else:
        device_map = {'': device}
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        base, quantization_config=bnb, device_map=device_map)
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return tok, model


def generate_with_bundle(tok, model, dp, bounds, max_new=200, forecast_context=None,
                         prompt_max_length=896):
    """已加载 bundle 上生成一点；非法/失败返回 None。"""
    import torch
    from worldstate import worldstate as ws
    from finetune.lora_train import eval_prompt

    w = ws.build_worldstate(dp, bounds)
    # The forecast context now contains current visible facts plus model vs
    # baseline comparisons.  576 tokens routinely truncated the 30/60/120s
    # heads and made Tier2 reason from an incomplete horizon set.  Keep the
    # bound explicit so callers on smaller GPUs can lower it deliberately.
    enc = tok(eval_prompt(w, forecast_context=forecast_context), return_tensors='pt',
              truncation=True, max_length=int(prompt_max_length))
    input_ids = enc.input_ids.to(model.device)
    attn = enc.attention_mask.to(model.device)
    with torch.no_grad():
        out = model.generate(
            input_ids, attention_mask=attn, max_new_tokens=max_new,
            do_sample=False, pad_token_id=tok.eos_token_id,
            eos_token_id=tok.eos_token_id, use_cache=True)
    text = tok.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True)
    return extract_decision(text)


def generate_batch_with_bundle(tok, model, requests, max_new=200,
                               prompt_max_length=896):
    """Generate several causal decisions in one padded batch.

    ``requests`` contains ``(dp, bounds, forecast_context)`` tuples.  Left
    padding is temporary and restored so callers sharing a tokenizer retain
    its original configuration.  A batch failure falls back to the proven
    single-row path, keeping evaluation resumable on older transformers or
    models with unusual generation constraints.
    """
    import torch
    from worldstate import worldstate as ws
    from finetune.lora_train import eval_prompt

    if not requests:
        return []
    prompts = []
    for dp, _bounds, forecast_context in requests:
        w = ws.build_worldstate(dp, _bounds)
        prompts.append(eval_prompt(w, forecast_context=forecast_context))
    previous_padding_side = getattr(tok, 'padding_side', 'right')
    try:
        tok.padding_side = 'left'
        enc = tok(prompts, return_tensors='pt', truncation=True,
                  max_length=int(prompt_max_length), padding=True)
        input_ids = enc.input_ids.to(model.device)
        attention_mask = enc.attention_mask.to(model.device)
        with torch.no_grad():
            out = model.generate(
                input_ids, attention_mask=attention_mask,
                max_new_tokens=int(max_new), do_sample=False,
                pad_token_id=tok.eos_token_id, eos_token_id=tok.eos_token_id,
                use_cache=True)
        start = int(input_ids.shape[1])
        return [extract_decision(tok.decode(row[start:], skip_special_tokens=True))
                for row in out]
    except Exception:
        return [generate_with_bundle(
            tok, model, dp, bounds, max_new=max_new,
            forecast_context=forecast_context,
            prompt_max_length=prompt_max_length)
                for dp, bounds, forecast_context in requests]
    finally:
        tok.padding_side = previous_padding_side


def generate_instructions_with_bundle(tok, model, instructions, max_new=200,
                                      prompt_max_length=896):
    """Generate decisions from prevalidated causal instructions in one batch.

    The regular helper accepts full replay decision points for compatibility.
    This companion is intentionally prompt-only so the visible-input path
    cannot accidentally reconstruct an omniscient world state during model
    evaluation or runtime integration.
    """
    return [item['decision'] for item in generate_instruction_results_with_bundle(
        tok, model, instructions, max_new=max_new,
        prompt_max_length=prompt_max_length)]


def generate_instruction_results_with_bundle(tok, model, instructions, max_new=200,
                                             prompt_max_length=896, contract_retry=True):
    """Generate visible-input decisions with strict and repair provenance.

    Invalid first passes get one explicit JSON-contract retry.  The first-pass
    status is retained so callers can measure raw reliability separately from
    runtime delivery reliability.
    """
    outputs = generate_instruction_texts_with_bundle(
        tok, model, instructions, max_new=max_new,
        prompt_max_length=prompt_max_length)
    result = []
    retry_indexes = []
    for index, text in enumerate(outputs):
        strict_decision, parse_error = extract_decision_with_error(text)
        decision, repairs, contract_error = extract_decision_with_repair(text)
        result.append({
            'decision': decision,
            'strict_parse_valid': strict_decision is not None,
            'parse_error': parse_error,
            'contract_repairs': repairs,
            'contract_error': contract_error,
            'first_pass_strict_parse_valid': strict_decision is not None,
            'first_pass_parse_error': parse_error,
            'first_pass_raw_output': text,
        })
        if contract_retry and strict_decision is None:
            retry_indexes.append(index)
    if retry_indexes:
        prefix = (
            '契约纠错重试：只输出一个可解析JSON对象，不要解释、代码围栏或多余字段。'
            'global_mode只能是进攻/防守/均衡；units必须含英雄、工程、步兵3、步兵4、空中、哨兵；'
            '每个单位必须含target、move、fire、risk四项，fire为布尔值，risk只能是高/中/低，禁止risk=无。\n')
        retry_outputs = generate_instruction_texts_with_bundle(
            tok, model, [prefix + str(instructions[i]) for i in retry_indexes],
            max_new=max_new, prompt_max_length=prompt_max_length)
        for index, retry_text in zip(retry_indexes, retry_outputs):
            retry_decision, retry_error = extract_decision_with_error(retry_text)
            retry_contract, retry_repairs, retry_contract_error = (
                extract_decision_with_repair(retry_text))
            item = result[index]
            item.update({
                'retry_raw_output': retry_text,
                'retry_strict_parse_valid': retry_decision is not None,
                'retry_parse_error': retry_error,
            })
            if retry_decision is not None:
                item.update({
                    'decision': retry_decision,
                    'strict_parse_valid': True,
                    'parse_error': None,
                    'contract_repairs': [],
                    'contract_error': None,
                })
            else:
                item.update({
                    'decision': retry_contract,
                    'strict_parse_valid': False,
                    'parse_error': retry_error,
                    'contract_repairs': retry_repairs,
                    'contract_error': retry_contract_error,
                })
    return result


def generate_instruction_texts_with_bundle(tok, model, instructions, max_new=200,
                                           prompt_max_length=896):
    """Generate prompt-only completions and return raw decoded text.

    Evaluation callers need the text even when schema extraction fails.  Keep
    this separate from the runtime helper so production paths remain
    fail-closed and continue to receive parsed decisions only.
    """
    import torch
    from finetune.lora_train import prompt_from_instruction

    if not instructions:
        return []
    prompts = [prompt_from_instruction(str(instruction)) for instruction in instructions]
    previous_padding_side = getattr(tok, 'padding_side', 'right')
    try:
        tok.padding_side = 'left'
        enc = tok(prompts, return_tensors='pt', truncation=True,
                  max_length=int(prompt_max_length), padding=True)
        input_ids = enc.input_ids.to(model.device)
        attention_mask = enc.attention_mask.to(model.device)
        with torch.no_grad():
            out = model.generate(
                input_ids, attention_mask=attention_mask,
                max_new_tokens=int(max_new), do_sample=False,
                pad_token_id=tok.eos_token_id, eos_token_id=tok.eos_token_id,
                use_cache=True)
        start = int(input_ids.shape[1])
        return [tok.decode(row[start:], skip_special_tokens=True) for row in out]
    finally:
        tok.padding_side = previous_padding_side


def generate_layered_decision(adapter_path, dp, bounds, base=None, device='cuda', max_new=200,
                              forecast_context=None, prompt_max_length=896):
    """一次性加载并生成单点决策（较慢；Prefer LazyLoraTier2 常驻）。失败返回 None。"""
    tok, model = load_lora_bundle(adapter_path, base=base, device=device)
    return generate_with_bundle(tok, model, dp, bounds, max_new=max_new,
                                forecast_context=forecast_context,
                                prompt_max_length=prompt_max_length)
