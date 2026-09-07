# -*- coding: utf-8 -*-
"""M2 留出评估: 加载 base+LoRA 适配器, 对决策点 decode 分层决策, 度量 legal/agreement/compliance/conflict。
torch/transformers/peft 惰性 import; 模块级零 torch。"""
import json
import re
from schema import layered as sl
from tree import global_mode as gm
from finetune.make_dataset import situation_digest

# 哨兵三态 -> move(用于与规则头 agreement 比对)
_SENTRY_TO_MOVE = {'def': '回防', 'hitbuilding': '推进', 'cruise': '游击'}


def _extract_decisions(text):
    m = re.search(r'\{.*\}', text, re.DOTALL)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except Exception:
        return None
    return d if sl.validate(d)[0] else None


def eval_model(adapter_path, dps, bounds, base=None, device='cuda', max_new=200, batch=None):
    """返回 {'n','legal_rate','agreement_with_rule','compliance_rate','conflict_rate'}。"""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    import torch
    from finetune.lora_train import LORA_BASE

    import os
    meta_path = os.path.join(os.path.dirname(adapter_path), 'meta.json')
    base = base or LORA_BASE
    if os.path.exists(meta_path):
        with open(meta_path, encoding='utf-8') as f:
            meta = json.load(f)
        base = meta.get('base', base)

    tok = AutoTokenizer.from_pretrained(adapter_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(base, load_in_4bit=True, device_map=device)
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()

    decs = []
    sent_moves = []
    rule_sent = []
    for dp in dps:
        from worldstate import worldstate as ws
        w = ws.build_worldstate(dp, bounds)
        input_ids = tok(situation_digest(w), return_tensors='pt').input_ids.to(model.device)
        with torch.no_grad():
            out = model.generate(input_ids, max_new_tokens=max_new, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
        text = tok.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True)
        d = _extract_decisions(text)
        if d is not None:
            decs.append(d)
            sent_moves.append(d['units'].get('哨兵', {}).get('move', '无'))
        rule_sent.append(_SENTRY_TO_MOVE.get(gm.derive_sentry_state(w), '游击'))

    n = len(dps)
    legal = sum(1 for d in decs if sl.validate(d)[0]) / max(n, 1)
    agree = sum(1 for a, b in zip(sent_moves, rule_sent) if a == b) / max(len(decs), 1)
    conf = sl.conflict_rate(decs) if len(decs) > 1 else 0.0
    return {'n': len(decs), 'legal_rate': legal,
            'agreement_with_rule': agree, 'conflict_rate': conf}
